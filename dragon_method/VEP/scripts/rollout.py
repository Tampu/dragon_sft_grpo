"""Backend-agnostic rollout interface.

    RolloutEngine
    ├── HFRolloutEngine     reference backend: InternVL HF generate(), exactly GRPO-R's call
    └── VLLMRolloutEngine   fast backend: vLLM V1, colocated on the trainer's GPU, LoRA hot-swap

Both return Rollout objects; parsing, Phi, rewards, advantages and the update are backend-independent.
Text is always decoded by the caller's HF tokenizer (skip_special_tokens=False), as in the R1 eval path.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from common import EOS_TOKEN, expand_image, hf_pixel_values, load_pil, prompt_text

TORCHRUN_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "GROUP_WORLD_SIZE",
                "TORCHELASTIC_USE_AGENT_STORE", "TORCH_NCCL_ASYNC_ERROR_HANDLING")


@dataclass
class Rollout:
    prompt_id: str
    token_ids: list
    text: str
    segments: object = None
    commit_answer: str = None
    finish_reason: str = ""
    logprobs: list = None
    prompt_token_ids: list = None


@dataclass
class GenConfig:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 640
    ngram: int = 6
    commit_stop: bool = False
    logprobs: bool = False
    seed: int = None
    extra: dict = field(default_factory=dict)


def completion_ids(ids, finish_reason, eos):
    """Up to and including the first EOS (GRPO-R's cleanup); EOS appended if a stop dropped it."""
    ids = list(ids)
    if eos in ids:
        return ids[:ids.index(eos) + 1]
    return ids + [eos] if finish_reason == "stop" else ids


def decode_response(tok, ids):
    return tok.decode(list(ids), skip_special_tokens=False).split(EOS_TOKEN)[0].strip()


class RolloutEngine:
    def generate(self, items, mode, cfg: GenConfig):
        raise NotImplementedError

    def set_lora(self, path):
        raise NotImplementedError


class VLLMRolloutEngine(RolloutEngine):
    def __init__(self, model_dir, tok, gpu_mem=0.25, max_model_len=6144, enable_lora=True, max_lora_rank=16,
                 seed=17, max_num_seqs=256, mm_cache_gb=4, enforce_eager=False, limit_images=1):
        from vllm import LLM
        from vep_logitsproc import VEPLogitsProcessor
        saved = {k: os.environ.pop(k) for k in TORCHRUN_ENV if k in os.environ}
        try:
            self.llm = LLM(model=str(model_dir), tokenizer=str(model_dir), trust_remote_code=True,
                           dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_mem,
                           max_model_len=max_model_len, enable_lora=enable_lora,
                           max_lora_rank=max_lora_rank, max_loras=1, max_cpu_loras=2,
                           enable_prefix_caching=True, limit_mm_per_prompt={"image": limit_images},
                           mm_processor_cache_gb=mm_cache_gb, logits_processors=[VEPLogitsProcessor],
                           max_num_seqs=max_num_seqs, enforce_eager=enforce_eager)
        finally:
            os.environ.update(saved)
        self.tok = tok
        self.eos = tok.convert_tokens_to_ids(EOS_TOKEN)
        self.lora = None
        self._lid = 0
        self._io = ThreadPoolExecutor(16)

    def set_lora(self, path, name=None):
        from vllm.lora.request import LoRARequest
        if path is None:
            self.lora = None
            return None
        self._lid += 1
        self.lora = LoRARequest(name or f"vep{self._lid}", self._lid, str(path))
        return self.lora

    def generate(self, items, mode, cfg: GenConfig, chunk=512):
        from vllm import SamplingParams
        sp = SamplingParams(n=cfg.n, temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
                            seed=cfg.seed, stop_token_ids=[self.eos], skip_special_tokens=False,
                            detokenize=False, logprobs=0 if cfg.logprobs else None,
                            extra_args={"vep_ngram": cfg.ngram, "vep_commit_stop": cfg.commit_stop})
        res = []
        for c0 in range(0, len(items), chunk):
            part = items[c0:c0 + chunk]
            imgs = list(self._io.map(lambda it: load_pil(it["image"]), part))
            prompts = [{"prompt": prompt_text(it, mode), "multi_modal_data": {"image": im}}
                       for it, im in zip(part, imgs)]
            outs = self.llm.generate(prompts, sp, lora_request=self.lora, use_tqdm=False)
            for it, o in zip(part, outs):
                group = []
                for c in o.outputs:
                    lps = None
                    if cfg.logprobs and c.logprobs is not None:
                        lps = [c.logprobs[j][t].logprob for j, t in enumerate(c.token_ids)]
                    ids = list(c.token_ids)
                    group.append(Rollout(it["id"], ids, decode_response(self.tok, ids),
                                         finish_reason=c.finish_reason, logprobs=lps,
                                         prompt_token_ids=list(o.prompt_token_ids)))
                res.append(group)
        return res


class _CommitStop:
    """HF StoppingCriteria: finish a row as soon as the <COMMIT> marker has been emitted."""

    def __init__(self, tok):
        self.tok = tok
        strs = tok.batch_decode([[i] for i in range(len(tok))])
        self.nl = {i for i, s in enumerate(strs) if "\n" in s}
        self.gt = {i for i, s in enumerate(strs) if ">" in s}

    def __call__(self, input_ids, scores, **kw):
        import torch
        done = []
        for row in input_ids.tolist():
            hit = False
            if row and row[-1] in self.gt:
                j = len(row) - 2
                while j >= 0 and row[j] not in self.nl:
                    j -= 1
                hit = "<COMMIT>" in self.tok.decode(row[j + 1:])
            done.append(hit)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


class HFRolloutEngine(RolloutEngine):
    """InternVL HF generate() with GRPO-R's exact arguments (one prompt at a time, G returns)."""

    def __init__(self, model, tok, device="cuda:0"):
        self.model, self.tok, self.device = model, tok, device
        self.eos = tok.convert_tokens_to_ids(EOS_TOKEN)
        self._stop = None

    def set_lora(self, path):
        raise NotImplementedError("HF backend scores the in-memory policy")

    def generate(self, items, mode, cfg: GenConfig):
        import torch
        from transformers import StoppingCriteriaList
        res = []
        for it in items:
            pv = hf_pixel_values(it["image"]).to(self.device)
            ids = self.tok(expand_image(prompt_text(it, mode), pv.shape[0]), return_tensors="pt").input_ids
            ids = ids.to(self.device)
            kw = dict(do_sample=cfg.temperature > 0, num_return_sequences=cfg.n, max_new_tokens=cfg.max_tokens,
                      eos_token_id=self.eos, pad_token_id=self.eos, repetition_penalty=1.0,
                      no_repeat_ngram_size=cfg.ngram)
            if cfg.temperature > 0:
                kw.update(temperature=cfg.temperature, top_p=cfg.top_p)
            else:
                kw.update(num_beams=1)
            if cfg.commit_stop:
                self._stop = self._stop or _CommitStop(self.tok)
                kw["stopping_criteria"] = StoppingCriteriaList([self._stop])
            self.model.language_model.config.use_cache = True
            with torch.no_grad():
                gen = self.model.generate(pixel_values=pv, input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
            group = []
            for row in gen.tolist():
                hit = self.eos in row
                cut = row[:row.index(self.eos) + 1] if hit else row
                fr = "stop" if hit or len(cut) < cfg.max_tokens else "length"
                group.append(Rollout(it["id"], cut, decode_response(self.tok, cut), finish_reason=fr,
                                     prompt_token_ids=ids[0].tolist()))
            res.append(group)
        return res
