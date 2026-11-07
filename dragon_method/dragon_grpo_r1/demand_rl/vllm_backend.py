#!/usr/bin/env python3
"""vLLM rollout backend for demand_rl, built from VEP's verified systems code.

Reused from /mnt/data2/traviku2/VEP/scripts (imported, not copied, not edited):
  rollout.VLLMRolloutEngine   vLLM V1 engine colocated on the trainer GPU, LoRA hot-swap,
                              prefix caching, HF-identical no-repeat-ngram logits processor
  rollout.completion_ids      GRPO-R's EOS cleanup for vLLM outputs
  common.export_merged        base + SFT LoRA (vision+LLM) + mlp1 merged into one HF dir that
                              both vLLM and the HF trainer load
  common.load_merged_hf       HF InternVLChatModel on those exact weights
  common.hf_pixel_values      the fork's tiling (bit-identical to build_batch_item, VEP gate)
  common.prompt_text / expand_image   internlm2-chat prompt with the PER-SAMPLE system turn
  common.vllm_lora_state / save_vllm_lora   per-step PEFT -> vLLM adapter sync
  packed.packed_logprobs      shared-prefix packed scoring (prompt once per group)

VEP's equivalence bench (VEP/outputs/bench_grpor.json) is what licenses this:
pixel values 30/30 identical, prompt ids 30/30 identical, teacher-forced logprob gap
0.022 nats (r = 0.9996), LoRA sync verified, sampled-rollout statistics HF vs vLLM all
with CIs containing 0. None of VEP's METHOD (Phi rewards, segment credit, COMMIT,
adaptive lambda) is imported.

What is new here is only the branch call: K continuations from (prompt + frozen
prefix). It is self-verifying: vLLM reports the prompt token ids it actually
conditioned on, and they are checked against HF's prompt ids + the prefix ids.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VEP_SCRIPTS = Path(os.environ.get("VEP_ROOT", "/mnt/data2/traviku2/VEP")) / "scripts"
if str(VEP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(VEP_SCRIPTS))

import common as VC  # noqa: E402  (VEP common: paths, export, prompts, pixels, LoRA sync)


def export_init(sft_ckpt: str, name: str) -> Path:
    """Merged HF dir for an adapter-only SFT checkpoint (idempotent; DONE marker)."""
    return VC.export_merged(VC.MERGED_ROOT / name, sft_ckpt=Path(sft_ckpt).resolve())


def resolve_base(name_or_path: str) -> Path:
    p = Path(name_or_path)
    return p if p.exists() else VC.MERGED_ROOT / name_or_path


class DemandEngine:
    """Thin wrapper over VEP's VLLMRolloutEngine with the two calls demand_rl needs."""

    def __init__(self, model_dir, tok, gpu_mem=0.25, max_model_len=6144, lora_rank=16,
                 seed=17, max_num_seqs=64, limit_images=1):
        from rollout import VLLMRolloutEngine
        self.eng = VLLMRolloutEngine(model_dir, tok, gpu_mem=gpu_mem, max_model_len=max_model_len,
                                     enable_lora=True, max_lora_rank=lora_rank, seed=seed,
                                     max_num_seqs=max_num_seqs, limit_images=limit_images)
        self.tok = tok
        self.eos = self.eng.eos
        self.n_prefix_retok = 0      # branch prefixes vLLM re-tokenized differently (used its ids)
        self.n_prefix_drop = 0       # branch calls dropped: prompt part itself disagreed

    def set_lora(self, path):
        return self.eng.set_lora(path)

    def _sp(self, n, temperature, top_p, max_tokens, ngram, logprobs=False):
        from vllm import SamplingParams
        # identical to rollout.VLLMRolloutEngine.generate's SamplingParams
        return SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                              stop_token_ids=[self.eos], skip_special_tokens=False,
                              detokenize=False, logprobs=0 if logprobs else None,
                              extra_args={"vep_ngram": ngram, "vep_commit_stop": False})

    def _run(self, prompts, sp):
        return self.eng.llm.generate(prompts, sp, lora_request=self.eng.lora, use_tqdm=False)

    def main(self, items: List[Dict], n, temperature, top_p, max_tokens, ngram,
             logprobs=False) -> List[Dict]:
        """One call for all prompts of a step. Per item: prompt_ids (what vLLM
        conditioned on) and n completions as (ids incl. EOS, truncated, logprobs)."""
        from rollout import completion_ids
        imgs = list(self.eng._io.map(lambda it: VC.load_pil(it["image"]), items))
        prompts = [{"prompt": VC.prompt_text(it, "grounding"), "multi_modal_data": {"image": im}}
                   for it, im in zip(items, imgs)]
        outs = self._run(prompts, self._sp(n, temperature, top_p, max_tokens, ngram, logprobs))
        res = []
        for o in outs:
            comps = []
            for c in o.outputs:
                ids = completion_ids(c.token_ids, c.finish_reason, self.eos)
                lps = None
                if logprobs and c.logprobs is not None:
                    lps = [c.logprobs[j][t].logprob for j, t in enumerate(c.token_ids)]
                comps.append({"ids": ids, "trunc": self.eos not in ids, "lps": lps})
            res.append({"prompt_ids": list(o.prompt_token_ids), "comps": comps})
        return res

    def branch(self, item: Dict, prompt_ids: List[int], prefix_ids: List[int], prefix_text: str,
               K, temperature, top_p, max_tokens, ngram) -> Tuple[Optional[List[int]], List[List[int]]]:
        """K continuations from exactly (prompt + frozen prefix). No text is added.

        Returns (prefix ids vLLM actually conditioned on, [continuation ids]).
        The prefix is passed as text (InternVL's vLLM processor expands the image
        placeholder from text); the returned prompt_token_ids are then checked:
          == prompt_ids + prefix_ids   -> exact, the normal case
          starts with prompt_ids only  -> prefix re-tokenized differently; the
                                          ids vLLM used are returned so HF scores
                                          the same context it sampled from
          otherwise                    -> (None, []) and the branch is dropped
        """
        from rollout import completion_ids
        im = VC.load_pil(item["image"])
        p = {"prompt": VC.prompt_text(item, "grounding") + prefix_text,
             "multi_modal_data": {"image": im}}
        o = self._run([p], self._sp(K, temperature, top_p, max_tokens, ngram))[0]
        got = list(o.prompt_token_ids)
        P = len(prompt_ids)
        if got[:P] != list(prompt_ids):
            self.n_prefix_drop += 1
            return None, []
        used = got[P:]
        if used != list(prefix_ids):
            self.n_prefix_retok += 1
        conts = [completion_ids(c.token_ids, c.finish_reason, self.eos) for c in o.outputs]
        return used, conts


def load_policy_merged(model_dir, lora_rank, device="cuda:0"):
    """Merged count-SFT + fresh rank-r LLM LoRA (alpha 2r), everything else frozen.

    Same construction as grpo_dragon6_v1.load_policy (merge SFT, wrap a fresh
    LoRA), on the merged dir vLLM serves. Reference policy = adapter disabled.
    """
    import torch
    from packed import install_dynamic_ckpt
    model, tok = VC.load_merged_hf(model_dir, device="cpu")
    model.wrap_llm_lora(r=lora_rank, lora_alpha=2 * lora_rank)
    model.language_model.config.use_cache = False
    model.to(device).eval()
    ctl = install_dynamic_ckpt(model.language_model)
    return model, tok, ctl


def sync_lora(engine: DemandEngine, model, out_dir, lora_rank):
    from peft.utils import get_peft_model_state_dict
    d = VC.save_vllm_lora(VC.vllm_lora_state(get_peft_model_state_dict(model.language_model)),
                          out_dir, lora_rank, 2 * lora_rank)
    engine.set_lora(d)
    return d
