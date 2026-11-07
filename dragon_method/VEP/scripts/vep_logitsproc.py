"""vLLM V1 logits processor used by every VEP / GRPO-R generation through vLLM.

  vep_ngram=n        HF NoRepeatNGramLogitsProcessor semantics over the *generated* tokens (InternVL's
                     HF generate() runs on inputs_embeds, so HF also sees only generated ids): a token
                     that would complete an n-gram already present in the output is set to -inf.
  vep_commit_stop    once the marker <COMMIT> has been emitted, only EOS is allowed (stop immediately).

Enabled per request through SamplingParams.extra_args. Output-token bookkeeping is done in apply(),
where vLLM guarantees the running output lists are final (async-scheduling placeholders replaced).
"""
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

try:
    from vllm.v1.sample.logits_processor.builtin import process_dict_updates
except ImportError:  # pragma: no cover
    from vllm.v1.sample.logits_processor.interface import MoveDirectionality

    def process_dict_updates(req_entries, batch_update, new_state):
        if not batch_update:
            return False
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            st = new_state(params, prompt_tok_ids, output_tok_ids)
            if st is not None:
                req_entries[index] = st
            else:
                req_entries.pop(index, None)
        for index in batch_update.removed:
            req_entries.pop(index, None)
        for a, b, direct in batch_update.moved:
            ea, eb = req_entries.pop(a, None), req_entries.pop(b, None)
            if ea is not None:
                req_entries[b] = ea
            if eb is not None and direct == MoveDirectionality.SWAP:
                req_entries[a] = eb
        return True

EOS = "<|im_end|>"
COMMIT = "<COMMIT>"


class _Row:
    __slots__ = ("n", "cs", "out", "table", "done", "cdone", "line_start", "force")

    def __init__(self, n, cs, out):
        self.n, self.cs, self.out = n, cs, out
        self.table, self.done, self.cdone, self.line_start, self.force = {}, 0, 0, 0, False


class VEPLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer, trust_remote_code=True)
        self.eos = self.tok.convert_tokens_to_ids(EOS)
        vocab = len(self.tok)
        strs = self.tok.batch_decode([[i] for i in range(vocab)])
        self.nl = {i for i, s in enumerate(strs) if "\n" in s}
        self.gt = {i for i, s in enumerate(strs) if ">" in s}
        self.device = device
        self.rows = {}

    def is_argmax_invariant(self):
        return False

    @staticmethod
    def _new(params, _prompt, out):
        ea = params.extra_args or {}
        n, cs = int(ea.get("vep_ngram", 0) or 0), bool(ea.get("vep_commit_stop", False))
        return _Row(n, cs, out) if (n > 1 or cs) else None

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.rows, batch_update, self._new)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.rows:
            return logits
        rr, tt, force = [], [], []
        for idx, st in self.rows.items():
            out = st.out
            L = len(out)
            if L and out[L - 1] < 0:
                L = out.index(-1) if -1 in out else L
            n = st.n
            if n > 1:
                tab = st.table
                for j in range(max(st.done, n - 1), L):
                    key = tuple(out[j - n + 1:j])
                    s = tab.get(key)
                    if s is None:
                        tab[key] = {out[j]}
                    else:
                        s.add(out[j])
                st.done = max(st.done, L)
                if L >= n - 1:
                    banned = tab.get(tuple(out[L - n + 1:L]))
                    if banned:
                        rr.extend([idx] * len(banned))
                        tt.extend(banned)
            if st.cs and not st.force:
                for j in range(st.cdone, L):
                    t = out[j]
                    if t in self.gt and COMMIT in self.tok.decode(out[st.line_start:j + 1]):
                        st.force = True
                        break
                    if t in self.nl:
                        st.line_start = j + 1
                st.cdone = L
            if st.force:
                force.append(idx)
        if rr:
            logits[torch.tensor(rr, device=logits.device), torch.tensor(tt, device=logits.device)] = -float("inf")
        if force:
            r = torch.tensor(force, device=logits.device)
            logits[r] = -float("inf")
            logits[r, self.eos] = 0.0
        return logits
