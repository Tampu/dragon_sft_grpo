"""Shared-prefix packed scoring of G completions of one prompt.

GRPO-R scores each completion with its own full forward ([prompt + c_i], ~3.5k image tokens each).
Here the sequence [prompt | c_1 | ... | c_G] goes through the LLM once with a prefix-sharing mask
(c_i attends to the whole prompt and causally to itself, never to c_j) and per-segment position ids,
which is the same function of the weights (gradients flow through the shared prompt states exactly
as through G copies) at ~1/G of the prompt cost. test_packed_logprobs.py checks the equivalence.
Vision features are computed once (no grad: vision tower and mlp1 are frozen, as in GRPO-R).
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def base_lm(language_model):
    return language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model


def install_dynamic_ckpt(language_model):
    """Per-decoder-layer activation checkpointing that can be switched per forward (eval-mode safe)."""
    ctl = {"on": False}
    for layer in base_lm(language_model).model.layers:
        fwd = layer.forward

        def wrapped(*a, _fwd=fwd, **kw):
            if ctl["on"] and torch.is_grad_enabled():
                return checkpoint(_fwd, *a, use_reentrant=False, **kw)
            return _fwd(*a, **kw)

        layer.forward = wrapped
    return ctl


def _chunk_lp(h, w, tgt):
    logits = F.linear(h, w).float()
    return logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, dim=-1)


def prefix_mask(P, lens, device, dtype=torch.bfloat16):
    T = P + sum(lens)
    seg = torch.repeat_interleave(torch.arange(len(lens) + 1, device=device),
                                  torch.tensor([P] + list(lens), device=device))
    ar = torch.arange(T, device=device)
    allowed = (ar[None, :] <= ar[:, None]) & ((seg[None, :] == 0) | (seg[None, :] == seg[:, None]))
    Tp = -(-T // 16) * 16
    m = torch.full((1, 1, T, Tp), float("-inf"), dtype=dtype, device=device)
    m[0, 0, :, :T].masked_fill_(allowed, 0.0)
    return m[..., :T]


def packed_logprobs(model, vit_embeds, prompt_ids, comps, img_context_id, chunk=1024):
    """Token logprobs of every completion (concatenated, in order) and their lengths.
    prompt_ids: LongTensor [P] incl. expanded <IMG_CONTEXT> tokens; comps: list of LongTensor [L_i]."""
    lm = base_lm(model.language_model)
    dev = prompt_ids.device
    P, lens = prompt_ids.numel(), [int(c.numel()) for c in comps]
    ids = torch.cat([prompt_ids] + list(comps))
    pos = torch.cat([torch.arange(P, device=dev)] + [torch.arange(P, P + L, device=dev) for L in lens])
    emb = lm.get_input_embeddings()(ids)
    img = (ids[:P] == img_context_id).nonzero(as_tuple=True)[0]
    vit = vit_embeds.reshape(-1, emb.shape[-1]).to(emb.dtype)
    if img.numel() != vit.shape[0]:
        raise ValueError(f"{img.numel()} <IMG_CONTEXT> tokens vs {vit.shape[0]} vision embeddings")
    emb = emb.index_put((img,), vit)
    h = lm.model(inputs_embeds=emb[None], attention_mask=prefix_mask(P, lens, dev, emb.dtype),
                 position_ids=pos[None], use_cache=False).last_hidden_state[0]
    starts = torch.cumsum(torch.tensor([P] + lens[:-1], device=dev), 0)
    pred = torch.cat([torch.cat([torch.tensor([P - 1], device=dev), torch.arange(s, s + L - 1, device=dev)])
                      for s, L in zip(starts.tolist(), lens)])
    tgt = torch.cat(list(comps))
    w = lm.lm_head.weight
    out = []
    for j in range(0, pred.numel(), chunk):
        hj, tj = h[pred[j:j + chunk]], tgt[j:j + chunk]
        out.append(checkpoint(_chunk_lp, hj, w, tj, use_reentrant=False) if torch.is_grad_enabled()
                   else _chunk_lp(hj, w, tj))
    return torch.cat(out), lens


def naive_logprobs(model, vit_embeds, prompt_ids, comps, img_context_id):
    """Reference: one full forward per completion (GRPO-R's completion_logprobs on embeddings)."""
    lm = base_lm(model.language_model)
    outs = []
    for c in comps:
        ids = torch.cat([prompt_ids, c])
        emb = lm.get_input_embeddings()(ids)
        img = (ids == img_context_id).nonzero(as_tuple=True)[0]
        emb = emb.index_put((img,), vit_embeds.reshape(-1, emb.shape[-1]).to(emb.dtype))
        logits = lm(inputs_embeds=emb[None], use_cache=False).logits[0]
        P = prompt_ids.numel()
        lp = logits[P - 1:-1].float().log_softmax(-1).gather(1, c[:, None])[:, 0]
        outs.append(lp)
    return torch.cat(outs), [int(c.numel()) for c in comps]
