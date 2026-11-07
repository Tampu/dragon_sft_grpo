#!/usr/bin/env python3
"""A2's prefix branching: freeze what the model already found, resample what
comes next.

This is the mechanism Test 4 pointed at. Test 4 established two things that
constrain the implementation:
  * plain prefix resampling recovers missed evidence (rec@4 0.331);
  * appending a natural-language instruction to the branch makes it WORSE
    (targeted 0.218, not distinguishable from a generic "Continue." at 0.225).
So a branch here is a pure continuation from a token prefix. No text is ever
appended to the prefix -- `assert_no_instruction()` exists to keep that true.

Cut points are box-list boundaries, i.e. token positions where the emitted text
ends on a complete "[x1, y1, x2, y2]" entry. Cutting there leaves a prefix the
model can continue with ", [" or "]</box>", so the continuation is in-distribution
rather than a repair of mangled syntax.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dragon_grpo"))

from count_reward import parse_boxes  # noqa: E402
from grpo_dragon6_v1 import box_iou  # noqa: E402

# pilot spec section 3: sampling priority only, never an inference-time rule and
# never revealed to the model
TYPE_PRIORITY = ["answer_bearing", "connector", "text", "legend", "mark"]


def type_rank(t: str) -> int:
    return TYPE_PRIORITY.index(t) if t in TYPE_PRIORITY else len(TYPE_PRIORITY)


def assert_no_instruction(prefix_text: str, continuation_prompt: Optional[str]) -> None:
    """A2 is defined by the ABSENCE of a textual hint. Fail loudly if one shows up."""
    if continuation_prompt:
        raise ValueError("A2 branches must be pure prefix continuations: Test 4 showed "
                         "textual targeting is worse than plain resampling "
                         "(0.218 vs 0.331 rec@4)")


# ---------------------------------------------------------------- token/char mapping

def _token_texts(tokenizer, ids: Sequence[int]) -> List[str]:
    """Per-token decoded strings whose concatenation equals decode(ids).

    One batch_decode call for the fast path (byte-level BPE, which is what Qwen3
    uses, reconstructs exactly). Falls back to incremental decoding if the
    concatenation does not match, so a tokenizer that normalizes across token
    boundaries cannot silently shift every cut point.
    """
    seq = list(ids)
    pieces = tokenizer.batch_decode([[i] for i in seq], skip_special_tokens=False)
    if "".join(pieces) == tokenizer.decode(seq, skip_special_tokens=False):
        return pieces
    pieces, prev = [], ""
    for t in range(1, len(seq) + 1):
        cur = tokenizer.decode(seq[:t], skip_special_tokens=False)
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces


def box_boundaries(tokenizer, ids: Sequence[int]) -> List[Tuple[int, int]]:
    """[(token_cut, n_boxes_in_prefix)] at every complete-box-entry boundary.

    `token_cut` is a count of leading tokens, so ids[:token_cut] is the prefix.
    Also includes the boundary immediately after the box list opens (0 boxes
    kept), which is the only available cut when a rollout emitted no box at all.
    """
    pieces = _token_texts(tokenizer, ids)
    out: List[Tuple[int, int]] = []
    text = ""
    n_closed = 0
    opened = False
    for t, p in enumerate(pieces, start=1):
        text += p
        if not opened and "<box>" in text:
            opened = True
            out.append((t, 0))
            continue
        if opened and "]" in p:
            # parse_boxes keeps the COMPLETED entries of an unclosed <box>
            # segment, so this counts complete [x1,y1,x2,y2] groups and ignores
            # a half-written one
            n_now = len(parse_boxes(text))
            if n_now > n_closed:
                n_closed = n_now
                out.append((t, n_closed))
    return out


@dataclass
class BranchPoint:
    """One frozen prefix plus the bookkeeping the credit audit (spec section 10) needs."""
    prompt_id: str
    src_rollout: int
    token_cut: int
    prefix_ids: torch.Tensor
    prefix_text: str
    prefix_boxes: List[Tuple[float, float, float, float]]
    target_gold: int            # the gold element this branch was triggered for
    target_etype: str
    target_sft_missed: bool
    reason: str


def select_branch_points(prompt_id: str,
                         gt: Sequence[Sequence[float]],
                         etypes: Sequence[str],
                         sft_missed: Sequence[bool],
                         rollouts: List[Dict],
                         tokenizer,
                         rng,
                         tau: float = 0.5,
                         max_points: int = 1,
                         cut: str = "random") -> List[BranchPoint]:
    """Trigger: a gold element missed by EVERY main rollout (spec section 3).

    `rollouts` entries need "ids" (completion token tensor), "text" and
    "r_ground". Prefer elements greedy SFT also missed, since breaking SFT
    inheritance is the mechanism under test; within that, follow TYPE_PRIORITY.

    Training-time use of gold here is privileged SAMPLING, not privileged
    inference: nothing about the target reaches the model's context.
    """
    if not gt or not rollouts:
        return []

    # which gold elements did the whole group miss
    group_hit = [False] * len(gt)
    for ro in rollouts:
        for b in parse_boxes(ro["text"]):
            for j, g in enumerate(gt):
                if not group_hit[j] and box_iou(tuple(g), b) >= tau:
                    group_hit[j] = True
    cand = [j for j in range(len(gt)) if not group_hit[j]]
    if not cand:
        return []

    cand.sort(key=lambda j: (not bool(sft_missed[j]),
                             type_rank(etypes[j] if j < len(etypes) else "?"),
                             j))

    # prefer a high-reward source trajectory: branching off the group's best
    # partial evidence state asks "what comes next" from a state worth extending
    order = sorted(range(len(rollouts)), key=lambda i: -rollouts[i]["r_ground"])

    pts: List[BranchPoint] = []
    used = set()
    for j in cand:
        if len(pts) >= max_points:
            break
        for i in order:
            if i in used:
                continue
            ro = rollouts[i]
            bnds = box_boundaries(tokenizer, ro["ids"].tolist())
            if not bnds:
                continue
            # the prefix must not already cover the target (spec section 3)
            ok = [(tc, nb) for tc, nb in bnds
                  if not any(box_iou(tuple(gt[j]), b) >= tau
                             for b in parse_boxes(
                                 tokenizer.decode(ro["ids"].tolist()[:tc],
                                                  skip_special_tokens=False)))]
            if not ok:
                continue
            # keep a cut that leaves something to continue: drop the final
            # boundary when there is an interior one
            interior = ok[:-1] if len(ok) > 1 else ok
            tc, nb = (rng.choice(interior) if cut == "random"
                      else interior[len(interior) // 2] if cut == "mid"
                      else interior[-1])
            ptext = tokenizer.decode(ro["ids"].tolist()[:tc], skip_special_tokens=False)
            assert_no_instruction(ptext, None)
            pts.append(BranchPoint(
                prompt_id=prompt_id, src_rollout=i, token_cut=tc,
                prefix_ids=ro["ids"][:tc].clone(), prefix_text=ptext,
                prefix_boxes=parse_boxes(ptext),
                target_gold=j,
                target_etype=etypes[j] if j < len(etypes) else "?",
                target_sft_missed=bool(sft_missed[j]),
                reason=f"missed_by_all_{len(rollouts)}_rollouts"
                       f"{'_sft_missed' if sft_missed[j] else ''}"))
            used.add(i)
            break
    return pts


@torch.no_grad()
def sample_continuations(model, tokenizer, item, bp: BranchPoint, K: int,
                         device, temperature: float, top_p: float,
                         max_new_tokens: int, no_repeat_ngram_size: int
                         ) -> List[Tuple[torch.Tensor, str]]:
    """K alternative continuations from exactly bp.prefix_ids. Only the
    continuation is resampled; the prefix is frozen context."""
    ids = torch.cat([item["prompt_ids"].to(device), bp.prefix_ids.to(device)]).unsqueeze(0)
    attn = torch.ones_like(ids)
    gen = model.generate(
        pixel_values=item["pixel_values"].to(device),
        input_ids=ids, attention_mask=attn,
        do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=K, max_new_tokens=max_new_tokens,
        eos_token_id=item["eos_token_id"],
        pad_token_id=tokenizer.eos_token_id or item["eos_token_id"],
        repetition_penalty=1.0, no_repeat_ngram_size=no_repeat_ngram_size,
        return_dict_in_generate=False)
    eos = item["eos_token_id"]
    out = []
    for row in gen:
        if tokenizer.pad_token_id is not None:
            row = row[row != tokenizer.pad_token_id]
        hits = (row == eos).nonzero()
        row = row[: int(hits[0]) + 1] if len(hits) else row
        txt = tokenizer.decode(row.tolist(), skip_special_tokens=False)
        txt = txt.split(tokenizer.decode([eos], skip_special_tokens=False))[0]
        out.append((row.cpu(), txt))
    return out
