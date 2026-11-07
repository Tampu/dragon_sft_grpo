"""Omission-aware credit for GRPO on evidence-box outputs (drop-in for grpo_dragon6_v1.py).

Output format assumed (vanilla DRAGON prompt, NO count token):
    <ref>evidence</ref><box>[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]</box><EOS>
Several <ref>..<box>[[..]]</box> groups are also handled. Boxes on the 0-1000 grid.

Per-token advantage = A_seq (vanilla, Dr.GRPO-corrected) + credit, where
  box j tokens     : beta_box  * box_credit_j  / std_box   (span: the "continue" separator before
                                                            box j through box j's last number)
  stop tokens      : beta_stop * stop_credit   / std_stop  (span: after the last box's last number
                                                            to the end, incl. "]]</box>" and EOS)
  everything else  : A_seq only
mode="replace": box tokens get only the box credit (no A_seq).

  s_j          = IoU          if Hungarian-matched with IoU >= tau
               = gamma * IoU  if matched with 0 < IoU < tau      (near-miss, below any hit)
               = 0            if unmatched / duplicate / invalid
  Phi          = 2 * sum(s) / (m + n)
  break-even   = max(Phi / 2, kappa)                             (kappa: junk < 0 even when Phi = 0)
  box_credit_j = s_j - break-even
  s_hat_i      = mean over siblings k != i of the best s-value any box of k achieves on a gold
                 element that rollout i missed (IoU < tau); 0 if none
  stop_credit  = break-even - s_hat_i                            (no stop credit if truncated)

Branching (demand-gated continuation): rollouts with stop_credit < 0 (siblings found reachable
evidence this one left out) are continued from the end of their last box with a forced "], [";
continuations are scored on the extended set with the same credits, no A_seq, prefix masked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Sequence[float]
NUM = r"(-?\d+(?:\.\d+)?)"
BOX_RE = re.compile(r"\[\s*" + NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\]")
GROUP_RE = re.compile(r"<box>(.*?)(?:</box>|$)", re.S)


@dataclass
class CreditConfig:
    tau: float = 0.5          # hit threshold (matches F1@0.5)
    gamma: float = 0.25       # near-miss factor: gamma*IoU < tau*... stays below any hit
    kappa: float = 0.05       # break-even floor
    beta_box: float = 1.0
    beta_stop: float = 1.0
    mode: str = "add"         # "add" | "replace"
    use_box: bool = True
    use_stop: bool = True
    shat_reduce: str = "mean"  # "mean" | "max" over siblings (pilot: "max", low-recall regime)
    # branching
    branch_k: int = 4         # continuations per selected rollout
    branch_max_per_group: int = 2
    branch_budget: float = 0.25   # max branch tokens / main rollout tokens per batch
    branch_max_new_tokens: int = 128
    forced_text: str = "], ["


# ---------------------------------------------------------------- parsing
@dataclass
class ParsedBoxes:
    boxes: List[List[float]] = field(default_factory=list)
    num_ends: List[int] = field(default_factory=list)    # char pos right after each box's 4th number
    starts: List[int] = field(default_factory=list)      # char pos of each box's "["


def parse_boxes(text: str) -> ParsedBoxes:
    out = ParsedBoxes()
    for g in GROUP_RE.finditer(text):
        base = g.start(1)
        for m in BOX_RE.finditer(g.group(1)):
            out.boxes.append([float(m.group(k)) for k in range(1, 5)])
            out.num_ends.append(base + m.end(4))
            out.starts.append(base + m.start())
    return out


# --------------------------------------------------------------- geometry
def _valid(b: Box) -> bool:
    return b[2] > b[0] and b[3] > b[1]


def iou(a: Box, b: Box) -> float:
    if not (_valid(a) and _valid(b)):
        return 0.0
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _s_value(v: float, cfg: CreditConfig) -> float:
    return v if v >= cfg.tau else cfg.gamma * v


def set_scores(P: Sequence[Box], G: Sequence[Box], cfg: CreditConfig):
    """Hungarian on raw IoU over the FULL set. Returns (s per box, Phi, set of gold hit at >= tau)."""
    s = np.zeros(len(P))
    hit = set()
    if P and G:
        M = np.array([[iou(p, g) for g in G] for p in P])
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            v = M[i, j]
            if v > 0:
                s[i] = _s_value(v, cfg)
                if v >= cfg.tau:
                    hit.add(int(j))
    if not P and not G:
        phi = 1.0
    else:
        phi = 2.0 * float(s.sum()) / (len(P) + len(G))
    return s, phi, hit


# ---------------------------------------------------------------- credits
@dataclass
class RolloutCredit:
    box: List[float]                 # raw box credits
    stop: Optional[float]            # raw stop credit (None if truncated)
    phi: float
    break_even: float
    missed: List[int]


def group_credits(box_lists: Sequence[Sequence[Box]], gold: Sequence[Box],
                  truncated: Sequence[bool], cfg: CreditConfig,
                  sibling_box_lists: Optional[Sequence[Sequence[Box]]] = None,
                  self_index: Optional[int] = None) -> List[RolloutCredit]:
    """Credits for every rollout of one group. For a branch, pass the extended box list as the
    only element of box_lists and the ORIGINAL group as sibling_box_lists with self_index set."""
    sibs = sibling_box_lists if sibling_box_lists is not None else box_lists
    out = []
    for i, P in enumerate(box_lists):
        s, phi, hit = set_scores(P, gold, cfg)
        be = max(phi / 2.0, cfg.kappa)
        missed = [g for g in range(len(gold)) if g not in hit]
        stop = None
        if not truncated[i]:
            me = self_index if sibling_box_lists is not None else i
            vals = []
            for k, Q in enumerate(sibs):
                if k == me:
                    continue
                best = 0.0
                for b in Q:
                    for g in missed:
                        best = max(best, _s_value(iou(b, gold[g]), cfg))
                vals.append(best)
            if not vals:
                s_hat = 0.0
            else:
                s_hat = float(np.max(vals)) if cfg.shat_reduce == "max" else float(np.mean(vals))
            stop = be - s_hat
        out.append(RolloutCredit([float(x) - be for x in s], stop, phi, be, missed))
    return out


def mcr_credit(P: Sequence[Box], gold: Sequence[Box], cfg: CreditConfig) -> List[float]:
    """MCR-GRPO baseline (Han et al. 2026): leave-one-out set value, z-scored within the response.
    Uses our Phi as the set value. Add with lambda=0.1 as in the paper."""
    if len(P) < 2:
        return [0.0] * len(P)
    _, full, _ = set_scores(P, gold, cfg)
    d = np.array([full - set_scores([b for k, b in enumerate(P) if k != j], gold, cfg)[1]
                  for j in range(len(P))])
    return list((d - d.mean()) / (d.std() + 1e-6))


# -------------------------------------------------------------- advantages
def seq_advantages(rewards_per_group: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """Dr.GRPO-corrected sequence advantage: center per group, ONE batch-level std."""
    centered = [np.asarray(r, float) - np.mean(r) for r in rewards_per_group]
    allc = np.concatenate(centered) if centered else np.zeros(1)
    sd = float(allc.std()) + 1e-6
    return [c / sd for c in centered]


def credit_scales(all_credits: Sequence[Sequence[RolloutCredit]]) -> Tuple[float, float]:
    """Batch std of raw box credits and of raw stop credits (credits are NOT centered)."""
    boxes = [x for grp in all_credits for rc in grp for x in rc.box]
    stops = [rc.stop for grp in all_credits for rc in grp if rc.stop is not None]
    sb = float(np.std(boxes)) + 1e-6 if len(boxes) > 1 else 1.0
    ss = float(np.std(stops)) + 1e-6 if len(stops) > 1 else 1.0
    return sb, ss


def char_spans(token_ids: Sequence[int], tokenizer) -> List[Tuple[int, int]]:
    """Character span of each token in tokenizer.decode(token_ids, skip_special_tokens=False).
    Use the SAME token ids that enter the loss (after eos/pad stripping)."""
    spans, prev = [], 0
    for t in range(1, len(token_ids) + 1):
        cur = len(tokenizer.decode(token_ids[:t], skip_special_tokens=False))
        spans.append((prev, cur))
        prev = cur
    return spans


def decision_spans(pb: ParsedBoxes, text_len: int):
    """Box j span: from the end of box j-1's numbers (the 'continue' separator) through box j's
    numbers; box 0 starts at its '['. Stop span: after the last box's numbers to the end."""
    spans = []
    for j in range(len(pb.boxes)):
        start = pb.starts[0] if j == 0 else pb.num_ends[j - 1]
        spans.append((start, pb.num_ends[j]))
    stop = (pb.num_ends[-1], text_len) if pb.boxes else None
    return spans, stop


def token_advantages(tok_spans: List[Tuple[int, int]], text_len: int, pb: ParsedBoxes,
                     a_seq: float, rc: Optional[RolloutCredit], sb: float, ss: float,
                     cfg: CreditConfig, include_seq: bool = True) -> np.ndarray:
    """Per-token advantages for one completion. rc=None (parse failure / vanilla) -> A_seq only."""
    base = a_seq if include_seq else 0.0
    adv = np.full(len(tok_spans), base, dtype=np.float32)
    if rc is None or not pb.boxes:
        return adv
    bspans, sspan = decision_spans(pb, text_len)
    for t, (a, b) in enumerate(tok_spans):
        mid = (a + b) / 2.0
        if sspan is not None and (sspan[0] <= mid < sspan[1] or (a == b and a >= sspan[0])):
            if cfg.use_stop and rc.stop is not None:
                adv[t] = base + cfg.beta_stop * rc.stop / ss
            continue
        for j, (s0, s1) in enumerate(bspans):
            if s0 <= mid < s1:
                if cfg.use_box:
                    c = cfg.beta_box * rc.box[j] / sb
                    adv[t] = c if (cfg.mode == "replace" and include_seq) else base + c
                break
    return adv


# --------------------------------------------------------------- branching
def select_branches(credits: Sequence[RolloutCredit], parsed: Sequence[ParsedBoxes],
                    cfg: CreditConfig) -> List[int]:
    """Rollouts whose stop was premature: stop credit < 0 and at least one box to continue from."""
    cand = [i for i, rc in enumerate(credits)
            if rc.stop is not None and rc.stop < 0 and parsed[i].boxes]
    cand.sort(key=lambda i: credits[i].stop)
    return cand[:cfg.branch_max_per_group]


def branch_prefix(token_ids: Sequence[int], tok_spans: List[Tuple[int, int]], text: str,
                  pb: ParsedBoxes, tokenizer, cfg: CreditConfig):
    """Token ids to feed after the prompt so the continuation starts a NEW box:
    prefix = completion tokens fully inside the text up to the last box's 4th number,
    forced = tokenize(remaining chars up to that point + '], [').
    Returns (prefix_ids, forced_ids, prefix_text) with prefix_text the exact decoded text."""
    cut = pb.num_ends[-1]
    t = 0
    while t < len(tok_spans) and tok_spans[t][1] <= cut:
        t += 1
    covered = tok_spans[t - 1][1] if t > 0 else 0
    forced_text = text[covered:cut] + cfg.forced_text
    forced_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    return list(token_ids[:t]), list(forced_ids), text[:cut] + cfg.forced_text


def score_branch(prefix_text: str, cont_text: str, base_boxes: Sequence[Box], gold: Sequence[Box],
                 group_boxes: Sequence[Sequence[Box]], self_index: int, truncated: bool,
                 cfg: CreditConfig):
    """Parse the extended output, return (full_text, ParsedBoxes, RolloutCredit, n_new_boxes).
    Only tokens of the continuation get gradient; spans are on full_text."""
    full = prefix_text + cont_text
    pb = parse_boxes(full)
    rc = group_credits([pb.boxes], gold, [truncated], cfg,
                       sibling_box_lists=group_boxes, self_index=self_index)[0]
    return full, pb, rc, len(pb.boxes) - len(base_boxes)
