#!/usr/bin/env python3
"""Rewards for the demand-aware arms.

Two separate scalars per rollout, never summed into one:

  r_ground  the FROZEN vanilla GRPO grounding reward, imported from
            dragon_grpo/grpo_dragon6_v1.py so A1/A2/A3 cannot drift from the A0
            baseline (pilot spec section 12: do not change the grounding reward
            during this pilot).

  r_count   -|log((n_hat+1)/(n+1))|, the spec's demand reward, bounded below so
            one absurd prediction cannot dominate a group's normalization.

They stay separate because they are routed to different tokens: r_count scores
the <count> span, r_ground scores the box tokens. Broadcasting the count reward
onto box tokens is the specific mistake this module exists to prevent.

Phi (the set-level objective branching scores improvement against) is the same
r_ground, so a branch's marginal reward is in the same units as the main
trajectory reward.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dragon_grpo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_dragon6_v1 import (RewardWeights, compute_reward,  # noqa: E402
                             parse_boxes_r1)
from count_format import parse_count  # noqa: E402

Box = Tuple[float, float, float, float]

# A prediction off by this log-ratio or more is already maximally wrong; the
# clamp keeps a degenerate n_hat (e.g. 999) from swamping the group std and
# zeroing every other sample's count advantage.
COUNT_FLOOR = -1.5
# No span at all is a format failure, scored at the floor rather than skipped:
# dropping it from the group would make "emit no count" a free action.
COUNT_MISSING = COUNT_FLOOR


def set_matching(mode: str) -> None:
    """Pin the 1:1 matching inside the frozen grounding reward.

    grpo_dragon6_v1.hungarian_match uses scipy when importable and a greedy
    max-IoU fallback otherwise. The A0 GRPO runs were trained in da_env_flash,
    which has NO scipy, so their reward used the greedy path; da_env2 (the vLLM
    env) has scipy. Without pinning, moving the trainer to vLLM would silently
    change the reward. 'greedy' (default) reproduces A0's training reward
    exactly in either env.
    """
    import grpo_dragon6_v1 as G
    if not hasattr(G, "_hungarian_orig"):
        G._hungarian_orig = G.hungarian_match
    if mode == "hungarian":
        G.hungarian_match = G._hungarian_orig
        return

    def greedy(gt, pred):
        if not gt or not pred:
            return []
        iou = [[G.box_iou(g, p) for p in pred] for g in gt]
        pairs = sorted(((iou[i][j], i, j) for i in range(len(gt))
                        for j in range(len(pred))), reverse=True)
        used_g, used_p, out = set(), set(), []
        for v, i, j in pairs:
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j); out.append((i, j, v))
        return out

    G.hungarian_match = greedy


def count_reward(n_hat: Optional[int], n_gold: int) -> float:
    if n_hat is None:
        return COUNT_MISSING
    r = -abs(math.log((n_hat + 1) / (n_gold + 1)))
    return max(COUNT_FLOOR, r)


def phi(gt: Sequence[Box], pred: Sequence[Box], w: RewardWeights) -> float:
    """Set-level grounding objective on an explicit box list.

    compute_reward() takes TEXT; branching needs to score a box set it assembled
    from a prefix plus a continuation, so this re-enters the same reward through
    a synthetic <box> string -- same code path, no second implementation of the
    reward to keep in sync.
    """
    if not pred:
        return compute_reward("", gt, w)["reward"]
    body = ", ".join("[%d, %d, %d, %d]" % tuple(round(c) for c in b) for b in pred)
    return compute_reward(f"<ref>evidence</ref><box>[{body}]</box>", gt, w)["reward"]


def score_rollout(text: str, gt: Sequence[Box], w: RewardWeights) -> Dict[str, float]:
    """Everything one rollout contributes, for both reward channels and logging."""
    n_gold = len(gt)
    n_hat = parse_count(text)
    g = compute_reward(text, gt, w)
    g["r_ground"] = g.pop("reward")
    g["n_hat"] = -1 if n_hat is None else n_hat
    g["n_gold"] = n_gold
    g["count_parsed"] = float(n_hat is not None)
    g["count_exact"] = float(n_hat == n_gold)
    g["count_abserr"] = float(abs(n_hat - n_gold)) if n_hat is not None else float("nan")
    g["compliant"] = float(n_hat is not None and g["n_pred"] == n_hat)
    g["r_count"] = count_reward(n_hat, n_gold)
    return g


def parse_boxes(text: str):
    return parse_boxes_r1(text)[0]
