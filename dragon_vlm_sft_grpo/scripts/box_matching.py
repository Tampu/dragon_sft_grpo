#!/usr/bin/env python3
"""
1:1 box matching + the GRPO reward. Shared by grpo_train.py (the reward
during RL) and export_for_eval.py (an optional bijective 1:1 rescoring
alongside eval_script.py's own many-to-many R/P/F1) -- kept in one place so
the two can't silently compute IoU differently.

Hungarian assignment via scipy when available, greedy fallback otherwise --
same caveat as dragon_sft_grpo/requirements.txt documents: if scipy isn't
installed when a run is produced, state that greedy matching was used,
since installing scipy later will change both the bijective eval numbers
and the RL reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Box = Tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def hungarian_match(gt: Sequence[Box], pred: Sequence[Box]) -> List[Tuple[int, int, float]]:
    """1:1 matching maximizing total IoU. Hungarian via scipy, greedy fallback."""
    if not gt or not pred:
        return []
    iou = [[box_iou(g, p) for p in pred] for g in gt]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-np.array(iou))
        return [(int(r), int(c), iou[r][c]) for r, c in zip(ri, ci)]
    except Exception:
        pairs = sorted(((iou[i][j], i, j) for i in range(len(gt))
                        for j in range(len(pred))), reverse=True)
        used_g, used_p, out = set(), set(), []
        for v, i, j in pairs:
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j); out.append((i, j, v))
        return out


@dataclass
class RewardWeights:
    """Same asymmetric shape as dragon_sft_grpo's GRPO reward, and the same
    reasoning: loose boxes should be penalized softly (IoU-graded), missing
    a gt box entirely should be penalized harder than an extra spurious box
    (under-prediction was the observed InternVL failure mode; whether it
    recurs here is itself worth watching in early GRPO metrics)."""
    w_iou: float = 0.45
    w_f1: float = 0.15
    w_fmt: float = 0.05
    w_miss: float = 0.25
    w_fp: float = 0.10


def compute_reward(pred: List[Box] | None, gt: Sequence[Box], w: RewardWeights) -> Dict[str, float]:
    n_gt = max(1, len(gt))
    if pred is None or len(pred) == 0:
        return {"reward": w.w_fmt * 0.0 - w.w_miss * 1.0, "fmt": 0.0,
                "mean_iou": 0.0, "f1": 0.0, "miss": 1.0, "fp": 0.0, "n_pred": 0}
    matches = hungarian_match(gt, pred)
    per_gt_iou = {i: v for i, _, v in matches}
    soft_recall = sum(per_gt_iou.get(i, 0.0) for i in range(len(gt))) / n_gt
    tp = sum(1 for _, _, v in matches if v >= 0.5)
    miss = (len(gt) - tp) / n_gt
    fp = (len(pred) - tp) / len(pred)
    prec, rec = tp / len(pred), tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    reward = (w.w_fmt * 1.0 + w.w_iou * soft_recall + w.w_f1 * f1
              - w.w_miss * miss - w.w_fp * fp)
    return {"reward": reward, "fmt": 1.0, "mean_iou": soft_recall, "f1": f1,
            "miss": miss, "fp": fp, "n_pred": len(pred)}
