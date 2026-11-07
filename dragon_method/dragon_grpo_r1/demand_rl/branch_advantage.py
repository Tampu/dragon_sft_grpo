#!/usr/bin/env python3
"""Local prefix-relative credit for A2's branch groups.

The point of the whole arm. In vanilla GRPO one scalar trajectory advantage is
applied to every token of a rollout, so a rare useful evidence action buried in
an otherwise poor trajectory can receive negative credit. Here the alternatives
share a prefix, so each continuation is compared only against the other things
the policy could have done from the SAME evidence state:

    r_j = Phi(E_t u c_j, G) - Phi(E_t, G)                     marginal improvement
    A_j = (r_j - mean_{k != j} r_k) / max(eps, std(r_0..r_K))  leave-one-out baseline

Two rules that are easy to get wrong and are enforced here:
  * the advantage applies ONLY to tokens generated after the branch point
    (the shared prefix gets zero: it is identical across the group and carries
    no preference information);
  * the branch group is normalized on its own and never merged into the main
    G=8 trajectory group.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dragon_grpo"))

from count_reward import parse_boxes, phi  # noqa: E402
from grpo_dragon6_v1 import box_iou  # noqa: E402

EPS = 1e-4


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def branch_group_advantages(bp, continuations: List[Tuple[object, str]],
                            gt: Sequence[Sequence[float]], w,
                            sft_missed: Sequence[bool],
                            etypes: Sequence[str],
                            tau: float = 0.5,
                            sigma_min: float = 0.0) -> List[Dict]:
    """One record per continuation: marginal reward, branch advantage, and the
    newly-hit gold elements the credit audit checks the sign against.

    `continuations` is [(ids, text)] with index 0 being the ORIGINAL
    continuation of the source rollout, per the spec's B(s_t) = {c_0..c_K}.
    """
    gtt = [tuple(map(float, g)) for g in gt]
    base_boxes = list(bp.prefix_boxes)
    phi_base = phi(gtt, base_boxes, w)

    # which gold elements the frozen prefix already covers
    pre_hit = [any(box_iou(g, b) >= tau for b in base_boxes) for g in gtt]

    recs: List[Dict] = []
    for j, (ids, text) in enumerate(continuations):
        add = parse_boxes(bp.prefix_text + text)
        # boxes contributed by the continuation = the full set minus the prefix
        # set (parsed from the concatenation, so a box the continuation finished
        # writing is attributed to the continuation)
        phi_after = phi(gtt, add, w)
        new_hits = [k for k, g in enumerate(gtt)
                    if not pre_hit[k] and any(box_iou(g, b) >= tau for b in add)]
        recs.append({
            "cont_index": j,
            "is_original": j == 0,
            "n_boxes_prefix": len(base_boxes),
            "n_boxes_after": len(add),
            "phi_before": phi_base,
            "phi_after": phi_after,
            "r_marginal": phi_after - phi_base,
            "new_gold": new_hits,
            "new_gold_etypes": [etypes[k] if k < len(etypes) else "?" for k in new_hits],
            "new_gold_sft_missed": [bool(sft_missed[k]) for k in new_hits],
            "hit_target": bp.target_gold in new_hits,
            "n_new_sft_missed": sum(1 for k in new_hits if sft_missed[k]),
        })

    rs = [r["r_marginal"] for r in recs]
    sd = _std(rs)
    n = len(rs)
    # a branch group with no spread carries no preference information, same rule
    # as the main group's sigma_min gate
    dead = sd < sigma_min
    for i, r in enumerate(recs):
        loo = _mean([rs[k] for k in range(n) if k != i])
        r["branch_group_mean"] = _mean(rs)
        r["branch_group_std"] = sd
        r["branch_advantage"] = 0.0 if dead else (rs[i] - loo) / max(EPS, sd)
    return recs


# ---------------------------------------------------------------- credit audit (spec section 10)

def audit(records: Sequence[Dict]) -> Dict[str, float]:
    """Does local credit actually reward the rare useful discovery?

    Gates from the spec:
        useful discoveries with positive advantage  >= 0.80
        junk continuations with non-positive advantage >= 0.70
    A run that fails these has a credit bug, whatever its F1 does.
    """
    useful = [r for r in records if r["n_new_sft_missed"] > 0]
    junk = [r for r in records if not r["new_gold"] and r["r_marginal"] <= 0]
    out = {
        "n_branch_samples": len(records),
        "n_useful": len(useful),
        "n_junk": len(junk),
        "useful_positive_frac": (sum(1 for r in useful if r["branch_advantage"] > 0)
                                 / len(useful)) if useful else float("nan"),
        "junk_nonpositive_frac": (sum(1 for r in junk if r["branch_advantage"] <= 0)
                                  / len(junk)) if junk else float("nan"),
        "nonzero_variance_frac": (sum(1 for r in records if r["branch_group_std"] > 0)
                                  / len(records)) if records else float("nan"),
        "discover_sft_missed_frac": (sum(1 for r in records if r["n_new_sft_missed"] > 0)
                                     / len(records)) if records else float("nan"),
        "discover_answer_bearing_frac": (
            sum(1 for r in records if "answer_bearing" in r["new_gold_etypes"])
            / len(records)) if records else float("nan"),
    }
    out["useful_sign_gate"] = out["useful_positive_frac"] >= 0.80
    out["junk_sign_gate"] = out["junk_nonpositive_frac"] >= 0.70
    return out
