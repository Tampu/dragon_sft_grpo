#!/usr/bin/env python3
"""The demand-aware branching METHOD, backend-free.

Both trainers -- the HF reference path (train_count_grpo.py) and the vLLM fast
path (train_count_grpo_vllm.py) -- call these functions and nothing else for
credit and the objective, so switching the rollout backend cannot change the
method. What lives here:

  main_micros()      count-channel / grounding-channel advantages, routed to
                     their own tokens (count reward never reaches a box token)
  branch_micros()    prefix-local credit for one branch group; c_0 is the
                     baseline only; credit and KL only after the cut
  masked_objective() the token-level clipped PG + k3 KL, token-mean per
                     completion over its MASKED tokens, summed / #completions
                     (GRPO-R's normalisation, with the branch mask)

Rollout generation (where the backends differ) is deliberately NOT in here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_format import count_span_token_end  # noqa: E402
import branch_advantage as BA  # noqa: E402


def normalize(vals: List[float], sigma_min: float) -> torch.Tensor:
    """Group-relative advantage, with the R1 near-tie gate: a group whose reward
    spread is below sigma_min carries no preference information, so A = 0."""
    t = torch.tensor(vals, dtype=torch.float32)
    if len(t) < 2 or float(t.std()) < sigma_min:
        return torch.zeros_like(t)
    return (t - t.mean()) / (t.std() + 1e-4)


def main_micros(ros: List[Dict], tokenizer, args) -> List[Dict]:
    """ros: [{"ids": LongTensor, "r_ground", "r_count", "st"}] for one prompt group.

    Two channels normalized over the SAME group, routed to different tokens:
    the first k tokens (the <count> span) carry the count advantage, the rest
    carry the grounding advantage.
    """
    A_g = normalize([r["r_ground"] for r in ros], args.sigma_min)
    A_c = normalize([r["r_count"] for r in ros], args.sigma_min)
    out = []
    for i, ro in enumerate(ros):
        T = len(ro["ids"])
        k = count_span_token_end(tokenizer, ro["ids"]) if args.w_count > 0 else 0
        k = min(k, T)
        adv = torch.full((T,), float(A_g[i]))
        if k:
            adv[:k] = float(A_c[i]) * args.w_count
        ro["st"]["count_tokens"] = k
        out.append({"ids": ro["ids"], "adv": adv, "mask": torch.ones(T),
                    "kind": "main", "count_tokens": k})
    return out


def branch_micros(bp, conts: List[Tuple[torch.Tensor, str]], gt, w, sftm, etypes,
                  args, sid: str, step: int) -> Tuple[List[Dict], List[Dict], int]:
    """One branch group -> (micros, audit records, generated branch tokens credited).

    conts[0] is the ORIGINAL continuation c_0 of the source rollout; it is the
    leave-one-out baseline and receives no gradient unless --credit-original
    (it already got the main trajectory advantage).
    """
    recs = BA.branch_group_advantages(bp, conts, gt, w, sftm, etypes,
                                      tau=args.tau, sigma_min=args.branch_sigma_min)
    micros, n_tok = [], 0
    for rec, (cids, _ctxt) in zip(recs, conts):
        rec.update(prompt_id=sid, step=step, src_rollout=bp.src_rollout,
                   token_cut=bp.token_cut, target_gold=bp.target_gold,
                   target_etype=bp.target_etype, target_sft_missed=bp.target_sft_missed,
                   reason=bp.reason, n_cont_tokens=len(cids))
        if rec["is_original"] and not args.credit_original:
            continue
        if not rec["is_original"]:
            n_tok += len(cids)
        npre = bp.prefix_ids.numel()
        full = torch.cat([bp.prefix_ids.cpu().long(), cids.cpu().long()])
        adv = torch.zeros(full.numel())
        mask = torch.zeros(full.numel())
        # credit and KL apply ONLY after the branch point: the shared prefix is
        # identical across the group and carries no preference information
        adv[npre:] = rec["branch_advantage"]
        mask[npre:] = 1.0
        micros.append({"ids": full, "adv": adv, "mask": mask,
                       "kind": "branch", "count_tokens": 0})
    return micros, recs, n_tok


def step_record(args, step: int, epoch: int, stats: List[Dict], branch_recs: List[Dict],
                kll: Dict[str, float], tot_loss: float, n_micros: int, n_branch_pts: int,
                tok_main: int, tok_branch: int, elapsed_min: float) -> Dict:
    """One grpo_metrics.jsonl row; shared so both backends log the same schema."""
    n = len(stats)

    def mean(k):
        v = [r[k] for r in stats if k in r and r[k] == r[k]]   # drop nan
        return round(sum(v) / len(v), 4) if v else None

    rec = {
        "step": step, "arm": args.arm, "epoch": epoch,
        "r_ground": mean("r_ground"), "r_count": mean("r_count"),
        "mean_iou": mean("mean_iou"), "f1@0.5": mean("f1"),
        "miss": mean("miss"), "fp": mean("fp"), "fmt_rate": mean("fmt"),
        "count_parse_rate": mean("count_parsed"),
        "count_exact": mean("count_exact"), "count_mae": mean("count_abserr"),
        "compliance": mean("compliant"),
        "boxes_per_gold": round(sum(r["n_pred"] / max(1, r["n_gold"]) for r in stats) / max(1, n), 3),
        "trunc_rate": mean("trunc"),
        "kl": round(kll.get("kl", 0.0) / max(1, n_micros), 5),
        "kl_box": round(kll["kl_box"] / kll["n_box"], 5) if kll.get("n_box") else None,
        "kl_count_tok": round(kll["kl_count"] / kll["n_count"], 5) if kll.get("n_count") else None,
        "kl_branch": round(kll["kl_branch"] / kll["n_branch"], 5) if kll.get("n_branch") else None,
        "loss": round(tot_loss / max(1, n_micros), 5),
        "n_micro": n_micros, "n_rollouts": n, "branch_points": n_branch_pts,
        "tok_main": tok_main, "tok_branch": tok_branch,
        "tok_extra_frac": round(tok_branch / max(1, tok_main), 4),
        "elapsed_min": round(elapsed_min, 1),
    }
    aud = BA.audit(branch_recs) if branch_recs else {}
    rec.update({f"audit_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in aud.items()})
    return rec


def add_log(acc: Dict[str, float], lg: Dict[str, float]) -> None:
    for k, v in lg.items():
        acc[k] = acc.get(k, 0) + v


def masked_objective(lp: torch.Tensor, ref: torch.Tensor, micros: Sequence[Dict],
                     args, n_micro: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    """lp/ref: concatenated per-token logprobs of `micros` (same order).

    loss = sum_i [ sum_t m_it (pg_it + beta kl_it) / sum_t m_it ] / n_micro
    mu = 1, so ratio == 1 and the clip is inert, exactly as in GRPO-R.
    Returns the loss tensor (call .backward()) and KL sums per token class
    (spec section 13).
    """
    dev = lp.device
    adv = torch.cat([m["adv"] for m in micros]).to(dev)
    msk = torch.cat([m["mask"] for m in micros]).to(dev)
    lens = [int(m["ids"].numel()) for m in micros]
    ratio = torch.exp(lp - lp.detach())
    pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv)
    d = ref - lp
    kl = torch.exp(d) - d - 1.0
    L = torch.tensor(lens, device=dev)
    seg = torch.repeat_interleave(torch.arange(len(lens), device=dev), L)
    num = torch.zeros(len(lens), device=dev).index_add_(0, seg, (pg + args.kl_beta * kl) * msk)
    den = torch.zeros(len(lens), device=dev).index_add_(0, seg, msk).clamp(min=1.0)
    loss = (num / den).sum() / n_micro

    kd = kl.detach()
    log = {"kl": 0.0, "kl_box": 0.0, "n_box": 0, "kl_count": 0.0, "n_count": 0,
           "kl_branch": 0.0, "n_branch": 0}
    o = 0
    for m, n in zip(micros, lens):
        k_i, m_i = kd[o:o + n], msk[o:o + n]
        log["kl"] += float((k_i * m_i).sum() / m_i.sum().clamp(min=1.0))
        if m["kind"] == "branch":
            log["kl_branch"] += float((k_i * m_i).sum() / m_i.sum().clamp(min=1.0)); log["n_branch"] += 1
        else:
            kc = m["count_tokens"]
            if kc:
                log["kl_count"] += float(k_i[:kc].mean()); log["n_count"] += 1
            if kc < n:
                log["kl_box"] += float(k_i[kc:].mean()); log["n_box"] += 1
        o += n
    return loss, log
