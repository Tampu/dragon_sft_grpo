#!/usr/bin/env python3
"""GRPO for the completer (--task completer) and the coverage grounder (--task coverage). NEEDS A GPU.

Same systems path as demand_rl/train_omission_vllm.py (colocated vLLM engine, LoRA sync after every
update, packed scoring, its Dr.GRPO objective): advantage = R - group mean, over ONE batch-wide std,
loss = ((pg + beta * kl) * mask).sum() / (n_completions * l_const). The reference is the base with the
adapter disabled. What differs is the prompt and the reward, both in gc_common.py.

--base is the merged frozen grounder in both tasks, so the trained LoRA sits on the same weights.
--init-adapter continues an earlier LoRA with a fresh optimizer (Stage 2 completer).
--ref-adapter makes that PEFT adapter the KL reference instead of the disabled adapter. The coverage
grounder uses it with --base sft --init-adapter <vanilla> --ref-adapter <vanilla>: the vanilla LoRA keeps
training on the SFT weights, so the grounder is served exactly. Merging it first would not do: in bf16
about 70% of the merged weight entries do not change.
--completer-adapter closes the loop for --task coverage: each step the current grounder's greedy E and
its samples go through that frozen completer (a vLLM engine in its own process, on --completer-base), and part of the
reward (--alpha) is paid only for gold boxes the completer's final set hits and greedy did not. See reward_parts.
--l-const 0 measures it once before the first update; --l-const-from reuses another run's value.

PATCH after seed 17 (see CHANGES.md):
  --reward-version v2 (default): grounder reward = (1-alpha) z(cov) + alpha z(use) - mu cap - trunc_pen [trunc],
      see gc_common.grounder_terms / mix_terms. v1 reproduces seed 17 exactly.
  coverage: a probe of pool recall on fixed train items every --div-every steps; two consecutive drops of
      more than --div-tol below the start stop the phase (the grounder is getting cleaner, not more complete).
  completer: --curriculum-frac of each batch comes from states where the current completer errs
      (rejected gold, adopted junk, dropped gold, kept junk; rare misses weighted twice); G defaults to 8;
      after a format warm-up the warm-up adapter becomes the KL reference.
  --completer-gpu runs the feedback completer on its own GPU; K / prompt-format consistency is checked;
      the worker is always shut down (try/finally + process group).

Before step 1 the completer's format rate at T = 1.0 is measured on 64 states. If under
--warmup-below, every completion would score the same penalty, every advantage would be zero, and GRPO
could not start, so the LoRA first gets a supervised format warm-up: cross-entropy on targets drawn
without looking at gold (E, each box dropped w.p. 0.15, each pool item added w.p. 0.15). It stops at
--warmup-target and is written to args.json and the log. Above the threshold it never runs.

    CUDA_VISIBLE_DEVICES=5 /mnt/data2/traviku2/da_env2/bin/python completer_pilot/train_gc.py \
        --task completer --states states/train.jsonl --K 4 --output-dir completer --max-steps 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gc_common as G  # noqa: E402
from prompts_v4 import v4_budget  # noqa: E402
from gc_engine import GCEngine, RemoteCompleter, VB, VC, ROOT, fresh_vllm_lora, hf_pixels, hf_prompt_ids  # noqa: E402
sys.path.insert(0, str(ROOT / "demand_rl"))
from train_omission_vllm import omission_objective  # noqa: E402
import credit as CR  # noqa: E402
import omission_grpo as OG  # noqa: E402
from grpo_dragon6_v1 import RewardWeights, compute_reward, parse_boxes_r1  # noqa: E402

DIFF_W = {}          # id -> per-gold Assembler-difficulty weights (--difficulty); kept out of args.json

TASK_DEFAULTS = {
    "completer": dict(group_size=8, prompts_per_step=8, max_new_tokens=64, no_repeat_ngram_size=0, max_model_len=7168),
    "coverage": dict(group_size=8, prompts_per_step=2, max_new_tokens=640, no_repeat_ngram_size=6, max_model_len=6144),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASK_DEFAULTS))
    ap.add_argument("--states", default=None, help="completer: generate_states.py output")
    ap.add_argument("--jsonl", default=None, help="coverage: item list (make_items.py output)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--base", default="grounder2633")
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--ref-adapter", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--l-const", type=float, default=0.0)
    ap.add_argument("--l-const-from", default=None)
    ap.add_argument("--lam", type=float, default=0.3, help="coverage: junk brake")
    ap.add_argument("--completer-adapter", default=None, help="coverage: frozen completer giving feedback")
    ap.add_argument("--completer-base", default="grounder2633")
    ap.add_argument("--completer-gpu-mem", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.5, help="coverage: share of the signal from the completer")
    ap.add_argument("--completer-gpu", default=None, help="coverage: CUDA device for the feedback completer worker")
    ap.add_argument("--allow-mismatch", action="store_true", help="skip the completer K / prompt-format check")
    # v2 grounder reward
    ap.add_argument("--grpo-norm", default="dr", choices=["dr", "standard"],
                    help="dr: group-centred advantages over one batch std, token-sum loss / l_const (Dr. GRPO, this "
                         "trainer's default). standard: the original grpo_dragon6_v1 recipe -- per-group z-scored "
                         "advantages (r - mean) / (std + 1e-4), loss = mean over each rollout's tokens / n rollouts")
    ap.add_argument("--reward-version", default="v2", choices=["v1", "v2", "vanilla"],
                    help="coverage: vanilla = grpo_dragon6_v1.compute_reward, the equal-compute control for a G round")
    ap.add_argument("--difficulty", default=None,
                    help="coverage v2: build_difficulty.py output; cov pays w_g / n_g (Variant B of the co-evolution plan)")
    ap.add_argument("--lam-junk", type=float, default=0.0, help="coverage v2: minus lam_junk * unmatched share of boxes")
    ap.add_argument("--mu", type=float, default=1.0, help="v2: weight of the spray cap")
    ap.add_argument("--cap-mult", type=float, default=3.0, help="v2: box budget = cap_mult * max(|G|, 2)")
    ap.add_argument("--trunc-pen", type=float, default=0.5, help="v2: penalty for a truncated sample")
    ap.add_argument("--nobox-r", type=float, default=-2.0, help="v2: reward of a sample with no closed box")
    ap.add_argument("--no-e-junk", action="store_true", help="v2: do not charge proposers of junk kept from E")
    ap.add_argument("--sig-momentum", type=float, default=0.9)
    ap.add_argument("--z-clip", type=float, default=3.0)
    # pool-diversity guard (coverage)
    ap.add_argument("--div-every", type=int, default=50)
    ap.add_argument("--div-items", type=int, default=200)
    ap.add_argument("--div-tol", type=float, default=0.02)
    # completer curriculum and prompt
    ap.add_argument("--curriculum-frac", type=float, default=0.5)
    ap.add_argument("--curriculum-scan", type=int, default=2000)
    ap.add_argument("--curriculum-refresh", type=int, default=100)
    ap.add_argument("--rare-max", type=int, default=2)
    ap.add_argument("--max-pool", type=int, default=30, help="pool clusters listed to the completer (seed 17: 30)")
    ap.add_argument("--show-e-support", action="store_true",
                    help="list agreement counts on E lines; only for a completer trained this way")
    ap.add_argument("--no-ref-warmup", action="store_true", help="completer: keep the bare base as KL reference")
    ap.add_argument("--ablate", default=None, choices=["blank_sheet", "blank_visual"],
                    help="completer: train with a blank crop sheet / blank sheet AND diagram (the trained no-vision "
                         "controls of Gate 1)")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v3", "v3t", "v4", "v4m", "v4t", "v4z"],
                    help="completer prompt (coverage: the feedback completer's). v3 = EDGE guideline, reading order, "
                         "no support counts, plus a crop sheet as a second image; see prompts_v3.py / crops.py. "
                         "v4 = verdict per candidate + fix + add, no tags (v4m: numbers drawn on the diagram, "
                         "v4t: with current/proposed tags, v4z: + zoom sheet of the small candidates); see prompts_v4.py")
    # v4 Assembler (prompts_v4.py)
    ap.add_argument("--types", default="data_count/elements_grpo_rest.json",
                    help="v4: evidence type per gold box of the train states (for --ab-weight and the add order)")
    ap.add_argument("--ab-weight", type=float, default=0.5,
                    help="v4: reward += ab_weight * (answer-bearing recall of the final set - that of E)")
    ap.add_argument("--verify-credit", type=float, default=0.5,
                    help="v4: token-level credit; each verdict / add's tokens get advantage += this * (its credit - "
                         "the group's mean credit for the same candidate / adds); 0 = sequence-level only")
    ap.add_argument("--credit", default="label", choices=["label", "loo"],
                    help="v4 token credit: label = fixed scores per verdict class (candidate_scores); loo = leave-one-"
                         "out change of the set reward when that decision is flipped (prompts_v4.loo_scores)")
    ap.add_argument("--oracle-adds", type=int, default=8, help="v4 oracle warm-up: at most this many adds per target")
    # task-dependent; see TASK_DEFAULTS
    ap.add_argument("--group-size", type=int)
    ap.add_argument("--prompts-per-step", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--no-repeat-ngram-size", type=int)
    ap.add_argument("--max-model-len", type=int)
    # shared GRPO settings
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    # format warm-up (completer)
    ap.add_argument("--warmup-below", type=float, default=0.5)
    ap.add_argument("--warmup-target", type=float, default=0.8)
    ap.add_argument("--warmup-max-steps", type=int, default=400)
    ap.add_argument("--warmup-lr", type=float, default=1e-4)
    ap.add_argument("--warmup-mode", default="format", choices=["format", "oracle"],
                    help="oracle: always run a supervised warm-up on oracle-edit targets of the train states")
    ap.add_argument("--oracle-steps", type=int, default=300, help="oracle warm-up length (x --prompts-per-step items)")
    ap.add_argument("--oracle-target", default="strict", choices=["strict", "empty_e", "near"],
                    help="oracle warm-up target (gc_common.oracle_target); strict reproduces v3b")
    # systems
    ap.add_argument("--gpu-mem", type=float, default=0.25)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--ckpt-tokens", type=int, default=0)
    ap.add_argument("--sync-dir", default="/dev/shm/demand_sync")
    a = ap.parse_args()
    for k, v in TASK_DEFAULTS[a.task].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.task == "completer" and a.prompt != "v1":
        a.max_model_len = max(a.max_model_len, 8192)       # diagram (<= 13 tiles) + sheet (<= 7) + ~1k text tokens
    a.v4 = a.prompt in G.V4_PROMPTS
    if a.v4:
        if a.task != "completer":
            raise SystemExit("--prompt v4* is a completer prompt; the coverage task has no v4 feedback path")
        if a.warmup_mode != "oracle":
            raise SystemExit("--prompt v4* needs --warmup-mode oracle (format targets mean nothing for verdicts)")
        a.max_model_len = max(a.max_model_len, 9216)       # + up to ~1k output tokens (verdicts, fixes, adds)
        if a.prompt == "v4z":
            a.max_model_len = max(a.max_model_len, 11264)  # + zoom sheet (<= 7 x 256 tokens)
    if a.prompt != "v1" and a.show_e_support:
        raise SystemExit("--show-e-support has no meaning with --prompt v3 (no agreement counts are shown)")
    return a


def load_items(a, tok):
    if a.task == "completer":
        recs = [json.loads(l) for l in open(a.states) if l.strip()]
        if a.v4:
            types = G.load_gold_types(ROOT / a.types if not Path(a.types).exists() else a.types)
            for r in recs:
                t = types.get(r["id"])
                r["gold_types"] = t if t is not None and len(t) == len(r["gold"]) else None
            have = sum(r["gold_types"] is not None for r in recs)
            print(f"[types] gold types for {have}/{len(recs)} train states from {a.types}", flush=True)
        items = [G.completer_item(r, a.K, max_pool=a.max_pool, show_e_support=a.show_e_support, prompt=a.prompt)
                 for r in recs]
        for it in items:
            it["budget"] = v4_budget(tok, len(it["crops"])) if a.v4 else \
                G.select_budget(tok, len(it["E"]) + len(it["pool"]), a.max_new_tokens)
            if a.ablate:
                it["ablate"] = a.ablate
        return items
    items = VC.load_split(a.jsonl)
    for it in items:
        it["budget"] = a.max_new_tokens
    return items


@contextmanager
def reference_policy(model, ref_adapter):
    """KL reference: the "ref" adapter if given, else the bare base. set_adapter also restores
    requires_grad on the trained adapter when it switches back."""
    lm = model.language_model
    if not ref_adapter:
        with lm.disable_adapter():
            yield
        return
    lm.set_adapter("ref")
    try:
        yield
    finally:
        lm.set_adapter("default")


def batch_indices(step, pps, n, seed, perms):
    """Reshuffled pass after pass; a function of the step alone, so --resume needs no data state."""
    out = []
    for pos in range(step * pps, (step + 1) * pps):
        ep, i = divmod(pos, n)
        if ep not in perms:
            perms[ep] = random.Random(f"{seed}:{ep}").sample(range(n), n)
        out.append(perms[ep][i])
    return out


def make_batch(step, a, n, perms, err_idx):
    """Uniform batch, with --curriculum-frac of it drawn from the current error states (completer only)."""
    base = batch_indices(step, a.prompts_per_step, n, a.seed, perms)
    if a.task != "completer" or not err_idx or a.curriculum_frac <= 0:
        return base
    k = min(len(err_idx), int(round(a.prompts_per_step * a.curriculum_frac)))
    return random.Random(f"{a.seed}:err:{step}").sample(err_idx, k) + base[: a.prompts_per_step - k]


def scan_errors(engine, items, tok, a, step):
    """Greedy pass of the CURRENT completer over a random subset; returns indices of states where it errs.
    Rare misses (a rejected gold cluster with support <= --rare-max) are listed twice."""
    rng = random.Random(f"{a.seed}:scan:{step}")
    sub = rng.sample(range(len(items)), min(a.curriculum_scan, len(items)))
    part = [items[i] for i in sub]
    gens = engine.generate(part, 1, 0.0, [it["budget"] for it in part], 0)
    err, n_err, n_rare = [], 0, 0
    for i, it, g in zip(sub, part, gens):
        text = G.canonical(tok.decode(g["comps"][0]["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0], it)
        e = G.edit_errors(it["E"], it["pool"], it["gold"], G.parse_select(text, len(it["E"]) + len(it["pool"])),
                          a.rare_max)
        if e["error"] or e["malformed"]:
            err.append(i)
            n_err += 1
            if e["rare_miss"]:
                err.append(i)
                n_rare += 1
    return err, {"scan_n": len(sub), "err_share": round(n_err / max(1, len(sub)), 4),
                 "rare_share": round(n_rare / max(1, len(sub)), 4)}


def probe_pool_recall(engine, probe, tok, a):
    """Mean pool recall (greedy E + group_size samples) of the CURRENT grounder on fixed train items."""
    budgets = [it["budget"] for it in probe]
    greedy = engine.generate(probe, 1, 0.0, budgets, a.no_repeat_ngram_size)
    samp = engine.generate(probe, a.group_size, a.temperature, budgets, a.no_repeat_ngram_size)
    dec = lambda c: [list(b) for b in parse_boxes_r1(
        tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0])[0]]
    vals = []
    for it, gg, g in zip(probe, greedy, samp):
        E = dec(gg["comps"][0])
        vals.append(G.pool_recall(E, G.build_pool(E, [dec(c) for c in g["comps"]], max_pool=a.max_pool), it["gold"]))
    return round(sum(vals) / max(1, len(vals)), 4)


def completer_run_args(adapter):
    """args.json of the run that trained a completer adapter (its dir or the parent), or {}."""
    p = Path(adapter)
    for d in (p, p.parent):
        f = d / "args.json"
        if f.exists():
            try:
                return json.load(open(f))
            except (OSError, ValueError):
                return {}
    return {}


def score_completer_v4(it, comps, tok, a):
    """reward = Phi(final) - Phi(E) + ab_weight * (answer-bearing recall gain); per-candidate credit for the tokens."""
    from prompts_v4 import candidate_scores, loo_scores
    E, gold, shown, types = it["E"], it["gold"], it["crops"], it.get("gold_types")
    base, ab0 = G.phi(E, gold), G.type_recall(E, gold, types)
    out = []
    for c in comps:
        text = tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0]
        S, p = G.v4_final(text, it)
        if p is None:
            out.append({"r": G.FORMAT_PENALTY, "valid": 0.0})
            continue
        acts = [x for x, _ in p["verdict"].values()]
        ab1 = G.type_recall(S, gold, types)
        cs = candidate_scores(shown, gold, p)
        fixes = [cs[("v", d)] for d, (x, _) in p["verdict"].items() if x == "fix"]
        adds = [cs[k] for k in cs if k[0] == "a"]
        r = G.phi(S, gold) - base + (a.ab_weight * (ab1 - ab0) if ab0 is not None else 0.0)
        out.append({"r": r, "valid": 1.0, "bpg": len(S) / len(gold), "n_gold": len(gold),
                    "n_keep": acts.count("keep"), "n_drop": acts.count("drop"), "n_fix": len(fixes), "n_add": len(adds),
                    "fix_hit": sum(f == 1.0 for f in fixes), "add_hit": sum(adds),
                    "drop_junk": sum(cs[("v", d)] == 1.0 for d, (x, _) in p["verdict"].items() if x == "drop"),
                    "verify_acc": sum(cs.values()) / max(1, len(cs)), "ab_gain": None if ab0 is None else ab1 - ab0,
                    "_credit": loo_scores(shown, gold, p, G.phi) if a.credit == "loo" else cs,
                    "_spans": p["spans"], "_text": text})
    return out


def token_advantages(groups, A, a, tok):
    """Per-token advantages for v4: the sequence advantage, plus verify_credit * (credit - group mean of the same key)
    on the tokens of each verdict / add (prompts_v4.candidate_scores). Adds share one key per group ("a"), so an add
    is credited against the group's add precision. Falls back to the sequence advantage where tokens do not map."""
    from prompts_v4 import token_spans
    out, n_fallback = [], 0
    for g, adv in zip(groups, A):
        sc = g.get("sc") or [{}] * len(g["ros"])
        key = lambda k: k if k[0] == "v" else ("a",)
        pool_ = {}
        for s in sc:
            for k, v in s.get("_credit", {}).items():
                pool_.setdefault(key(k), []).append(v)
        mean = {k: sum(v) / len(v) for k, v in pool_.items()}
        rows = []
        for c, x, s in zip(g["ros"], adv, sc):
            t = [float(x)] * len(c["ids"])
            if a.verify_credit > 0 and "_credit" in s:
                m = token_spans(tok, c["ids"], s["_text"], s["_spans"])
                if m is None:
                    n_fallback += 1
                else:
                    for i, k in enumerate(m):
                        if k is not None and k in s["_credit"]:
                            t[i] += a.verify_credit * (s["_credit"][k] - mean[key(k)])
            rows.append(t)
        out.append(rows)
    return out, n_fallback


def group_z(r):
    """grpo_dragon6_v1's advantage: (r - mean) / (std + 1e-4) within the group, unbiased std as torch.std."""
    import numpy as np
    r = np.asarray(r, float)
    return (r - r.mean()) / ((r.std(ddof=1) if len(r) > 1 else 0.0) + 1e-4)


def standard_objective(lp, ref, micros, a, n_main):
    """grpo_dragon6_v1's loss: per rollout, mean over its tokens of the clipped PG + beta * k3 KL, summed and
    divided by the number of rollouts (one update per batch, so ratio = 1 and the clip is inert, as there)."""
    import torch
    dev = lp.device
    ratio = torch.exp(lp - lp.detach())
    adv = torch.cat([m["adv"] for m in micros]).to(dev)
    pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - a.clip_eps, 1 + a.clip_eps) * adv)
    d = ref - lp
    kl = torch.exp(d) - d - 1.0
    tok = pg + a.kl_beta * kl
    loss, o, log = 0.0, 0, {"kl": 0.0, "kl_box": 0.0, "n_box": 0}
    for m in micros:
        n = int(m["ids"].numel())
        loss = loss + tok[o:o + n].mean() / n_main
        v = float(kl[o:o + n].detach().mean())
        log["kl"] += v; log["kl_box"] += v; log["n_box"] += 1
        o += n
    return loss, log


def score_completer(it, comps, tok, a):
    if a.v4:
        return score_completer_v4(it, comps, tok, a)
    E, pool, gold = it["E"], it["pool"], it["gold"]
    base = G.phi(E, gold)
    out = []
    for c in comps:
        full = tok.decode(c["ids"], skip_special_tokens=False)
        text = G.canonical(full.split(VC.EOS_TOKEN)[0], it)
        idx = G.parse_select(text, len(E) + len(pool))
        if idx is None:
            out.append({"r": G.FORMAT_PENALTY, "valid": 0.0})
            continue
        S = G.selected(E, pool, idx)
        st = G.edit_stats(E, pool, gold, idx)
        st.pop("by_support")
        out.append({"r": G.phi(S, gold) - base, "valid": 1.0, "bpg": len(S) / len(gold), "n_gold": len(gold), **st})
    return out


def feedback_items(batch, gens, greedy, tok, a):
    """Completer items for a batch: this grounder's greedy E + its samples -> pool."""
    items = []
    for it, g, gg in zip(batch, gens, greedy):
        boxes = lambda c: [list(b) for b in parse_boxes_r1(
            tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0])[0]]
        q, ans = G.split_question(it["user"])
        items.append(G.completer_item({"id": it["id"], "domain": it["domain"], "image": it["image"], "question": q,
                                       "answer": ans, "gold": it["gold"], "E": boxes(gg["comps"][0]),
                                       "samples": [boxes(c) for c in g["comps"]]}, a.group_size,
                                      max_pool=a.max_pool, show_e_support=a.show_e_support, prompt=a.prompt))
    return items


def select_finals(items, comp_engine, tok):
    """The completer's greedy step, with the settings eval_completer.py uses. (final, idx, malformed, text) per item;
    idx is None when malformed, and the final set is then E."""
    budgets = [G.select_budget(tok, len(i["E"]) + len(i["pool"])) for i in items]
    out = []
    for i, r in zip(items, comp_engine.generate(items, 1, 0.0, budgets, 0)):
        text = G.canonical(tok.decode(r["comps"][0]["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0], i)
        out.append((*G.select_with_idx(text, i["E"], i["pool"]), text))
    return out


def completer_finals(batch, gens, engine, comp_engine, tok, a):
    """The pipeline the grounder serves: its greedy E + these samples -> pool -> completer greedy -> final set."""
    greedy = engine.generate(batch, 1, 0.0, [it["budget"] for it in batch], a.no_repeat_ngram_size)
    items = feedback_items(batch, gens, greedy, tok, a)
    res = select_finals(items, comp_engine, tok)
    return ([{"final": f, "idx": idx, "E": i["E"], "pool": i["pool"], "hits_E": G.hungarian_hits(i["E"], i["gold"])}
             for i, (f, idx, _, _) in zip(items, res)],
            sum(m for _, _, m, _ in res))


def score_coverage(it, comps, tok, a, fb=None):
    gold = it["gold"]
    parsed = []
    for c in comps:
        text = tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0]
        boxes, _, unclosed = parse_boxes_r1(text)
        parsed.append(([list(b) for b in boxes], bool(boxes) and not unclosed and not c["trunc"]))
    gh = G.group_hit_counts([p for p, _ in parsed], gold)
    out = []
    for P, ok in parsed:
        hit = G.hungarian_hits(P, gold)
        parts = G.reward_parts(P, gold, gh, fb["final"], fb["hits_E"], a.alpha, a.lam, ok) if fb else \
            {"r": G.coverage_reward(P, gold, gh, a.lam, ok)}
        out.append({**parts, "valid": float(ok),
                    "cov": sum(1.0 / gh[g] for g in hit) / len(gold),
                    "junk": (len(P) - len(hit)) / max(1, len(P)),
                    "recall": len(hit) / len(gold), "bpg": len(P) / len(gold), "n_gold": len(gold)})
    return out


def score_vanilla(it, comps, tok):
    """The original GRPO reward (grpo_dragon6_v1.compute_reward): the equal-compute control of a Grounder round."""
    out = []
    for c in comps:
        text = tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0]
        r = compute_reward(text, it["gold"], RewardWeights())
        P = [list(b) for b in parse_boxes_r1(text)[0]]
        hit = G.hungarian_hits(P, it["gold"])
        out.append({"r": float(r["reward"]), "valid": float(r["fmt"] > 0 and not r["unclosed"] and not c["trunc"]),
                    "recall": len(hit) / max(1, len(it["gold"])), "junk": (len(P) - len(hit)) / max(1, len(P)),
                    "bpg": len(P) / max(1, len(it["gold"])), "n_gold": len(it["gold"])})
    return out


def score_coverage_v2(it, comps, tok, a, fb=None):
    """Raw v2 terms for one group (rewards are mixed batch-wide in main): closed boxes count even when the
    sample was truncated, as at eval; truncation and no-box are penalised in gc_common.mix_terms."""
    P_list, has_box, trunc = [], [], []
    for c in comps:
        text = tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0]
        boxes, _, unclosed = parse_boxes_r1(text)
        P_list.append([list(b) for b in boxes])
        has_box.append(bool(boxes))
        trunc.append(bool(unclosed or c["trunc"]))
    w = DIFF_W.get(it["id"])
    t = G.grounder_terms(P_list, has_box, trunc, it["gold"], fb, a.cap_mult, use_e_junk=not a.no_e_junk,
                         weights=w if w is not None and len(w) == len(it["gold"]) else None)
    if fb:                                    # seed-17 metrics, for comparison across reward versions
        hF, hE = G.hungarian_hits(fb["final"], it["gold"]), set(fb["hits_E"])
        for P, row in zip(P_list, t["rows"]):
            hit = G.hungarian_hits(P, it["gold"])
            row.update(n_hit_new=len(hit - hE), n_hit_E=len(hit & hE),
                       n_acc_pool=len((hit & hF) - hE), n_acc_E=len(hit & hF & hE))
    return t


def step_summary(a, stats):
    def mean(k):
        v = [s[k] for s in stats if s.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None

    def ratio(k, n):
        d = sum(s.get(n, 0) for s in stats)
        return round(sum(s.get(k, 0) for s in stats) / d, 4) if d else None
    rec = {"reward": mean("r"), "valid": mean("valid"), "bpg": mean("bpg")}
    for lab, lo, hi in G.BINS:
        v = [s["bpg"] for s in stats if "bpg" in s and lo <= s["n_gold"] <= hi]
        rec[f"bpg_{lab}"] = round(sum(v) / len(v), 3) if v else None
    if a.task == "completer":
        rec.update(adds=mean("n_add"), drops=mean("n_drop"), add_prec=ratio("add_hit", "n_add"),
                   drop_prec=ratio("drop_junk", "n_drop"), keep_all=mean("keep_all"))
        if a.v4:
            rec.update(fixes=mean("n_fix"), fix_prec=ratio("fix_hit", "n_fix"), verify_acc=mean("verify_acc"),
                       ab_gain=mean("ab_gain"))
    else:
        rec.update(cov=mean("cov"), junk=mean("junk"), recall=mean("recall"), uncond=mean("uncond"), cond=mean("cond"),
                   accepted_from_pool=ratio("n_acc_pool", "n_hit_new"), accepted_from_E=ratio("n_acc_E", "n_hit_E"),
                   use=mean("use"), cap=mean("cap"))
        if any("trunc" in s for s in stats):
            rec.update(trunc=round(sum(s.get("trunc", 0) for s in stats) / len(stats), 4),
                       nobox=round(sum(not s.get("has_box", True) for s in stats) / len(stats), 4))
    return rec


def format_check(engine, items, tok, a, n, G_):
    part = items[:n]
    gens = engine.generate(part, G_, a.temperature, [it["budget"] for it in part], a.no_repeat_ngram_size)
    score = score_completer if a.task == "completer" else score_coverage
    valid, lens, shown, rew = [], [], [], []
    for it, g in zip(part, gens):
        for s, c in zip(score(it, g["comps"], tok, a), g["comps"]):
            valid.append(s["valid"])
            lens.append(len(c["ids"]))
            rew.append(s.get("r", 0.0))
        if len(shown) < 3:
            shown.append(tok.decode(g["comps"][0]["ids"], skip_special_tokens=False)[:120])
    return sum(valid) / len(valid), sum(lens) / len(lens), shown, sum(rew) / len(rew)


def format_warmup(model, engine, items, tok, a, img_id, ctl, sync_root, params, dev):
    """--warmup-mode format: teach the output format only (E with random drops, random pool adds); runs when
    GRPO would otherwise see all-equal rewards, stops at --warmup-target.
    --warmup-mode oracle: supervised on the oracle edit of each TRAIN state (E boxes that hit gold + pool clusters
    hitting the rest), --oracle-steps steps, no early stop. With the v3 prompt the format targets are noise to the
    model (it cannot tell E from pool), which is what sank the first v3 run; the oracle targets carry the task."""
    import torch
    from packed import packed_logprobs
    rng = random.Random(a.seed)
    opt = torch.optim.AdamW(params, lr=a.warmup_lr, weight_decay=0.0, betas=(0.9, 0.95))
    eos = tok.convert_tokens_to_ids(VC.EOS_TOKEN)
    pps, n = a.prompts_per_step, len(items)
    n_steps = a.oracle_steps if a.warmup_mode == "oracle" else a.warmup_max_steps
    rate = 0.0
    for step in range(n_steps):
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for it in rng.sample(items, min(pps, n)):
            if a.v4:
                from prompts_v4 import format_v4, oracle_v4
                target = format_v4(*oracle_v4(it["crops"], it["gold"], it.get("gold_types"),
                                              max_add=a.oracle_adds, E_disp=it["E_disp"]))
            else:
                if a.warmup_mode == "oracle":
                    idx = G.oracle_target(it["E"], it["pool"], it["gold"], a.oracle_target)
                else:
                    nE = len(it["E"])
                    idx = [i + 1 for i in range(nE) if rng.random() >= 0.15]
                    idx += [nE + i + 1 for i in range(len(it["pool"])) if rng.random() < 0.15]
                target = G.select_text(G.to_display(idx, it))
            ids = torch.tensor(tok(target, add_special_tokens=False).input_ids + [eos], device=dev)
            pv, nt = hf_pixels(it)
            pid = torch.tensor(hf_prompt_ids(it, tok, nt))
            with torch.no_grad():
                vit = model.extract_feature(pv.to(dev))
            ctl["on"] = True
            lp, _ = packed_logprobs(model, vit, pid.to(dev), [ids], img_id)
            loss = -lp.mean() / pps
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        opt.step()
        if (step + 1) % 20 == 0:
            VB.sync_lora(engine, model, sync_root / f"w{step + 1}", a.lora_rank)
            shutil.rmtree(sync_root / f"w{step - 19}", ignore_errors=True)
            rate, _, shown, rew = format_check(engine, items, tok, a, 32, a.group_size)
            print(f"[warmup {step + 1}] ce={tot:.3f} valid@T1={rate:.3f} edit_gain@T1={rew:+.4f} e.g. {shown[0]!r}",
                  flush=True)
            if a.warmup_mode == "format" and rate >= a.warmup_target:
                return step + 1, rate
    return n_steps, rate


def main():
    a = parse_args()
    VC.setup_caches()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from packed import packed_logprobs
    dev = "cuda:0"
    torch.manual_seed(a.seed)

    out = Path(a.output_dir)
    if (out / "grpo_metrics.jsonl").exists() and not a.resume:
        raise SystemExit(f"{out} already has a run: pick a new --output-dir or pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    old_args = json.load(open(out / "args.json")) if a.resume and (out / "args.json").exists() else {}
    if not a.ref_adapter and old_args.get("ref_adapter"):
        a.ref_adapter = old_args["ref_adapter"]          # e.g. the warm-up adapter chosen on the first launch

    # the feedback completer must see the prompt format and K it was trained with
    if a.task == "coverage" and a.completer_adapter and not a.allow_mismatch:
        ca = completer_run_args(a.completer_adapter)
        if ca.get("K") is not None and int(ca["K"]) != a.group_size:
            raise SystemExit(f"completer trained with K={ca['K']} but the pool here has {a.group_size} samples "
                             f"(--group-size); pass --allow-mismatch to override")
        if bool(ca.get("show_e_support", False)) != a.show_e_support:
            raise SystemExit("completer prompt format (show_e_support) differs from this run; --allow-mismatch to override")
        if int(ca.get("max_pool", 30)) != a.max_pool:
            raise SystemExit(f"completer trained with --max-pool {ca.get('max_pool', 30)}, this run lists {a.max_pool}")
        if ca.get("prompt", "v1") != a.prompt:
            raise SystemExit(f"completer trained with --prompt {ca.get('prompt', 'v1')}, this run uses {a.prompt}")

    mdir = VB.resolve_base(a.base)
    if not (mdir / "DONE").exists():
        raise SystemExit(f"no merged base at {mdir}: run completer_pilot/export_grounder.py first")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(mdir), trust_remote_code=True, use_fast=False)
    items = load_items(a, tok)
    if a.difficulty:
        if a.task != "coverage" or a.reward_version != "v2":
            raise SystemExit("--difficulty applies to --task coverage --reward-version v2 only")
        DIFF_W.update({k: v["w"] for k, v in json.load(open(a.difficulty))["weights"].items()})
        have = sum(it["id"] in DIFF_W and len(DIFF_W[it["id"]]) == len(it["gold"]) for it in items)
        if have < 0.99 * len(items):
            raise SystemExit(f"--difficulty covers {have} of {len(items)} training items (ids or gold counts differ)")
        print(f"[difficulty] weights for {have}/{len(items)} items from {a.difficulty}", flush=True)
    n_img = 1 if a.prompt == "v1" else 3 if a.prompt == "v4z" else 2
    engine = GCEngine(mdir, tok, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len, lora_rank=a.lora_rank,
                      seed=a.seed, max_num_seqs=a.max_num_seqs, limit_images=n_img if a.task == "completer" else 1)
    comp_engine, comp_tmp = None, None
    sync_root = Path(a.sync_dir) / f"gc_{out.name}_{os.getpid()}"
    try:
        if a.task == "coverage" and a.completer_adapter:
            cb = VB.resolve_base(a.completer_base)
            comp_engine = RemoteCompleter(cb, a.completer_gpu_mem, seed=a.seed, max_num_seqs=a.max_num_seqs,
                                          gpu=a.completer_gpu, limit_images=n_img,
                                          max_model_len=7168 if a.prompt == "v1" else 8192)
            comp_tmp = fresh_vllm_lora(a.completer_adapter, f"{out.name}_feedback")
            comp_engine.set_lora(comp_tmp)
        _train(a, out, old_args, mdir, tok, items, engine, comp_engine, sync_root, torch, packed_logprobs, dev)
    finally:
        shutil.rmtree(sync_root, ignore_errors=True)
        if comp_engine is not None:
            comp_engine.close()
        if comp_tmp:
            shutil.rmtree(comp_tmp, ignore_errors=True)


def _train(a, out, old_args, mdir, tok, items, engine, comp_engine, sync_root, torch, packed_logprobs, dev):
    model, _, ctl = VB.load_policy_merged(mdir, a.lora_rank, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if a.ref_adapter:
        model.language_model.load_adapter(str(a.ref_adapter), adapter_name="ref", is_trainable=False)
        model.language_model.set_adapter("default")
    img_id = tok.convert_tokens_to_ids(VC.IMG_CONTEXT)
    score = score_completer if a.task == "completer" else score_coverage
    pps, K_G = a.prompts_per_step, a.group_size
    v2 = a.task == "coverage" and a.reward_version == "v2"

    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    start, warm_steps = 0, 0
    a.valid_at_init = old_args.get("valid_at_init")
    if a.resume and (out / "trainer_state.pt").exists():
        st = torch.load(out / "trainer_state.pt", map_location="cpu", weights_only=False)
        start, a.l_const = st["step"], st["l_const"]
        set_peft_model_state_dict(model.language_model,
                                  load_file(str(out / f"grpo-step{start}" / "adapter_model.safetensors")))
        VB.sync_lora(engine, model, sync_root / f"s{start}", a.lora_rank)
        print(f"[resume] from step {start}, l_const={a.l_const}", flush=True)
    else:
        if a.init_adapter:
            set_peft_model_state_dict(model.language_model,
                                      load_file(str(Path(a.init_adapter) / "adapter_model.safetensors")))
            VB.sync_lora(engine, model, sync_root / "init", a.lora_rank)
            print(f"[init] LoRA from {a.init_adapter}", flush=True)
        n_chk = 64 if a.task == "completer" else 16
        rate, mean_len, shown, rew = format_check(engine, items, tok, a, n_chk, K_G)
        a.valid_at_init = rate
        print(f"[format] valid at T={a.temperature}: {rate:.3f}, edit gain {rew:+.4f}, mean length {mean_len:.1f} tokens\n  " +
              "\n  ".join(repr(s) for s in shown), flush=True)
        if a.task == "completer" and (rate < a.warmup_below or a.warmup_mode == "oracle"):
            print(f"[warmup] mode={a.warmup_mode}, valid at init {rate:.3f}", flush=True)
            warm_steps, rate = format_warmup(model, engine, items, tok, a, img_id, ctl, sync_root, params, dev)
            if rate < a.warmup_below:
                raise SystemExit(f"[warmup] valid rate still {rate:.3f} after {warm_steps} steps")
            VB.sync_lora(engine, model, sync_root / "warm", a.lora_rank)
            model.language_model.save_pretrained(str(out / "warmup"))
            _, mean_len, _, rew = format_check(engine, items, tok, a, n_chk, K_G)
            print(f"[warmup] done: edit gain at T={a.temperature} {rew:+.4f}", flush=True)
            if not a.ref_adapter and not a.no_ref_warmup:
                # KL to the bare base would pull every <select> token toward a model that never emits them
                model.language_model.load_adapter(str(out / "warmup"), adapter_name="ref", is_trainable=False)
                model.language_model.set_adapter("default")
                a.ref_adapter = str(out / "warmup")
                print(f"[warmup] KL reference = {a.ref_adapter}", flush=True)
        if a.l_const_from:
            a.l_const = json.load(open(Path(a.l_const_from) / "args.json"))["l_const"]
        elif not a.l_const:
            a.l_const = round(mean_len, 2)
    a.warmup_steps = warm_steps if not old_args else old_args.get("warmup_steps", warm_steps)

    # pool-diversity guard (coverage): baseline pool recall on fixed train items, before any update
    probe = []
    if a.task == "coverage" and a.div_every > 0:
        probe = [items[i] for i in random.Random(f"{a.seed}:probe").sample(range(len(items)),
                                                                           min(a.div_items, len(items)))]
        a.div_base = old_args.get("div_base")
        if a.div_base is None:
            a.div_base = probe_pool_recall(engine, probe, tok, a)
        print(f"[diversity] pool recall at start on {len(probe)} train items: {a.div_base}", flush=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2, default=str))

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    if start:
        opt.load_state_dict(st["opt"])
    metrics_f = open(out / "grpo_metrics.jsonl", "a")
    io = ThreadPoolExecutor(4)
    perms, t_run = {}, time.time()
    sig, err_idx, below, scan_rec = {}, [], 0, {}
    print(f"[gc] task={a.task} reward={a.reward_version if a.task == 'coverage' else '-'} {a.max_steps} steps x {pps} "
          f"x G={K_G}, base={mdir}, l_const={a.l_const}, warmup_steps={warm_steps}, {len(items)} items", flush=True)

    step = start - 1
    for step in range(start, a.max_steps):
        t0 = time.time()
        if a.task == "completer" and a.curriculum_frac > 0 and (step == start or (step - start) % a.curriculum_refresh == 0):
            err_idx, scan_rec = scan_errors(engine, items, tok, a, step)
            print(f"[curriculum] step {step}: {scan_rec}", flush=True)
        batch = [items[i] for i in make_batch(step, a, len(items), perms, err_idx)]
        prep = io.submit(lambda b=batch: [hf_pixels(it) for it in b])
        gens = engine.generate(batch, K_G, a.temperature, [it["budget"] for it in batch],
                               a.no_repeat_ngram_size, logprobs=True)
        pvs = prep.result()
        finals, comp_bad = [None] * len(batch), 0
        if comp_engine is not None:
            finals, comp_bad = completer_finals(batch, gens, engine, comp_engine, tok, a)
        t_gen = time.time()

        groups, stats, metas, v2_rows, n_bad_prompt = [], [], [], [], 0
        for it, g, (pv, n_tiles), final in zip(batch, gens, pvs, finals):
            pid = hf_prompt_ids(it, tok, n_tiles)
            if pid != g["prompt_ids"]:
                n_bad_prompt += 1      # HF would score a different context than vLLM sampled from
                continue
            if v2:
                t = score_coverage_v2(it, g["comps"], tok, a, final)
                for s, c in zip(t["rows"], g["comps"]):
                    s["n_tokens"] = len(c["ids"])
                v2_rows.append(t["rows"])
                metas.append(t["meta"])
                stats.extend(t["rows"])
                groups.append({"ros": g["comps"], "r": None, "pv": pv, "pid": pid})
                continue
            if a.task == "coverage":
                sc = score_vanilla(it, g["comps"], tok) if a.reward_version == "vanilla" else \
                    score_coverage(it, g["comps"], tok, a, final)
            else:
                sc = score(it, g["comps"], tok, a)
            for s, c in zip(sc, g["comps"]):
                s["n_tokens"] = len(c["ids"])
            if a.task == "completer":
                for s in sc:
                    if s["valid"]:
                        s["keep_all"] = float(s["n_drop"] == 0 and s["n_add"] == 0 and s.get("n_fix", 0) == 0)
            stats.extend(sc)
            groups.append({"ros": g["comps"], "r": [s["r"] for s in sc], "pv": pv, "pid": pid, "sc": sc})
        if not groups:
            print(f"[warn] step {step + 1}: no scoreable prompts", flush=True)
            continue
        mix = {}
        if v2:
            mix = G.mix_terms(v2_rows, a.alpha, sig, a.mu, a.trunc_pen, a.nobox_r, a.sig_momentum, a.z_clip,
                              lam_junk=a.lam_junk)
            for grp, rr, rows in zip(groups, mix["r"], v2_rows):
                grp["r"] = rr
                for s, r in zip(rows, rr):
                    s["r"] = r

        A = OG.seq_advantages([g["r"] for g in groups]) if a.grpo_norm == "dr" else \
            [group_z(g["r"]) for g in groups]
        TA, n_fallback = token_advantages(groups, A, a, tok) if a.v4 else (None, 0)
        n_main = sum(len(g["ros"]) for g in groups)
        opt.zero_grad(set_to_none=True)
        tot_loss, kll, gap_num, gap_den = 0.0, {}, 0.0, 0.0
        for gi, (g, adv) in enumerate(zip(groups, A)):
            micros = [{"ids": torch.tensor(c["ids"], dtype=torch.long),
                       "adv": torch.tensor(TA[gi][j], dtype=torch.float32) if TA is not None else
                       torch.full((len(c["ids"]),), float(x), dtype=torch.float32),
                       "mask": torch.ones(len(c["ids"]))} for j, (c, x) in enumerate(zip(g["ros"], adv))]
            with torch.no_grad():
                vit = model.extract_feature(g["pv"].to(dev))
            P = torch.tensor(g["pid"], device=dev)
            C = [m["ids"].to(dev) for m in micros]
            ctl["on"] = P.numel() + sum(len(c) for c in C) > a.ckpt_tokens
            with torch.no_grad(), reference_policy(model, a.ref_adapter):
                ref, _ = packed_logprobs(model, vit, P, C, img_id)
            lp, lens = packed_logprobs(model, vit, P, C, img_id)
            loss, lg = (omission_objective if a.grpo_norm == "dr" else standard_objective)(lp, ref, micros, a, n_main)
            loss.backward()
            tot_loss += loss.item() * n_main
            CR.add_log(kll, lg)
            o, lpd = 0, lp.detach().float().cpu()
            for c, L in zip(g["ros"], lens):
                if c["lps"]:
                    k = min(len(c["lps"]), L)
                    gap_num += float((lpd[o:o + k] - torch.tensor(c["lps"][:k])).abs().sum())
                    gap_den += k
                o += L
            del vit, ref, lp
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
        opt.step()
        t_train = time.time()
        VB.sync_lora(engine, model, sync_root / f"s{step + 1}", a.lora_rank)
        shutil.rmtree(sync_root / f"s{step - 1}", ignore_errors=True)

        rec = {"step": step + 1, **step_summary(a, stats),
               "kl": round(kll.get("kl", 0.0) / max(1, kll.get("n_box", 0)), 5),
               "loss": round(tot_loss / max(1, n_main), 5),
               "mean_tokens": round(sum(s["n_tokens"] for s in stats) / max(1, len(stats)), 1),
               "lp_gap": round(gap_num / max(1.0, gap_den), 4), "grad_norm": round(gnorm, 4),
               "prompt_id_mismatch": n_bad_prompt, "completer_malformed": comp_bad, "gen_s": round(t_gen - t0, 2),
               "credit_fallback": n_fallback if a.v4 else None,
               "train_s": round(t_train - t_gen, 2), "step_s": round(time.time() - t0, 2),
               "elapsed_min": round((time.time() - t_run) / 60, 1)}
        if v2:
            n_ad = sum(m.get("n_adopt", 0) for m in metas)
            rec.update(use_share=mix["use_share"], sd_cov=round(mix["sd_cov"], 5), sd_use=round(mix["sd_use"], 5),
                       use_nz=round(sum(s["use"] != 0 for s in stats) / max(1, len(stats)), 4),
                       adopt_per_prompt=round(n_ad / max(1, len(metas)), 3),
                       adopt_prec=round(sum(m.get("n_adopt_hit", 0) for m in metas) / n_ad, 4) if n_ad else None,
                       ekept_junk_per_prompt=round(sum(m.get("n_ekept_junk", 0) for m in metas) / max(1, len(metas)), 3))
        if scan_rec:
            rec.update({f"cur_{k}": v for k, v in scan_rec.items()})
        stop_div = False
        if probe and (step + 1) % a.div_every == 0:
            pr = probe_pool_recall(engine, probe, tok, a)
            rec["pool_recall_probe"] = pr
            below = below + 1 if pr < a.div_base - a.div_tol else 0
            stop_div = below >= 2
        eta = (time.time() - t_run) / (step + 1 - start) * (a.max_steps - step - 1) / 3600
        print(f"[{a.task} step {step + 1}/{a.max_steps}] " +
              " ".join(f"{k}={v}" for k, v in rec.items() if k != "step" and v is not None) +
              f" eta_h={eta:.2f}", flush=True)
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

        if (step + 1) % a.save_every == 0 or step + 1 == a.max_steps or stop_div:
            ck = out / f"grpo-step{step + 1}"
            model.language_model.save_pretrained(str(ck), selected_adapters=["default"])
            tok.save_pretrained(str(ck))
            torch.save({"step": step + 1, "opt": opt.state_dict(), "l_const": a.l_const}, out / "trainer_state.pt")
            print(f"[ckpt] {ck}", flush=True)
        if stop_div:
            msg = (f"stopped at step {step + 1}: pool recall {rec['pool_recall_probe']} fell more than {a.div_tol} "
                   f"below the start ({a.div_base}) on two consecutive probes")
            print(f"[diversity] {msg}", flush=True)
            (out / "DIVERSITY_STOP").write_text(msg + "\n")
            break

    (out / "DONE").write_text(f"{step + 1} steps, {(time.time() - t_run) / 3600:.2f} h\n")
    print(f"done: task={a.task} {step + 1} steps in {(time.time() - t_run) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
