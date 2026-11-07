#!/usr/bin/env python3
"""Omission pilot P0 / P1 / P2 with vLLM rollouts (omission_pilot/PILOT.md, sections 2a-2c).

Same systems path as train_count_grpo_vllm.py (colocated vLLM engine, LoRA sync after every
update, packed scoring) and the same grounding reward. What differs:

  * no count token: original SFT init, vanilla prompt, the data_i35 pool;
  * A_seq = centered per group, ONE batch-wide std, no sigma_min gate;
  * per-token advantages from omission_grpo (P0: A_seq only; P1: + box; P2: + box + stop);
  * loss = ((pg + beta * kl) * mask).sum() / (n_main * l_const);
  * no branching.

--l-const is the mean completion length of the init policy, measured once before training.
--debug-adv prints one completion's tokens with their advantages at step 1.

    CUDA_VISIBLE_DEVICES=5 /mnt/data2/traviku2/da_env2/bin/python demand_rl/train_omission_vllm.py \
        --arm p2 --l-const 150 --output-dir outputs/p2 --max-steps 1000 --save-every 100
"""
from __future__ import annotations

import os

_WS = int(os.environ.get("WORLD_SIZE", 1))

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "dragon_grpo"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "omission_pilot"))

import vllm_backend as VB  # noqa: E402  (puts VEP/scripts on sys.path first)
from grpo_dragon6_v1 import RewardWeights  # noqa: E402
from count_reward import score_rollout, set_matching  # noqa: E402
import branch_sampler as BS  # noqa: E402
import credit as CR  # noqa: E402
import omission_grpo as OG  # noqa: E402

BINS = [("1-2", 1, 2), ("3-8", 3, 8), ("9+", 9, 10 ** 6)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["p0", "p1", "p2"])
    ap.add_argument("--l-const", type=float, required=True)
    ap.add_argument("--init-name", default="sft")
    ap.add_argument("--jsonl", default=str(ROOT / "data_i35/i35_grpo_rest_v4.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    # credit
    ap.add_argument("--beta-box", type=float, default=2.0)
    ap.add_argument("--beta-stop", type=float, default=2.0)
    ap.add_argument("--credit-mode", default="add", choices=["add", "replace"])
    ap.add_argument("--shat-reduce", default="max", choices=["max", "mean"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--debug-adv", action="store_true")
    # vLLM / systems
    ap.add_argument("--gpu-mem", type=float, default=0.25)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--ckpt-tokens", type=int, default=7168)
    ap.add_argument("--sync-dir", default="/dev/shm/demand_sync")
    # GRPO
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--w-iou", type=float, default=0.45)
    ap.add_argument("--w-f1", type=float, default=0.15)
    ap.add_argument("--w-fmt", type=float, default=0.05)
    ap.add_argument("--w-miss", type=float, default=0.25)
    ap.add_argument("--w-fp", type=float, default=0.10)
    ap.add_argument("--matching", default="greedy", choices=["greedy", "hungarian"])
    # misc
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def credit_config(args) -> OG.CreditConfig:
    return OG.CreditConfig(tau=args.tau, gamma=args.gamma, kappa=args.kappa,
                           beta_box=args.beta_box, beta_stop=args.beta_stop, mode=args.credit_mode,
                           use_box=args.arm in ("p1", "p2"), use_stop=args.arm == "p2",
                           shat_reduce=args.shat_reduce)


def token_spans(tok, ids: List[int]) -> List[Tuple[int, int]]:
    """omission_grpo.char_spans without its quadratic decode, on the same text."""
    spans, o = [], 0
    for p in BS._token_texts(tok, ids):
        spans.append((o, o + len(p)))
        o += len(p)
    return spans


def batch_advantages(groups: List[Dict], tok, cfg: OG.CreditConfig) -> Tuple[List[List], Dict[str, float]]:
    """groups: [{"gold", "ros": [{"ids" (list), "full" (text incl. EOS), "r", "trunc"}]}].

    Returns per-token advantages per group/rollout, and the online sign rates.
    """
    A = OG.seq_advantages([[ro["r"] for ro in g["ros"]] for g in groups])
    parsed = [[OG.parse_boxes(ro["full"]) for ro in g["ros"]] for g in groups]
    creds = [OG.group_credits([p.boxes for p in P], g["gold"], [ro["trunc"] for ro in g["ros"]], cfg)
             for g, P in zip(groups, parsed)]
    sb, ss = OG.credit_scales(creds)
    advs, hit, nonhit, stops = [], [], [], []
    for g, a, P, cr in zip(groups, A, parsed, creds):
        out = []
        for ro, ai, pb, rc in zip(g["ros"], a, P, cr):
            spans = token_spans(tok, ro["ids"])
            if spans[-1][1] != len(ro["full"]):
                raise RuntimeError("token spans do not cover the decoded text")
            out.append(OG.token_advantages(spans, len(ro["full"]), pb, float(ai),
                                           rc if pb.boxes else None, sb, ss, cfg))
            for b in rc.box:
                c = cfg.beta_box * b / sb if cfg.use_box else 0.0
                v = c if (cfg.use_box and cfg.mode == "replace") else float(ai) + c
                (hit if b + rc.break_even >= cfg.tau else nonhit).append(v)
            if rc.stop is not None:
                stops.append(rc.stop)
        advs.append(out)
    mean = lambda xs: round(float(sum(xs)) / len(xs), 4) if xs else None
    return advs, {"sign_hit_pos": mean([v > 0 for v in hit]), "sign_nonhit_neg": mean([v < 0 for v in nonhit]),
                  "stop_credit_neg": mean([s < 0 for s in stops]), "n_hit_boxes": len(hit),
                  "n_nonhit_boxes": len(nonhit), "box_std": round(sb, 4), "stop_std": round(ss, 4)}


def omission_objective(lp, ref, micros, args, n_main):
    """((pg + beta * kl) * mask).sum() / (n_main * l_const), per-token advantages. mu = 1, so the
    clip is inert. The KL log matches credit.masked_objective's."""
    import torch
    dev = lp.device
    adv = torch.cat([m["adv"] for m in micros]).to(dev)
    msk = torch.cat([m["mask"] for m in micros]).to(dev)
    ratio = torch.exp(lp - lp.detach())
    pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv)
    d = ref - lp
    kl = torch.exp(d) - d - 1.0
    loss = ((pg + args.kl_beta * kl) * msk).sum() / (n_main * args.l_const)
    log, o = {"kl": 0.0, "kl_box": 0.0, "n_box": 0}, 0
    kd = kl.detach()
    for m in micros:
        n = int(m["ids"].numel())
        v = float(kd[o:o + n].mean())
        log["kl"] += v; log["kl_box"] += v; log["n_box"] += 1
        o += n
    return loss, log


def main() -> None:
    args = parse_args()
    VB.VC.setup_caches()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_matching(args.matching)
    import torch
    from packed import packed_logprobs
    dev = "cuda:0"

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    if (out / "grpo_metrics.jsonl").exists() and not args.resume:
        raise SystemExit(f"{out} already has a run: pick a new --output-dir or pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    cfg = credit_config(args)

    mdir = VB.resolve_base(args.init_name)
    if not (mdir / "DONE").exists() and not (mdir / "config.json").exists():
        raise SystemExit(f"no merged init at {mdir}: run demand_rl/export_init.py first")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(mdir), trust_remote_code=True, use_fast=False)
    engine = VB.DemandEngine(mdir, tok, gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
                             lora_rank=args.lora_rank, seed=args.seed, max_num_seqs=args.max_num_seqs)
    model, _, ctl = VB.load_policy_merged(mdir, args.lora_rank, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    img_id = tok.convert_tokens_to_ids(VB.VC.IMG_CONTEXT)
    w = RewardWeights(args.w_iou, args.w_f1, args.w_fmt, args.w_miss, args.w_fp)

    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    random.Random(args.seed).shuffle(samples)
    items = [VB.VC.to_item(s) for s in samples]
    pps, G = args.prompts_per_step, args.group_size
    n_steps = min(args.max_steps, -(-len(items) // pps))

    start, tok_main = 0, 0
    sync_root = Path(args.sync_dir) / f"{out.name}_{os.getpid()}"
    if args.resume and (out / "trainer_state.pt").exists():
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(out / "trainer_state.pt", map_location="cpu", weights_only=False)
        start, tok_main = st["step"], st["tok_main"]
        set_peft_model_state_dict(model.language_model,
                                  load_file(str(out / f"grpo-step{start}" / "adapter_model.safetensors")))
        opt.load_state_dict(st["opt"])
        rng.setstate(st["rng"])
        VB.sync_lora(engine, model, sync_root / f"s{start}", args.lora_rank)
        print(f"[resume] from step {start}", flush=True)

    metrics_f = open(out / "grpo_metrics.jsonl", "a")
    io = ThreadPoolExecutor(4)
    t_run = time.time()
    print(f"[omission] arm={args.arm} {n_steps} steps x {pps} x G={G}, init={mdir}, l_const={args.l_const}, "
          f"box={cfg.use_box} stop={cfg.use_stop} mode={cfg.mode} beta={cfg.beta_box}/{cfg.beta_stop} "
          f"shat={cfg.shat_reduce}", flush=True)

    for step in range(start, n_steps):
        t0 = time.time()
        batch = items[step * pps:(step + 1) * pps]
        prep = io.submit(lambda b=batch: [VB.VC.hf_pixel_values(it["image"]) for it in b])
        gens = engine.main(batch, G, args.temperature, args.top_p, args.max_new_tokens,
                           args.no_repeat_ngram_size, logprobs=True)
        pvs = prep.result()
        t_gen = time.time()

        groups, stats = [], []
        n_bad_prompt = step_tok_main = 0
        for it, g, pv in zip(batch, gens, pvs):
            pid = tok(VB.VC.expand_image(VB.VC.prompt_text(it, "grounding"), pv.shape[0])).input_ids
            if pid != g["prompt_ids"]:
                n_bad_prompt += 1      # HF would score a different context than vLLM sampled from
                continue
            ros = []
            for c in g["comps"]:
                full = tok.decode(c["ids"], skip_special_tokens=False)
                st = score_rollout(full.split(VB.VC.EOS_TOKEN)[0], it["gold"], w)
                st["trunc"] = float(c["trunc"])
                st["n_tokens"] = len(c["ids"])
                ros.append({"ids": list(c["ids"]), "full": full, "r": st["r_ground"],
                            "trunc": bool(c["trunc"]), "lps": c["lps"], "st": st})
                step_tok_main += len(c["ids"])
            stats.extend(ro["st"] for ro in ros)
            groups.append({"gold": it["gold"], "ros": ros, "pv": pv, "pid": pid})
        if not groups:
            print(f"[warn] step {step + 1}: no scoreable prompts", flush=True)
            continue

        advs, signs = batch_advantages(groups, tok, cfg)
        n_main = sum(len(g["ros"]) for g in groups)
        work = []
        for g, A in zip(groups, advs):
            micros = [{"ids": torch.tensor(ro["ids"], dtype=torch.long), "adv": torch.from_numpy(a),
                       "mask": torch.ones(len(ro["ids"])), "kind": "main", "count_tokens": 0,
                       "vllm_lps": ro["lps"]} for ro, a in zip(g["ros"], A)]
            work.append((g["pv"], g["pid"], micros))
        if args.debug_adv and step == start:
            ro, a = next((ro, a) for g, A in zip(groups, advs) for ro, a in zip(g["ros"], A)
                         if OG.parse_boxes(ro["full"]).boxes)
            print("[debug-adv] token -> advantage:\n" + "\n".join(
                f"{p!r:>14} {float(v):+.3f}" for p, v in zip(BS._token_texts(tok, ro["ids"]), a)), flush=True)

        opt.zero_grad(set_to_none=True)
        tot_loss, kll = 0.0, {}
        lp_gap_num = lp_gap_den = 0.0
        for pv, pid, micros in work:
            with torch.no_grad():
                vit = model.extract_feature(pv.to(dev))
            P = torch.tensor(pid, device=dev)
            C = [m["ids"].to(dev) for m in micros]
            ctl["on"] = P.numel() + sum(len(c) for c in C) > args.ckpt_tokens
            with torch.no_grad(), model.language_model.disable_adapter():
                ref, _ = packed_logprobs(model, vit, P, C, img_id)
            lp, lens = packed_logprobs(model, vit, P, C, img_id)
            loss, lg = omission_objective(lp, ref, micros, args, n_main)
            loss.backward()
            tot_loss += loss.item() * n_main
            CR.add_log(kll, lg)
            o = 0
            lpd = lp.detach().float().cpu()
            for m, L in zip(micros, lens):
                v = m.get("vllm_lps")
                if v:
                    k = min(len(v), L)
                    lp_gap_num += float((lpd[o:o + k] - torch.tensor(v[:k])).abs().sum())
                    lp_gap_den += k
                o += L
            del vit, ref, lp
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
        opt.step()
        t_train = time.time()
        VB.sync_lora(engine, model, sync_root / f"s{step + 1}", args.lora_rank)
        shutil.rmtree(sync_root / f"s{step - 1}", ignore_errors=True)
        t_sync = time.time()
        tok_main += step_tok_main

        rec = CR.step_record(args, step + 1, 0, stats, [], kll, tot_loss, n_main,
                             0, tok_main, 0, (time.time() - t_run) / 60)
        for k in ("r_count", "count_parse_rate", "count_exact", "count_mae", "compliance", "kl_count_tok",
                  "kl_branch", "branch_points", "tok_branch", "tok_extra_frac"):
            rec.pop(k, None)
        rec.update(signs)
        for lab, lo, hi in BINS:
            v = [r["n_pred"] / r["n_gold"] for r in stats if lo <= r["n_gold"] <= hi]
            rec[f"bpg_{lab}"] = round(sum(v) / len(v), 3) if v else None
        rec.update({"mean_tokens": round(step_tok_main / max(1, n_main), 1),
                    "lp_gap": round(lp_gap_num / max(1.0, lp_gap_den), 4), "grad_norm": round(gnorm, 4),
                    "prompt_id_mismatch": n_bad_prompt, "gen_s": round(t_gen - t0, 2),
                    "train_s": round(t_train - t_gen, 2), "step_s": round(t_sync - t0, 2)})
        eta = (time.time() - t_run) / (step + 1 - start) * (n_steps - step - 1) / 3600
        print(f"[{args.arm} step {step + 1}/{n_steps}] " +
              " ".join(f"{k}={v}" for k, v in rec.items() if k not in ("step", "arm", "epoch") and v is not None)
              + f" eta_h={eta:.2f}", flush=True)
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

        if (step + 1) % args.save_every == 0 or step + 1 == n_steps:
            ck = out / f"grpo-step{step + 1}"
            model.language_model.save_pretrained(str(ck))
            tok.save_pretrained(str(ck))
            torch.save({"step": step + 1, "opt": opt.state_dict(), "rng": rng.getstate(),
                        "tok_main": tok_main}, out / "trainer_state.pt")
            print(f"[ckpt] {ck}", flush=True)

    shutil.rmtree(sync_root, ignore_errors=True)
    (out / "DONE").write_text(f"{n_steps} steps, {(time.time() - t_run) / 3600:.2f} h, tok_main={tok_main}\n")
    print(f"done: arm={args.arm} {n_steps} steps in {(time.time() - t_run) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
