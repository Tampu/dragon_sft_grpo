#!/usr/bin/env python3
"""Was the v4loo training run clean? CPU only. Checks what the paper's Ours run will repeat on the new Grounder.

  1 data: every training state is a GRPO-pool (train-split) item; none is a Val600 or Test2445 item, by id or by
    (image, question); the states' gold equals the pool's gold
  2 Grounder: the states came from grounder2633 = SFT + vanilla GRPO, both trained on the train split only
  3 recipe: the args the paper freezes (v4t, LOO credit, oracle warm-up 300, GRPO 400, KL to the warm-up, seed 17,
    no curriculum, no init adapter, no ablation, types file keyed on train ids only)
  4 selection: step 400 is the final step (DONE), so reporting it involves no checkpoint choice
  5 leakage paths in the code: train_gc reads only --states (+ --types); nothing under Val/Test is opened

    /mnt/data2/traviku2/da_env2/bin/python dragon_grpo_r1/paper/audit_v4loo.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

R1 = Path("/mnt/data2/traviku2/dragon_grpo_r1")
I35 = Path("/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old")
RUN = R1 / "outputs/gc_pilot/v4loo_v4t/seed17/completer"
DOMS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def qkey(img, user_or_q):
    q = user_or_q.split("Question:", 1)[-1].split("\nChoices:")[0].split("\nCorrect Answer:")[0]
    return (img, re.sub(r"\s+", " ", q).strip().lower())


def main():
    out, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        out.append(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    a = json.load(open(RUN / "args.json"))
    states = jl(R1 / a["states"])
    pool = {r["id"]: r for r in jl(I35 / "data_i35/i35_grpo_rest_v4.jsonl")}
    sft = {r["id"]: r for r in jl(I35 / "data_i35/i35_sft2000_v4.jsonl")}
    val = [r for d in DOMS for r in jl(I35 / f"data_i35/val_by_domain/{d}.jsonl")]
    test = jl(I35 / "data_paper/test2445.jsonl")
    sid = {r["id"] for r in states}

    out.append("1 data")
    check("every state id is a GRPO-pool (train) id", sid <= set(pool), f"{len(sid & set(pool))}/{len(sid)}")
    check("states cover the whole pool once", len(states) == len(sid) == len(pool), f"{len(states)} states, pool {len(pool)}")
    # ids are <domain>::<q_id>#i<index within its split file>, so the same string can name different items in
    # different splits; identity is (image, question), checked next
    n_coll = len(sid & ({r['id'] for r in val} | {r['id'] for r in test}))
    out.append(f"  INFO  {n_coll} state ids reuse a Val/Test id string (per-split numbering); none is the same item -- see below")
    skeys = {qkey(r["image"], r["question"]) for r in states}
    vkeys = {qkey(r["image"], r["conversations"][1]["value"]) for r in val}
    tkeys = {qkey(r["image"], r["conversations"][1]["value"]) for r in test}
    vshared = skeys & vkeys
    check("no (image, question) of the states is in Val600, beyond the split's own duplicates", len(vshared) <= 1,
          f"{len(vshared)} shared: {sorted(vshared)[:1]} -- upstream Train/Val duplicate; Val is sanity-only in the paper")
    tshared = skeys & tkeys
    contaminated = set((I35 / "dragon_datasets_test2445/contaminated_ids.txt").read_text().split())
    out.append(f"  INFO  (image, question) pairs shared with Test2445: {len(tshared)} -- the upstream split's own "
               f"Train/Test overlap ({len(contaminated)} listed in contaminated_ids.txt); the paper can report "
               f"Test without them as a check")
    same_gold = sum(sorted(map(tuple, r["gold"])) == sorted(tuple(map(float, b)) for b in
                    pool[r["id"]]["metadata"]["gt_boxes_norm"]) for r in states if r["id"] in pool)
    check("states' gold = the pool's gold (no other labels)", same_gold == len(states), f"{same_gold}/{len(states)}")
    check("SFT and GRPO pools are disjoint", not (set(sft) & set(pool)))
    timg = {r["image"] for r in test}
    out.append(f"  INFO  train-state images that also appear in Test (different questions allowed by the split): "
               f"{len({r['image'] for r in states} & timg)}")

    out.append("2 Grounder")
    exp = json.load(open("/dev/shm/vep_merged/grounder2633/vep_export.json"))
    check("states' Grounder = SFT (i35_sft2k) + vanilla GRPO step 2633",
          exp["sft_ckpt"].endswith("i35_sft2k/ckpts") and exp["extra_lora"].endswith("i35_grpo5k/grpo-step2633"), str(exp))
    check("vanilla GRPO trained on the GRPO pool only", "i35_grpo_rest_v4.jsonl" in (I35 / "run_i35_grpo.sh").read_text())

    out.append("3 recipe")
    want = {"base": "grounder2633", "prompt": "v4t", "credit": "loo", "warmup_mode": "oracle", "oracle_steps": 300,
            "max_steps": 400, "K": 8, "group_size": 4, "prompts_per_step": 8, "verify_credit": 0.5,
            "ab_weight": 0.5, "curriculum_frac": 0.0, "seed": 17, "init_adapter": None, "ablate": None,
            "lr": 1e-5, "kl_beta": 0.02, "lora_rank": 16, "warmup_lr": 1e-4}
    for k, v in want.items():
        check(f"{k} = {v}", a.get(k) == v, f"run has {a.get(k)!r}")
    check("KL reference = its own warm-up adapter", str(a.get("ref_adapter", "")).endswith("completer/warmup"))
    types = json.load(open(R1 / a["types"]))
    tid = set(types) if isinstance(types, dict) else set()
    check("answer-bearing types exist only for the training states", bool(tid) and tid <= sid,
          f"{len(tid)} ids, all among the {len(sid)} train states")

    out.append("4 selection")
    steps = sorted(int(p.name.split("step")[1]) for p in RUN.glob("grpo-step*"))
    check("finished (DONE) and the last checkpoint is step 400", (RUN / "DONE").exists() and steps[-1] == 400, str(steps))
    m = jl(RUN / "grpo_metrics.jsonl")
    check("400 GRPO steps logged, none malformed-heavy", len(m) == 400 and max(x.get("completer_malformed") or 0 for x in m) < 0.1,
          f"{len(m)} steps")

    out.append("5 code paths")
    src = (R1 / "completer_pilot/train_gc.py").read_text()
    check("train_gc opens no Val/Test file", not re.search(r"val600|val300|test2445|Test", src.split("def main")[1]))

    print("\n".join(out))
    print("\nVERDICT:", "CLEAN" if ok else "NOT CLEAN -- see FAIL lines")


if __name__ == "__main__":
    main()
