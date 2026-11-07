#!/usr/bin/env python3
"""Greedy completer selections on stored states: one sel_<tag>.jsonl per adapter. NEEDS A GPU (da_env2).
All scoring is in report.py, on the CPU. PATH 'none' runs the bare grounder on the completer prompt.

    CUDA_VISIBLE_DEVICES=5 /mnt/data2/traviku2/da_env2/bin/python completer_pilot/eval_completer.py \
        --states states/val300.jsonl --K 4 --lora step100=completer/grpo-step100 --out-dir eval/val300
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gc_common as G  # noqa: E402
from gc_engine import GCEngine, VB, VC, fresh_vllm_lora  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="grounder2633")
    ap.add_argument("--states", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lora", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=7168)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v3", "v3t", "v4", "v4m", "v4t", "v4z"],
                    help="the prompt the adapters were trained with")
    ap.add_argument("--ablate", default=None, choices=["blank_sheet", "blank_diagram", "blank_visual"],
                    help="inference-only ablation: that image replaced by white of the same size")
    a = ap.parse_args()
    v4 = a.prompt in G.V4_PROMPTS
    if a.prompt != "v1":
        a.max_model_len = max(a.max_model_len, 11264 if a.prompt == "v4z" else 9216 if v4 else 8192)

    VC.setup_caches()
    from transformers import AutoTokenizer
    base = VB.resolve_base(a.base)
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, use_fast=False)
    items = [G.completer_item(json.loads(l), a.K, prompt=a.prompt) for l in open(a.states) if l.strip()]
    for it in items:
        it["ablate"] = a.ablate
    if v4:
        from prompts_v4 import v4_budget
        budgets = [v4_budget(tok, len(it["crops"])) for it in items]
    else:
        budgets = [G.select_budget(tok, len(it["E"]) + len(it["pool"]), a.max_new_tokens) for it in items]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [s.split("=", 1) for s in a.lora]
    specs = [(t, p) for t, p in specs if not (a.skip_existing and (out / f"sel_{t}.jsonl").exists())]
    if not specs:
        return
    eng = GCEngine(base, tok, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len, seed=a.seed, max_num_seqs=256,
                   limit_images=1 if a.prompt == "v1" else 3 if a.prompt == "v4z" else 2)

    for tag, path in specs:
        tmp = None if path == "none" else fresh_vllm_lora(path, f"{out.name}_{tag}")
        eng.set_lora(tmp)
        t0 = time.time()
        recs = []
        for c0 in range(0, len(items), a.chunk):
            part, mt = items[c0:c0 + a.chunk], budgets[c0:c0 + a.chunk]
            for it, g in zip(part, eng.generate(part, 1, 0.0, mt, 0)):
                c = g["comps"][0]
                raw = tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0]
                if v4:   # final sets are boxes, not indices: fixes and adds are new coordinates
                    F, p = G.v4_final(raw, it)
                    rec = {"id": it["id"], "text": raw, "idx": None if p is None else [], "final": F,
                           "final_nofix": G.v4_final(raw, it, use_fix=False)[0],
                           "final_noadd": G.v4_final(raw, it, use_add=False)[0],
                           "final_select": G.v4_final(raw, it, use_fix=False, use_add=False)[0],
                           "verdict": None if p is None else {str(d): v for d, v in p["verdict"].items()},
                           "adds": None if p is None else p["adds"], "n_tokens": len(c["ids"]), "trunc": c["trunc"]}
                    recs.append(rec)
                    continue
                txt = G.canonical(raw, it)       # internal indices (E first, then pool), as report.py expects
                recs.append({"id": it["id"], "text": txt, "idx": G.parse_select(txt, len(it["E"]) + len(it["pool"])),
                             "n_tokens": len(c["ids"]), "trunc": c["trunc"], "raw": raw})
        with (out / f"sel_{tag}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        bad = sum(r["idx"] is None for r in recs) / len(recs)
        print(f"[{tag}] {len(recs)} in {(time.time() - t0) / 60:.1f} min, malformed {bad:.3f}, "
              f"example {recs[0]['text']!r}", flush=True)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
