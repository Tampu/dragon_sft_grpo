#!/usr/bin/env python3
"""Grounder states: 1 greedy output (E) + K samples at T = 1.0, per item. NEEDS A GPU (da_env2).

Resumes from what is already in --out. Greedy and sampling settings are the ones every Val and Test
number was scored with (640 tokens, no_repeat_ngram 6), through the same engine.

  --base   merged dir name (grounder2633; sft for row A with --K 0)
  --lora   optional PEFT adapter on top of --base (the retrained grounder in Stage 2)
  --split  val300 | test | grpo_pool | a jsonl path

    CUDA_VISIBLE_DEVICES=5 /mnt/data2/traviku2/da_env2/bin/python completer_pilot/generate_states.py \
        --base grounder2633 --split data/train2000.jsonl --K 4 --out states/train.jsonl
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
from grpo_dragon6_v1 import parse_boxes_r1  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="grounder2633")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--grounder-prompt", default="v1", choices=["v1", "v3"],
                    help="v3 replaces the system turn with prompts_v3.GROUNDER_SYSTEM_V3 (EDGE-based)")
    a = ap.parse_args()

    VC.setup_caches()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = VC.load_split(a.split)
    if a.grounder_prompt == "v3":
        from prompts_v3 import GROUNDER_SYSTEM_V3
        for it in items:
            it["system"] = GROUNDER_SYSTEM_V3
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"[states] {len(items)} items, {len(done)} done, K={a.K}, base={a.base}, lora={a.lora}", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    base = VB.resolve_base(a.base)
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, use_fast=False)
    eng = GCEngine(base, tok, gpu_mem=a.gpu_mem, seed=a.seed, max_num_seqs=256)
    tmp = None
    if a.lora:
        tmp = fresh_vllm_lora(a.lora, f"{out.stem}_{Path(a.lora).parent.name}_{Path(a.lora).name}")
        eng.set_lora(tmp)

    def decode(c):
        txt = tok.decode(c["ids"], skip_special_tokens=False).split(VC.EOS_TOKEN)[0]
        return txt, [list(b) for b in parse_boxes_r1(txt)[0]]

    t0 = time.time()
    with out.open("a") as f:
        for c0 in range(0, len(todo), a.chunk):
            part = todo[c0:c0 + a.chunk]
            mt = [a.max_new_tokens] * len(part)
            greedy = eng.generate(part, 1, 0.0, mt, a.no_repeat_ngram_size)
            smp = eng.generate(part, a.K, a.temperature, mt, a.no_repeat_ngram_size) if a.K else [None] * len(part)
            for it, g, s in zip(part, greedy, smp):
                c = g["comps"][0]
                txt, E = decode(c)
                q, ans = G.split_question(it["user"])
                f.write(json.dumps({
                    "id": it["id"], "domain": it["domain"], "image": it["image"], "question": q, "answer": ans,
                    "gold": [list(b) for b in it["gold"]], "E": E, "E_response": txt, "E_tokens": len(c["ids"]),
                    "E_trunc": c["trunc"], "samples": [decode(x)[1] for x in s["comps"]] if s else [],
                    "sample_trunc": [x["trunc"] for x in s["comps"]] if s else []}) + "\n")
            f.flush()
            n = min(c0 + a.chunk, len(todo))
            print(f"[states] {n}/{len(todo)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
