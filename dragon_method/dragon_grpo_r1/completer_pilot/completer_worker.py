#!/usr/bin/env python3
"""One vLLM engine serving the frozen completer to train_gc.py over stdin/stdout. Replies are lines starting
with @@; everything else on stdout is engine logging, which the parent forwards. See gc_engine.RemoteCompleter.

The engine is released explicitly when stdin closes (and on any exception), so no EngineCore is left
holding GPU memory; the parent additionally runs this worker in its own process group."""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gc_engine import GCEngine, VB, VC  # noqa: E402


def reply(obj):
    print("@@" + json.dumps(obj), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gpu-mem", type=float, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-model-len", type=int, default=7168)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--limit-images", type=int, default=1, help="2 for a v3 completer (diagram + crop sheet)")
    a = ap.parse_args()
    VC.setup_caches()
    from transformers import AutoTokenizer
    base = VB.resolve_base(a.base)
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, use_fast=False)
    eng = GCEngine(base, tok, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len, seed=a.seed,
                   max_num_seqs=a.max_num_seqs, limit_images=a.limit_images)
    try:
        reply({"ready": True})
        for line in sys.stdin:
            m = json.loads(line)
            if "lora" in m:
                eng.set_lora(m["lora"])
                reply({"ok": True})
            else:
                gens = eng.generate(m["items"], m["n"], m["temperature"], m["max_tokens"], m["ngram"])
                reply({"gens": [{"comps": [{"ids": c["ids"], "trunc": c["trunc"]} for c in g["comps"]]}
                                for g in gens]})
    finally:
        del eng
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
