#!/usr/bin/env python3
"""Merge SFT + the vanilla GRPO adapter into one HF dir: the frozen grounder, and the base under the
completer's LoRA. CPU only, ~17 GB in /dev/shm/vep_merged, idempotent. Fails if the adapter changed nothing.

    /mnt/data2/traviku2/da_env2/bin/python completer_pilot/export_grounder.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gc_engine import VC  # noqa: E402

KEY = "language_model.model.layers.0.self_attn.q_proj.weight"


def weight(d):
    from safetensors import safe_open
    fn = json.load(open(d / "model.safetensors.index.json"))["weight_map"][KEY]
    with safe_open(str(d / fn), "pt") as f:
        return f.get_tensor(KEY).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", default=str(VC.ORIG_GRPO))
    ap.add_argument("--name", default="grounder2633")
    ap.add_argument("--sft-ckpt", default=str(VC.SFT_CKPT), help="SFT adapter dir merged into the base")
    a = ap.parse_args()
    VC.setup_caches()
    lora = None if a.lora == "none" else Path(a.lora).resolve()
    out = VC.export_merged(VC.MERGED_ROOT / a.name, extra_lora=lora, sft_ckpt=Path(a.sft_ckpt).resolve())
    if lora is None:
        print(out, json.load(open(out / "vep_export.json")) if (out / "vep_export.json").exists() else "")
        return
    print(out, json.load(open(out / "vep_export.json")))
    sft = VC.MERGED_ROOT / "sft"
    if (sft / "DONE").exists():
        d = float((weight(out) - weight(sft)).norm())
        print(f"[check] |merged - sft| on {KEY} = {d:.4f}")
        if d == 0:
            raise SystemExit("the GRPO adapter did not change the weights")


if __name__ == "__main__":
    main()
