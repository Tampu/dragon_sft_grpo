#!/usr/bin/env python3
"""
SFT baseline vs. grpo-stepN, on the identical held-out slice and the
identical metric function GRPO's periodic val used (grpo_dragon6_v1.run_val:
mean matched IoU, F1@0.5, per-box recall@0.9) -- so the reported delta is a
genuine SFT->GRPO comparison, not GRPO-step-50 vs GRPO-step-250 mislabeled
as a baseline.

SFT policy  = load_lora_checkpoint(sft_ckpt, merge_lora=True)  (no GRPO adapter)
GRPO policy = load_lora_checkpoint(sft_ckpt, merge_lora=True)
              + PeftModel.from_pretrained(model.language_model, grpo_ckpt)
              (GRPO adapter applied, NOT merged -- functionally identical at
              inference; merging is a speed/consolidation step, not needed
              for a single eval pass)

Usage:
    python compare_sft_vs_grpo.py \
        --sft-ckpt /mnt/data2/traviku2/outputs/dragon6_sft2k_v4/ckpts \
        --grpo-ckpt /mnt/data2/traviku2/outputs/dragon6_grpo_v1/grpo-step250 \
        --val-jsonl /mnt/data2/traviku2/dragon_datasets_sft2k/holdout600_grounding_v4.jsonl \
        --val-size 100 \
        --image-root /mnt/data2/traviku2/Diagram_Attribution_Dataset \
        --hf-cache-dir /mnt/data2/traviku2/hf_home \
        --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import grpo_dragon6_v1 as g  # noqa: E402
from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path, load_lora_checkpoint  # noqa: E402

_ensure_internvl_repo_on_path()
from internvl.train.constants import IMG_CONTEXT_TOKEN  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-ckpt", required=True)
    ap.add_argument("--grpo-ckpt", required=True,
                    help="a grpo-stepN dir, e.g. outputs/dragon6_grpo_v1/grpo-step250")
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--val-size", type=int, default=100,
                    help="MUST match the --val-size GRPO training used (default 100) "
                         "so the SFT row is scored on the exact same slice.")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--hf-cache-dir", type=str, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conv-style", default="internlm2-chat")
    ap.add_argument("--max-tiles", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    return ap.parse_args()


def load_sft_only(args):
    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=args.sft_ckpt, device=args.device,
        dtype=torch.bfloat16, hf_cache_dir=args.hf_cache_dir, merge_lora=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.language_model.config.use_cache = True
    return model, tokenizer


def load_sft_plus_grpo(args):
    from peft import PeftModel
    model, tokenizer = load_sft_only(args)
    model.language_model = PeftModel.from_pretrained(model.language_model, args.grpo_ckpt)
    model.language_model = model.language_model.to(args.device)
    model.language_model.config.use_cache = True
    return model, tokenizer


def main() -> None:
    args = parse_args()
    val_samples = [json.loads(l) for l in Path(args.val_jsonl).read_text().splitlines()
                   if l.strip()][: args.val_size]
    print(f"[data] scoring both checkpoints on the same {len(val_samples)} held-out samples "
          f"from {args.val_jsonl}")

    metrics_f = Path(args.grpo_ckpt).parent / "sft_vs_grpo_comparison.jsonl"
    results = {}

    for label, loader in (("SFT (dragon6_sft2k_v4)", load_sft_only),
                          (f"GRPO ({Path(args.grpo_ckpt).name})", load_sft_plus_grpo)):
        print(f"\n[load] {label}")
        model, tokenizer = loader(args)
        # run_val prints "[val step {step}] ..." and writes {"val": {...}} to metrics_f;
        # step is just a label here, not a real training step.
        with metrics_f.open("a") as f:
            g.run_val(model, tokenizer, val_samples, Path(args.image_root), args,
                      step=label, metrics_f=f)
        del model
        torch.cuda.empty_cache()

    print(f"\nWrote comparison rows -> {metrics_f}")
    print("Read them back with: cat", metrics_f)


if __name__ == "__main__":
    main()
