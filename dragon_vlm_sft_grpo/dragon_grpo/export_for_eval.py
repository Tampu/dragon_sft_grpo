#!/usr/bin/env python3
"""
Run a model (base / SFT / SFT+GRPO arm) on a converted domain JSONL and
write pred_<domain>.json in the schema eval_script.py consumes -- generic
across model_configs.

    python3 export_for_eval.py \
        --model-config ../configs/models/qwen3vl_8b_thinking.json \
        --jsonl ../dragon_datasets/test_split/chartqa.jsonl \
        --dataset-name chartqa --image-root ../Diagram_Attribution_Dataset \
        --out-dir preds_for_eval/base

Arms:
  base  = omit --sft-checkpoint and --grpo-checkpoint: pristine model_config
          weights, no LoRA at all.
  sft   = --sft-checkpoint <dir from train_sft.py>, merged into the weights.
  grpo  = --sft-checkpoint <...> --grpo-checkpoint <dir from grpo_train.py>,
          SFT merged then the GRPO adapter applied on top (not merged, but
          equivalent for inference -- merge_and_unload() afterward for a
          faster forward pass if profiling shows it matters).

We report image_width=image_height=1000 for every item since our boxes
already live in the normalized [0,1000] xyxy space grounding_prompts.py and
box_geometry.py use throughout -- eval_script.py's own normalize_boxes()
divides by these values only if it decides the input ISN'T already
normalized (values > 1.5), so this keeps that branch well-defined; IoU is
scale-invariant either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from grounding_prompts import build_chat_messages, parse_pred_boxes
from model_config import load_model_and_processor, load_model_config, render_chat_text


def load_arm(args, cfg):
    if args.sft_checkpoint:
        from lora_checkpoint_io import load_adapter_checkpoint
        model, processor, _ = load_adapter_checkpoint(
            args.sft_checkpoint, device=args.device, merge=True,
            hf_cache_dir=args.hf_cache_dir, model_cfg=cfg)
        print(f"[load] SFT arm: merged {args.sft_checkpoint}")
        if args.grpo_checkpoint:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.grpo_checkpoint)
            model = model.merge_and_unload()
            print(f"[load] GRPO arm: merged {args.grpo_checkpoint} on top")
    else:
        model, processor = load_model_and_processor(cfg, device=args.device, hf_cache_dir=args.hf_cache_dir)
        print("[load] BASE arm: pristine model_config weights, no adapters")
    model.eval()  # already placed on args.device via load_model_and_processor's device_map
    return model, processor


def xyxy_to_xywh(b) -> Dict[str, float]:
    x1, y1, x2, y2 = b
    return {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}


def run_inference(model, processor, samples: List[Dict[str, Any]], image_root: Path,
                  cfg, args) -> List[Dict[str, Any]]:
    from PIL import Image

    gen = cfg.generation_defaults
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out_items = []
    for k, s in enumerate(samples, 1):
        image_abs = str((image_root / s["image"]).resolve())
        image = Image.open(image_abs).convert("RGB")
        messages = build_chat_messages(s, image_abs, include_assistant=False)
        prompt_text = render_chat_text(processor, messages, cfg, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(args.device)

        gen_out = model.generate(
            **inputs, do_sample=gen.get("do_sample", False), num_beams=gen.get("num_beams", 1),
            max_new_tokens=args.max_new_tokens or gen["max_new_tokens"],
            repetition_penalty=1.0, no_repeat_ngram_size=gen.get("no_repeat_ngram_size", 0),
            pad_token_id=pad_id)
        # skip_special_tokens=False: preserve <think>/</think> and any other
        # special tokens the response may rely on -- see grpo_train.py's
        # module docstring for why this is non-negotiable.
        text = processor.tokenizer.decode(
            gen_out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        gt_boxes = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
        pred_boxes = parse_pred_boxes(text) or []

        out_items.append({
            "sample_id": s.get("id"), "dataset": args.dataset_name,
            "image_width": 1000, "image_height": 1000,
            "gt_boxes_raw": [{**xyxy_to_xywh(b), "source": "gt", "kind": "bbox"} for b in gt_boxes],
            "pred_boxes_parsed": [xyxy_to_xywh(b) for b in pred_boxes],
        })
        if k % 25 == 0 or k == len(samples):
            print(f"  [{k}/{len(samples)}] inferred (last pred had "
                  f"{len(pred_boxes)} boxes vs {len(gt_boxes)} gt)")
    return out_items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--sft-checkpoint", default=None)
    ap.add_argument("--grpo-checkpoint", default=None)
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_model_config(args.model_config)
    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if args.max_samples:
        samples = samples[: args.max_samples]

    model, processor = load_arm(args, cfg)
    out_items = run_inference(model, processor, samples, Path(args.image_root), cfg, args)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{args.dataset_name}.json"
    out_path.write_text(json.dumps(out_items, indent=2))
    print(f"\nWrote {len(out_items)} items -> {out_path}")
    print(f"Score with: python3 eval_script.py --pred_dir {out_dir}")


if __name__ == "__main__":
    main()
