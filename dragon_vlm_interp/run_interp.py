#!/usr/bin/env python3
"""
CLI entry point: explain one (image, question, answer) DRAGON grounding
example with a given interpretability backend, on a given (optionally
fine-tuned) checkpoint.

    python3 run_interp.py \
        --model-config configs/models/internvl3_5_8b.json \
        --sft-checkpoint ../dragon_vlm_sft_grpo/outputs/sft/final \
        --image path/to/chart.png \
        --question "What is the tallest bar?" --answer "Revenue" \
        --out-dir results/

Omit --target-text to explain the MODEL'S OWN prediction (generated once,
greedily, then re-serialized through grounding_prompts.build_target() for a
canonical token layout -- see box_spans.py). Pass --target-text explicitly
to run the WRONG-ANSWER / no-answer ablation: give it a different answer's
predicted boxes, or hand-craft a counterfactual box list, and compare the
resulting heatmap against the model's own-answer heatmap for the same
image -- this is the concrete mechanism behind the "does attention shift
with a wrong answer" ablation discussed for this project's causal-transfer
experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dragon_vlm_sft_grpo" / "scripts"))

from base import InterpRequest  # noqa: E402
from box_spans import compute_box_token_spans  # noqa: E402
from grounding_prompts import (  # noqa: E402
    GROUNDING_SYSTEM_PROMPT, build_chat_messages, build_target, build_user_prompt, parse_pred_boxes,
)
from lora_checkpoint_io import load_adapter_checkpoint  # noqa: E402
from model_config import load_model_and_processor, load_model_config, render_chat_text  # noqa: E402
from registry import get_backend  # noqa: E402


def load_model_for_interp(args, cfg):
    """Mirrors dragon_vlm_sft_grpo/dragon_grpo/grpo_train.py's own arm
    logic exactly, including the case that used to be silently dropped
    here: --grpo-checkpoint given WITHOUT --sft-checkpoint. grpo_train.py
    records which SFT checkpoint (if any -- None for a cold-start GRPO run,
    see its "no SFT, only GRPO" support) each GRPO checkpoint was trained
    on top of, in that checkpoint's own checkpoint_meta.json -- read it
    rather than requiring the caller to remember and re-supply it, and
    rather than silently falling back to the base arm if they don't."""
    sft_checkpoint = args.sft_checkpoint
    if args.grpo_checkpoint and not sft_checkpoint:
        meta_path = Path(args.grpo_checkpoint) / "checkpoint_meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"load_model_for_interp: --grpo-checkpoint given without --sft-checkpoint, and "
                f"{meta_path} doesn't exist to auto-discover it. Pass --sft-checkpoint explicitly, "
                f"or confirm this GRPO checkpoint dir is really the one grpo_train.py wrote."
            )
        meta = json.loads(meta_path.read_text())
        sft_checkpoint = meta.get("sft_checkpoint")
        print(f"[load_model_for_interp] --sft-checkpoint not given; auto-discovered "
              f"{sft_checkpoint!r} from {meta_path} (None = this was a cold-start GRPO run)")

    if sft_checkpoint:
        model, processor, _ = load_adapter_checkpoint(
            sft_checkpoint, device=args.device, merge=True,
            hf_cache_dir=args.hf_cache_dir, model_cfg=cfg)
    else:
        model, processor = load_model_and_processor(cfg, device=args.device, hf_cache_dir=args.hf_cache_dir)

    if args.grpo_checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.grpo_checkpoint)
        model = model.merge_and_unload()
    return model, processor


def render_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    hm_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return np.clip(hm_color * alpha + img_bgr * (1 - alpha), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--sft-checkpoint", default=None)
    ap.add_argument("--grpo-checkpoint", default=None)
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default="tam")

    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--choices", nargs="*", default=[])
    ap.add_argument("--answer", required=True)
    ap.add_argument("--target-text", default=None,
                    help="explicit assistant text to explain, e.g. for the wrong-answer ablation. "
                         "Omit to explain the model's own greedy prediction.")
    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_model_config(args.model_config)
    model, processor = load_model_for_interp(args, cfg)
    model.eval()

    # tiling params are this project's own concern (interp/tiling.py), not part of the shared
    # ModelConfig dataclass dragon_vlm_sft_grpo's training/eval scripts use -- read the raw JSON
    # directly rather than polluting that shared abstraction with a field only this repo needs.
    raw_cfg = json.loads(Path(args.model_config).read_text())
    tiling_cfg = raw_cfg.get("tiling", {})
    backend_kwargs = {}
    if tiling_cfg.get("scheme", "internvl_dynamic_tile") == "internvl_dynamic_tile":
        backend_kwargs = {
            "min_tiles": tiling_cfg.get("min_tiles", 1),
            "max_tiles": tiling_cfg.get("max_tiles", 12),
            "tile_image_size": tiling_cfg.get("tile_image_size", 448),
            "use_thumbnail": tiling_cfg.get("use_thumbnail", True),
        }

    image = Image.open(args.image).convert("RGB")
    sample = {"conversations": [
        {"from": "system", "value": GROUNDING_SYSTEM_PROMPT},
        {"from": "user", "value": build_user_prompt(args.question, args.choices, args.answer)},
    ]}
    messages = build_chat_messages(sample, args.image, include_assistant=False)

    if args.target_text is None:
        prompt_text = render_chat_text(processor, messages, cfg, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(args.device)
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False, num_beams=1,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=pad_id)
        raw_text = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        boxes = parse_pred_boxes(raw_text)
        if not boxes:
            raise SystemExit(f"run_interp: model's own greedy generation had no parseable box array. "
                             f"Raw output: {raw_text!r}\nPass --target-text explicitly instead.")
        target_text = build_target([[int(v) for v in b] for b in boxes])
        print(f"[run_interp] explaining the model's own prediction: {target_text}")
    else:
        target_text = args.target_text
        print(f"[run_interp] explaining the given --target-text: {target_text}")

    box_spans = compute_box_token_spans(processor.tokenizer, target_text)
    print(f"[run_interp] {len(box_spans)} box(es) found in target_text -> per-box + combined heatmaps")

    backend = get_backend(args.backend, cfg, **backend_kwargs)
    request = InterpRequest(image=image, messages=messages, target_text=target_text, box_token_spans=box_spans or None)
    heatmap = backend.explain(model, processor, request)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(image, heatmap.array)
    cv2.imwrite(str(out_dir / "heatmap_overlay.jpg"), overlay)
    np.save(out_dir / "heatmap.npy", heatmap.array)
    (out_dir / "meta.json").write_text(json.dumps({**heatmap.meta, "target_text": target_text}, indent=2))
    print(f"[run_interp] wrote {out_dir / 'heatmap_overlay.jpg'} and meta.json -- meta: {heatmap.meta}")


if __name__ == "__main__":
    main()
