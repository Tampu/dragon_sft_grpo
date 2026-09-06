#!/usr/bin/env python3
"""
export_for_eval_script.py -- run a checkpoint on a DRAGON jsonl split and
write pred_<dataset>.json in the exact schema eval_script.py consumes, so
comparisons land in the paper's own units (MPIoU / GroupIoU / F1@tau) instead
of the training-time reward proxy.

eval_script.py expects, per item:
    {
      "sample_id": ..., "dataset": ...,
      "image_width": W, "image_height": H,
      "gt_boxes_raw":      [{"x":..,"y":..,"w":..,"h":..,"source":"gt","kind":"bbox"}, ...],
      "pred_boxes_parsed": [{"x":..,"y":..,"w":..,"h":..}, ...]
    }
and normalises coords by dividing by image_width/image_height IF they exceed
1.5 -- so we deliberately report image_width=image_height=1000 for every item,
since our boxes already live in InternVL's normalized [0,1000] xyxy space.
IoU is a ratio, so this choice doesn't change the resulting scores; it just
keeps eval_script's own normalisation branch well-defined instead of skipped.

Usage (one call per domain, since eval_script's batch mode buckets by
filename, not by an internal 'dataset' field):

    python export_for_eval_script.py export \
        --checkpoint /path/to/merged_or_adapter_ckpt \
        --jsonl dragon_datasets_sft2k/val400_grounding_v4.jsonl \
        --dataset-name chartqa \
        --image-root /mnt/data2/traviku2/dragon_datasets \
        --out-dir preds_for_eval/sft2k

This writes preds_for_eval/sft2k/pred_chartqa.json AND immediately prints the
1:1 (bijective) vs many-to-many R/P/F1@tau comparison on those same
predictions. Then, for the paper's official numbers, run eval_script.py
unmodified (many-to-many, as originally written):

    python eval_script.py --pred_dir preds_for_eval/sft2k

To rescore predictions you (or eval_script.py itself, or anyone) already
produced, without touching the model:

    python export_for_eval_script.py rescore \
        --pred-json preds_for_eval/sft2k/pred_chartqa.json

Repeat --checkpoint / --out-dir per model (SFT vs GRPO) to get directly
comparable pred_dir trees, then diff their all_datasets_summary.json (for the
official many-to-many numbers) and their *_matching_compare.json (for the
1:1 delta).

Repeat --checkpoint / --out-dir per model (SFT vs GRPO) to get directly
comparable pred_dir trees, then diff their all_datasets_summary.json.

---
Patched for this pipeline's actual checkpoint layout (see load_model() and
run_inference() below): our checkpoints are adapter-only (LoRA + mlp1 saved
in vision_lora/llm_lora subdirs, not a merged HF-loadable directory), and
model.chat() strips the <ref>/<box> special tokens via skip_special_tokens=
True during decode -- both bugs already found and fixed elsewhere this
session (internvl_lora_checkpoint_io.load_lora_checkpoint and
sft_v4_phase0.chat_keep_special_tokens respectively). load_model()/
run_inference() below reuse those fixes instead of the original
InternVLChatModel.from_pretrained()/model.chat() calls, per this file's own
documented escape hatch ("If neither applies to your checkpoint layout, load
the model yourself and call run_inference() directly").
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]

NATIVE_BOX_RE = re.compile(r"<box>\s*(\[.*?\])\s*</box>", re.S)
V3_ANSWER_RE = re.compile(r"Answer:\s*([0-9][0-9,;\s]*)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ---------------------------------------------------------------------------
# Bijective (Hungarian, 1:1) Recall/Precision/F1 -- rescoring layer that sits
# ALONGSIDE eval_script.py, not a replacement for it. eval_script.py's own
# Recall/Precision/F1@tau is many-to-many: several pred boxes can each
# individually match the SAME gt box and all count as "matched" for
# precision (COCO-style protocols disallow this on purpose -- it's how mAP
# avoids being gamed by duplicate detections). We report BOTH numbers from
# the SAME exported predictions so the delta itself is visible:
#   many-to-many  = exactly eval_script.py's own recall_precision_f1()
#   bijective/1:1 = optimal one-to-one assignment (scipy Hungarian, greedy
#                   fallback), extra boxes on an already-matched gt count
#                   as false positives instead of free matches.
# ---------------------------------------------------------------------------

def _iou_xywh(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def many_to_many_rpf1(gt_boxes: List[Dict], pred_boxes: List[Dict], tau: float) -> Dict[str, float]:
    """Exact reimplementation of eval_script.py's recall_precision_f1, kept
    separate so the two matching strategies are computed identically on the
    same box lists and only the assignment rule differs."""
    if not gt_boxes or not pred_boxes:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    gt_covered = sum(1 for g in gt_boxes if any(_iou_xywh(p, g) >= tau for p in pred_boxes))
    pred_matched = sum(1 for p in pred_boxes if any(_iou_xywh(p, g) >= tau for g in gt_boxes))
    recall = gt_covered / len(gt_boxes)
    precision = pred_matched / len(pred_boxes)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def bijective_rpf1(gt_boxes: List[Dict], pred_boxes: List[Dict], tau: float) -> Dict[str, float]:
    """1:1 optimal assignment. A pred box only counts as a true positive if
    it is the box assigned to some gt under the matching that maximizes
    total IoU -- extra boxes near an already-assigned gt are false
    positives, matching COCO-style detection evaluation."""
    if not gt_boxes or not pred_boxes:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    iou = [[_iou_xywh(p, g) for g in gt_boxes] for p in pred_boxes]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = -np.array(iou)
        ri, ci = linear_sum_assignment(cost)
        tp = sum(1 for r, c in zip(ri, ci) if iou[r][c] >= tau)
    except Exception:
        pairs = sorted(((iou[i][j], i, j) for i in range(len(pred_boxes))
                        for j in range(len(gt_boxes))), reverse=True)
        used_p, used_g, tp = set(), set(), 0
        for v, i, j in pairs:
            if v < tau or i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j); tp += 1
    recall = tp / len(gt_boxes)
    precision = tp / len(pred_boxes)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


_TAU_LIST = [0.1, 0.3, 0.5, 0.7]


def rescore_matching(pred_json_path: Path, tau_list: List[float] = None) -> Dict[str, Any]:
    """Reads a pred_<dataset>.json already in eval_script.py's format and
    computes BOTH matching strategies per sample, then averages -- same
    aggregation convention as eval_script.py (average R/P per sample first,
    then recompute F1 from averaged R/P, not average of per-sample F1)."""
    tau_list = tau_list or _TAU_LIST
    items = json.loads(pred_json_path.read_text())

    sums = {"m2m": {t: {"recall": 0.0, "precision": 0.0} for t in tau_list},
            "1to1": {t: {"recall": 0.0, "precision": 0.0} for t in tau_list}}
    n = len(items)
    per_sample = []
    for it in items:
        gt = it.get("gt_boxes_raw", [])
        pred = it.get("pred_boxes_parsed", [])
        row = {"sample_id": it.get("sample_id"), "n_gt": len(gt), "n_pred": len(pred)}
        for t in tau_list:
            m = many_to_many_rpf1(gt, pred, t)
            b = bijective_rpf1(gt, pred, t)
            sums["m2m"][t]["recall"] += m["recall"]; sums["m2m"][t]["precision"] += m["precision"]
            sums["1to1"][t]["recall"] += b["recall"]; sums["1to1"][t]["precision"] += b["precision"]
            row[f"m2m_R@{int(t*100)}"] = m["recall"]; row[f"m2m_P@{int(t*100)}"] = m["precision"]
            row[f"1to1_R@{int(t*100)}"] = b["recall"]; row[f"1to1_P@{int(t*100)}"] = b["precision"]
        per_sample.append(row)

    summary = {"num_samples": n, "tau_list": tau_list, "many_to_many": {}, "bijective_1to1": {}}
    for strat_key, out_key in [("m2m", "many_to_many"), ("1to1", "bijective_1to1")]:
        for t in tau_list:
            pct = int(t * 100)
            r = sums[strat_key][t]["recall"] / n if n else 0.0
            p = sums[strat_key][t]["precision"] / n if n else 0.0
            f1 = (2 * r * p / (r + p)) if (r + p) > 0 else 0.0
            summary[out_key][f"R@{pct}"] = round(r, 4)
            summary[out_key][f"P@{pct}"] = round(p, 4)
            summary[out_key][f"F1@{pct}"] = round(f1, 4)
    summary["per_sample"] = per_sample
    return summary


def cmd_rescore(args: argparse.Namespace) -> None:
    """Standalone: rescore an existing pred_<dataset>.json (already produced
    by this script, or by anyone else's export in the same schema) under
    both matching strategies without re-running inference."""
    pred_path = Path(args.pred_json)
    out = rescore_matching(pred_path, args.tau)
    # Deliberately does NOT start with "pred_": eval_script.py globs
    # pred_*.json in --pred_dir and only excludes a few specific suffixes
    # (_eval_summary.json, _eval_per_sample.json, pred_all.json) -- a
    # "pred_<dataset>_matching_compare.json" name would silently get picked
    # up and fed through extract_gt_boxes() as if it were a prediction list,
    # crashing on 'str' object has no attribute 'get' (its top-level JSON is
    # a summary dict, not a list of items).
    stem = pred_path.stem[len("pred_"):] if pred_path.stem.startswith("pred_") else pred_path.stem
    out_path = pred_path.with_name(stem + "_matching_compare.json")
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n{pred_path.name}  (n={out['num_samples']})")
    print(f"{'tau':>5}  {'m2m R':>7} {'m2m P':>7} {'m2m F1':>7}   "
          f"{'1:1 R':>7} {'1:1 P':>7} {'1:1 F1':>7}   {'ΔF1 (1:1 - m2m)':>16}")
    for t in out["tau_list"]:
        pct = int(t * 100)
        m2m, b = out["many_to_many"], out["bijective_1to1"]
        dF1 = b[f"F1@{pct}"] - m2m[f"F1@{pct}"]
        print(f"{t:>5.1f}  {m2m[f'R@{pct}']:>7.4f} {m2m[f'P@{pct}']:>7.4f} {m2m[f'F1@{pct}']:>7.4f}   "
              f"{b[f'R@{pct}']:>7.4f} {b[f'P@{pct}']:>7.4f} {b[f'F1@{pct}']:>7.4f}   {dF1:>+16.4f}")
    print(f"\nSaved -> {out_path}")
    print("Negative ΔF1 at low tau with matched R (m2m R == 1:1 R roughly) is the "
          "signature of duplicate-box spam: many-to-many gives free precision credit "
          "to extra boxes clustered on one gt region; 1:1 does not.")


def parse_pred_boxes(text: str) -> List[Box]:
    """Same tolerant parser as sft_v4_phase0.py / grpo_dragon6_v1.py: native
    <box>[[..]]</box> first, legacy 'Answer: x1,y1,x2,y2;...' as fallback."""
    boxes: List[Box] = []
    for m in NATIVE_BOX_RE.finditer(text):
        try:
            payload = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if payload and isinstance(payload[0], (int, float)):
            payload = [payload]
        for b in payload:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                # len==4 does not imply 4 NUMBERS: models sometimes emit a text
                # label or a nested list in a coordinate slot, and the bare
                # float() below raised OUTSIDE the literal_eval try. Skip the
                # malformed entry rather than crash the caller.
                try:
                    boxes.append(tuple(float(v) for v in b))
                except (TypeError, ValueError):
                    continue
    if boxes:
        return boxes
    m = V3_ANSWER_RE.search(text)
    if m:
        for chunk in m.group(1).split(";"):
            parts = [p for p in re.split(r"[,\s]+", chunk.strip()) if p]
            if len(parts) == 4:
                try:
                    boxes.append(tuple(float(p) for p in parts))
                except ValueError:
                    pass
    return boxes


def xyxy_to_xywh(b: Box) -> Dict[str, float]:
    x1, y1, x2, y2 = b
    return {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}


def load_model(args):
    """Adapter-only loader for this pipeline's checkpoints, replacing the
    original InternVLChatModel.from_pretrained(args.checkpoint) call (which
    only works for a merged/full checkpoint -- ours never merge SFT LoRA
    into a saved model.safetensors, see internvl_lora_checkpoint_io.py).

    --checkpoint = the SFT adapter-only dir (LoRA + mlp1), always required.
    --grpo-checkpoint = optional grpo-stepN adapter dir; if given, it's
        applied (not merged) on top of the SFT-merged base, same
        reconstruction compare_sft_vs_grpo.py already validated.
    --no-adapters = BASE arm: load the pristine InternVL3-8B weights with the
        LoRA adapters present-but-disabled, rather than merged. Everything
        else -- tokenizer (incl. the <ref>/<box> special tokens), tiling,
        decode path, and the grounding system prompt carried in the SFT
        checkpoint's config.json -- stays identical, so the only variable
        across arms is the trained weights. Same construction as
        causal_transfer_eval.load_arm's base arm.
    """
    import torch
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path, load_lora_checkpoint
    _ensure_internvl_repo_on_path()  # must run before ANY internvl.* import below
    from internvl.train.constants import IMG_CONTEXT_TOKEN

    no_adapters = getattr(args, "no_adapters", False)
    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=args.checkpoint, device=args.device,
        dtype=torch.bfloat16, hf_cache_dir=args.hf_cache_dir,
        merge_lora=not no_adapters)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    if no_adapters:
        for sub in (model.language_model, model.vision_model):
            try:
                sub.disable_adapter_layers()
            except Exception as e:
                print(f"[warn] disable_adapter_layers failed on {type(sub).__name__}: {e}")
        print("[load] BASE arm: SFT/vision LoRA disabled (not merged)")
    elif getattr(args, "grpo_checkpoint", None):
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, args.grpo_checkpoint)
        model.language_model = model.language_model.to(args.device)

    model.config.max_dynamic_patch = args.max_tiles
    model.language_model.config.use_cache = True
    return model, tokenizer


def run_inference(model, tokenizer, samples: List[Dict[str, Any]],
                  image_root: Path, args) -> List[Dict[str, Any]]:
    import torch
    from PIL import Image
    from internvl.train.dataset import build_transform, dynamic_preprocess
    # Reuse the fixed decode path, not model.chat(): model.chat()'s
    # tokenizer.batch_decode(..., skip_special_tokens=True) silently strips
    # <ref>/<box> (registered as special tokens during training), which is
    # the exact bug behind Phase 0's earlier 0/32-all-empty-predictions
    # result -- see sft_v4_phase0.chat_keep_special_tokens's docstring.
    from sft_v4_phase0 import chat_keep_special_tokens

    transform = build_transform(is_train=False, input_size=448,
                                pad2square=False, normalize_type="imagenet")
    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                   num_beams=1, repetition_penalty=1.0)  # never 1.1 -- see prior notes

    out_items = []
    for k, s in enumerate(samples, 1):
        img = Image.open(image_root / s["image"]).convert("RGB")
        tiles = dynamic_preprocess(img, min_num=1, max_num=args.max_tiles,
                                   image_size=448, use_thumbnail=True)
        pixel_values = torch.stack([transform(t) for t in tiles]).to(
            torch.bfloat16).to(args.device)

        question = s["conversations"][1]["value"]
        if "<image>" not in question:
            question = "<image>\n" + question
        response = chat_keep_special_tokens(model, tokenizer, pixel_values, question,
                                            gen_cfg, args.device)

        gt_boxes = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
        pred_boxes = parse_pred_boxes(response)

        out_items.append({
            "sample_id": s.get("id"),
            "dataset": args.dataset_name,
            # coords are pre-normalized to [0,1000]; report as the image "size"
            # so eval_script's own division lands them in [0,1] -- IoU is
            # scale-invariant either way, this just keeps normalise_boxes()
            # well-defined rather than skipped (image_width/height default 0).
            "image_width": 1000,
            "image_height": 1000,
            "gt_boxes_raw": [
                {**xyxy_to_xywh(b), "source": "gt", "kind": "bbox"} for b in gt_boxes
            ],
            "pred_boxes_parsed": [xyxy_to_xywh(b) for b in pred_boxes],
        })
        if k % 25 == 0 or k == len(samples):
            print(f"  [{k}/{len(samples)}] inferred (last pred had "
                  f"{len(pred_boxes)} boxes vs {len(gt_boxes)} gt)")
    return out_items


def cmd_export(args: argparse.Namespace) -> None:
    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if args.max_samples:
        samples = samples[: args.max_samples]

    if args.from_predictions:
        preds = {json.loads(l)["id"]: json.loads(l)["response"]
                for l in Path(args.from_predictions).read_text().splitlines() if l.strip()}
        out_items = []
        for s in samples:
            gt_boxes = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
            pred_boxes = parse_pred_boxes(preds.get(s.get("id"), ""))
            out_items.append({
                "sample_id": s.get("id"), "dataset": args.dataset_name,
                "image_width": 1000, "image_height": 1000,
                "gt_boxes_raw": [{**xyxy_to_xywh(b), "source": "gt", "kind": "bbox"}
                                for b in gt_boxes],
                "pred_boxes_parsed": [xyxy_to_xywh(b) for b in pred_boxes],
            })
    else:
        model, tokenizer = load_model(args)
        out_items = run_inference(model, tokenizer, samples, Path(args.image_root), args)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{args.dataset_name}.json"
    out_path.write_text(json.dumps(out_items, indent=2))
    print(f"\nWrote {len(out_items)} items -> {out_path}")
    print(f"Run eval_script.py unmodified for the paper's official many-to-many numbers:")
    print(f"  python eval_script.py --pred_dir {out_dir}")

    if not args.skip_matching_compare:
        print(f"\n--- 1:1 vs many-to-many comparison on the same predictions ---")
        cmd_rescore(argparse.Namespace(pred_json=str(out_path), tau=args.tau))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="Run a checkpoint and write pred_<dataset>.json "
                                      "for eval_script.py, plus 1:1 vs many-to-many compare.")
    p.add_argument("--checkpoint", required=True,
                   help="SFT adapter-only checkpoint dir (LoRA + mlp1), e.g. "
                        ".../dragon6_sft2k_v4/ckpts. Always the SFT dir, even when scoring "
                        "a GRPO checkpoint -- pass --grpo-checkpoint for that on top of it.")
    p.add_argument("--grpo-checkpoint", default=None,
                   help="Optional grpo-stepN adapter dir, applied on top of --checkpoint's "
                        "merged SFT weights. Omit to score the plain SFT checkpoint.")
    p.add_argument("--no-adapters", action="store_true",
                   help="BASE arm: disable (don't merge) the SFT LoRA, scoring pristine "
                        "InternVL3-8B through an otherwise identical harness.")
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.add_argument("--jsonl", required=True,
                   help="A single-domain DRAGON jsonl (from sft_v4_phase0.py convert).")
    p.add_argument("--dataset-name", required=True,
                   help="Written into each item's 'dataset' field, e.g. chartqa.")
    p.add_argument("--image-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-tiles", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--from-predictions", default=None,
                   help="Optional jsonl with {'id':..., 'response': <raw model text>} "
                       "aligned to --jsonl, to reuse predictions without re-running the model.")
    p.add_argument("--tau", type=float, nargs="*", default=_TAU_LIST)
    p.add_argument("--skip-matching-compare", action="store_true",
                   help="Only write pred_<dataset>.json, skip the 1:1/many-to-many printout.")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("rescore", help="Rescore an EXISTING pred_<dataset>.json under both "
                                       "1:1 (bijective) and many-to-many matching, no model needed.")
    p.add_argument("--pred-json", dest="pred_json", required=True)
    p.add_argument("--tau", type=float, nargs="*", default=_TAU_LIST)
    p.set_defaults(fn=cmd_rescore)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
