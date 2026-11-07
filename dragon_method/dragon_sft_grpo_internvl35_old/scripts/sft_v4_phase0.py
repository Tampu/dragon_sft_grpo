#!/usr/bin/env python3
"""
sft_v4_phase0.py -- Phase-0 rerun with every fix from the failure analysis,
in one file, ordered so each change is attributable.

    STEP 1 (no retraining):  rescore the EXISTING checkpoint's verify log
        under a per-box IoU >= 0.9 bijective-match criterion instead of
        (near-)exact coordinate equality. Tells you how many of the 24
        "failures" were tolerance artifacts (expected: #2, #17, #23, #31
        and friends flip to PASS; the grid-collapse ones stay FAIL).

            python sft_v4_phase0.py rescore \
                --log logs_phase0_verify_v2.log \
                --jsonl dragon_datasets_phase0/phase0_grounding.jsonl

    STEP 2a: rebuild the 32-sample JSONL with (i) InternVL-NATIVE box
        syntax  <ref>evidence</ref><box>[[x1, y1, x2, y2], ...]</box>
        using the special tokens the tokenizer already carries, and
        (ii) a contradiction check that fails loudly if any
        (image, question) pair maps to two different gold box sets
        (the suspected #9/#11 situation -- unfittable by definition).

            python sft_v4_phase0.py convert \
                --in raw_phase0_records.jsonl \
                --image-root /path/to/images \
                --out dragon_datasets_phase0_v4/phase0_grounding_v4.jsonl \
                --stage A

    STEP 2b: emit the meta file + the exact config overrides for
        DYNAMIC TILING (dynamic_image_size=True, use_thumbnail=True,
        max_dynamic_patch=12) and a max_seq_length actually sized for
        ~13 tiles x 256 image tokens, so charts/infographic text is
        legible to the ViT. Pass these to internvl_chat_finetune.py.

            python sft_v4_phase0.py make-meta \
                --jsonl dragon_datasets_phase0_v4/phase0_grounding_v4.jsonl \
                --image-root /path/to/images \
                --out dragon_datasets_phase0_v4/meta_phase0_v4.json \
                --repeat-time 50

    STEP 3: verify the retrained checkpoint with PURE GREEDY decoding
        (do_sample=False, num_beams=1, repetition_penalty=1.0 -- the
        1.1 penalty used in the benchmark eval corrupts long digit
        strings and must NOT be active here), tiling the image at
        inference exactly as in training, parsing BOTH the native
        <box> format and the legacy v3 "Answer:" format, and scoring
        with the same IoU>=0.9 criterion as Step 1 so the two runs are
        directly comparable. Prints the pass-rate split by gt box count.

            python sft_v4_phase0.py verify \
                --checkpoint /path/to/merged_or_adapter_ckpt \
                --jsonl dragon_datasets_phase0_v4/phase0_grounding_v4.jsonl \
                --image-root /path/to/images

Reuses dragon_grounding_convert_core + sft_v3_patches for everything that
was already correct (normalization to [0,1000], xyxy, spatial sort, dedup,
explanation handling). Only the target SYNTAX, the tiling config, the
decode settings, and the pass criterion change.
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Shared: IoU >= 0.9 bijective-match pass criterion (used by Steps 1 and 3)
# --------------------------------------------------------------------------

IOU_PASS_THRESHOLD = 0.9

Box = Tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt: Sequence[Box], pred: Sequence[Box],
                thr: float = IOU_PASS_THRESHOLD) -> Tuple[int, List[float]]:
    """One-to-one matching, Hungarian if scipy is available, else greedy.
    Returns (n_matched_at_thr, per-gt best IoU list for diagnostics)."""
    if not gt or not pred:
        return 0, [0.0] * len(gt)
    iou = [[box_iou(g, p) for p in pred] for g in gt]
    best_per_gt = [max(row) for row in iou]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = -np.array(iou)
        ri, ci = linear_sum_assignment(cost)
        matched = sum(1 for r, c in zip(ri, ci) if iou[r][c] >= thr)
    except Exception:  # greedy fallback, fine at these sizes
        pairs = sorted(
            ((iou[i][j], i, j) for i in range(len(gt)) for j in range(len(pred))),
            reverse=True,
        )
        used_g, used_p, matched = set(), set(), 0
        for v, i, j in pairs:
            if v < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j); matched += 1
    return matched, best_per_gt


def sample_passes(gt: Sequence[Box], pred: Sequence[Box],
                  thr: float = IOU_PASS_THRESHOLD) -> Dict[str, Any]:
    """PASS = same box count AND every gt box matched 1:1 at IoU >= thr
    (i.e. box-level precision = recall = 1.0 at the threshold)."""
    matched, best = match_boxes(gt, pred, thr)
    ok = (len(gt) == len(pred)) and (matched == len(gt)) and len(gt) > 0
    return {
        "pass": ok,
        "n_gt": len(gt),
        "n_pred": len(pred),
        "n_matched": matched,
        "min_best_iou": min(best) if best else 0.0,
    }


def summarize(results: List[Dict[str, Any]], label: str) -> None:
    n = len(results)
    n_pass = sum(r["pass"] for r in results)
    few = [r for r in results if r["n_gt"] <= 5]
    many = [r for r in results if r["n_gt"] > 5]
    print(f"\n=== {label}: {n_pass}/{n} PASS at per-box IoU >= {IOU_PASS_THRESHOLD} ===")
    if few:
        print(f"  <=5 boxes: {sum(r['pass'] for r in few)}/{len(few)} pass")
    if many:
        print(f"  >5 boxes:  {sum(r['pass'] for r in many)}/{len(many)} pass")
    near = [r for r in results
            if not r["pass"] and r["n_gt"] == r["n_pred"] and r["min_best_iou"] >= 0.5]
    if near:
        print(f"  near-misses (right count, all IoU >= 0.5, but < {IOU_PASS_THRESHOLD}): {len(near)}")


# --------------------------------------------------------------------------
# STEP 1: rescore the existing verify log under the IoU criterion
# --------------------------------------------------------------------------

def cmd_rescore(args: argparse.Namespace) -> None:
    """Parse '[k/32] PASS|FAIL', 'target: [...]', 'pred: [...]' blocks out of
    the existing log and rescore FAILs (and sanity-check PASSes) at IoU>=0.9.
    No model, no GPU -- pure re-analysis of the run you already have."""
    text = Path(args.log).read_text()
    blocks = re.split(r"(?=\[\d+/\d+\] (?:PASS|FAIL))", text)
    results, flipped = [], []
    for blk in blocks:
        m = re.match(r"\[(\d+)/\d+\] (PASS|FAIL)", blk)
        if not m:
            continue
        idx, old_verdict = int(m.group(1)), m.group(2)

        def grab(name: str) -> Optional[List[Box]]:
            mm = re.search(rf"{name}:\s*(\[[^\n]*\])", blk)
            if not mm:
                return None
            try:
                return [tuple(map(float, b)) for b in ast.literal_eval(mm.group(1))]
            except Exception:
                return None

        gt, pred = grab("target"), grab("pred")
        if old_verdict == "PASS" and (gt is None or pred is None):
            # old harness typically doesn't print boxes for passes
            results.append({"pass": True, "n_gt": -1, "n_pred": -1,
                            "n_matched": -1, "min_best_iou": 1.0, "idx": idx})
            continue
        if gt is None or pred is None:
            print(f"  [warn] sample {idx}: could not parse boxes, keeping old verdict {old_verdict}")
            results.append({"pass": old_verdict == "PASS", "n_gt": -1, "n_pred": -1,
                            "n_matched": -1, "min_best_iou": 0.0, "idx": idx})
            continue
        r = sample_passes(gt, pred)
        r["idx"] = idx
        results.append(r)
        if old_verdict == "FAIL" and r["pass"]:
            flipped.append(idx)
            print(f"  sample {idx}: FAIL -> PASS (tolerance artifact; "
                  f"min per-box IoU {r['min_best_iou']:.3f})")
        elif old_verdict == "FAIL":
            print(f"  sample {idx}: still FAIL "
                  f"(gt={r['n_gt']} pred={r['n_pred']} matched={r['n_matched']} "
                  f"min_best_iou={r['min_best_iou']:.3f})")

    # attach gt box counts from jsonl if given, for the <=5 / >5 split
    if args.jsonl:
        samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
        for r in results:
            i = r["idx"] - 1
            if 0 <= i < len(samples) and r["n_gt"] < 0:
                tgt = samples[i]["conversations"][2]["value"]
                r["n_gt"] = tgt.count(";") + 1 if "Answer:" in tgt else r["n_gt"]
    summarize(results, "STEP 1 rescore of existing checkpoint")
    print(f"  verdicts flipped FAIL->PASS: {flipped or 'none'}")
    print("  Remaining FAILs are the real signal (grid collapse etc.) -> proceed to Step 2.")


# --------------------------------------------------------------------------
# STEP 2a: v4 converter -- InternVL-NATIVE box syntax + contradiction check
# --------------------------------------------------------------------------
# Native grounding format InternVL was pretrained on (special tokens <ref>,
# </ref>, <box>, </box> are already added by internvl_chat_finetune.py's
# token_list). We emit ONE ref with ALL evidence boxes in its list, sorted
# by the same sort_boxes_xyxy rule as v3, so the target stays deterministic
# while riding the model's existing box-emission prior.

GROUNDING_SYSTEM_PROMPT_V4 = (
    "Given an image, a question, and its answer, identify the minimal set of "
    "regions a human would need to justify the answer: the region depicting "
    "the answer itself, its text label if visibly present, and any supporting "
    "visual evidence (legend swatch, axis/tick, comparator region, "
    "arrow/connector, or linked text).\n"
    "Respond with the evidence regions in this exact format:\n"
    "<ref>evidence</ref><box>[[x1, y1, x2, y2], [x1, y1, x2, y2]]</box>\n"
    "Coordinates are integers from 0 to 1000, normalized to the image width "
    "and height (x1,y1 = top-left, x2,y2 = bottom-right). List boxes "
    "top-to-bottom, then left-to-right. Do not repeat a box or include "
    "irrelevant regions.\n"
    "Then, on a new line, write: Explanation: <one paragraph describing each "
    "region, its role, and why it was necessary>"
)

GROUNDING_SYSTEM_PROMPT_V4_STAGE_A = GROUNDING_SYSTEM_PROMPT_V4.split(
    "Then, on a new line")[0].rstrip()


def _import_v3():
    """Import the pieces of the existing pipeline that were already correct."""
    from sft_v3_patches import (  # noqa: F401
        normalize_and_sort_gt_boxes, build_reasoning_prompt_v3)
    from dragon_grounding_convert_core import (  # noqa: F401
        extract_explanation_text, substitute_ids_with_labels,
        explanation_has_unresolved_ids)
    return (normalize_and_sort_gt_boxes, build_reasoning_prompt_v3,
            extract_explanation_text, substitute_ids_with_labels,
            explanation_has_unresolved_ids)


def build_native_target(boxes: List[Tuple[int, int, int, int]],
                        explanation: Optional[str]) -> str:
    box_list = ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in boxes)
    line = f"<ref>evidence</ref><box>[{box_list}]</box>"
    if explanation:
        line += f"\nExplanation: {explanation}"
    return line


def cmd_convert(args: argparse.Namespace) -> None:
    (normalize_and_sort_gt_boxes, build_reasoning_prompt_v3,
     extract_explanation_text, substitute_ids_with_labels,
     explanation_has_unresolved_ids) = _import_v3()
    from PIL import Image

    stage_a = args.stage.upper() == "A"
    image_root = Path(args.image_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]

    # ---- contradiction check: same (image, question) -> different gold set?
    key_to_boxes: Dict[Tuple[str, str], List[Tuple[str, tuple]]] = defaultdict(list)
    converted, skipped = [], 0
    for rec in records:
        image_rel = rec.get("image_path") or rec.get("image")
        image_abs = (image_root / image_rel).resolve()
        if not image_abs.exists():
            skipped += 1
            continue
        with Image.open(image_abs) as img:
            image_size = img.size

        boxes = normalize_and_sort_gt_boxes(rec.get("bbox") or [], image_size)
        if not boxes:
            skipped += 1
            continue

        explanation = None
        if not stage_a:
            explanation = extract_explanation_text(rec.get("explanation_raw"))
            if explanation:
                explanation = substitute_ids_with_labels(explanation, rec.get("bbox") or [])
            if not explanation or explanation_has_unresolved_ids(explanation, rec.get("bbox") or []):
                skipped += 1
                continue

        key = (str(image_rel), (rec.get("question") or "").strip().lower())
        key_to_boxes[key].append((str(rec.get("id")), tuple(boxes)))

        converted.append({
            "id": rec.get("id"),
            "image": image_rel,
            "conversations": [
                {"from": "system", "value": (GROUNDING_SYSTEM_PROMPT_V4_STAGE_A
                                             if stage_a else GROUNDING_SYSTEM_PROMPT_V4)},
                {"from": "user", "value": build_reasoning_prompt_v3(
                    question=rec.get("question", ""),
                    choices=rec.get("choices") or [],
                    answer=rec.get("answer", ""))},
                {"from": "assistant", "value": build_native_target(
                    boxes, explanation if not stage_a else None)},
            ],
            "metadata": {
                "answer": rec.get("answer", ""),
                "image_size": list(image_size),
                "gt_boxes_norm": [list(b) for b in boxes],  # GRPO reward reads this
            },
        })

    contradictions = {k: v for k, v in key_to_boxes.items()
                      if len({bx for _, bx in v}) > 1}
    if contradictions:
        print("\n!!! CONTRADICTORY DUPLICATES -- fix before training, these are "
              "unfittable by definition:")
        for (img, q), entries in contradictions.items():
            ids = ", ".join(i for i, _ in entries)
            print(f"  image={img!r} question={q[:80]!r} ids=[{ids}] "
                  f"-> {len({bx for _, bx in entries})} distinct gold sets")
        if not args.allow_contradictions:
            sys.exit("Aborting (pass --allow-contradictions to write anyway).")

    with out_path.open("w") as f:
        for s in converted:
            f.write(json.dumps(s) + "\n")
    print(f"\nWrote {len(converted)} samples -> {out_path} "
          f"(skipped {skipped}; stage {'A: box line only' if stage_a else 'B: box line + explanation'})")
    print("Example target:\n  " + converted[0]["conversations"][2]["value"][:160] if converted else "")


# --------------------------------------------------------------------------
# STEP 2b: meta file + dynamic-tiling config overrides
# --------------------------------------------------------------------------
# Single 448x448 tile (~256 image tokens) makes chart/legend/axis text
# illegible -> the model can see repeated structure but can't bind the
# question to a region, which is exactly the observed grid-collapse mode.
# Restore InternVL's grounding-pretraining regime: up to 12 tiles + thumbnail.

TILE_TOKENS = 256          # 448/14 patches, 0.5 downsample -> 256 tokens/tile
MAX_TILES = 12             # + 1 thumbnail
TEXT_BUDGET = 1024         # prompt + target headroom for phase 0


def cmd_make_meta(args: argparse.Namespace) -> None:
    image_tokens = TILE_TOKENS * (MAX_TILES + 1)          # 3328
    max_seq_len = image_tokens + TEXT_BUDGET               # 4352 -> round up
    max_seq_len = ((max_seq_len + 255) // 256) * 256       # 4352 -> 4352; keep aligned

    meta = {
        "dragon_phase0_v4": {
            "root": str(Path(args.image_root).resolve()) + "/",
            "annotation": str(Path(args.jsonl).resolve()),
            "data_augment": False,
            "repeat_time": args.repeat_time,
            "max_dynamic_patch": MAX_TILES,
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta -> {out}")
    print("\nSet these in the finetune config / CLI (the fixes, explicitly):")
    print(json.dumps({
        "dynamic_image_size": True,       # WAS False -> single illegible tile
        "use_thumbnail": True,            # WAS False
        "max_dynamic_patch": MAX_TILES,
        "force_image_size": 448,
        "max_seq_length": max_seq_len,    # sized for ~13 tiles x 256 tokens
        # NOT "internvl2_5": that routes to preprocess_internvl2_5, which
        # requires conversation['from'] in {'human','gpt'} -- this whole
        # pipeline uses {'system','user','assistant'} (preprocess_internlm's
        # convention, already fixed this session to handle a leading system
        # turn). Native <ref>/<box> tokens tokenize the same regardless of
        # which conv_style formats the surrounding turns, so nothing is lost
        # by keeping this.
        "conv_style": "internlm2-chat",
    }, indent=2))
    print("\nPhase-0 training recipe unchanged otherwise: ~300 steps, lr 1e-4, "
          "LoRA on LLM (+ optionally use_backbone_lora>0 as an ablation AFTER "
          "the resolution fix is attributed).")


# --------------------------------------------------------------------------
# STEP 3: clean greedy verify harness
# --------------------------------------------------------------------------

NATIVE_BOX_RE = re.compile(r"<box>\s*(\[.*?\])\s*</box>", re.S)
V3_ANSWER_RE = re.compile(r"Answer:\s*([0-9][0-9,;\s]*)")


def parse_pred_boxes(text: str) -> List[Box]:
    """Accept native <box>[[...]]</box> (possibly several ref/box groups)
    and fall back to the v3 'Answer: x1,y1,x2,y2;...' format."""
    boxes: List[Box] = []
    for m in NATIVE_BOX_RE.finditer(text):
        try:
            payload = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if payload and isinstance(payload[0], (int, float)):
            payload = [payload]  # single flat box
        for b in payload:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append(tuple(float(v) for v in b))
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


def chat_keep_special_tokens(model, tokenizer, pixel_values, question, generation_config,
                              device, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                              IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
    """Reimplementation of InternVLChatModel.chat() (modeling_internvl_chat.py
    lines 343-392) that decodes with skip_special_tokens=False.

    The stock chat() calls tokenizer.batch_decode(..., skip_special_tokens=True)
    -- but <ref>/</ref>/<box>/</box> were registered as special tokens during
    training (internvl_chat_finetune.py: tokenizer.add_tokens(token_list,
    special_tokens=True)), so the stock decode silently erases the exact tags
    this v4 format and parse_pred_boxes() depend on. That produces empty
    predictions no matter what the model actually generated -- this is the
    root cause of the 0/32-with-all-empty-preds result, not a training failure.
    """
    import torch
    from internvl.conversation import get_conv_template

    if pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    generation_config = dict(generation_config)
    generation_config['eos_token_id'] = eos_token_id
    with torch.no_grad():
        generation_output = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
    response = response.split(template.sep.strip())[0].strip()
    return response


def cmd_verify(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    # Must run before ANY `internvl.*` import: when this script is invoked
    # standalone (not through train_dragon6_fast_v2b.py, which sets
    # PYTHONPATH=InternVL/internvl_chat for the torchrun subprocess),
    # nothing else puts that package on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path, load_lora_checkpoint
    _ensure_internvl_repo_on_path()

    from internvl.train.dataset import build_transform, dynamic_preprocess

    # This pipeline never merges LoRA into the base weights (save_lora_
    # checkpoint saves vision_lora/, llm_lora/, trainable_extra.safetensors
    # separately -- there is no merged model.safetensors to hand a plain
    # InternVLChatModel.from_pretrained()). Reuse the loader every other
    # eval script in this repo already uses.
    device = args.device
    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
        hf_cache_dir=args.hf_cache_dir,
    )

    # Inference tiling MUST mirror training tiling.
    transform = build_transform(is_train=False, input_size=448,
                                pad2square=False, normalize_type="imagenet")

    # PURE GREEDY. repetition_penalty=1.0 is deliberate: the benchmark
    # eval's 1.1 penalizes legitimately repeated digits/commas in long
    # coordinate strings and corrupts them -- it must not leak in here.
    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                   num_beams=1, repetition_penalty=1.0)

    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    image_root = Path(args.image_root)
    results = []
    for k, s in enumerate(samples, 1):
        img = Image.open(image_root / s["image"]).convert("RGB")
        tiles = dynamic_preprocess(img, min_num=1, max_num=MAX_TILES,
                                   image_size=448, use_thumbnail=True)
        pixel_values = torch.stack([transform(t) for t in tiles]).to(
            torch.bfloat16).to(device)

        question = s["conversations"][1]["value"]
        if "<image>" not in question:
            question = "<image>\n" + question
        # chat_keep_special_tokens() reads model.system_message, which was
        # baked into the checkpoint's config from THIS jsonl's system turn,
        # so training and inference framing agree by construction. It is a
        # drop-in for model.chat() except it decodes with
        # skip_special_tokens=False so <ref>/<box> survive (see docstring).
        response = chat_keep_special_tokens(model, tokenizer, pixel_values, question,
                                            gen_cfg, device)

        gt = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
        pred = parse_pred_boxes(response)
        r = sample_passes(gt, pred)
        r["idx"] = k
        results.append(r)
        tag = "PASS" if r["pass"] else "FAIL"
        print(f"[{k}/{len(samples)}] {tag}  id={s.get('id')}")
        if not r["pass"]:
            print(f"    target: {[tuple(int(v) for v in b) for b in gt]}")
            print(f"    pred:   {[tuple(int(v) for v in b) for b in pred]}")

    summarize(results, "STEP 3 verify (v4: native syntax + dynamic tiling, greedy)")
    print("\nInterpretation: expect ~30+/32 if the resolution/format diagnosis "
          "holds. If grid collapse persists even here, memorization itself "
          "requires legible question-to-label binding -- report that in the "
          "camera-ready training-split discussion.")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rescore", help="Step 1: rescore existing verify log at IoU>=0.9")
    p.add_argument("--log", required=True)
    p.add_argument("--jsonl", default=None, help="phase0 jsonl (for box-count split)")
    p.set_defaults(fn=cmd_rescore)

    p = sub.add_parser("convert", help="Step 2a: build v4 jsonl (native box syntax)")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=["A", "a", "B", "b"], default="A")
    p.add_argument("--allow-contradictions", action="store_true")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("make-meta", help="Step 2b: meta file + tiling config overrides")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeat-time", type=int, default=50)
    p.set_defaults(fn=cmd_make_meta)

    p = sub.add_parser("verify", help="Step 3: greedy verify of retrained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
