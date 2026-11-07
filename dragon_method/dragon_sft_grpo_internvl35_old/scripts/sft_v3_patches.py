#!/usr/bin/env python3
"""
SFT v3 patches for DRAGON grounding fine-tuning.

Three fixes over exp2, in order of importance:

  1. COORDINATE REPRESENTATION: normalized integer coords in [0, 1000]
     (InternVL's native grounding convention), xyxy order, so the target is
     learnable from resized tiles and aligned with the base model's
     pretraining prior. Absolute pixel coords were unlearnable: the prompt
     carried no image size and tiling destroys pixel scale.

  2. DETERMINISTIC TARGET: boxes sorted top-to-bottom, left-to-right
     (restores the sort_boxes_xyxy behavior the exp2 rewrite dropped),
     deduped after normalization.

  3. LOSS SHAPING: (a) two-stage schedule -- Stage A supervises only the
     "Answer:" line (Explanation labels masked to -100), Stage B trains the
     full target; (b) WeightedCETrainer upweights Answer-line tokens ~4x in
     Stage B so coordinate gradient isn't drowned by explanation prose.

Drop-in replacements for dragon_grounding_convert_core functions, plus a
Trainer subclass for internvl_chat_finetune.py.

Note on this repo's actual tiling config (checked against configs/
internvl3_8b_ai2d_sft.json + train_dragon6_fast_v2b.py): dynamic_image_size
is False and use_thumbnail is False, i.e. single-tile 448x448 -> ~256 image
tokens, not the 12-tile+thumbnail ~3328 tokens this file's docstring
originally assumed. max_seq_length should be sized off that real number, not
4096 -- see prepare_dragon6_exp3.py.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Reuse these from dragon_grounding_convert_core:
from dragon_grounding_convert_core import (
    clamp,
    dedup_bbox_items_by_id,
    extract_explanation_text,
    extract_xyxy_from_bbox_item,
    explanation_has_unresolved_ids,
    sort_boxes_xyxy,
    substitute_ids_with_labels,
)

COORD_SCALE = 1000  # InternVL-native normalized coordinate range

# ---------------------------------------------------------------------------
# 1+2. Fixed system prompt, user prompt, and target builder
# ---------------------------------------------------------------------------

GROUNDING_SYSTEM_PROMPT_V3 = (
    "Given an image, a question, and its answer, identify the minimal set of "
    "bounding boxes a human would need to justify the answer: the region "
    "depicting the answer itself, its text label if visibly present, and any "
    "supporting visual evidence (legend swatch, axis/tick, comparator region, "
    "arrow/connector, or linked text).\n"
    "Respond with exactly two lines:\n"
    "Answer: x1,y1,x2,y2;x1,y1,x2,y2;...\n"
    "Explanation: <one paragraph describing each region, its role, and why "
    "it was necessary>\n"
    "Coordinates are integers from 0 to 1000, normalized to the image width "
    "and height (x1,y1 = top-left corner, x2,y2 = bottom-right corner). "
    "List boxes top-to-bottom, then left-to-right. Do not repeat a box or "
    "include irrelevant regions."
)


def build_reasoning_prompt_v3(question: str, choices: List[str], answer: str) -> str:
    """Same pure-grounding input as v2, minus the Question_ID line (a noise
    token stream the model can't use) -- image + question + choices + answer.
    No image-size metadata needed: coordinates are normalized."""
    choices_str = (
        "; ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        if choices else "None"
    )
    return (
        "<image>\n"
        f"Question: {question}\n"
        f"Choices: {choices_str}\n"
        f"Correct Answer: {answer}"
    )


def normalize_and_sort_gt_boxes(
    bbox_items: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    """gt-preferred dedup by id -> xyxy -> normalize to [0,1000] ints ->
    spatial sort -> near-duplicate collapse in normalized space."""
    width, height = image_size
    deduped = dedup_bbox_items_by_id(bbox_items)
    gt_items = [it for it in deduped if it.get("source") == "gt"] or deduped

    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    for it in gt_items:
        xyxy = extract_xyxy_from_bbox_item(it)
        if xyxy is None:
            continue
        boxes_xyxy.append(xyxy)
    if not boxes_xyxy:
        return []

    normed = []
    for x1, y1, x2, y2 in boxes_xyxy:
        nx1 = round(clamp(x1 / width, 0.0, 1.0) * COORD_SCALE)
        ny1 = round(clamp(y1 / height, 0.0, 1.0) * COORD_SCALE)
        nx2 = round(clamp(x2 / width, 0.0, 1.0) * COORD_SCALE)
        ny2 = round(clamp(y2 / height, 0.0, 1.0) * COORD_SCALE)
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        normed.append((float(nx1), float(ny1), float(nx2), float(ny2)))

    normed = sort_boxes_xyxy(normed)  # THE fix exp2 dropped: stable ordering

    # dedup near-identical boxes (tol=5 in 0-1000 space ~= 0.5% of image dim)
    out: List[Tuple[int, int, int, int]] = []
    for cand in normed:
        dup = any(all(abs(cand[k] - ex[k]) <= 5 for k in range(4)) for ex in out)
        if not dup:
            out.append(tuple(int(v) for v in cand))
    return out


def build_reasoning_target_v3(
    bbox_items: List[Dict[str, Any]],
    raw_explanation: Optional[str],
    image_size: Tuple[int, int],
    include_explanation: bool = True,
) -> Optional[str]:
    """Answer line in normalized sorted xyxy. Stage A: include_explanation=
    False writes ONLY the Answer line (the collator/preprocessor then
    supervises the whole assistant turn, which is just coordinates).
    Stage B: full two-line target."""
    boxes = normalize_and_sort_gt_boxes(bbox_items, image_size)
    if not boxes:
        return None
    answer_line = "Answer: " + ";".join(
        f"{x1},{y1},{x2},{y2}" for (x1, y1, x2, y2) in boxes
    )
    if not include_explanation:
        return answer_line

    explanation = extract_explanation_text(raw_explanation)
    if not explanation:
        return None
    explanation = substitute_ids_with_labels(explanation, bbox_items)
    if not explanation or explanation_has_unresolved_ids(explanation, bbox_items):
        return None
    return f"{answer_line}\nExplanation: {explanation}"


def convert_annotated_reasoning_sample_v3(
    record: Dict[str, Any],
    image_root: Path,
    include_explanation: bool = True,
) -> Optional[Dict[str, Any]]:
    """v3 analogue of convert_annotated_reasoning_sample. The converted JSONL
    additionally stores gt_boxes_norm in metadata -- this is what the GRPO
    reward reads later, so training and reward can never disagree about the
    coordinate space."""
    image_rel = record.get("image_path") or record.get("image")
    if image_rel is None:
        raise KeyError("Sample is missing image_path/image field")
    image_abs = (image_root / image_rel).resolve()
    if not image_abs.exists():
        return None
    with Image.open(image_abs) as img:
        image_size = img.size

    target = build_reasoning_target_v3(
        record.get("bbox") or [],
        record.get("explanation_raw"),
        image_size,
        include_explanation=include_explanation,
    )
    if target is None:
        return None

    prompt = build_reasoning_prompt_v3(
        question=record.get("question", ""),
        choices=record.get("choices") or [],
        answer=record.get("answer", ""),
    )
    gt_boxes_norm = normalize_and_sort_gt_boxes(record.get("bbox") or [], image_size)

    return {
        "id": record.get("id"),
        "image": image_rel,
        "conversations": [
            {"from": "system", "value": GROUNDING_SYSTEM_PROMPT_V3},
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": target},
        ],
        "metadata": {
            "answer": record.get("answer", ""),
            "image_size": list(image_size),
            "gt_boxes_norm": [list(b) for b in gt_boxes_norm],
        },
    }


# ---------------------------------------------------------------------------
# 3b. Weighted cross-entropy: upweight Answer-line (coordinate) tokens.
#     Use in Stage B. Stage A doesn't need it (target IS the answer line).
# ---------------------------------------------------------------------------

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer

IGNORE_INDEX = -100


def build_answer_token_weight(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    answer_weight: float = 4.0,
) -> torch.Tensor:
    """Per-token weights aligned with `labels`: answer_weight for tokens of
    the assistant's Answer line, 1.0 for the rest of the supervised span,
    0.0 for ignored positions.

    Locates the Answer line by decoding the supervised span once per sample
    and counting tokens up to the first newline. Cheap (runs on CPU tensors
    in the collator) and robust to tokenizer merges at the boundary within
    +/-1 token, which is immaterial at weight granularity.
    """
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for b in range(labels.size(0)):
        sup = (labels[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if sup.numel() == 0:
            continue
        weights[b, sup] = 1.0
        start = sup[0].item()
        span_ids = input_ids[b, sup].tolist()
        text = tokenizer.decode(span_ids, skip_special_tokens=False)
        nl = text.find("\n")
        if nl == -1:  # answer-only target: everything is the Answer line
            weights[b, sup] = answer_weight
            continue
        n_answer_tokens = len(
            tokenizer(text[:nl], add_special_tokens=False).input_ids
        )
        end = min(start + n_answer_tokens, labels.size(1))
        weights[b, start:end] = answer_weight
    return weights


class WeightedCETrainer(Trainer):
    """Same causal-LM objective, but the scalar loss is a weighted mean:
    coordinate tokens contribute answer_weight x the gradient of prose
    tokens. Nothing else about backprop changes -- LoRA adapters + mlp1
    still receive the gradient exactly as before.
    """

    def __init__(self, *args, answer_weight: float = 4.0, weight_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_weight = answer_weight
        self.weight_tokenizer = weight_tokenizer or self.processing_class

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        token_w = build_answer_token_weight(
            labels.cpu(), inputs["input_ids"].cpu(),
            self.weight_tokenizer, self.answer_weight,
        ).to(labels.device)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_w = token_w[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_tok = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        w = shift_w.view(-1)
        loss = (per_tok * w).sum() / w.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Phase 0 sanity harness: overfit 32 samples. If the model cannot memorize
# 32 targets to ~verbatim reproduction in a few hundred steps, the failure
# is a pipeline bug (template mismatch, truncation, label masking), not a
# capacity or algorithm problem. Run this BEFORE any full training.
#
#   - Build a 32-line JSONL with convert_annotated_reasoning_sample_v3
#     (include_explanation=False), repeat_time=50 in the meta file.
#   - Train ~300 steps, lr 1e-4, then greedy-decode all 32 prompts and diff
#     against targets. Expect exact or near-exact coordinate reproduction.
# ---------------------------------------------------------------------------
