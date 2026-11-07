#!/usr/bin/env python3
"""
Shared, dataset-agnostic core for converting QA-conditioned grounding samples
into InternVL conversation JSONL format.

Goal (QA-conditioned grounding):
Input  (human):   <image>\nQuestion ...\n[Options...]\nAnswer: <answer>
Target (gpt):     bounding boxes needed to answer, in a stable, label-agnostic format.

This module holds the box math, prompt building, and InternVL record
serialization logic extracted from prepare_ai2d_internvl_sft_v2b.py so it can
be reused by every per-dataset converter (ai2d, infographics, mapwise,
mapIQ, circuitvqa, chartqa) without duplicating box-format handling 6 times.

Two ways to use this module:
  1. Call convert_sample() directly if your dataset needs custom per-sample
     resolution logic (e.g. AI2D's questions/<image>.json lookup).
  2. Call convert_samples_jsonl() for the generic case where each line of
     samples.jsonl already carries "question"/"choices" inline (the
     "adapter contract" used by the 5 new datasets' prepare_<name>_samples.py
     scripts, produced via scripts/prepare_dragon_grounding_sft.py).

Notes:
- Label-agnostic: we do NOT include object labels/ids in the target.
- Determinism: boxes are sorted (top-to-bottom, left-to-right) for stable targets.
- Optional: normalize coords to [0,1] with normalize=True (requires reading image size).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Task framing for QA-conditioned grounding. Injected as conversations[0] with
# from="system" so InternVL's dataset preprocessing (train/dataset.py) picks it
# up instead of falling back to the internlm2-chat template's generic default
# system message -- and mirrored onto InternVLChatConfig.system_message at
# training time (internvl_chat_finetune.py) so the saved checkpoint carries the
# same prompt for inference via model.chat(), which reads it from config.
GROUNDING_SYSTEM_PROMPT = (
    "Given an image, a question, and its answer, look at the image and "
    "identify the minimal set of bounding boxes a human would need to "
    "justify the answer: the region that depicts the answer itself (e.g. a "
    "map region, bar/point/segment, flow node, component, or labeled shape), "
    "its text label if visibly present, and any supporting visual evidence "
    "(legend swatch/text, axis/tick, comparator, neighboring region, "
    "arrow/connector, or linked text) -- located purely from the image, not "
    "from any list given to you.\n"
    "Respond with exactly two lines, nothing else, no extra spaces or line "
    "breaks within a line:\n"
    "Answer: x,y,w,h;x,y,w,h;...\n"
    "Explanation: <step-by-step reasoning, in your own words, describing "
    "each region you selected, its role (the answer itself / its label / "
    "supporting evidence), and why it was necessary>\n"
    "Coordinates are absolute pixel x,y,w,h. Do not repeat a box or include "
    "irrelevant regions."
)


def build_prompt(question_text: str, choices: List[str]) -> str:
    prompt = (question_text or "").strip()
    if choices:
        lines = [f"({chr(65 + idx)}) {choice}" for idx, choice in enumerate(choices)]
        prompt = f"{prompt}\nOptions:\n" + "\n".join(lines)
    return prompt


def extract_choices(
    sample: Dict[str, Any],
    question_meta: Dict[str, Any],
    annotation_payload: Optional[Dict[str, Any]],
) -> List[str]:
    """Best-effort choice extraction from all known sample/annotation schemas."""
    candidates: List[Any] = []
    if question_meta:
        candidates.append(question_meta.get("answerTexts"))
    candidates.append(sample.get("choices"))
    candidates.append(sample.get("options"))
    candidates.append((sample.get("meta") or {}).get("choices"))
    if annotation_payload:
        candidates.append(annotation_payload.get("choices"))
        candidates.append((annotation_payload.get("answers") or {}).get("choices"))

    for item in candidates:
        if isinstance(item, list) and item:
            cleaned = [str(x).strip() for x in item if str(x).strip()]
            if cleaned:
                return cleaned
    return []


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sort_boxes_xyxy(boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """Sort boxes top-to-bottom, then left-to-right, based on centers."""

    def key(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cy, cx)

    return sorted(boxes, key=key)


def dedup_boxes(boxes: List[Tuple[float, float, float, float]], tol: float = 1.0) -> List[Tuple[float, float, float, float]]:
    """Deduplicate near-identical boxes using a simple tolerance check."""
    deduped: List[Tuple[float, float, float, float]] = []
    for candidate in boxes:
        keep = True
        for existing in deduped:
            if (
                abs(candidate[0] - existing[0]) <= tol and
                abs(candidate[1] - existing[1]) <= tol and
                abs(candidate[2] - existing[2]) <= tol and
                abs(candidate[3] - existing[3]) <= tol
            ):
                keep = False
                break
        if keep:
            deduped.append(candidate)
    return deduped


def extract_xyxy_from_bbox_item(item: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Support common bbox formats -> (x1, y1, x2, y2)."""
    if all(k in item for k in ("x", "y", "w", "h")):
        x1 = float(item["x"])
        y1 = float(item["y"])
        x2 = x1 + float(item["w"])
        y2 = y1 + float(item["h"])
        return (x1, y1, x2, y2)
    if isinstance(item.get("bbox"), dict):
        bbox = item["bbox"]
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x1 = float(bbox["x"])
            y1 = float(bbox["y"])
            x2 = x1 + float(bbox["w"])
            y2 = y1 + float(bbox["h"])
            return (x1, y1, x2, y2)
    rect = item.get("rectangle")
    if isinstance(rect, list) and len(rect) == 2:
        (x1, y1), (x2, y2) = rect
        return (float(x1), float(y1), float(x2), float(y2))
    return None


def load_image_size(image_abs_path: Path) -> Tuple[int, int]:
    with Image.open(image_abs_path) as img:
        width, height = img.size
    return width, height


def load_annotation_payload(sample: Dict[str, Any], annotations_root: Path) -> Optional[Dict[str, Any]]:
    """Load the annotation JSON referenced by the sample, if available."""
    ann_rel = sample.get("annotation") or sample.get("meta", {}).get("annotation")
    if not ann_rel:
        return None
    ann_path = Path(ann_rel)
    if not ann_path.is_absolute():
        ann_path = annotations_root / ann_path
    ann_path = ann_path.resolve()
    if not ann_path.exists():
        return None
    try:
        return json.loads(ann_path.read_text())
    except json.JSONDecodeError:
        return None


def collect_bbox_items(
    sample: Dict[str, Any],
    annotations_root: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return bbox list for the sample, loading the annotation file if needed."""
    bbox_items = sample.get("bbox")
    annotation_payload: Optional[Dict[str, Any]] = None
    if not isinstance(bbox_items, list) or len(bbox_items) == 0:
        annotation_payload = load_annotation_payload(sample, annotations_root)
        if annotation_payload is not None:
            bbox_items = annotation_payload.get("bbox", [])
        else:
            bbox_items = []
    return bbox_items, annotation_payload


def build_boxes_string(
    bbox_items: List[Dict[str, Any]],
    normalize: bool,
    image_size: Optional[Tuple[int, int]],
    bbox_source: str,
    dedup_tol: float,
    round_ndigits: int = 4,
) -> str:
    """Convert bbox list to InternVL-friendly <boxes> serialization."""
    if bbox_source == "reviewed":
        bbox_source = "all"

    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    for box in bbox_items or []:
        src = (box.get("source") or "").lower()
        if bbox_source != "all" and src != bbox_source:
            continue
        xyxy = extract_xyxy_from_bbox_item(box)
        if xyxy is None:
            continue
        boxes_xyxy.append(xyxy)

    if not boxes_xyxy:
        return "<boxes>\n</boxes>"

    if normalize:
        if image_size is None:
            raise ValueError("normalize=True requires image_size=(width, height)")
        width, height = image_size
        normed: List[Tuple[float, float, float, float]] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            normed.append(
                (
                    clamp(x1 / width, 0.0, 1.0),
                    clamp(y1 / height, 0.0, 1.0),
                    clamp(x2 / width, 0.0, 1.0),
                    clamp(y2 / height, 0.0, 1.0),
                )
            )
        boxes_xyxy = normed

    boxes_xyxy = dedup_boxes(sort_boxes_xyxy(boxes_xyxy), tol=dedup_tol)

    lines = ["<boxes>"]
    for x1, y1, x2, y2 in boxes_xyxy:
        if normalize:
            coords = (
                round(x1, round_ndigits),
                round(y1, round_ndigits),
                round(x2, round_ndigits),
                round(y2, round_ndigits),
            )
        else:
            coords = (
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            )
        lines.append(f"<box> {coords[0]} {coords[1]} {coords[2]} {coords[3]} </box>")
    lines.append("</boxes>")
    return "\n".join(lines)


def convert_sample(
    sample: Dict[str, Any],
    question_text: str,
    question_meta: Dict[str, Any],
    include_answer_in_prompt: bool,
    normalize_boxes: bool,
    image_root: Path,
    bbox_source: str,
    dedup_tol: float,
    bbox_items: List[Dict[str, Any]],
    annotation_payload: Optional[Dict[str, Any]],
    require_choices: bool,
) -> Dict[str, Any]:
    choices = extract_choices(sample, question_meta, annotation_payload)
    if require_choices and not choices:
        raise ValueError(f"missing choices for sample id={sample.get('id')}")
    prompt = build_prompt(question_text or sample.get("question", ""), choices)

    answer = (sample.get("answer") or "").strip()
    if include_answer_in_prompt:
        human_value = f"<image>\n{prompt}\nAnswer: {answer}"
    else:
        human_value = f"<image>\n{prompt}"
    human_turn = {"from": "user", "value": human_value}

    image_rel = sample.get("image_path") or sample.get("image")
    if image_rel is None:
        raise KeyError("Sample is missing image_path/image field")

    image_abs = (image_root / image_rel).resolve()
    image_size = load_image_size(image_abs) if normalize_boxes else None

    assistant_turn = {
        "from": "assistant",
        "value": build_boxes_string(
            bbox_items=bbox_items,
            normalize=normalize_boxes,
            image_size=image_size,
            bbox_source=bbox_source,
            dedup_tol=dedup_tol,
        ),
    }

    metadata = {
        "question_id": sample.get("question_id") or sample.get("meta", {}).get("question_id"),
        "question_path": sample.get("question_path") or sample.get("meta", {}).get("question_path"),
        "annotation_path": sample.get("annotation") or sample.get("meta", {}).get("annotation"),
        "answer": answer,
    }
    if choices:
        metadata["choices"] = choices
        metadata["correct_choice_index"] = question_meta.get("correctAnswer")

    system_turn = {"from": "system", "value": GROUNDING_SYSTEM_PROMPT}

    return {
        "id": sample.get("id"),
        "image": image_rel,
        "conversations": [system_turn, human_turn, assistant_turn],
        "metadata": metadata,
    }


def convert_samples_jsonl(
    samples_path: Path,
    image_root: Path,
    annotations_root: Path,
    output_path: Path,
    normalize: bool = False,
    bbox_source: str = "all",
    dedup_tol: float = 1.0,
    include_answer: bool = True,
    require_choices: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """
    Generic converter for datasets whose samples.jsonl already carries
    "question"/"choices" inline (the adapter-contract schema) rather than
    needing an external questions/<image>.json lookup like AI2D.

    Each samples.jsonl line is expected to look like:
      {"id": "...", "question": "...", "choices": ["..."], "answer": "...",
       "image_path": "images/<file>", "bbox": [{"x":.., "y":.., "w":.., "h":..}]}
    (bbox may also live in a side annotation file referenced by "annotation".)

    Returns counts: {"written", "skipped_no_boxes", "skipped_missing_choices"}.
    """
    samples_path = Path(samples_path).resolve()
    image_root = Path(image_root).resolve()
    annotations_root = Path(annotations_root).resolve()
    output_path = Path(output_path).resolve()

    out_lines: List[str] = []
    skipped_no_boxes = 0
    skipped_missing_choices = 0

    with samples_path.open("r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            bbox_items, annotation_payload = collect_bbox_items(sample, annotations_root)
            if not isinstance(bbox_items, list) or len(bbox_items) == 0:
                skipped_no_boxes += 1
                continue

            try:
                record = convert_sample(
                    sample=sample,
                    question_text=sample.get("question", ""),
                    question_meta={},
                    include_answer_in_prompt=include_answer,
                    normalize_boxes=normalize,
                    image_root=image_root,
                    bbox_source=bbox_source,
                    dedup_tol=dedup_tol,
                    bbox_items=bbox_items,
                    annotation_payload=annotation_payload,
                    require_choices=require_choices,
                )
            except ValueError:
                skipped_missing_choices += 1
                continue
            out_lines.append(json.dumps(record, ensure_ascii=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return {
        "written": len(out_lines),
        "skipped_no_boxes": skipped_no_boxes,
        "skipped_missing_choices": skipped_missing_choices,
    }


# ---------------------------------------------------------------------------
# Pure visual-grounding format: Answer: x,y,w,h;x,y,w,h;... + Explanation: <CoT>
# Input is ONLY image + question + answer -- no candidate region list -- so
# the model must localize boxes from the image itself rather than selecting
# from ids it was already handed. Coordinates are absolute pixel x,y,w,h.
# predicted_explanation (upstream Gemini/InternVL3 CoT text) references
# region ids like "BBOX_1" that make no sense without a shown candidate list,
# so it's rewritten: id mentions become their label ("the region labeled
# 'resistor'") where available, and dropped where not. Shared by
# training-data conversion (prepare_dragon6_exp1.py) and inference
# (infer_ai2d_grounding_lora_v2b.py) so the two prompts can't drift apart the
# way the old system prompt did.
# ---------------------------------------------------------------------------

import re as _re

_EXPLANATION_PREFIX_RE = _re.compile(r"^[A-Za-z0-9][\w.\-]*\s*:\s*")
_MENTIONED_ID_RE = _re.compile(r"\b([A-Za-z]+_?\d+[A-Za-z0-9_]*)\b")


def substitute_ids_with_labels(explanation: str, bbox_items: List[Dict[str, Any]]) -> str:
    """Rewrite region-id mentions in upstream explanation text so it reads
    correctly with no candidate list shown to the model: "BBOX_1 is the
    answer object" -> "the region labeled 'resistor' is the answer object"
    when BBOX_1 has a label. Falls back to a generic "this text"/"this
    region" descriptor (from the item's kind) when there's no label, rather
    than deleting the mention outright -- ai2d and ChartQA have essentially
    no labels at all (0.1% / 1.9% of items across their full train pools),
    so outright deletion there leaves subject-less, broken sentences (e.g.
    "is the Answer object, representing...")."""
    deduped = dedup_bbox_items_by_id(bbox_items)
    id_to_label = {
        str(it.get("id", "")).strip(): it.get("label")
        for it in deduped
        if it.get("id") and it.get("label")
    }
    id_to_kind = {str(it.get("id", "")).strip(): it.get("kind") for it in deduped if it.get("id")}

    def replace(match: "_re.Match") -> str:
        token = match.group(1)
        label = id_to_label.get(token)
        if label:
            return f"the region labeled '{label}'"
        # Also covers ids the upstream explanation mentions that aren't in
        # *this* sample's bbox list at all (e.g. ChartQA's "_val"-suffixed
        # ids) -- those are phantom references just as much as a known-but-
        # unlabeled id, so they get the same generic fallback rather than
        # leaking a meaningless token like "BBOX_15_val" into the target.
        return "this text" if str(id_to_kind.get(token, "")).lower() == "text" else "this region"

    rewritten = _MENTIONED_ID_RE.sub(replace, explanation)
    # Collapse whitespace/punctuation artifacts left by deleted mentions
    # (e.g. "BBOX_1 is the answer" -> " is the answer" -> "is the answer").
    rewritten = _re.sub(r"\s+", " ", rewritten).strip()
    rewritten = _re.sub(r"^[,;]\s*", "", rewritten)
    rewritten = _re.sub(r"\s+([,.;])", r"\1", rewritten)
    return rewritten


def explanation_has_unresolved_ids(explanation: str, bbox_items: List[Dict[str, Any]]) -> bool:
    """After substitute_ids_with_labels, no id-shaped token should remain at
    all -- every match gets replaced with a label or a generic descriptor in
    a single pass, known or not. (bbox_items is unused now that unknown ids
    get the same fallback treatment as known-but-unlabeled ones; kept for a
    stable call signature.) A leftover match means something didn't resolve
    cleanly -- treat the sample as unusable rather than train on a stray id
    token with no meaning in this input."""
    return bool(_MENTIONED_ID_RE.search(explanation))


def format_bbox_coords(item: Dict[str, Any]) -> Optional[str]:
    """Plain "x,y,w,h" -- no id prefix, since there's no candidate list the
    model was shown for an id to refer back to."""
    xyxy_or_xywh = extract_xyxy_from_bbox_item(item)
    if xyxy_or_xywh is None:
        return None
    # extract_xyxy_from_bbox_item returns (x1,y1,x2,y2); convert back to x,y,w,h
    # "as in the annotations" per the task spec, rather than xyxy.
    x1, y1, x2, y2 = xyxy_or_xywh
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    # Integer pixel coords: sub-pixel precision isn't meaningful for boxes on
    # images this size, and every digit here is repeated across every region
    # in every sample -- a real token cost against the 1024 max_seq_length
    # budget (see the truncation investigation this fix is part of).
    return f"{round(x)},{round(y)},{round(w)},{round(h)}"


def dedup_bbox_items_by_id(bbox_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate pred/gt copies of the same region id, preferring gt."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in bbox_items or []:
        bid = str(item.get("id", "")).strip()
        if not bid:
            continue
        existing = by_id.get(bid)
        if existing is None or (item.get("source") == "gt" and existing.get("source") != "gt"):
            by_id[bid] = item
    return list(by_id.values())


def extract_explanation_text(raw_explanation: Optional[str]) -> str:
    """First model's explanation from a "Model-A: ...\\nModel-B: ..." blob,
    with the leading "Model-A:" prefix stripped so the trained model doesn't
    learn to name-drop the labeling model. Collapsed to one line."""
    if not raw_explanation:
        return ""
    first_segment = raw_explanation.split("\n")[0].strip()
    first_segment = _EXPLANATION_PREFIX_RE.sub("", first_segment).strip()
    return " ".join(first_segment.split())


def build_reasoning_prompt(
    question_id: str,
    question: str,
    choices: List[str],
    answer: str,
) -> str:
    """Pure grounding input: image + question + choices + answer only -- no
    candidate region list. The model must localize boxes from the image
    itself, not select from ids it was already handed."""
    choices_str = "; ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)) if choices else "None"
    return (
        "<image>\n"
        f"Question_ID: {question_id}\n"
        f"Question: {question}\n"
        f"Choices: {choices_str}\n"
        f"Correct Answer: {answer}"
    )


def build_reasoning_target(bbox_items: List[Dict[str, Any]], raw_explanation: Optional[str]) -> Optional[str]:
    """Answer line uses only reviewed/gt-tagged regions (the minimal
    sufficient set); falls back to all available regions if none are
    explicitly tagged "gt" (some datasets only carry one copy per id)."""
    deduped = dedup_bbox_items_by_id(bbox_items)
    gt_items = [it for it in deduped if it.get("source") == "gt"]
    if not gt_items:
        gt_items = deduped
    entries = [format_bbox_coords(it) for it in gt_items]
    entries = [e for e in entries if e is not None]
    if not entries:
        return None
    explanation = extract_explanation_text(raw_explanation)
    if not explanation:
        return None
    explanation = substitute_ids_with_labels(explanation, bbox_items)
    if not explanation or explanation_has_unresolved_ids(explanation, bbox_items):
        return None
    return f"Answer: {';'.join(entries)}\nExplanation: {explanation}"


def convert_annotated_reasoning_sample(record: Dict[str, Any], image_root: Path) -> Optional[Dict[str, Any]]:
    """Build one training conversation in the Answer:/Explanation: format.

    `record` is the enriched adapter-contract dict produced by
    prepare_dragon6_exp1.py's to_adapter_record(): id, question, choices,
    answer, image_path, bbox (full raw items incl. id/kind/label/source),
    explanation_raw. Returns None if there's no usable target (no gt boxes,
    or no explanation text to train on).
    """
    target = build_reasoning_target(record.get("bbox") or [], record.get("explanation_raw"))
    if target is None:
        return None

    image_rel = record.get("image_path") or record.get("image")
    if image_rel is None:
        raise KeyError("Sample is missing image_path/image field")
    image_abs = (image_root / image_rel).resolve()
    if not image_abs.exists():
        return None

    prompt = build_reasoning_prompt(
        question_id=str(record.get("id", "")),
        question=record.get("question", ""),
        choices=record.get("choices") or [],
        answer=record.get("answer", ""),
    )

    return {
        "id": record.get("id"),
        "image": image_rel,
        "conversations": [
            {"from": "system", "value": GROUNDING_SYSTEM_PROMPT},
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": target},
        ],
        "metadata": {"answer": record.get("answer", "")},
    }
