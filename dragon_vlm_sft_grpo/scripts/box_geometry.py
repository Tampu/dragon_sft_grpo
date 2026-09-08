#!/usr/bin/env python3
"""
Model-agnostic box geometry: extraction from the raw split_reviewed-2 bbox
schema, dedup, spatial sort, and normalization to [0, 1000] integer xyxy.

This is a trimmed, purpose-built reimplementation of the box-math shared by
dragon_sft_grpo's dragon_grounding_convert_core.py and sft_v3_patches.py --
same behavior (dedup-by-id preferring source=='gt', xyxy extraction from
{x,y,w,h} / {bbox:{x,y,w,h}} / "rectangle" corners, spatial top-to-bottom/
left-to-right sort, near-duplicate collapse), reduced to just what this
pipeline's plain-JSON-box target needs -- nothing here is model-specific,
so it is shared unchanged by every model config this repo runs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]

COORD_SCALE = 1000  # normalized coordinate range shared by target, reward, and eval


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def extract_xyxy_from_bbox_item(item: Dict[str, Any]) -> Optional[Box]:
    """Support the raw annotation schema's bbox shapes -> (x1, y1, x2, y2)."""
    if all(k in item for k in ("x", "y", "w", "h")):
        x1 = float(item["x"]); y1 = float(item["y"])
        return (x1, y1, x1 + float(item["w"]), y1 + float(item["h"]))
    if isinstance(item.get("bbox"), dict):
        bbox = item["bbox"]
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x1 = float(bbox["x"]); y1 = float(bbox["y"])
            return (x1, y1, x1 + float(bbox["w"]), y1 + float(bbox["h"]))
    rect = item.get("rectangle")
    if isinstance(rect, list) and len(rect) == 2:
        (x1, y1), (x2, y2) = rect
        return (float(x1), float(y1), float(x2), float(y2))
    return None


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


def sort_boxes_xyxy(boxes: List[Box]) -> List[Box]:
    """Sort boxes top-to-bottom, then left-to-right, based on centers."""
    def key(box: Box) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((y1 + y2) / 2.0, (x1 + x2) / 2.0)
    return sorted(boxes, key=key)


def normalize_and_sort_gt_boxes(
    bbox_items: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    """gt-preferred dedup by id -> xyxy -> normalize to [0,1000] ints ->
    spatial sort -> near-duplicate collapse in normalized space.

    Every downstream consumer -- the SFT/GRPO/val/test converters, the GRPO
    reward, and export_for_eval.py -- calls this same function, so training
    target, RL reward, and eval_script.py's input can never disagree about
    the coordinate space, regardless of which model is plugged in.
    """
    width, height = image_size
    deduped = dedup_bbox_items_by_id(bbox_items)
    gt_items = [it for it in deduped if it.get("source") == "gt"] or deduped

    boxes_xyxy: List[Box] = []
    for it in gt_items:
        xyxy = extract_xyxy_from_bbox_item(it)
        if xyxy is not None:
            boxes_xyxy.append(xyxy)
    if not boxes_xyxy:
        return []

    normed: List[Box] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        nx1 = round(clamp(x1 / width, 0.0, 1.0) * COORD_SCALE)
        ny1 = round(clamp(y1 / height, 0.0, 1.0) * COORD_SCALE)
        nx2 = round(clamp(x2 / width, 0.0, 1.0) * COORD_SCALE)
        ny2 = round(clamp(y2 / height, 0.0, 1.0) * COORD_SCALE)
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        normed.append((float(nx1), float(ny1), float(nx2), float(ny2)))

    normed = sort_boxes_xyxy(normed)

    out: List[Tuple[int, int, int, int]] = []
    for cand in normed:
        dup = any(all(abs(cand[k] - ex[k]) <= 5 for k in range(4)) for ex in out)
        if not dup:
            out.append(tuple(int(v) for v in cand))
    return out
