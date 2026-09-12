#!/usr/bin/env python3
"""
The one genuinely new piece of engineering the vendored tool (TAM) doesn't
solve on its own: mapping a flat sequence of per-vision-token relevance
scores back onto the ORIGINAL, full-resolution image when the model's
processor split that image into multiple tiles internally, instead of
resizing it to one tile the way TAM's own InternVL demo does.

Different model families tile completely differently -- this module is a
small strategy registry (Open/Closed: add a new family's resolver as a new
function + registry entry, nothing else changes), not a single hardcoded
scheme:

  "internvl_dynamic_tile"  InternVL's public dynamic_preprocess scheme:
                            up to max_num 448x448 tiles chosen by closest-
                            aspect-ratio search, plus an optional whole-
                            image thumbnail tile when >1 tile was chosen.
                            THIS IS THE ONE IMPLEMENTED BELOW.
  (add "qwen_smart_resize" etc. later -- Qwen-VL-family tiling is a
   fundamentally different scheme -- variable-resolution patch grid via
   min/max_pixels, not discrete fixed-size tiles -- and needs its own
   resolver, not a variant of this one.)

HONESTY ABOUT CONFIDENCE: `dynamic_preprocess`/`find_closest_aspect_ratio`
below reproduce InternVL's own PUBLICLY PUBLISHED reference tiling code
(the same function, near-verbatim, appears across InternVL's official model
cards and usage examples) -- high confidence this is the right ALGORITHM.
What is NOT independently verified in this session (no GPU/model access):
whether OpenGVLab/InternVL3_5-*-HF's specific `AutoProcessor` port calls
this exact function with these exact defaults, and whether the token
layout it produces has zero separators between tiles. Both are checked by
the runtime asserts in `resolve_tile_layout` and `unflatten_vision_scores`
below -- a mismatch raises loudly instead of silently mis-painting the
heatmap. Run resolve_tile_layout() on one real example and inspect
`layout.meta` before trusting a full evaluation pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class Tile:
    box_in_original: Tuple[int, int, int, int]  # (x1, y1, x2, y2), ORIGINAL image pixel coords
    is_thumbnail: bool                          # True = whole-image context tile, not a spatial sub-region


@dataclass
class TileLayout:
    tiles: List[Tile]              # in the SAME order the processor concatenates their tokens
    grid_side: int                 # sqrt(vision tokens per tile) -- e.g. 16 for 256 tokens/tile
    tokens_per_tile: int
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InternVL's own dynamic_preprocess, reproduced (see module docstring on
# confidence level). Kept as a near-literal port so it stays diffable
# against the original if InternVL's own reference code changes.
# ---------------------------------------------------------------------------

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def internvl_dynamic_tile_boxes(
    orig_width: int, orig_height: int,
    min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], bool]:
    """Returns (list of tile boxes in ORIGINAL-image pixel coords, has_thumbnail).
    Tile order matches InternVL's own row-major block order, thumbnail (if
    any) always LAST -- this ordering is what the token-layout assumption
    in resolve_tile_layout() depends on."""
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tr = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    grid_w, grid_h = tr
    target_width, target_height = image_size * grid_w, image_size * grid_h
    blocks = grid_w * grid_h

    # boxes in the RESIZED (target_width x target_height) space, then rescaled
    # back to ORIGINAL pixel coords -- callers need original coords to paint
    # onto the untouched input image, not the model's internal resize.
    sx, sy = orig_width / target_width, orig_height / target_height
    boxes = []
    for i in range(blocks):
        col, row = i % grid_w, i // grid_w
        x1, y1 = col * image_size, row * image_size
        x2, y2 = x1 + image_size, y1 + image_size
        boxes.append((round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)))

    has_thumbnail = use_thumbnail and blocks != 1
    if has_thumbnail:
        boxes.append((0, 0, orig_width, orig_height))  # thumbnail covers the WHOLE original image
    return boxes, has_thumbnail


def resolve_tile_layout(
    image: Image.Image, n_image_tokens_total: int,
    min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = True,
) -> TileLayout:
    """The verification point: replicate the tiling DECISION from image
    size + config, then check it against the REAL number of image-context
    tokens the model's processor actually produced for this image
    (n_image_tokens_total, i.e. how many times image_token_id appears in
    input_ids). If they disagree, the replicated algorithm doesn't match
    this checkpoint's real processor -- raise, don't guess."""
    boxes, has_thumbnail = internvl_dynamic_tile_boxes(
        image.width, image.height, min_num, max_num, image_size, use_thumbnail)
    n_tiles = len(boxes)

    if n_image_tokens_total % n_tiles != 0:
        raise ValueError(
            f"resolve_tile_layout: replicated dynamic_preprocess predicts {n_tiles} tiles "
            f"(thumbnail={has_thumbnail}), but the real image-token count "
            f"{n_image_tokens_total} isn't evenly divisible by that. The replicated tiling "
            f"algorithm does not match this checkpoint's real processor -- inspect this "
            f"model's actual dynamic_preprocess/image_processor config (min/max tiles, "
            f"image_size, use_thumbnail) and pass matching kwargs instead of the defaults."
        )
    tokens_per_tile = n_image_tokens_total // n_tiles
    grid_side = int(round(math.sqrt(tokens_per_tile)))
    if grid_side * grid_side != tokens_per_tile:
        raise ValueError(
            f"resolve_tile_layout: {tokens_per_tile} tokens/tile is not a perfect square "
            f"(got tile count {n_tiles} from {n_image_tokens_total} total image tokens) -- "
            f"cannot reshape into a square patch grid. Check whether this checkpoint's ViT "
            f"downsample factor differs from the 256-tokens/448px-tile assumption."
        )

    tiles = [Tile(box_in_original=b, is_thumbnail=(has_thumbnail and i == n_tiles - 1))
             for i, b in enumerate(boxes)]
    return TileLayout(
        tiles=tiles, grid_side=grid_side, tokens_per_tile=tokens_per_tile,
        meta={"n_tiles": n_tiles, "has_thumbnail": has_thumbnail,
              "image_size_px": (image.width, image.height), "tile_px": image_size},
    )


def unflatten_vision_scores(scores: np.ndarray, layout: TileLayout, pre_upsample_filter=None) -> np.ndarray:
    """Paint a flat, per-vision-token relevance array (length == n_tiles *
    tokens_per_tile, in tile order) onto a full-resolution canvas matching
    the ORIGINAL image. The thumbnail tile is EXCLUDED from spatial
    painting -- it represents the whole image as global context, not a
    location, so attributing it to one region would be wrong; its scalar
    contribution is dropped from the spatial canvas (not averaged into
    other tiles) rather than silently mixed in.

    `pre_upsample_filter`: optional callable (2D grid -> 2D grid, same
    shape) applied to each tile's SMALL grid_side x grid_side grid BEFORE
    upsampling to its full pixel size -- e.g. TAM's rank_guassian_filter
    denoising. Deliberately a caller-supplied hook, not hardcoded here:
    this module stays filter-agnostic (Dependency Inversion), and different
    backends can supply different denoising or none. Applying a filter
    AFTER upsampling instead (i.e. at full image resolution) would be both
    architecturally backwards from how TAM's own filter is designed to be
    used (on the small pre-upsample grid) and, for rank_guassian_filter's
    pure-Python per-pixel nested loop specifically, prohibitively slow at
    real image resolutions -- always filter before this function upsamples,
    never after it returns.

    Returns canvas (H x W float32, in the SAME size as the ORIGINAL image)."""
    if scores.shape[0] != layout.meta["n_tiles"] * layout.tokens_per_tile:
        raise ValueError(
            f"unflatten_vision_scores: got {scores.shape[0]} scores, expected "
            f"{layout.meta['n_tiles'] * layout.tokens_per_tile} "
            f"({layout.meta['n_tiles']} tiles x {layout.tokens_per_tile} tokens/tile)"
        )

    w, h = layout.meta["image_size_px"]
    canvas = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)  # handles overlapping tile boxes, if any, by averaging

    import cv2
    for i, tile in enumerate(layout.tiles):
        chunk = scores[i * layout.tokens_per_tile: (i + 1) * layout.tokens_per_tile]
        if tile.is_thumbnail:
            continue  # global context, not a spatial location -- see docstring
        grid = chunk.reshape(layout.grid_side, layout.grid_side).astype(np.float32)
        if pre_upsample_filter is not None:
            grid = pre_upsample_filter(grid)   # denoise at grid_side x grid_side, e.g. 16x16 -- cheap
        x1, y1, x2, y2 = tile.box_in_original
        tw, th = max(1, x2 - x1), max(1, y2 - y1)
        resized = cv2.resize(grid, (tw, th), interpolation=cv2.INTER_LINEAR)  # THEN upsample to full tile size
        canvas[y1:y2, x1:x2] += resized
        weight[y1:y2, x1:x2] += 1.0

    nonzero = weight > 0
    canvas[nonzero] /= weight[nonzero]
    return canvas
