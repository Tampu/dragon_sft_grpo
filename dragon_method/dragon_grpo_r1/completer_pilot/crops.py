"""Crop sheet for the v3 completer: one image holding a numbered, zoomed crop of every candidate. CPU only.

The sheet is exactly tc x tr tiles of 448 px, so InternVL's dynamic tiling (max 12 tiles, HF and vLLM alike)
cuts it into exactly those tiles (plus the thumbnail when tc * tr > 1); no tile mixes two layouts. Up to 24
candidates get 4 cells per tile (224 px), up to 54 get 9 (149 px), more get 16 (112 px); at most 6 tiles,
i.e. at most 7 x 256 vision tokens next to the diagram's 13 x 256.

Each cell: a header strip with the candidate's number, then the box and its surroundings (half the box size
on each side plus 2% of the image, widened to the cell's aspect ratio), scaled to fill the cell, with the box
outlined in red.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

TILE = 448
GRIDS = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2)}
LAYOUTS = [(24, 2), (54, 3), (96, 4)]          # (max candidates, cells per tile side)


def reading_order(boxes: Sequence[Sequence[float]], band: float = 50.0) -> List[int]:
    """Indices of boxes top-to-bottom (in bands of `band` on the 0-1000 grid), then left-to-right."""
    return sorted(range(len(boxes)), key=lambda i: (int(boxes[i][1] // band), boxes[i][0], boxes[i][1], i))


def layout(n: int) -> Tuple[int, int, int]:
    """(tiles across, tiles down, cells per tile side) for n candidates."""
    for cap, k in LAYOUTS:
        if n <= cap:
            t = max(1, math.ceil(n / (k * k)))
            return (*GRIDS[t], k)
    raise ValueError(f"{n} candidates do not fit on one sheet (max {LAYOUTS[-1][0]})")


def _font(px):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                           # Pillow < 10.1
        return ImageFont.load_default()


def _context(b, W, H, aspect, margin=0.5, pad=0.02):
    """Pixel crop window around box b (pixels): `margin` x the box size plus `pad` x the image on each side, then
    widened to `aspect` (w / h), clamped to the image."""
    x1, y1, x2, y2 = b
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    w, h = bw * (1 + 2 * margin) + 2 * pad * W, bh * (1 + 2 * margin) + 2 * pad * H
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, W), min(h, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l = min(max(0.0, cx - w / 2), W - w)
    t = min(max(0.0, cy - h / 2), H - h)
    return l, t, l + w, t + h


def crop_sheet(img, boxes: Sequence[Sequence[float]], labels: Sequence[int] = None, k: int = None,
               margin: float = 0.5, pad: float = 0.02):
    """img: PIL RGB; boxes: candidates in display order, xyxy on the 0-1000 grid. Returns a PIL RGB sheet.
    labels: the number printed on each cell (default 1..n); k: cells per tile side (default from layout())."""
    from PIL import Image, ImageDraw
    n = len(boxes)
    labels = list(labels) if labels is not None else list(range(1, n + 1))
    if k is None:
        tc, tr, k = layout(max(1, n))
    else:
        tc, tr = GRIDS[max(1, math.ceil(n / (k * k)))]
    cell = TILE // k
    sheet = Image.new("RGB", (TILE * tc, TILE * tr), "white")
    draw = ImageDraw.Draw(sheet)
    head = max(16, cell // 8)
    font = _font(head - 3)
    W, H = img.size
    cols = tc * k
    for d, b in enumerate(boxes):
        r, c = divmod(d, cols)
        ox, oy = c * cell, r * cell
        px = [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]
        px = [min(px[0], px[2]), min(px[1], px[3]), max(px[0], px[2]), max(px[1], px[3])]
        cw, ch = cell - 2, cell - head - 2
        l, t, rr, bb = _context(px, W, H, cw / ch, margin, pad)
        sx, sy = cw / (rr - l), ch / (bb - t)
        s = min(sx, sy)
        pw, ph = max(1, int(round((rr - l) * s))), max(1, int(round((bb - t) * s)))
        patch = img.crop((int(l), int(t), int(math.ceil(rr)), int(math.ceil(bb)))).resize((pw, ph), Image.BICUBIC)
        x0, y0 = ox + 1 + (cw - pw) // 2, oy + head + 1 + (ch - ph) // 2
        sheet.paste(patch, (x0, y0))
        bx = [x0 + (px[0] - l) * s, y0 + (px[1] - t) * s, x0 + (px[2] - l) * s, y0 + (px[3] - t) * s]
        draw.rectangle(bx, outline=(230, 0, 0), width=2)
        draw.rectangle([ox, oy, ox + cell - 1, oy + head - 1], fill=(0, 0, 0))
        draw.text((ox + 3, oy + 1), str(labels[d]), fill=(255, 255, 255), font=font)
        draw.rectangle([ox, oy, ox + cell - 1, oy + cell - 1], outline=(160, 160, 160), width=1)
    return sheet


ZOOM_AREA = 0.005        # candidates below 0.5% of the image get a zoom cell (size_bins.py: v4 loses them)
ZOOM_MAX = 12


def zoom_pick(boxes: Sequence[Sequence[float]]) -> List[int]:
    """Display numbers (1-based, ascending) of the candidates to zoom: area < ZOOM_AREA, the ZOOM_MAX smallest."""
    area = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) / 1e6 for b in boxes]
    small = sorted((a, d) for d, a in enumerate(area, 1) if a < ZOOM_AREA)[:ZOOM_MAX]
    return sorted(d for _, d in small)


def zoom_sheet(img, boxes: Sequence[Sequence[float]], labels: Sequence[int]):
    """Small candidates enlarged: a whole 448-px tile each (up to 6), else 224-px cells; tight context (a quarter
    of the box plus 1% of the image on each side), so a box that is ~1 visual token in the diagram gets dozens."""
    return crop_sheet(img, boxes, labels, k=1 if len(boxes) <= 6 else 2, margin=0.25, pad=0.01)


def sheet_tiles(n: int) -> int:
    """Image tiles InternVL makes of the sheet: tc * tr, plus the thumbnail when that is more than one."""
    tc, tr, _ = layout(max(1, n))
    return tc * tr + (1 if tc * tr > 1 else 0)


MARK = (255, 0, 200)          # magenta: rarer than red in charts and maps, so a mark is not read as a data series


def marked_diagram(img, boxes: Sequence[Sequence[float]], numbers: bool = True, width: int = 0):
    """Set-of-marks: the diagram with every candidate outlined (magenta on a white halo) and, with numbers=True, its display number in a
    small black tag at the box's top-left corner (inside the image). boxes in display order, 0-1000 grid.
    One box with numbers=False is the single-candidate view of probe_verify.py."""
    from PIL import ImageDraw
    out = img.copy()
    W, H = out.size
    draw = ImageDraw.Draw(out)
    lw = width or max(2, round(min(W, H) / 400))
    font = _font(max(11, round(min(W, H) / 60)))
    for d, b in enumerate(boxes):
        x1, y1, x2, y2 = b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(255, 255, 255), width=lw + 2)   # halo
        draw.rectangle([x1, y1, x2, y2], outline=MARK, width=lw)
        if numbers:
            t = str(d + 1)
            l, tp, r, bt = draw.textbbox((0, 0), t, font=font)
            tx, ty = min(max(0, x1), W - (r - l) - 4), min(max(0, y1), H - (bt - tp) - 4)
            draw.rectangle([tx, ty, tx + (r - l) + 4, ty + (bt - tp) + 4], fill=(0, 0, 0))
            draw.text((tx + 2 - l, ty + 2 - tp), t, fill=(255, 255, 255), font=font)
    return out
