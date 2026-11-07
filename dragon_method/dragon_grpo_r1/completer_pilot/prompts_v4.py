"""v4 Assembler: verify every candidate, fix its extent, add what nobody proposed. CPU only.

Three changes over v3t, one per finding of the 100-item error taxonomy (outputs/gc_pilot/error_taxonomy/):
  * verification: one explicit verdict per candidate instead of a list of kept numbers, so every candidate gets
    its own decision tokens and its own credit (train_gc.py --verify-credit); no (current)/(proposed) tags, which
    let v3c copy the Grounder's agreement (it kept 81% of junk that 6-8 of 8 samples proposed); optionally the
    candidates' numbers drawn on the diagram itself (prompt v4m, set-of-marks) next to the crop sheet.
  * resize: "fix [box]" replaces a candidate's box. A perfect resize of v3c's near-miss boxes would take Val600
    GroupIoU 0.378 -> 0.442 and F1@50 0.465 -> 0.612 (resize_ceiling.py); keep/drop cannot reach any of it.
  * answer-bearing: "add" proposes boxes no candidate covers. Answer-bearing gold (chart data points, 117 of 600
    Val items, median area 0.16% of the image) is 35% "discovery" (no proposal within IoU 0.1), so selection
    alone cannot find it; the guideline names data elements explicitly (item-independent, like EDGE's own list)
    and the reward pays answer-bearing recall separately (train_gc.py --ab-weight).

Output (display numbers, every candidate exactly once, any order):
  <verify>1 keep; 2 drop; 3 fix [x1, y1, x2, y2]; 4 drop</verify><add>[x1, y1, x2, y2], ...</add>
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from prompts_v3 import GUIDE, fmt_box

MAX_ADD = 12

V4_GUIDE = (
    "For charts, plots and tables, box each individual data element whose value is used to get the answer "
    "(the bar, point, slice, segment or cell, and its label), not the whole plot area."
)

COMPLETER_SYSTEM_V4 = (
    "You verify candidate evidence regions for a question about a diagram. Look at each candidate in the image: "
    "keep it if a human needs it to verify the answer, fix its box if it shows a needed element but the box is "
    "too large, too small or shifted, and drop it otherwise. Then add any needed region no candidate covers."
)


def completer_user_v4(question: str, answer: str, cands: Sequence, marks: bool = False,
                      tags: Optional[Sequence[str]] = None, zoom: Optional[Sequence[int]] = None) -> str:
    """cands in display order; number d (1-based) in the text, on the crop sheet and (marks) on the diagram.
    zoom (v4z): display numbers that also get an enlarged cell on a third image (crops.zoom_sheet)."""
    where = ("The first image shows every candidate outlined in magenta with its number at the top-left corner. "
             if marks else "")
    lines = ["<image>",
             "Candidate crops: <image>"] + (["Zoomed small candidates: <image>"] if zoom else []) + [
             f"Question: {question}",
             f"Answer: {answer}",
             "Candidate evidence regions (x1, y1, x2, y2 on a 0-1000 grid). " + where +
             "The second image shows each candidate with the same number, outlined in red, with some of its surroundings:"]
    if zoom:
        lines.append("The third image enlarges the small candidates " + ", ".join(f"[{d}]" for d in zoom) +
                     " (same numbers, outlined in red, less surroundings).")
    lines += [f"[{d}] {fmt_box(b)}" + (f"  ({tags[d - 1]})" if tags else "") for d, b in enumerate(cands, 1)]
    lines.append("The evidence is the minimal set of regions required to verify the answer. " + GUIDE + "\n" + V4_GUIDE)
    lines.append("Give a verdict for every candidate, in the form <verify>1 keep; 2 drop; 3 fix [x1, y1, x2, y2]; ..."
                 "</verify> (fix = keep with the corrected box). Then list needed regions that no candidate covers, "
                 f"at most {MAX_ADD}, as <add>[x1, y1, x2, y2], ...</add>; write <add></add> if there are none.")
    return "\n".join(lines)


# ------------------------------------------------------------------ parse / format
BOX = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
FULL_RE = re.compile(r"\s*<verify>(.*?)</verify>\s*<add>(.*?)</add>\s*", re.S)
ITEM_RE = re.compile(r"(\d+)\s+(keep|drop|fix)\s*(?:" + BOX + r")?")
BOX_RE = re.compile(BOX)


def _box(m, off=0) -> List[float]:
    x1, y1, x2, y2 = (min(1000.0, max(0.0, float(m.group(off + k)))) for k in range(1, 5))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def parse_v4(text: str, n: int) -> Optional[Dict]:
    """{"verdict": {d: (action, box or None)}, "adds": [box], "spans": [(start, end, key)]} or None if malformed.
    Malformed: no <verify>..</verify><add>..</add>, a candidate missing / repeated / out of range, fix without a
    box, a degenerate box, or text between items that is not a separator. Adds past MAX_ADD are ignored.
    spans: character ranges of each verdict ("v", d) and each add ("a", k), for token-level credit."""
    m = FULL_RE.fullmatch(text)
    if m is None:
        return None
    verdict, spans = {}, []
    body, o = m.group(1), m.start(1)
    pos = 0
    for it in ITEM_RE.finditer(body):
        if body[pos:it.start()].strip(" ;,\n\t"):
            return None
        pos = it.end()
        d, act = int(it.group(1)), it.group(2)
        if not 1 <= d <= n or d in verdict:
            return None
        box = _box(it, 2) if it.group(3) is not None else None
        if act == "fix" and (box is None or box[2] - box[0] < 1 or box[3] - box[1] < 1):
            return None
        verdict[d] = (act, box if act == "fix" else None)
        spans.append((o + it.start(), o + it.end(), ("v", d)))
    if body[pos:].strip(" ;,\n\t") or len(verdict) != n:
        return None
    adds, body, o, pos = [], m.group(2), m.start(2), 0
    for k, bm in enumerate(BOX_RE.finditer(body)):
        if body[pos:bm.start()].strip(" ,;\n\t"):
            return None
        pos = bm.end()
        b = _box(bm)
        if k < MAX_ADD and b[2] - b[0] >= 1 and b[3] - b[1] >= 1:
            adds.append(b)
            spans.append((o + bm.start(), o + bm.end(), ("a", len(adds) - 1)))
    if body[pos:].strip(" ,;\n\t"):
        return None
    return {"verdict": verdict, "adds": adds, "spans": spans}


def format_v4(verdict: Dict[int, tuple], adds: Sequence) -> str:
    items = [f"{d} {a}" + (f" {fmt_box(b)}" if a == "fix" else "") for d, (a, b) in sorted(verdict.items())]
    return "<verify>" + "; ".join(items) + "</verify><add>" + ", ".join(fmt_box(b) for b in adds) + "</add>"


# ------------------------------------------------------------------ final set
def final_v4(shown: Sequence, p: Dict, use_fix: bool = True, use_add: bool = True, dup: float = 0.7) -> List:
    """Final evidence set. use_fix=False keeps a fixed candidate at its ORIGINAL box, use_add=False drops the adds
    (the ablations that say what each capability contributed). An add with IoU >= dup to a box already in the set
    is a duplicate and is left out."""
    from gc_common import iou
    out = []
    for d in sorted(p["verdict"]):
        a, b = p["verdict"][d]
        if a == "keep" or (a == "fix" and not use_fix):
            out.append(list(shown[d - 1]))
        elif a == "fix":
            out.append(list(b))
    if use_add:
        for b in p["adds"]:
            if all(iou(b, f) < dup for f in out):
                out.append(list(b))
    return out


# ------------------------------------------------------------------ oracle target (warm-up)
def oracle_v4(shown: Sequence, gold: Sequence, types: Optional[Sequence[str]] = None, near: float = 0.3,
              max_add: int = 8, E_disp: Optional[Sequence[int]] = None):
    """(verdict, adds) the warm-up imitates. Candidates matched 1:1 to gold at IoU >= 0.5 -> keep; for each gold box
    still unmatched, the best free candidate with IoU >= near -> fix to the gold box; the rest -> drop; gold no
    candidate reaches -> add (answer-bearing first, then reading order), at most max_add. If that leaves the set
    empty, the Grounder's own boxes (E_disp: their display numbers) are kept, as oracle_target's "near" mode does."""
    from crops import reading_order
    from gc_common import iou, match_pairs
    pairs = match_pairs(list(shown), list(gold))
    verdict = {d + 1: ("keep", None) for d in pairs}
    used = set(pairs.values())
    cand = sorted(((iou(shown[d], gold[j]), d, j) for d in range(len(shown)) if d not in pairs
                   for j in range(len(gold)) if j not in used), reverse=True)
    taken = set()
    for v, d, j in cand:
        if v < near:
            break
        if d in taken or j in used:
            continue
        verdict[d + 1] = ("fix", [float(round(c)) for c in gold[j]])
        taken.add(d); used.add(j)
    for d in range(len(shown)):
        verdict.setdefault(d + 1, ("drop", None))
    left = [j for j in range(len(gold)) if j not in used]
    ro = {j: r for r, j in enumerate(reading_order([gold[j] for j in left]))}
    ro = {left[k]: r for k, r in ro.items()}
    ab = lambda j: 0 if types and j < len(types) and types[j] == "answer_bearing" else 1
    left.sort(key=lambda j: (ab(j), ro[j]))
    adds = [[float(round(c)) for c in gold[j]] for j in left[:max_add]]
    if not adds and all(a == "drop" for a, _ in verdict.values()) and E_disp:
        for d in E_disp:
            verdict[d] = ("keep", None)
    return verdict, adds


# ------------------------------------------------------------------ per-candidate credit (training)
def candidate_scores(shown: Sequence, gold: Sequence, p: Dict, near: float = 0.3) -> Dict:
    """Credit in [0, 1] for each verdict ("v", d) and each add ("a", k) of one parsed output.
    A candidate's label comes from a 1:1 match of ALL candidates to gold: hit (matched, IoU >= 0.5); near (unmatched,
    IoU >= near to a gold box no candidate matched); junk otherwise (duplicates included).
      keep: hit 1, near 0.5, junk 0        drop: hit 0, near 0.3, junk 1
      fix:  lands (IoU >= 0.5 to the candidate's gold, or for junk/near to any free gold) -> 1, and for a hit only if
            it does not lose more than 0.05 IoU (else 0.5); a fix that misses: near 0.25, hit 0, junk 0
    An add scores 1 if it takes a gold box the rest of the final set leaves unmatched (1:1, IoU >= 0.5), else 0."""
    from gc_common import iou, match_pairs
    pairs = match_pairs(list(shown), list(gold))
    free = [j for j in range(len(gold)) if j not in pairs.values()]
    out = {}
    for d, (a, b) in p["verdict"].items():
        k = d - 1
        if k in pairs:
            g = gold[pairs[k]]
            if a == "keep":
                s = 1.0
            elif a == "drop":
                s = 0.0
            else:
                ni = iou(b, g)
                s = 1.0 if ni >= 0.5 and ni >= iou(shown[k], g) - 0.05 else (0.5 if ni >= 0.5 else 0.0)
        else:
            best = max((iou(shown[k], gold[j]) for j in free), default=0.0)
            lab = "near" if best >= near else "junk"
            if a == "keep":
                s = 0.5 if lab == "near" else 0.0
            elif a == "drop":
                s = 0.3 if lab == "near" else 1.0
            else:
                lands = max((iou(b, gold[j]) for j in free), default=0.0) >= 0.5
                s = 1.0 if lands else (0.25 if lab == "near" else 0.0)
        out[("v", d)] = s
    if p["adds"]:
        rest = final_v4(shown, p, use_add=False)
        left = [gold[j] for j in range(len(gold)) if j not in set(match_pairs(rest, list(gold)).values())]
        hit = match_pairs(p["adds"], left) if left else {}
        for k in range(len(p["adds"])):
            out[("a", k)] = 1.0 if k in hit else 0.0
    return out


def token_spans(tok, ids: Sequence[int], text: str, spans) -> Optional[List]:
    """Span key of each completion token (None outside every span), by the token's first character. None when the
    per-token decode does not concatenate to `text` (then the caller falls back to sequence-level credit)."""
    pieces = [tok.decode([t], skip_special_tokens=False) for t in ids]
    if not "".join(pieces).startswith(text):
        return None
    out, c = [], 0
    for piece in pieces:
        key = None
        for s, e, k in spans:
            if s <= c < e or (c < s < c + len(piece)):
                key = k
                break
        out.append(key)
        c += len(piece)
    return out


def loo_scores(shown: Sequence, gold: Sequence, p: Dict, phi) -> Dict:
    """Leave-one-out credit, same keys as candidate_scores: what each decision did to the set reward. A verdict's
    credit is phi(final) - phi(final with that verdict flipped: keep / fix -> drop, drop -> keep); an add's is
    phi(final) - phi(final without it). Scaled by the number of gold boxes (one box ~ one unit) and clipped to
    [-1, 1]. Dropping a hit now costs what the hit was worth to the set, not one verdict out of ~20."""
    S = final_v4(shown, p)
    base, ng = phi(S, gold), max(1, len(gold))
    clip = lambda x: max(-1.0, min(1.0, x * ng))
    out = {}
    for d, (a, b) in p["verdict"].items():
        v = dict(p["verdict"])
        v[d] = ("keep", None) if a == "drop" else ("drop", None)
        out[("v", d)] = clip(base - phi(final_v4(shown, {"verdict": v, "adds": p["adds"]}), gold))
    for k in range(len(p["adds"])):
        out[("a", k)] = clip(base - phi(final_v4(shown, {"verdict": p["verdict"],
                                                          "adds": p["adds"][:k] + p["adds"][k + 1:]}), gold))
    return out


def v4_budget(tok, n: int) -> int:
    """max_tokens for a v4 output: every candidate fixed plus MAX_ADD adds, all with 3-digit boxes."""
    worst = format_v4({d: ("fix", [999, 999, 999, 999]) for d in range(1, n + 1)}, [[999, 999, 999, 999]] * MAX_ADD)
    return len(tok(worst, add_special_tokens=False).input_ids) + 4
