"""Grounder-completer pilot: pool, completer prompt, rewards, edit statistics. CPU only.

Phi is omission_grpo.set_scores (Hungarian 1:1 on IoU, s = IoU if IoU >= 0.5 else 0.25 * IoU,
Phi = 2 * sum(s) / (|S| + |G|)), the same function as the credit pilot. Boxes are xyxy on the
0-1000 grid. A state record is what generate_states.py writes.

PATCH (after seed 17), all backward compatible -- every v1 function and the default completer prompt
are unchanged, so the existing round-0 completer sees exactly the prompts it was trained on:
  * build_pool keeps "sids" (which samples proposed each cluster). The prompt does not show them.
  * grounder_terms + mix_terms: v2 grounder reward (see their docstrings).
  * e_support / voting_drop / tune_voting: an untrained baseline that can also DROP E boxes.
  * edit_errors: why a state is worth training the completer on (curriculum).
  * completer_user(..., e_sup=...): optional agreement count on E lines, for a NEW completer only.
  * select_with_idx: final set plus the selected indices.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omission_pilot"))
import omission_grpo as OG  # noqa: E402

CFG = OG.CreditConfig()
iou = OG.iou
FORMAT_PENALTY = -0.1
BINS = [("1-2", 1, 2), ("3-8", 3, 8), ("9+", 9, 10 ** 6)]
# The grounder's system turn demands <ref>/<box> output, which contradicts <select>, so the completer
# gets its own. All format instructions stay in the user turn.
COMPLETER_SYSTEM = "You review candidate evidence regions for a question about an image."
SELECT_RE = re.compile(r"<select>\s*(\d+(?:\s*,\s*\d+)*)?\s*</select>")
V4_PROMPTS = ("v4", "v4m", "v4t", "v4z")


def phi(S, gold, cfg=CFG) -> float:
    return OG.set_scores(S, gold, cfg)[1]


def match_pairs(P, G, thr: float = 0.5) -> Dict[int, int]:
    """Hungarian 1:1 on IoU; {index in P: index in G} for pairs with IoU >= thr."""
    if not len(P) or not len(G):
        return {}
    M = np.array([[iou(p, g) for g in G] for p in P])
    r, c = linear_sum_assignment(-M)
    return {int(i): int(j) for i, j in zip(r, c) if M[i, j] >= thr}


def hungarian_hits(P, G, thr: float = 0.5) -> Set[int]:
    return set(match_pairs(P, G, thr).values())


# ------------------------------------------------------------------- pool
def _dedupe(boxes, thr):
    out = []
    for b in boxes:
        if all(iou(b, k) < thr for k in out):
            out.append(b)
    return out


def _medoid(members):
    if len(members) == 1:
        return members[0]
    means = [np.mean([iou(a, b) for j, b in enumerate(members) if j != i]) for i, a in enumerate(members)]
    return members[int(np.argmax(means))]


def build_pool(E, samples, iou_merge=0.5, iou_covered=0.5, max_pool=30) -> List[Dict]:
    """Clusters of the samples' boxes, minus those covered by E, by descending support.
    support = number of samples containing the cluster, after per-sample dedupe.
    sids = those samples' indices (used by the v2 grounder reward; never shown to the completer)."""
    clusters = []
    for si, s in enumerate(samples):
        for b in _dedupe(s, iou_merge):
            for c in clusters:
                if iou(b, c["rep"]) >= iou_merge:
                    c["members"].append(b)
                    c["sids"].add(si)
                    break
            else:
                clusters.append({"rep": b, "members": [b], "sids": {si}})
    pool = []
    for c in clusters:
        rep = _medoid(c["members"])
        if all(iou(rep, e) < iou_covered for e in E):
            pool.append({"rep": list(rep), "support": len(c["sids"]), "sids": sorted(c["sids"])})
    pool.sort(key=lambda c: -c["support"])
    return pool[:max_pool]


def e_support(E, samples, thr: float = 0.5) -> List[int]:
    """Per E box: how many samples contain a box with IoU >= thr to it (agreement with greedy)."""
    return [sum(any(iou(e, b) >= thr for b in s) for s in samples) for e in E]


# ----------------------------------------------------------------- prompt
def split_question(user: str):
    """(question text incl. any Choices line, answer) from the grounder's user turn; answer "" when the turn has
    no "Correct Answer:" line (the no-answer condition of paper analysis 3)."""
    u = user.replace("<image>", "").strip()
    q, a = u.rsplit("Correct Answer:", 1) if "Correct Answer:" in u else (u, "")
    q = q.strip()
    return (q[len("Question:"):].strip() if q.startswith("Question:") else q), a.strip()


def completer_user(question: str, answer: str, E, pool, K: int, e_sup: Optional[Sequence[int]] = None) -> str:
    """e_sup=None reproduces the seed-17 prompt byte for byte. Pass e_sup only for a completer trained with it."""
    lines = [f"Question: {question}", f"Answer: {answer}", "Candidate evidence regions (coordinates 0-1000):"]
    fmt = lambda b: "[" + ", ".join(str(int(round(c))) for c in b) + "]"
    for j, b in enumerate(E):
        tag = "(current)" if e_sup is None else f"(current, support {e_sup[j]}/{K})"
        lines.append(f"[{len(lines) - 2}] {fmt(b)}  {tag}")
    for c in pool:
        lines.append(f"[{len(lines) - 2}] {fmt(c['rep'])}  (proposed, support {c['support']}/{K})")
    lines.append("Select every region a human needs to verify the answer, and only those.")
    lines.append("Output the indices as <select>i, j, ...</select>. Output <select></select> if none are needed.")
    return "\n".join(lines)


def completer_item(rec: Dict, K: int, max_pool: int = 30, show_e_support: bool = False, prompt: str = "v1") -> Dict:
    """State record -> engine item (prompt_text reads "system" and "user") with the pool attached.

    prompt="v3t": v3 plus a (current) / (proposed) tag per line -- which boxes greedy E holds, never support.
    prompt="v3": EDGE guideline, E and pool merged into one list in reading order with no tags or support,
    and item["crops"] (the candidates in display order) for the crop sheet. item["order"][d - 1] is the
    0-based internal index (E first, then pool) of display number d; to_internal maps the model's output."""
    E = [list(b) for b in rec["E"]]
    samples = rec["samples"][:K]
    pool = build_pool(E, samples, max_pool=max_pool)
    e_sup = e_support(E, samples)
    it = {"id": rec["id"], "domain": rec["domain"], "image": rec["image"],
          "gold": [list(b) for b in rec["gold"]], "E": E, "pool": pool, "K": K, "e_sup": e_sup, "prompt": prompt}
    if prompt == "v1":
        it.update(system=COMPLETER_SYSTEM,
                  user=completer_user(rec["question"], rec["answer"], E, pool, K, e_sup if show_e_support else None))
        return it
    if prompt not in ("v3", "v3t") + V4_PROMPTS:
        raise ValueError(f"unknown completer prompt {prompt!r}")
    from crops import reading_order
    from prompts_v3 import COMPLETER_SYSTEM_V3, completer_user_v3
    cands = E + [c["rep"] for c in pool]
    order = reading_order(cands)
    shown = [cands[i] for i in order]
    tags = ["current" if i < len(E) else "proposed" for i in order] if prompt in ("v3t", "v4t") else None
    if prompt in V4_PROMPTS:
        # v4: verdict per candidate + fix + add (prompts_v4.py); v4m also draws the numbers on the diagram
        # v4z also enlarges the small candidates on a third image (crops.zoom_sheet)
        from crops import zoom_pick
        from prompts_v4 import COMPLETER_SYSTEM_V4, completer_user_v4
        zoom = zoom_pick(shown) if prompt == "v4z" else []
        it.update(system=COMPLETER_SYSTEM_V4, crops=shown, order=order, marks=prompt == "v4m", zoom=zoom,
                  user=completer_user_v4(rec["question"], rec["answer"], shown, prompt == "v4m", tags, zoom),
                  E_disp=sorted(d + 1 for d, i in enumerate(order) if i < len(E)), gold_types=rec.get("gold_types"))
        return it
    it.update(system=COMPLETER_SYSTEM_V3, user=completer_user_v3(rec["question"], rec["answer"], shown, tags),
              crops=shown, order=order)
    return it


def to_internal(text: str, it: Dict) -> Optional[List[int]]:
    """The completer's output as sorted internal 1-based indices (E first, then pool), None if malformed."""
    idx = parse_select(text, len(it["E"]) + len(it["pool"]))
    if idx is None or "order" not in it:
        return idx
    return sorted(it["order"][d - 1] + 1 for d in idx)


def to_display(idx: Sequence[int], it: Dict) -> List[int]:
    """Internal 1-based indices -> the numbers the item's prompt shows (identity for v1)."""
    if "order" not in it:
        return sorted(idx)
    pos = {k: d for d, k in enumerate(it["order"], 1)}
    return sorted(pos[i - 1] for i in idx)


def select_text(idx: Sequence[int]) -> str:
    return "<select>" + ", ".join(str(i) for i in idx) + "</select>"


def canonical(text: str, it: Dict) -> str:
    """The output rewritten in internal indices, so every v1 function (parse_select, select_with_idx,
    edit_stats, report.py) reads a v3 completer unchanged. Malformed stays malformed ("")."""
    if "order" not in it:
        return text
    idx = to_internal(text, it)
    return "" if idx is None else select_text(idx)


def select_budget(tok, n_listed: int, floor: int = 64) -> int:
    """max_tokens for one prompt: the spec's 64, raised where keeping every listed region needs more.
    A flat 64 fits about 17 indices, which would make "keep E" unreachable on dense items."""
    full = "<select>" + ", ".join(str(i) for i in range(1, n_listed + 1)) + "</select>"
    return max(floor, len(tok(full, add_special_tokens=False).input_ids) + 2)


# ----------------------------------------------------------------- reward
def parse_select(text: str, n_listed: int) -> Optional[List[int]]:
    """Sorted unique 1-based indices; None if malformed or an index is out of range."""
    m = SELECT_RE.fullmatch(text.strip())
    if m is None:
        return None
    idx = sorted({int(x) for x in m.group(1).split(",")}) if m.group(1) else []
    return idx if all(1 <= i <= n_listed for i in idx) else None


def selected(E, pool, idx: Sequence[int]) -> List:
    items = list(E) + [c["rep"] for c in pool]
    return [items[i - 1] for i in idx]


def completer_reward(text: str, E, pool, gold, cfg=CFG) -> float:
    idx = parse_select(text, len(E) + len(pool))
    if idx is None:
        return FORMAT_PENALTY
    return phi(selected(E, pool, idx), gold, cfg) - phi(E, gold, cfg)


def coverage_reward(P, gold, group_hits, lam: float = 0.3, fmt_ok: bool = True) -> float:
    """v1 (seed 17)."""
    if not fmt_ok:
        return -0.5
    hit = hungarian_hits(P, gold)
    cov = sum(1.0 / group_hits[g] for g in hit) / max(1, len(gold))
    junk = (len(P) - len(hit)) / max(1, len(P))
    return cov - lam * junk


def reward_parts(P, gold, group_hits, final_set, hits_E, alpha: float = 0.5, lam: float = 0.3,
                 fmt_ok: bool = True) -> Dict:
    """v1 (seed 17). A gold box this sample hits pays (1 - alpha) * 1/n_g, plus alpha * 1/n_g if the
    completer's final set hits it and greedy E did not; minus lam * junk fraction. alpha = 0 is coverage_reward."""
    hit = hungarian_hits(P, gold)
    in_final = hit & hungarian_hits(final_set, gold)
    from_pool, from_E = in_final - set(hits_E), in_final & set(hits_E)
    pay = lambda gs: sum(1.0 / group_hits[g] for g in gs) / max(1, len(gold))
    uncond, cond = (1 - alpha) * pay(hit), alpha * pay(from_pool)
    junk = lam * (len(P) - len(hit)) / max(1, len(P))
    return {"r": uncond + cond - junk if fmt_ok else -0.5, "uncond": uncond, "cond": cond, "junk_pen": junk,
            "n_hit_new": len(hit - set(hits_E)), "n_hit_E": len(hit & set(hits_E)),
            "n_acc_pool": len(from_pool), "n_acc_E": len(from_E)}


def final_from_text(text: str, E, pool):
    """(final set, malformed). A malformed selection falls back to E, as at eval."""
    idx = parse_select(text, len(E) + len(pool))
    return (list(E), True) if idx is None else (selected(E, pool, idx), False)


def select_with_idx(text: str, E, pool):
    """(final set, idx or None, malformed). idx is None when malformed; the final set is then E."""
    idx = parse_select(text, len(E) + len(pool))
    return (list(E), None, True) if idx is None else (selected(E, pool, idx), idx, False)


def group_hit_counts(box_lists, gold) -> Counter:
    return Counter(g for P in box_lists for g in hungarian_hits(P, gold))


# ------------------------------------------------------- v2 grounder reward
def final_set_credit(E, pool, idx, samples, gold, use_e_junk: bool = True, cfg=CFG) -> Dict:
    """Credit each sample for what it put into the completer's final set F.

    For every box f in F, credit(f) = s_F(f) - max(Phi(F)/2, kappa), the box credit of f inside F
    (positive for a gold hit, negative for junk or a loose box), and it is shared equally by the
    samples that proposed f:
      * an ADOPTED pool cluster -> its proposers are the cluster's sids (positive or negative credit);
      * a KEPT E box whose credit is negative (junk that survived the completer) -> its proposers are the
        samples containing a box with IoU >= 0.5 to it (only if use_e_junk);
      * a kept E box with positive credit pays nothing: it was already in greedy (cov pays for it).
    Junk the completer rejects is not in F and costs nothing. Returns {"use": [per sample, / |G|],
    "n_adopt", "n_adopt_hit", "n_ekept_junk"}."""
    n_s, n = len(samples), max(1, len(gold))
    use = [0.0] * n_s
    out = {"use": use, "n_adopt": 0, "n_adopt_hit": 0, "n_ekept_junk": 0}
    if idx is None:
        return out
    F = selected(E, pool, idx)
    s, phiF, _ = OG.set_scores(F, gold, cfg)
    be = max(phiF / 2.0, cfg.kappa)
    nE = len(E)
    for pos, i in enumerate(idx):
        credit = float(s[pos]) - be
        if i > nE:                                           # adopted pool cluster
            c = pool[i - nE - 1]
            out["n_adopt"] += 1
            out["n_adopt_hit"] += int(s[pos] >= cfg.tau)
            sids = c.get("sids") or []
            for sid in sids:
                use[sid] += credit / len(sids) / n
        elif use_e_junk and credit < 0:                      # junk / loose E box the completer kept
            f = F[pos]
            sids = [k for k, S in enumerate(samples) if any(iou(f, b) >= 0.5 for b in S)]
            out["n_ekept_junk"] += 1
            for sid in sids:
                use[sid] += credit / len(sids) / n
    return out


def grounder_terms(P_list, has_box, truncated, gold, fb: Optional[Dict] = None, cap_mult: float = 3.0,
                   use_e_junk: bool = True, weights: Optional[Sequence[float]] = None) -> Dict:
    """Raw per-sample terms of the v2 grounder reward for ONE prompt's group.

      cov_i  rarity-weighted gold hits: sum over gold g that sample i hits of w_g / n_g, divided by |G|.
             w_g = 1 (v2) unless `weights` is given: then w_g is the Assembler-difficulty weight of gold g
             (build_difficulty.py; the plan's Variant B, w_g = rarity * assembler_difficulty).
             Completer-free at train time: the weights are a frozen record of the previous Assembler.
      use_i  final_set_credit: new gold the completer adopted from sample i (+), junk from sample i that
             ended up in the final set (-). The coupling. Zero when fb is None.
      cap_i  boxes beyond B = cap_mult * max(|G|, 2), as a fraction of B. Completer-free spray guard.

    P_list: each sample's CLOSED boxes (a truncated sample keeps the boxes it finished, as eval does).
    fb: {"E", "pool" (with sids), "idx" (completer selection or None)} for this prompt."""
    n = max(1, len(gold))
    gh = group_hit_counts(P_list, gold)
    B = cap_mult * max(len(gold), 2)
    cred = final_set_credit(fb["E"], fb["pool"], fb["idx"], P_list, gold, use_e_junk) if fb else None
    if weights is not None and len(weights) != len(gold):
        raise ValueError(f"{len(weights)} difficulty weights for {len(gold)} gold boxes")
    w = list(weights) if weights is not None else [1.0] * len(gold)
    rows = []
    for i, P in enumerate(P_list):
        hit = hungarian_hits(P, gold)
        rows.append({"cov": sum(w[g] / gh[g] for g in hit) / n,
                     "use": cred["use"][i] if cred else 0.0,
                     "cap": max(0.0, len(P) - B) / B,
                     "has_box": bool(has_box[i]), "trunc": bool(truncated[i]),
                     "valid": float(bool(has_box[i]) and not truncated[i]),
                     "recall": len(hit) / n, "junk": (len(P) - len(hit)) / max(1, len(P)),
                     "bpg": len(P) / n, "n_gold": len(gold)})
    meta = {k: v for k, v in (cred or {}).items() if k != "use"}
    return {"rows": rows, "meta": meta}


def mix_terms(groups_rows: List[List[Dict]], alpha: float, sig: Dict, mu: float = 1.0, trunc_pen: float = 0.5,
              nobox_r: float = -2.0, momentum: float = 0.9, clip: float = 3.0, lam_junk: float = 0.0) -> Dict:
    """v2 grounder reward for a batch of groups.

    Each of cov and use is centered WITHIN its group (GRPO only sees within-group differences), divided by an
    EMA of its pooled within-group std (carried across steps in `sig`), and clipped to [-clip, clip]. Then
        r = (1 - alpha) * z_cov + alpha * z_use - mu * cap - trunc_pen * [truncated] - lam_junk * junk
    and a sample with no closed box gets nobox_r. junk = unmatched fraction of the sample's boxes; lam_junk = 0
    is v2 (seed-17 alpha=0 grew E to 4.7 boxes/gold on 1-2 gold items without it). So alpha really is the share of the learning signal
    that comes from the completer: in seed 17 the raw conditional pay was ~8% of the positive reward and
    far less of its within-group variance. alpha = 0 is the uncoupled control (same cap, same scaling).
    Returns {"r": per-group lists, "sd_cov", "sd_use", "use_share"} and updates sig in place."""
    def pooled_sd(key):
        dev = []
        for rows in groups_rows:
            v = np.array([r[key] for r in rows], float)
            dev.extend((v - v.mean()).tolist())
        return float(np.sqrt(np.mean(np.square(dev)))) if dev else 0.0

    sd = {}
    for k in ("cov", "use"):
        sd[k] = pooled_sd(k)
        sig[k] = sd[k] if sig.get(k) is None else momentum * sig[k] + (1 - momentum) * sd[k]
    out_r, zc_all, zu_all = [], [], []
    for rows in groups_rows:
        c = np.array([r["cov"] for r in rows], float)
        u = np.array([r["use"] for r in rows], float)
        zc = np.clip((c - c.mean()) / max(sig["cov"], 1e-6), -clip, clip) if sig["cov"] > 1e-6 else 0 * c
        zu = np.clip((u - u.mean()) / max(sig["use"], 1e-6), -clip, clip) if sig["use"] > 1e-6 else 0 * u
        r = (1 - alpha) * zc + alpha * zu
        rr = []
        for j, row in enumerate(rows):
            if not row["has_box"]:
                rr.append(float(nobox_r))
            else:
                rr.append(float(r[j]) - mu * row["cap"] - (trunc_pen if row["trunc"] else 0.0)
                          - lam_junk * row["junk"])
        out_r.append(rr)
        zc_all.extend(((1 - alpha) * zc).tolist())
        zu_all.extend((alpha * zu).tolist())
    a, b = float(np.std(zc_all)) if zc_all else 0.0, float(np.std(zu_all)) if zu_all else 0.0
    return {"r": out_r, "sd_cov": sd["cov"], "sd_use": sd["use"],
            "use_share": round(b / (a + b), 4) if a + b > 0 else 0.0}


# ---------------------------------------------------------- reference sets
def voting(E, pool, k: int) -> List:
    """Add-only voting (seed 17 rows)."""
    return list(E) + [c["rep"] for c in pool if c["support"] >= k]


def voting_drop(E, pool, e_sup, k_add: int, k_keep: int) -> List:
    """Voting that can also drop: keep an E box only if >= k_keep samples agree with it, add pool clusters
    with support >= k_add. k_keep = 0 is add-only voting. The fair untrained counterpart of the completer,
    which both adds and drops."""
    return [e for e, s in zip(E, e_sup) if s >= k_keep] + [c["rep"] for c in pool if c["support"] >= k_add]


def tune_voting(records: Sequence[Dict], K: int, score: Optional[Callable] = None):
    """Grid-search (k_add, k_keep) on TRAIN states. records: state dicts with E, samples, gold.
    score(S, gold) -> float, default phi; pass your GroupIoU or F1 to match the headline metric.
    Returns ((k_add, k_keep), mean score) and the full grid."""
    score = score or (lambda S, g: phi(S, g))
    prepared = []
    for r in records:
        E = [list(b) for b in r["E"]]
        smp = r["samples"][:K]
        prepared.append((E, build_pool(E, smp), e_support(E, smp), r["gold"]))
    grid = {}
    for k_add in range(1, K + 1):
        for k_keep in range(0, K + 1):
            grid[(k_add, k_keep)] = float(np.mean([score(voting_drop(E, p, es, k_add, k_keep), g)
                                                   for E, p, es, g in prepared]))
    best = max(grid, key=grid.get)
    return best, grid[best], grid


def oracle_edit(E, pool, gold) -> List:
    """E boxes that hit gold + a pool cluster for each still-unmatched gold box it can hit."""
    pairs = match_pairs(E, gold)
    left = [g for j, g in enumerate(gold) if j not in pairs.values()]
    reps = [c["rep"] for c in pool]
    return [E[i] for i in sorted(pairs)] + [reps[i] for i in sorted(match_pairs(reps, left))]


def oracle_idx(E, pool, gold) -> List[int]:
    """oracle_edit as internal 1-based indices (E first, then pool): the supervised warm-up target."""
    pairs = match_pairs(E, gold)
    left = [g for j, g in enumerate(gold) if j not in pairs.values()]
    reps = [c["rep"] for c in pool]
    return sorted(i + 1 for i in pairs) + sorted(len(E) + i + 1 for i in match_pairs(reps, left))


def pool_recall(E, pool, gold) -> float:
    return len(hungarian_hits(list(E) + [c["rep"] for c in pool], gold)) / max(1, len(gold))


def pool_hit_flags(E, pool, gold) -> List[bool]:
    """Per pool cluster: can it hit a gold box E leaves unmatched? The pool's base hit rate."""
    used = set(match_pairs(E, gold).values())
    left = [g for j, g in enumerate(gold) if j not in used]
    return [any(iou(c["rep"], g) >= 0.5 for g in left) for c in pool]


def edit_stats(E, pool, gold, idx: Sequence[int]) -> Dict:
    """Adds and drops of one valid selection. An add hits if it takes a gold box the KEPT E boxes
    leave unmatched; a drop is junk if the dropped box is not matched to gold in the full E."""
    n_E = len(E)
    kept = [i for i in range(n_E) if i + 1 in idx]
    adds = [i - n_E - 1 for i in idx if i > n_E]
    e_pairs = match_pairs(E, gold)
    drops = [i for i in range(n_E) if i + 1 not in idx]
    used = set(match_pairs([E[i] for i in kept], gold).values())
    left = [g for j, g in enumerate(gold) if j not in used]
    add_pairs = match_pairs([pool[i]["rep"] for i in adds], left)
    by_support = Counter()
    for a, i in enumerate(adds):
        by_support[(pool[i]["support"], a in add_pairs)] += 1
    return {"n_add": len(adds), "add_hit": len(add_pairs), "n_drop": len(drops),
            "drop_junk": sum(i not in e_pairs for i in drops), "n_keep": len(kept),
            "by_support": by_support}


def edit_errors(E, pool, gold, idx: Optional[Sequence[int]], rare_max: int = 2) -> Dict:
    """Why a state is worth training the completer on, given its current (greedy) selection.
      missed_add  an unselected pool cluster hits a gold box left unmatched by the kept E boxes and the adds
      junk_add    an added cluster takes no new gold
      gold_drop   a dropped E box was matched to gold
      junk_kept   a kept E box is not matched to gold
      rare_miss   a missed add has support <= rare_max (where co-evolution needs the completer to improve)
      malformed   the selection did not parse
    "error" is any of them."""
    if idx is None:
        return {"error": True, "malformed": True, "missed_add": False, "junk_add": False, "gold_drop": False,
                "junk_kept": False, "rare_miss": False}
    n_E = len(E)
    kept = [i for i in range(n_E) if i + 1 in idx]
    add_ix = [i - n_E - 1 for i in idx if i > n_E]
    e_pairs = match_pairs(E, gold)
    used = set(match_pairs([E[i] for i in kept], gold).values())
    left = [j for j in range(len(gold)) if j not in used]
    add_pairs = match_pairs([pool[i]["rep"] for i in add_ix], [gold[j] for j in left])
    taken = {left[j] for j in add_pairs.values()}
    left2 = [gold[j] for j in left if j not in taken]
    missed = [i for i in range(len(pool)) if i not in add_ix and any(iou(pool[i]["rep"], g) >= 0.5 for g in left2)]
    out = {"malformed": False,
           "missed_add": bool(missed),
           "junk_add": len(add_pairs) < len(add_ix),
           "gold_drop": any(i in e_pairs for i in range(n_E) if i + 1 not in idx),
           "junk_kept": any(i not in e_pairs for i in kept),
           "rare_miss": any(pool[i]["support"] <= rare_max for i in missed)}
    out["error"] = any(out[k] for k in ("missed_add", "junk_add", "gold_drop", "junk_kept"))
    return out


# ------------------------------------------------- failure taxonomy (co-evolution plan, Experiment 0)
GEO_BINS = [("hit", 0.5), ("near_miss", 0.3), ("weak", 0.1), ("discovery", -1.0)]   # best-IoU floors


def geometry_bucket(best: float) -> str:
    """Best IoU any proposal reaches on a gold box -> hit >= 0.5, near_miss [0.3, 0.5), weak [0.1, 0.3),
    discovery < 0.1 (no proposal is even close: the semantic element was never found)."""
    for name, floor in GEO_BINS:
        if best >= floor:
            return name
    return "discovery"


def gold_reach(E, pool, samples, gold) -> List[Dict]:
    """Per gold box: how the grounder's proposals reach it.
      greedy  E hits it (Hungarian 1:1 at 0.5, as the metrics count it)
      pool    E + the clustered, capped pool hits it (Hungarian, what the completer can actually select)
      raw     some single box of E or of any sample has IoU >= 0.5 with it (no clustering, no cap)
      best    the best IoU of any box of E or of any sample"""
    reps = [c["rep"] for c in pool]
    hE = hungarian_hits(E, gold)
    hP = hungarian_hits(list(E) + reps, gold)
    allb = list(E) + [b for s in samples for b in s]
    out = []
    for j, g in enumerate(gold):
        best = max((iou(b, g) for b in allb), default=0.0)
        out.append({"greedy": j in hE, "pool": j in hP, "raw": best >= 0.5, "best": float(best)})
    return out


def load_gold_types(path) -> Dict[str, List[str]]:
    """id -> evidence type per gold box, from demand_rl/precompute_elements.py output ({"etypes": [...]}) or
    from a state file carrying "gold_types". Ids are unique within one split; never mix splits in one file."""
    import json
    p = str(path)
    if p.endswith(".json"):
        return {k: v["etypes"] for k, v in json.load(open(p)).items()}
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if "gold_types" in r:
                out[r["id"]] = r["gold_types"]
    return out


def oracle_target(E, pool, gold, mode: str = "strict", near: float = 0.3) -> List[int]:
    """Supervised Assembler target (internal 1-based indices).
      strict  oracle_idx: E boxes matched to gold at IoU >= 0.5 + pool clusters for the rest; empty if nothing
              matches (13% of train states). GroupIoU / MeanIoU give partial credit that the empty set throws
              away, and a model imitating it learns to answer nothing (v3b warm-up: 5.7% empty on Val300).
      empty_e strict, but an empty target becomes "keep E"            (Val300 ceiling GIoU 0.546 -> 0.559, F1 same)
      near    strict + every E box with IoU >= `near` to some gold, empty -> keep E   (ceiling GIoU 0.581, F1 0.697;
              keeps 50% of E instead of 37%, so imitation errors cost less: a wrong drop costs ~3x a wrong keep)"""
    idx = oracle_idx(E, pool, gold)
    if mode == "near":
        keep = {i + 1 for i, e in enumerate(E) if max((iou(e, g) for g in gold), default=0.0) >= near}
        idx = sorted(set(idx) | keep)
    elif mode != "strict" and mode != "empty_e":
        raise ValueError(f"unknown oracle target {mode!r}")
    if not idx and mode != "strict":
        idx = list(range(1, len(E) + 1))
    return idx


# ------------------------------------------------------------- v4 Assembler (prompts_v4.py)
def v4_final(text: str, it: Dict, use_fix: bool = True, use_add: bool = True):
    """(final set, parsed or None). A malformed output falls back to E, as for <select>."""
    from prompts_v4 import final_v4, parse_v4
    p = parse_v4(text, len(it["crops"]))
    return (list(it["E"]), None) if p is None else (final_v4(it["crops"], p, use_fix, use_add), p)


def type_recall(S, gold, types: Optional[Sequence[str]], kind: str = "answer_bearing") -> Optional[float]:
    """Share of the gold boxes of one type that S hits (1:1 over ALL gold, IoU >= 0.5); None if the item has none."""
    if not types:
        return None
    js = [j for j, t in enumerate(types) if t == kind]
    if not js:
        return None
    hit = hungarian_hits(S, gold)
    return sum(j in hit for j in js) / len(js)
