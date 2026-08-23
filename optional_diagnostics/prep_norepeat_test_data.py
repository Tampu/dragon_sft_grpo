#!/usr/bin/env python3
"""
Data prep for test_norepeat_fix.py -- builds a pool jsonl with UNIQUE ids
plus the --ids-file (bad) and --good-ids-file (good) id lists.

Why the pool has to be rebuilt: test_norepeat_fix.py indexes its sample pool
as `samples = {s.get("id"): s for s in ...}` and resolves --ids-file entries
through that dict. That is correct for data with unique ids, but this repo's
holdout files reuse per-image local q_ids -- verified: infographics has 100
samples sharing only 2 distinct ids ("...::Q_0", "...::Q_1"), mapiq likewise.
Feeding those files in directly would collapse the pool to 2 samples per
domain and silently match the wrong records. So each record here gets
id = "<orig_id>#i<positional_index>", which is unique by construction and
still traces back to the source file and line.

Bad set  = every zero-box prediction from GRPO-step250 on infographics+mapiq,
           read positionally from dragon_grpo/empty_classification.json
           (all 76 were classified as truncation loops).
Good set = previously-correct, multi-box samples drawn across all 6 domains
           (n_gt >= 3, non-empty prediction, soft-recall >= 0.5 vs its own
           gt), so the regression check covers legitimate dense coordinate
           output, which is exactly what an n-gram filter might corrupt.

Matching between pred_<domain>.json and holdout_per_domain/<domain>.jsonl is
POSITIONAL: export_for_eval_script.py iterates samples in file order and
appends predictions in the same order, and ids are not unique so they cannot
be used as keys.
"""
import json
from pathlib import Path
from typing import List, Tuple

ROOT = Path("/mnt/data2/traviku2")
HOLDOUT_DIR = ROOT / "dragon_datasets_sft2k/holdout_per_domain"
PRED_DIR = ROOT / "preds_for_eval/grpo_step250"
EMPTY_CLS = ROOT / "dragon_grpo/empty_classification.json"
OUT_DIR = ROOT / "dragon_grpo/norepeat_test_data"

DOMAINS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]

Box = Tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def soft_recall(gt: List[Box], pred: List[Box]) -> float:
    if not gt or not pred:
        return 0.0
    return sum(max(box_iou(g, p) for p in pred) for g in gt) / len(gt)


def uid(orig_id: str, idx: int) -> str:
    return f"{orig_id}#i{idx}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = []            # every record, with a unique id
    bad_ids: List[str] = []
    good_cands = []      # (soft_recall, n_gt, uid)

    empty_cls = json.loads(EMPTY_CLS.read_text())
    bad_idx_by_domain = {d: {r["idx"] for r in empty_cls.get(d, [])}
                         for d in ("infographics", "mapiq")}

    for domain in DOMAINS:
        samples = [json.loads(l) for l in
                   (HOLDOUT_DIR / f"{domain}.jsonl").read_text().splitlines() if l.strip()]
        preds = json.loads((PRED_DIR / f"pred_{domain}.json").read_text())
        assert len(samples) == len(preds), \
            f"{domain}: {len(samples)} samples vs {len(preds)} preds -- positional match broken"

        for i, (s, p) in enumerate(zip(samples, preds)):
            rec = dict(s)
            rec["id"] = uid(s.get("id"), i)
            rec["_source"] = {"domain": domain, "line": i, "orig_id": s.get("id")}
            pool.append(rec)

            if i in bad_idx_by_domain.get(domain, ()):
                bad_ids.append(rec["id"])
                continue

            gt = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
            pred_boxes = [(b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
                          for b in p["pred_boxes_parsed"]]
            if len(gt) >= 3 and pred_boxes:
                sr = soft_recall(gt, pred_boxes)
                if sr >= 0.5:
                    good_cands.append((sr, len(gt), rec["id"], domain))

    # good set: spread across domains (round-robin by domain, best-first within)
    good_cands.sort(key=lambda t: -t[0])
    by_domain = {}
    for sr, n_gt, gid, domain in good_cands:
        by_domain.setdefault(domain, []).append((sr, n_gt, gid))
    good_ids: List[str] = []
    round_i = 0
    while len(good_ids) < 15:
        added = False
        for domain in DOMAINS:
            lst = by_domain.get(domain, [])
            if round_i < len(lst) and len(good_ids) < 15:
                good_ids.append(lst[round_i][2])
                added = True
        if not added:
            break
        round_i += 1

    pool_path = OUT_DIR / "pool.jsonl"
    pool_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pool) + "\n")
    (OUT_DIR / "bad_ids.txt").write_text("\n".join(bad_ids) + "\n")
    (OUT_DIR / "good_ids.txt").write_text("\n".join(good_ids) + "\n")

    ids = [r["id"] for r in pool]
    assert len(ids) == len(set(ids)), "pool ids are not unique -- the whole point of this script"

    print(f"pool:      {len(pool)} records ({len(set(ids))} unique ids) -> {pool_path}")
    print(f"bad set:   {len(bad_ids)} ids (all previously-truncated) -> {OUT_DIR/'bad_ids.txt'}")
    print(f"good set:  {len(good_ids)} ids -> {OUT_DIR/'good_ids.txt'}")

    dom_count = {}
    for gid in good_ids:
        d = next(r["_source"]["domain"] for r in pool if r["id"] == gid)
        dom_count[d] = dom_count.get(d, 0) + 1
    print(f"good set domain spread: {dom_count}")

    # split bad + good across two GPUs (interleaved so each half is a fair
    # mix of domains and box counts, not one domain per GPU)
    for half in (0, 1):
        (OUT_DIR / f"bad_ids_gpu{half}.txt").write_text(
            "\n".join(bad_ids[half::2]) + "\n")
        (OUT_DIR / f"good_ids_gpu{half}.txt").write_text(
            "\n".join(good_ids[half::2]) + "\n")
        print(f"  gpu{half}: {len(bad_ids[half::2])} bad, {len(good_ids[half::2])} good")


if __name__ == "__main__":
    main()
