#!/usr/bin/env python3
"""
build_split_manifest.py -- exact file-level manifest of split_reviewed-2.

Emits a row per raw annotation file recording which split and domain it
belongs to, the image it points at, its question id, whether it carries a
bbox annotation, and an md5 so a recipient can verify they hold the same
data. This is the artifact that makes "Train 7,285 / Val 2,434 / Test 2,445"
checkable rather than asserted.

Also records the known upstream defect: the released splits are NOT disjoint.
Questions keyed by (image_path, q_id) that appear in more than one split are
flagged in the summary, so a reproduction can exclude them or report both
contaminated and clean numbers.

Outputs (to dragon_datasets_manifest/):
  split_manifest.csv          one row per file, sorted by split/domain/file
  split_manifest_summary.json counts per split x domain + cross-split dupes
"""
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "split_reviewed-2"
OUT = ROOT / "dragon_datasets_manifest"
SPLITS = ["Train", "Val", "Test"]
DOMAINS = ["ai2d", "ChartQA", "Circuit-VQA", "Infographics", "MapIQ", "Mapwise"]


def has_bbox(raw):
    bb = raw.get("bbox")
    if not isinstance(bb, list) or not bb:
        bb = raw.get("boxes")
    return isinstance(bb, list) and len(bb) > 0, (len(bb) if isinstance(bb, list) else 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, by_key = [], defaultdict(list)

    for split in SPLITS:
        for dom in DOMAINS:
            d = SP / split / dom
            if not d.is_dir():
                continue
            for fp in sorted(d.glob("*.json")):
                blob = fp.read_bytes()
                try:
                    raw = json.loads(blob)
                except Exception:
                    rows.append(dict(split=split, domain=dom, file=fp.name,
                                     image_path="", q_id="", has_bbox="PARSE_ERROR",
                                     n_boxes=0, md5=hashlib.md5(blob).hexdigest()))
                    continue
                hb, nb = has_bbox(raw)
                img, qid = raw.get("image_path", ""), str(raw.get("q_id", ""))
                rows.append(dict(split=split, domain=dom, file=fp.name,
                                 image_path=img, q_id=qid,
                                 has_bbox=int(hb), n_boxes=nb,
                                 md5=hashlib.md5(blob).hexdigest()))
                if img:
                    by_key[(img, qid)].append(split)

    csv_path = OUT / "split_manifest.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "domain", "file", "image_path",
                                          "q_id", "has_bbox", "n_boxes", "md5"])
        w.writeheader()
        w.writerows(rows)

    # counts
    per = Counter((r["split"], r["domain"]) for r in rows)
    per_split = Counter(r["split"] for r in rows)
    boxed = Counter(r["split"] for r in rows if r["has_bbox"] == 1)
    imgs = defaultdict(set)
    for r in rows:
        if r["image_path"]:
            imgs[r["split"]].add(r["image_path"])

    # the upstream defect: same (image,q_id) in >1 split
    dupes = {f"{k[0]}|{k[1]}": sorted(set(v)) for k, v in by_key.items() if len(set(v)) > 1}

    summary = {
        "root": str(SP),
        "files_per_split": dict(per_split),
        "files_per_split_domain": {f"{s}/{d}": n for (s, d), n in sorted(per.items())},
        "with_bbox_per_split": dict(boxed),
        "unique_images_per_split": {k: len(v) for k, v in imgs.items()},
        "image_overlap_between_splits": {
            "Train&Val": len(imgs["Train"] & imgs["Val"]),
            "Train&Test": len(imgs["Train"] & imgs["Test"]),
            "Val&Test": len(imgs["Val"] & imgs["Test"]),
        },
        "cross_split_duplicate_questions": {
            "count": len(dupes),
            "note": "same (image_path,q_id) present in more than one split -- "
                    "a defect in the released data, not introduced by this pipeline",
            "items": dupes,
        },
    }
    (OUT / "split_manifest_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {csv_path}  ({len(rows)} rows)")
    print(f"wrote {OUT/'split_manifest_summary.json'}\n")
    print(f"  {'split':<7}{'files':>8}{'with bbox':>11}{'uniq images':>13}")
    for s in SPLITS:
        print(f"  {s:<7}{per_split[s]:>8}{boxed[s]:>11}{len(imgs[s]):>13}")
    print(f"  {'TOTAL':<7}{sum(per_split.values()):>8}{sum(boxed.values()):>11}")
    print(f"\n  per split x domain:")
    for s in SPLITS:
        print("   ", s, {d: per[(s, d)] for d in DOMAINS})
    print(f"\n  cross-split duplicate questions: {len(dupes)}")
    print(f"  image overlap: {summary['image_overlap_between_splits']}")


if __name__ == "__main__":
    main()
