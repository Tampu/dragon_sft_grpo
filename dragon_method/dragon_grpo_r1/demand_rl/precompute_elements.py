#!/usr/bin/env python3
"""CPU prepass: per-gold-element evidence types for any converted pool.

The branch trigger prefers answer_bearing > connector > text > legend > mark
(pilot spec section 3), so the trainer needs an evidence type per gold box of
every TRAINING prompt. diagnosis/common.gold_evidence_types() already derives
them from the reviewed annotations, but only indexes the Val split; this
generalizes the index to Train/Val/Test and dumps the result so the GPU trainer
never has to touch the annotation tree.

Optionally merges precomputed greedy-SFT hit labels (--sft-miss-json, a map
id -> [bool per gold box]) so the trigger can prefer elements SFT also missed.
Without them the trigger still works -- "missed by every main rollout" is the
trigger; SFT-miss is only a preference -- and falls back to TYPE_PRIORITY alone.

Usage (da_env2, CPU):
    python demand_rl/precompute_elements.py \
        --jsonl data_count/count_grpo_rest.jsonl --split Train \
        --out data_count/elements_grpo_rest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnosis"))

import common as C  # noqa: E402

SPLIT_ROOT = ROOT.parent / "split_reviewed-2"


def raw_index(split: str) -> Dict[Tuple[str, str], Dict]:
    """(image_path, q_id) -> reviewed annotation record, for one split.

    Same keying as common.val_raw_index (that pair is unique within a split),
    generalized off the hardcoded Val directory.
    """
    idx: Dict[Tuple[str, str], Dict] = {}
    base = SPLIT_ROOT / split
    for folder in C.DOMAINS:
        d = base / folder
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.json")):
            raw = json.loads(fp.read_text())
            idx[(str(raw.get("image_path")), str(raw.get("q_id")))] = raw
    return idx


def dom_of(sample: Dict) -> str:
    sid = str(sample.get("id", ""))
    return sid.split("::")[0] if "::" in sid else str(sample.get("image", "")).split("/")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--split", default="Train", choices=["Train", "Val", "Test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-miss-json", default=None,
                    help="optional {id: [bool per gold box]} greedy-SFT hit labels; "
                         "False = SFT missed that element")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N prompts (the pilot trains far fewer than the pool)")
    args = ap.parse_args()

    samples = C.load_jsonl(args.jsonl)
    if args.limit:
        samples = samples[: args.limit]
    idx = raw_index(args.split)
    print(f"[raw] {len(idx)} reviewed {args.split} annotations indexed")

    miss = {}
    if args.sft_miss_json:
        miss = json.loads(Path(args.sft_miss_json).read_text())
        print(f"[sft] greedy-SFT labels for {len(miss)} prompts")

    out, hist, unlinked = {}, Counter(), 0
    for s in samples:
        sid = str(s.get("id"))
        # id is "<dom>::<q_id>#i<k>" and q_id is itself "<image>.png-<n>", so the
        # q_id is everything between "::" and "#" -- NOT the trailing "-<n>".
        qid = sid.split("::", 1)[1].split("#")[0] if "::" in sid else None
        raw = idx.get((str(s.get("image")), qid)) if qid else None
        if raw is None:
            # fall back to matching on image alone when the q_id suffix does not
            # round-trip; better an image-level match than no evidence types
            cands = [r for (ip, _), r in idx.items() if ip == str(s.get("image"))]
            raw = cands[0] if len(cands) == 1 else None
        dom = dom_of(s)
        et = C.gold_evidence_types(s, raw, dom)
        hist.update(et)
        if raw is None:
            unlinked += 1
        rec = {"etypes": et, "n_gt": len(s["metadata"]["gt_boxes_norm"]), "dom": dom}
        if sid in miss:
            hits = miss[sid]
            rec["sft_missed"] = [not bool(h) for h in hits][: rec["n_gt"]]
        out[sid] = rec

    Path(args.out).write_text(json.dumps(out))
    print(f"[out] {len(out)} prompts -> {args.out}"
          f"  ({unlinked} with no reviewed annotation match)")
    tot = sum(hist.values())
    for k, v in hist.most_common():
        print(f"   {k:16s} {v:6d}  {v / tot:6.1%}")


if __name__ == "__main__":
    main()
