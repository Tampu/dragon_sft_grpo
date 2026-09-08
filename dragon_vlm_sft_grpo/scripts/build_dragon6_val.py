#!/usr/bin/env python3
"""
Combine the six per-domain Val JSONLs (build_dragon6_raw.py's output) into
one round-robin DOMAIN-INTERLEAVED file, so that taking any prefix
`val_samples[:val_size]` during training-time validation is automatically
domain-balanced.

This fixes, by construction, a bug documented in the InternVL pipeline this
repo's design is informed by (dragon_sft_grpo/README.md section 4.3): its
holdout file was grouped by domain (ai2d first), so `run_val`'s default
`--val-size 100` silently scored ai2d only, and the same slice-before-
shuffle pattern hit `--max-prompts N` on the GRPO pool. Interleaving here
instead of shuffling means the ordering is also deterministic and
inspectable (domain 0,1,2,3,4,5,0,1,2,... rather than a shuffled blob).

No cap, no quota: every uniformly-filtered Val record is kept and written
out, just reordered.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_VAL_DIR = ROOT / "dragon_datasets" / "raw_by_split" / "Val"
OUT_PATH = ROOT / "dragon_datasets" / "raw_val_interleaved.jsonl"

DOMAIN_KEYS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]


def main() -> None:
    per_domain = {}
    for key in DOMAIN_KEYS:
        path = RAW_VAL_DIR / f"{key}.jsonl"
        per_domain[key] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"  [{key}] {len(per_domain[key])} val records")

    interleaved = []
    idx = [0] * len(DOMAIN_KEYS)
    while True:
        progressed = False
        for i, key in enumerate(DOMAIN_KEYS):
            if idx[i] < len(per_domain[key]):
                interleaved.append(per_domain[key][idx[i]])
                idx[i] += 1
                progressed = True
        if not progressed:
            break

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in interleaved) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(interleaved)} domain-interleaved val records -> {OUT_PATH}")
    print("Any prefix of this file is domain-balanced -- safe to slice with --val-size.")


if __name__ == "__main__":
    main()
