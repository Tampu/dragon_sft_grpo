#!/usr/bin/env python3
"""
Split the uniformly-filtered Train pool (build_dragon6_raw.py's output) into
a domain-balanced, mutually DISJOINT SFT pool and GRPO pool.

Disjointness is structural: each domain's shuffled Train records are
quota-allocated to SFT first, then GRPO draws its quota from what's LEFT
over. No re-walking + diffing against already-built files is needed because
build_dragon6_raw.py never caps the raw pool -- there is no "was this
already spent on a smaller cap" subtlety to reproduce.

Default sizes (2000 SFT / 5000 GRPO) are a starting point sized to be
comparable across different model_configs run through this same pipeline,
not tied to any one model. Override --total-sft/--total-grpo for a
different scale; both pools stay disjoint and domain-balanced regardless.

Domain-balance quota logic: equal target per domain with remainder assigned
to the first domains alphabetically, then any shortfall (a domain whose
pool can't fill its quota -- MapIQ typically, since its Train split has far
fewer box-annotated files than the other five domains) redistributed
round-robin across domains with remaining surplus.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN_DIR = ROOT / "dragon_datasets" / "raw_by_split" / "Train"
OUT_DIR = ROOT / "dragon_datasets"

SEED = 42
DOMAIN_KEYS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]


def allocate_quotas(pool_sizes: dict, total: int) -> dict:
    keys = list(pool_sizes.keys())
    n = len(keys)
    base = total // n
    remainder = total - base * n
    quota = {k: base + (1 if i < remainder else 0) for i, k in enumerate(keys)}

    shortfall = 0
    for k in keys:
        if quota[k] > pool_sizes[k]:
            shortfall += quota[k] - pool_sizes[k]
            quota[k] = pool_sizes[k]

    while shortfall > 0:
        capacity = {k: pool_sizes[k] - quota[k] for k in keys if pool_sizes[k] - quota[k] > 0}
        if not capacity:
            break
        share = max(1, shortfall // len(capacity))
        progressed = False
        for k in capacity:
            if shortfall <= 0:
                break
            add = min(share, capacity[k], shortfall)
            if add > 0:
                quota[k] += add
                shortfall -= add
                progressed = True
        if not progressed:
            break
    return quota


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-sft", type=int, default=2000)
    ap.add_argument("--total-grpo", type=int, default=5000)
    args = ap.parse_args()

    rng = random.Random(SEED)

    pools = {}
    for key in DOMAIN_KEYS:
        path = RAW_TRAIN_DIR / f"{key}.jsonl"
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rng.shuffle(records)
        pools[key] = records

    pool_sizes = {k: len(v) for k, v in pools.items()}
    print("Train pool sizes (uniform-filtered, no per-domain cap):")
    for k in DOMAIN_KEYS:
        print(f"  {k}: {pool_sizes[k]}")
    print(f"  total: {sum(pool_sizes.values())}")

    sft_quota = allocate_quotas(pool_sizes, args.total_sft)
    sft_records, remainder = [], {}
    print(f"\nSFT allocation (target {args.total_sft}):")
    for k in DOMAIN_KEYS:
        n = sft_quota[k]
        sft_records.extend(pools[k][:n])
        remainder[k] = pools[k][n:]
        print(f"  {k}: {n} / {pool_sizes[k]}  (remaining for GRPO: {len(remainder[k])})")
    print(f"  SFT total: {len(sft_records)} (target {args.total_sft})")

    remainder_sizes = {k: len(v) for k, v in remainder.items()}
    grpo_quota = allocate_quotas(remainder_sizes, args.total_grpo)
    grpo_records = []
    print(f"\nGRPO allocation (target {args.total_grpo}, drawn only from what SFT left behind):")
    for k in DOMAIN_KEYS:
        n = grpo_quota[k]
        grpo_records.extend(remainder[k][:n])
        print(f"  {k}: {n} / {remainder_sizes[k]}")
    print(f"  GRPO total: {len(grpo_records)} (target {args.total_grpo})")
    if len(grpo_records) < args.total_grpo:
        print(f"  [note] {args.total_grpo - len(grpo_records)} short of target -- "
              f"combined SFT+GRPO capacity across all 6 domains is the binding "
              f"constraint (MapIQ is typically the tightest domain).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_sft_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sft_records) + "\n", encoding="utf-8")
    (OUT_DIR / "raw_grpo_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in grpo_records) + "\n", encoding="utf-8")

    sft_keys = {(r["image_path"], r["id"]) for r in sft_records}
    grpo_keys = {(r["image_path"], r["id"]) for r in grpo_records}
    overlap = sft_keys & grpo_keys
    print(f"\nSFT ∩ GRPO overlap: {len(overlap)} (must be 0 -- disjoint by construction)")
    assert not overlap, "SFT/GRPO pools are not disjoint -- quota allocation bug"

    print(f"\nWrote {OUT_DIR / 'raw_sft_records.jsonl'} ({len(sft_records)})")
    print(f"Wrote {OUT_DIR / 'raw_grpo_records.jsonl'} ({len(grpo_records)})")


if __name__ == "__main__":
    main()
