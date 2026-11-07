#!/usr/bin/env python3
"""
InternVL3.5 fork data split: 2000 SFT + ALL remaining Train records for GRPO.
Val is never trained on -- it is only the GRPO monitoring set. Test is untouched.

Source is the UNFILTERED raw pool in data_i35/raw_by_split/{Train,Val}, built by
dragon_vlm_sft_grpo's build_dragon6_raw.py. Its only drops are records with no
bbox annotation or a missing image file. Unlike the InternVL3 run, there is NO
>15-box, full-image-box or contradictory-duplicate filtering.

SFT allocation is an equal quota per domain (2000 // 6, remainder to the first
domains alphabetically) after a seed-42 shuffle -- the same allocate_quotas
logic both pipelines use. GRPO is everything SFT did not take, so the two pools
are disjoint by construction.

Writes, under data_i35/:
  raw_sft2000.jsonl           2000 records
  raw_grpo_rest.jsonl         all remaining Train records
  raw_val_interleaved.jsonl   Val, domain round-robin. grpo_dragon6_v1.py takes
                              the FIRST --val-size lines, so an unordered file
                              would validate on ai2d only.

    python3 scripts/build_i35_split.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_i35"
SRC = DATA / "raw_by_split"
SEED = 42
TOTAL_SFT = 2000
DOMAINS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]


def load(split: str, dom: str) -> list:
    path = SRC / split / f"{dom}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def allocate_quotas(pool_sizes: dict, total: int) -> dict:
    keys = list(pool_sizes)
    base, rem = divmod(total, len(keys))
    quota = {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}
    shortfall = 0
    for k in keys:
        if quota[k] > pool_sizes[k]:
            shortfall += quota[k] - pool_sizes[k]
            quota[k] = pool_sizes[k]
    while shortfall > 0:
        capacity = {k: pool_sizes[k] - quota[k] for k in keys if pool_sizes[k] > quota[k]}
        if not capacity:
            break
        share = max(1, shortfall // len(capacity))
        for k in capacity:
            add = min(share, capacity[k], shortfall)
            quota[k] += add
            shortfall -= add
            if shortfall <= 0:
                break
    return quota


def write(name: str, records: list) -> None:
    path = DATA / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(records)})")


def main() -> None:
    rng = random.Random(SEED)
    pools = {}
    for dom in DOMAINS:
        recs = load("Train", dom)
        rng.shuffle(recs)
        pools[dom] = recs
    sizes = {d: len(v) for d, v in pools.items()}
    quota = allocate_quotas(sizes, TOTAL_SFT)

    sft, grpo = [], []
    print(f"{'domain':<14}{'train':>7}{'sft':>6}{'grpo':>6}")
    for dom in DOMAINS:
        sft.extend(pools[dom][:quota[dom]])
        grpo.extend(pools[dom][quota[dom]:])
        print(f"{dom:<14}{sizes[dom]:>7}{quota[dom]:>6}{sizes[dom] - quota[dom]:>6}")
    print(f"{'total':<14}{sum(sizes.values()):>7}{len(sft):>6}{len(grpo):>6}")

    ids_sft = {r["id"] for r in sft}
    ids_grpo = {r["id"] for r in grpo}
    assert not ids_sft & ids_grpo, "SFT and GRPO pools overlap"
    assert len(sft) + len(grpo) == sum(sizes.values()), "records lost in the split"

    val_by_dom = {dom: load("Val", dom) for dom in DOMAINS}
    val = []
    for i in range(max(len(v) for v in val_by_dom.values())):
        for dom in DOMAINS:
            if i < len(val_by_dom[dom]):
                val.append(val_by_dom[dom][i])

    write("raw_sft2000.jsonl", sft)
    write("raw_grpo_rest.jsonl", grpo)
    write("raw_val_interleaved.jsonl", val)


if __name__ == "__main__":
    main()
