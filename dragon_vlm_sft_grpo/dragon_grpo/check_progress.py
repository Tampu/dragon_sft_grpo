#!/usr/bin/env python3
"""Tail a running GRPO job's grpo_metrics.jsonl and print the latest train
and val rows. Usage: python3 check_progress.py --out-dir outputs/grpo"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tail", type=int, default=10)
    args = ap.parse_args()

    path = Path(args.out_dir) / "grpo_metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"no metrics file at {path} yet")

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    train_rows = [r for r in rows if "val" not in r]
    val_rows = [r["val"] for r in rows if "val" in r]

    print(f"=== last {args.tail} train steps ===")
    for r in train_rows[-args.tail:]:
        print(" ".join(f"{k}={v}" for k, v in r.items()))

    if val_rows:
        print(f"\n=== last {args.tail} val checkpoints ===")
        for r in val_rows[-args.tail:]:
            print(" ".join(f"{k}={v}" for k, v in r.items()))

    print(f"\ntotal steps logged: {len(train_rows)}")


if __name__ == "__main__":
    main()
