#!/usr/bin/env python3
"""
Merge the two half-runs of test_norepeat_fix.py (GPU 4 + GPU 5) into one
summary, and add the sub-diagnosis the four built-in verdicts can't express.

Reproduces the test script's own tables exactly (same counts, same DEGRADED
rule: soft-recall drop >0.15 vs that sample's own ngram=0 baseline, computed
within the run that produced it -- safe to merge because each sample's whole
ngram sweep always ran inside a single process).

Adds one breakdown the built-in verdicts fold together: an EMPTY generation
can mean two very different things.
  - EMPTY/no_tag    : no <box> at all -- genuine non-answer.
  - EMPTY/unclosed  : opened <box>, emitted coordinates, but never closed it
                      (frequently with whitespace padded INSIDE the numbers,
                      e.g. "[848, 160, 948,   175]" -- the model evading the
                      n-gram block by inserting spaces rather than repeating
                      a blocked token sequence). This is an evasion mode, not
                      a clean stop, and it is invisible in the CLEAN/JITTER/
                      LOOP/EMPTY table.
The distinction decides the go/no-go: an ngram value that converts LOOP into
EMPTY/unclosed has not fixed anything -- it reshaped the degeneracy, which is
the same conclusion the DECISION GUIDE draws from JITTER dominance.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/data2/traviku2")
# every run dir under norepeat_test/ that actually holds generations, so this
# works for a 2-GPU split or a single-GPU run without editing paths. Excludes
# partial_backup/ (kept only as the record of the crashed first attempt).
RUN_DIRS = sorted(p.parent for p in
                  (ROOT / "dragon_grpo/norepeat_test").glob("*/generations.jsonl")
                  if p.parent.name != "partial_backup")

VERDICTS = ["CLEAN", "JITTER", "LOOP", "EMPTY"]
# whitespace padded inside a coordinate list: two+ spaces, or a space before a comma
WS_EVASION_RE = re.compile(r"\d\s{2,}|\s+,")
# letters inside the box payload = a text label leaked into a coordinate slot
BOX_PAYLOAD_RE = re.compile(r"<box>(.*?)(?:</box>|$)", re.S)
MALFORMED_RE = re.compile(r"[A-Za-z]")


def has_malformed_payload(resp: str) -> bool:
    m = BOX_PAYLOAD_RE.search(resp)
    return bool(m and MALFORMED_RE.search(m.group(1)))


def empty_subtype(resp: str) -> str:
    if "<box>" not in resp:
        return "no_tag"
    if "</box>" not in resp:
        return "unclosed"
    return "other"


def main() -> None:
    rows = []
    for d in RUN_DIRS:
        p = d / "generations.jsonl"
        if not p.exists():
            print(f"[warn] missing {p} -- run not finished?")
            continue
        rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    if not rows:
        raise SystemExit("no generations found")

    ngrams = sorted({r["ngram"] for r in rows})
    counts = defaultdict(int)
    for r in rows:
        key = ("good" if r["set"] == "good" else r["arm"], r["ngram"], r["verdict"])
        counts[key] += 1

    print(f"merged {len(rows)} generations from {len([d for d in RUN_DIRS if (d/'generations.jsonl').exists()])} runs")
    n_bad = len({r["id"] for r in rows if r["set"] == "bad"})
    n_good = len({r["id"] for r in rows if r["set"] == "good"})
    print(f"bad set: {n_bad} samples   good set: {n_good} samples\n")

    print("=" * 72)
    print("SUMMARY  (bad set: want LOOP -> CLEAN, fear LOOP -> JITTER/unclosed)")
    print("=" * 72)
    for arm in ("greedy", "sampled"):
        if not any(r["arm"] == arm and r["set"] == "bad" for r in rows):
            continue
        print(f"\n[bad set, {arm}]")
        print(f"  {'ngram':>6} " + " ".join(f"{v:>7}" for v in VERDICTS)
              + f" | {'E:no_tag':>9} {'E:unclosed':>11} {'ws_evade':>9} {'malformed':>10}")
        for ng in ngrams:
            row = " ".join(f"{counts[(arm, ng, v)]:>7}" for v in VERDICTS)
            sub = defaultdict(int)
            ws = mal = 0
            for r in rows:
                if r["set"] == "bad" and r["arm"] == arm and r["ngram"] == ng:
                    if r["verdict"] == "EMPTY":
                        sub[empty_subtype(r["response"])] += 1
                    if WS_EVASION_RE.search(r["response"]):
                        ws += 1
                    if has_malformed_payload(r["response"]):
                        mal += 1
            print(f"  {ng:>6} {row} | {sub['no_tag']:>9} {sub['unclosed']:>11} "
                  f"{ws:>9} {mal:>10}")

    # DEGRADED on the good set, recomputed from the merged rows
    base = {}
    for r in rows:
        if r["set"] == "good" and r["ngram"] == 0:
            base[r["id"]] = r["soft_recall"]
    degraded = defaultdict(int)
    for r in rows:
        if r["set"] == "good" and r["ngram"] != 0:
            if base.get(r["id"], 0.0) - r["soft_recall"] > 0.15:
                degraded[r["ngram"]] += 1

    print(f"\n[good set, greedy]  DEGRADED = soft_recall drop >0.15 vs ngram=0")
    print(f"  {'ngram':>6} " + " ".join(f"{v:>7}" for v in VERDICTS)
          + f" {'DEGRADED':>9} {'mean_sr':>8}")
    for ng in ngrams:
        row = " ".join(f"{counts[('good', ng, v)]:>7}" for v in VERDICTS)
        srs = [r["soft_recall"] for r in rows if r["set"] == "good" and r["ngram"] == ng]
        mean_sr = sum(srs) / len(srs) if srs else 0.0
        print(f"  {ng:>6} {row} {degraded[ng]:>9} {mean_sr:>8.3f}")

    # headline: per-ngram CLEAN rate on the bad set, both arms pooled
    print(f"\n{'='*72}\nBAD-SET CLEAN RATE (the number the decision turns on)\n{'='*72}")
    print(f"  {'ngram':>6} {'greedy':>18} {'sampled':>18}")
    for ng in ngrams:
        cells = []
        for arm in ("greedy", "sampled"):
            tot = sum(counts[(arm, ng, v)] for v in VERDICTS)
            c = counts[(arm, ng, "CLEAN")]
            cells.append(f"{c}/{tot} ({c/tot*100:.0f}%)" if tot else "n/a")
        print(f"  {ng:>6} {cells[0]:>18} {cells[1]:>18}")

    out = ROOT / "dragon_grpo/norepeat_test/merged_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_generations": len(rows), "n_bad": n_bad, "n_good": n_good,
        "ngrams": ngrams,
        "counts": {f"{a}|{n}|{v}": c for (a, n, v), c in counts.items()},
        "degraded": dict(degraded),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nMerged summary -> {out}")


if __name__ == "__main__":
    main()
