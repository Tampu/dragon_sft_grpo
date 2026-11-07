#!/usr/bin/env python3
"""
build_test2445.py -- prepare the OFFICIAL DRAGON test split (split_reviewed-2/Test,
2,445 instances) for evaluation, so a row can go in Table 3.

Deliberately DIFFERENT from the training-pool builders: this applies **no
filters at all**. build_dragon6_sft2k.py drops >15-box targets, full-image
boxes and contradictory duplicates because those hurt *training*; applying
them here would silently evaluate on an easier subset than the published
baselines and make the comparison invalid. The only records dropped are the
12 with no bbox annotation at all (6 Circuit-VQA, 6 MapIQ), which cannot be
scored by any metric; that count is reported so the denominator is explicit.

Emits, per domain, a v4-native-syntax JSONL ready for
export_for_eval_script.py, plus a `contaminated_ids.txt` listing the test
questions that also appear in our SFT/GRPO training pools -- an upstream
defect ((Train∪Val)∩Test = 19 questions in the released split, not something
this pipeline introduced) -- so a contamination-free variant can be scored
alongside the full-set number.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SP = ROOT / "split_reviewed-2" / "Test"
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
OUT = ROOT / "dragon_datasets_test2445"

# split_reviewed-2 folder -> our short domain name (matches preds_for_eval naming)
DOMAINS = {"ai2d": "ai2d", "ChartQA": "chartqa", "Circuit-VQA": "circuitvqa",
           "Infographics": "infographics", "MapIQ": "mapiq", "Mapwise": "mapwise"}
ANSWER_KEYS = ["gt_answer", "ground_truth_answer"]


def clean_choices(raw):
    if not isinstance(raw, list) or not raw:
        return []
    c = [str(x).strip() for x in raw if str(x).strip()]
    if any(any(ch in x for ch in "{}':") for x in c):
        return []          # malformed stringified-dict fragments (MapIQ upstream bug)
    return c


def to_adapter_record(raw, fallback):
    if isinstance(raw.get("answers"), dict):
        answer = raw["answers"].get("correct", "")
        expl = raw["answers"].get("predicted_explanation", "")
    else:
        answer = next((raw[k] for k in ANSWER_KEYS if raw.get(k)), "")
        expl = raw.get("predicted_explanation", "")
    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        bbox = raw.get("boxes", [])
    return {"id": raw.get("q_id", fallback), "question": raw.get("question_text", ""),
            "choices": clean_choices(raw.get("choices")), "answer": str(answer).strip(),
            "image_path": raw["image_path"], "bbox": bbox, "explanation_raw": expl}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # training-pool keys, to mark upstream-inherited contamination
    trainkeys = set()
    for p in ("dragon_datasets_sft2k/raw_sft2k_records.jsonl",
              "dragon_datasets_grpo/raw_grpo5k_records.jsonl"):
        for l in (ROOT / p).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                trainkeys.add(((r.get("image_path") or r.get("image")), str(r.get("id"))))

    all_raw, contaminated, n_nobox = [], [], 0
    for folder, dom in DOMAINS.items():
        recs = []
        for i, fp in enumerate(sorted((SP / folder).glob("*.json"))):
            raw = json.loads(fp.read_text())
            bb = raw.get("bbox") or raw.get("boxes") or []
            if not (isinstance(bb, list) and bb):
                n_nobox += 1
                continue
            rec = to_adapter_record(raw, f"{folder}-Test-{i}")
            key = (rec["image_path"], str(rec["id"]))
            rec["id"] = f"{dom}::{rec['id']}#i{len(recs)}"     # unique + traceable
            if key in trainkeys:
                contaminated.append(rec["id"])
            recs.append(rec)
        (OUT / f"raw_{dom}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
        all_raw += recs
        print(f"  [{dom}] {len(recs)} usable")

    (OUT / "contaminated_ids.txt").write_text("\n".join(contaminated) + "\n")
    print(f"\ntotal usable: {len(all_raw)} (dropped {n_nobox} with no bbox annotation)")
    print(f"test questions also present in SFT/GRPO training pools: {len(contaminated)}"
          f"  -> {OUT/'contaminated_ids.txt'}")
    print(f"\nNext, per domain:\n"
          f"  python3 scripts/sft_v4_phase0.py convert --in {OUT}/raw_<dom>.jsonl \\\n"
          f"    --image-root Diagram_Attribution_Dataset --out {OUT}/<dom>.jsonl \\\n"
          f"    --stage A --allow-contradictions")


if __name__ == "__main__":
    main()
