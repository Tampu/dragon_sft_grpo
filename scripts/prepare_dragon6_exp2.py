#!/usr/bin/env python3
"""
Build the second 6-dataset DRAGON SFT experiment: pure visual grounding (no
candidate region list in the input -- see dragon_grounding_convert_core's
build_reasoning_prompt/build_reasoning_target), scaled up from exp1's 1000
train / 200 val to 2800 train / 600 val.

Why the scale-up: exp1's ID-consistency filter (explanation_mentions_unknown_
ids) rejected samples whose predicted_explanation referenced a region id
that wasn't in that sample's own bbox list, capping ChartQA at ~199 usable
examples. The pure-grounding rewrite's substitute_ids_with_labels() now
resolves every id mention (to a label, or a generic "this region"/"this
text" descriptor) instead of rejecting the sample outright, so that ceiling
is gone -- rechecked availability: ai2d 1400/1400, ChartQA 1157/1157,
Circuit-VQA 1215/1226, Infographics 1196/1218, MapIQ 336/1122, Mapwise
1162/1162 (Train split). MapIQ is the only real bottleneck now, so it gets a
smaller share (300/100) while the other five get 500/100 each:
  5 x (500 train + 100 val) + 1 x (300 train + 100 val) = 2800 train / 600 val

At batch size 4 that's 700 steps/epoch; 20 epochs = 14,000 steps, about 3x
the ~4,680 steps where the historical precedent (README_AI2D_SFT_GRPO.md,
ai2d-only, no candidate list) first showed non-zero signal at 10 epochs.

Steps: identical pipeline to prepare_dragon6_exp1.py otherwise -- walk each
dataset's full split_reviewed-2/{Train,Val}/<folder> pool (shuffled, fixed
seed), keep converting candidates until N_train/N_val successful conversions
are collected, copy only the kept raw JSONs into a new experiment folder as
the audit trail, write the adapter-contract + converted InternVL JSONL, and
configs/dragon6_exp2_meta.json for train_dragon6_fast_v2b.py.
"""
import json
import random
import shutil
from pathlib import Path

from dragon_grounding_convert_core import convert_annotated_reasoning_sample

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "split_reviewed-2"
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
EXP_DIR = ROOT / "experiments" / "exp2_puregrounding_2800train_600val"
DRAGON_DIR = ROOT / "dragon_datasets"
SEED = 42

# key -> (split_reviewed-2 folder name, n_train, n_val)
DATASETS = {
    "ai2d_grounding": ("ai2d", 500, 100),
    "chartqa_grounding": ("ChartQA", 500, 100),
    "circuitvqa_grounding": ("Circuit-VQA", 500, 100),
    "infographics_grounding": ("Infographics", 500, 100),
    "mapiq_grounding": ("MapIQ", 300, 100),
    "mapwise_grounding": ("Mapwise", 500, 100),
}

ANSWER_KEYS = ["gt_answer", "ground_truth_answer"]  # ai2d handled separately via answers.correct


def clean_choices(raw):
    if not isinstance(raw, list) or not raw:
        return []
    cleaned = [str(x).strip() for x in raw if str(x).strip()]
    # Drop lists that leak stringified-dict fragments (upstream MapIQ bug):
    # entries containing "{" / "'" / ":" aren't plausible multiple-choice text.
    if any(any(ch in c for ch in "{}':") for c in cleaned):
        return []
    return cleaned


def has_boxes(raw: dict) -> bool:
    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        bbox = raw.get("boxes")
    return isinstance(bbox, list) and len(bbox) > 0


def to_adapter_record(raw: dict, q_id_fallback: str) -> dict:
    if "answers" in raw and isinstance(raw["answers"], dict):
        answer = raw["answers"].get("correct", "")
        explanation_raw = raw["answers"].get("predicted_explanation", "")
    else:
        answer = ""
        for key in ANSWER_KEYS:
            if raw.get(key):
                answer = raw[key]
                break
        explanation_raw = raw.get("predicted_explanation", "")

    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        bbox = raw.get("boxes", [])

    return {
        "id": raw.get("q_id", q_id_fallback),
        "question": raw.get("question_text", ""),
        "choices": clean_choices(raw.get("choices")),
        "answer": str(answer).strip(),
        "image_path": raw["image_path"],
        "bbox": bbox,
        "explanation_raw": explanation_raw,
    }


def process_split(key: str, folder_name: str, n: int, split: str, exp_split_dir: Path) -> tuple:
    """Walk the full shuffled pool, keeping the first n candidates that
    survive both box-presence and the pure-grounding target builder. Returns
    (adapter_samples_path, converted_internvl_path, written_count)."""
    src_folder = SPLIT_DIR / split / folder_name
    dst_folder = exp_split_dir / folder_name
    dst_folder.mkdir(parents=True, exist_ok=True)

    candidates = sorted(src_folder.glob("*.json"))
    rng = random.Random(SEED)
    rng.shuffle(candidates)

    adapter_lines = []
    converted_lines = []
    kept = 0
    for i, fp in enumerate(candidates):
        if kept >= n:
            break
        raw = json.loads(fp.read_text())
        if not has_boxes(raw):
            continue
        record = to_adapter_record(raw, q_id_fallback=f"{folder_name}-{split}-{i}")
        converted = convert_annotated_reasoning_sample(record, IMAGE_ROOT)
        if converted is None:
            continue
        shutil.copy2(fp, dst_folder / fp.name)
        adapter_lines.append(record)
        converted_lines.append(json.dumps(converted, ensure_ascii=False))
        kept += 1

    if kept < n:
        raise RuntimeError(
            f"[{key}/{split}] only found {kept}/{n} usable, box-annotated candidates "
            f"across the full {len(candidates)}-file pool. Lower this dataset's target count."
        )

    out_name = "samples.jsonl" if split == "Train" else "samples_infer_holdout.jsonl"
    adapter_path = DRAGON_DIR / key / out_name
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in adapter_lines) + "\n",
        encoding="utf-8",
    )

    converted_path = None
    if split == "Train":
        converted_path = DRAGON_DIR / key / f"{key}_internvl_grounding.jsonl"
        converted_path.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")

    return adapter_path, converted_path, kept


def main() -> None:
    meta = {}
    total_train = 0
    total_val = 0

    for key, (folder_name, n_train, n_val) in DATASETS.items():
        _, converted_path, kept_train = process_split(key, folder_name, n_train, "Train", EXP_DIR / "Train")
        _, _, kept_val = process_split(key, folder_name, n_val, "Val", EXP_DIR / "Val")
        total_train += kept_train
        total_val += kept_val
        print(f"[{key}] kept {kept_train}/{n_train} train + {kept_val}/{n_val} val")

        meta[key] = {
            "root": "Diagram_Attribution_Dataset",
            "annotation": str(converted_path.relative_to(ROOT)),
            "data_augment": True,
            "repeat_time": 1,
        }

    meta_path = ROOT / "configs" / "dragon6_exp2_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {meta_path} ({len(meta)} datasets, {total_train} train samples total)")
    print(f"Held out {total_val} val samples total for post-training testing "
          f"(dragon_datasets/<key>/samples_infer_holdout.jsonl)")
    print(f"Raw JSONs used for this experiment copied to {EXP_DIR}")


if __name__ == "__main__":
    main()
