#!/usr/bin/env python3
"""Paper data (method frozen, Nov 6 2026): the i35 data with the approved Grounder prompt, plus the full
2,445-instance Test set. CPU only.

Prompt change (approved): the system turn loses the sentence "List boxes top-to-bottom, then left-to-right."
Nothing else changes: same items, same user turns (question, choices, "Correct Answer:"), same SFT targets
(gold boxes sorted by centre, top-to-bottom then left-to-right, as the converter always wrote them).

Writes to dragon_sft_grpo_internvl35_old/data_paper/:
  sft2000.jsonl      2,000 SFT items        (from data_i35/i35_sft2000_v4.jsonl)
  grpo_pool.jsonl    5,266 GRPO items       (from data_i35/i35_grpo_rest_v4.jsonl; also the Assembler's train states)
  val600.jsonl       600 Val items          (from data_i35/val_by_domain/*.jsonl; sanity checks only)
  test2445.jsonl     2,445 Test items       = the 2,432 converted ones (dragon_datasets_test2445/<dom>.jsonl)
                     + the 13 the converter dropped: 12 with no bbox annotation (6 Circuit-VQA, 6 MapIQ) and
                     1 ChartQA item whose boxes all have zero height. Their gold is [] -- the official
                     eval_script counts them in the denominator with 0 for every method.
  test2432_ids.txt   ids of the scorable 2,432 (appendix number)
  sft2000_meta.json  meta file for the InternVL SFT trainer

    /mnt/data2/traviku2/da_env2/bin/python dragon_grpo_r1/paper/build_data.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

I35 = Path("/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old")
OUT = I35 / "data_paper"
OLD = " List boxes top-to-bottom, then left-to-right."
DOMS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]
OFFICIAL = {"ai2d": 483, "chartqa": 389, "circuitvqa": 404, "infographics": 411, "mapiq": 369, "mapwise": 389}


def fix(rec):
    s = rec["conversations"][0]
    assert s["from"] == "system" and s["value"].count(OLD) == 1, rec["id"]
    s["value"] = s["value"].replace(OLD, "")
    return rec


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write(name, recs):
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
    with open(OUT / name, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"  {name}: {len(recs)}  {dict(Counter(r['id'].split('::')[0] for r in recs))}")


def dropped_test_records(system):
    """The 13 official Test instances the old pipeline left out, built with the converter's own functions."""
    sys.path.insert(0, str(I35 / "scripts"))
    sys.path.insert(0, str(I35 / "dragon_grpo"))
    from PIL import Image
    from sft_v3_patches import build_reasoning_prompt_v3, normalize_and_sort_gt_boxes
    import build_test2445 as BT
    have = {r["id"] for d in DOMS for r in load(I35 / f"dragon_datasets_test2445/{d}.jsonl")}
    out = []
    for folder, dom in BT.DOMAINS.items():
        n_usable = 0                                       # build_test2445's own id rule: #i<index among usable>
        for i, fp in enumerate(sorted((BT.SP / folder).glob("*.json"))):
            raw = json.loads(fp.read_text())
            bb = raw.get("bbox") or raw.get("boxes") or []
            rec = BT.to_adapter_record(raw, f"{folder}-Test-{i}")
            if isinstance(bb, list) and bb:
                new_id = f"{dom}::{rec['id']}#i{n_usable}"
                n_usable += 1
                if new_id in have:
                    continue                               # converted: already in the 2,432
            else:
                new_id = f"{dom}::{rec['id']}#nobox{i}"
            with Image.open(BT.IMAGE_ROOT / rec["image_path"]) as im:
                size = im.size
            boxes = normalize_and_sort_gt_boxes(rec["bbox"] or [], size) if isinstance(rec["bbox"], list) else []
            out.append({"id": new_id, "image": rec["image_path"], "conversations": [
                {"from": "system", "value": system},
                {"from": "user", "value": build_reasoning_prompt_v3(
                    question=rec["question"], choices=rec["choices"], answer=rec["answer"])},
                {"from": "assistant", "value": "<ref>evidence</ref><box>[" +
                 ", ".join(f"[{a}, {b}, {c}, {d}]" for a, b, c, d in boxes) + "]</box>"}],
                "metadata": {"answer": rec["answer"], "image_size": list(size),
                             "gt_boxes_norm": [list(b) for b in boxes], "unscorable": not boxes}})
    return out


def main():
    OUT.mkdir(exist_ok=True)
    print(f"[paper data] -> {OUT}")
    write("sft2000.jsonl", [fix(r) for r in load(I35 / "data_i35/i35_sft2000_v4.jsonl")])
    write("grpo_pool.jsonl", [fix(r) for r in load(I35 / "data_i35/i35_grpo_rest_v4.jsonl")])
    write("val600.jsonl", [fix(r) for d in DOMS for r in load(I35 / f"data_i35/val_by_domain/{d}.jsonl")])
    test = [fix(r) for d in DOMS for r in load(I35 / f"dragon_datasets_test2445/{d}.jsonl")]
    scorable = [r["id"] for r in test]
    system = test[0]["conversations"][0]["value"]
    extra = dropped_test_records(system)
    test = sorted(test + extra, key=lambda r: (DOMS.index(r["id"].split("::")[0]), r["id"]))
    cnt = Counter(r["id"].split("::")[0] for r in test)
    assert dict(cnt) == OFFICIAL, f"Test counts {dict(cnt)} != official {OFFICIAL}"
    assert sum(1 for r in extra if r["metadata"]["unscorable"]) == len(extra) == 13, [r["id"] for r in extra]
    write("test2445.jsonl", test)
    (OUT / "test2432_ids.txt").write_text("\n".join(scorable) + "\n")
    print(f"  added {len(extra)} Test items with no usable gold: {[r['id'] for r in extra]}")
    (OUT / "sft2000_meta.json").write_text(json.dumps({"paper_sft2000": {
        "root": "/mnt/data2/traviku2/Diagram_Attribution_Dataset/", "annotation": str(OUT / "sft2000.jsonl"),
        "data_augment": False, "repeat_time": 1, "max_dynamic_patch": 12}}, indent=2))
    for name in ["sft2000.jsonl", "grpo_pool.jsonl", "val600.jsonl", "test2445.jsonl"]:
        assert all(OLD.strip() not in json.loads(l)["conversations"][0]["value"] for l in open(OUT / name))
    print("ok: no file contains the removed sentence")


if __name__ == "__main__":
    main()
