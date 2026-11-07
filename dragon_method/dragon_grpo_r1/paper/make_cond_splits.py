#!/usr/bin/env python3
"""Analysis 3 inputs: Test2445 with a wrong answer, and with no answer. CPU only.

Only the "Correct Answer:" line of the user turn changes; image, question, choices, gold stay. Fixed once, seed 17.
  wrong   an answer that differs from the correct one, chosen in this order:
            1 another listed choice of the same question (multiple choice: the natural wrong answer)
            2 yes <-> no, true <-> false
            3 the answer of another question on the same image, of the same type (yes/no, number, list, text)
            4 an answer of another item of the same domain and the same type, seeded
  none    the "Correct Answer:" line removed

Writes data_paper/test2445_wrong.jsonl, test2445_none.jsonl and test2445_wrong_map.json (id -> how it was chosen).

    /mnt/data2/traviku2/da_env2/bin/python dragon_grpo_r1/paper/make_cond_splits.py
"""
from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old/data_paper")
LINE = re.compile(r"\nCorrect Answer:[^\n]*")


def answer_of(user):
    m = re.search(r"\nCorrect Answer:([^\n]*)", user)
    return m.group(1).strip() if m else ""


def choices_of(user):
    m = re.search(r"\nChoices:([^\n]*)", user)
    if not m or m.group(1).strip() in ("", "None"):
        return []
    return [re.sub(r"^\([A-Z]\)\s*", "", c.strip()) for c in m.group(1).split(";") if c.strip()]


def is_num(s):
    return re.fullmatch(r"[-+]?[\d,]*\.?\d+%?", s.replace(" ", "")) is not None


FLIP = {"yes": "No", "no": "Yes", "true": "False", "false": "True"}


def kind(s):
    """Answer type: yes/no, number, list / structured, or text."""
    t = s.strip().lower()
    return "yesno" if t in FLIP else "num" if is_num(s) else "list" if t[:1] in "[{" else "text"


def main():
    recs = [json.loads(l) for l in open(DATA / "test2445.jsonl")]
    rng = random.Random(17)
    by_img, by_dom_type = defaultdict(list), defaultdict(list)
    for r in recs:
        ans = answer_of(r["conversations"][1]["value"])
        by_img[r["image"]].append(ans)
        if ans:
            by_dom_type[(r["id"].split("::")[0], kind(ans))].append(ans)
    wrong, none, how = [], [], {}
    for r in recs:
        u = r["conversations"][1]["value"]
        ans = answer_of(u)
        norm = ans.strip().lower()
        cands = [c for c in choices_of(u) if c.strip().lower() != norm]
        rule = "choice"
        if not cands and norm in FLIP:
            cands, rule = [FLIP[norm]], "flip_yesno"
        if not cands:
            cands, rule = sorted({a for a in by_img[r["image"]]
                                  if a and a.strip().lower() != norm and kind(a) == kind(ans)}), "same_image"
        if not cands:
            pool = by_dom_type[(r["id"].split("::")[0], kind(ans))]
            cands, rule = sorted({a for a in pool if a.strip().lower() != norm}), "same_domain_type"
        w = rng.choice(cands) if cands else ""
        how[r["id"]] = {"rule": rule if cands else "none_available", "correct": ans, "wrong": w}
        rw = copy.deepcopy(r)
        rw["conversations"][1]["value"] = LINE.sub(f"\nCorrect Answer: {w}", u)
        rw["metadata"]["answer"] = w
        wrong.append(rw)
        rn = copy.deepcopy(r)
        rn["conversations"][1]["value"] = LINE.sub("", u)
        rn["metadata"]["answer"] = ""
        none.append(rn)
    for name, rs in [("test2445_wrong.jsonl", wrong), ("test2445_none.jsonl", none)]:
        with open(DATA / name, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    (DATA / "test2445_wrong_map.json").write_text(json.dumps(how, indent=0))
    print(f"wrong-answer rules: {dict(Counter(v['rule'] for v in how.values()))}")
    print("examples:", [(k, v) for k, v in list(how.items())[:3]])
    print(f"none: 'Correct Answer' lines left = {sum('Correct Answer' in r['conversations'][1]['value'] for r in none)}")


if __name__ == "__main__":
    main()
