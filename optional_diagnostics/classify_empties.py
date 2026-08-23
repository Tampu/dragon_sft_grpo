#!/usr/bin/env python3
"""
Classify GRPO-step250's zero-box predictions on infographics/mapiq into:
  - no_box_tag   : response never opens a <box> tag at all -- true abstention,
                   a reward-shaping problem.
  - truncated    : response opens <box> but never closes it before hitting
                   max_new_tokens (or the list inside doesn't parse because
                   it's cut off mid-number/mid-list) -- a token-budget
                   problem, not a reward problem. max_new_tokens=320 was the
                   pilot's setting; mapiq/infographics are exactly the
                   two domains with the densest gt box counts, making this a
                   live suspect per the plan.
  - malformed_other: has both <box> and </box> but still fails to parse for
                   another reason (residual bucket, should be small).

Matches predictions to source records by POSITION, not by 'id' -- ids like
"Q_0"/"Q_01" repeat across different images within a dataset (established
earlier this session), so matching by id string would silently pick the
wrong record. export_for_eval_script.py's run_inference() iterates samples
in file order and writes predictions in the same order, so pred_<domain>.json
line i <-> holdout_per_domain/<domain>.jsonl line i.
"""
import ast
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path, load_lora_checkpoint  # noqa: E402
_ensure_internvl_repo_on_path()
from internvl.train.dataset import build_transform, dynamic_preprocess  # noqa: E402
from internvl.train.constants import IMG_CONTEXT_TOKEN  # noqa: E402
from sft_v4_phase0 import chat_keep_special_tokens  # noqa: E402

ROOT = Path("/mnt/data2/traviku2")
SFT_CKPT = ROOT / "outputs/dragon6_sft2k_v4/ckpts"
GRPO_CKPT = ROOT / "outputs/dragon6_grpo_v1/grpo-step250"
HF_CACHE = ROOT / "hf_home"
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
DEVICE = "cuda:7"
MAX_TILES = 12
MAX_NEW_TOKENS = 320  # exact pilot setting, so this is a faithful reproduction

NATIVE_BOX_OPEN = "<box>"
NATIVE_BOX_CLOSE = "</box>"
NATIVE_BOX_RE = re.compile(r"<box>\s*(\[.*?\])\s*</box>", re.S)


def find_empty_indices(domain: str):
    pred_path = ROOT / f"preds_for_eval/grpo_step250/pred_{domain}.json"
    items = json.loads(pred_path.read_text())
    return [i for i, it in enumerate(items) if len(it["pred_boxes_parsed"]) == 0]


def classify(text: str) -> str:
    if NATIVE_BOX_OPEN not in text:
        return "no_box_tag"
    if NATIVE_BOX_CLOSE not in text:
        return "truncated"
    # has both tags but parse_pred_boxes-style extraction still yields nothing
    m = NATIVE_BOX_RE.search(text)
    if m is None:
        return "truncated"  # tags present but content between them doesn't match the list pattern
    try:
        payload = ast.literal_eval(m.group(1))
        if payload:
            return "malformed_other_nonempty_but_unparsed"  # shouldn't normally hit this path
    except Exception:
        return "malformed_other"
    return "malformed_other"  # e.g. an explicit empty [] list


def main():
    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=str(SFT_CKPT), device=DEVICE, dtype=torch.bfloat16,
        hf_cache_dir=str(HF_CACHE), merge_lora=True)
    from peft import PeftModel
    model.language_model = PeftModel.from_pretrained(model.language_model, str(GRPO_CKPT))
    model.language_model = model.language_model.to(DEVICE)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.language_model.config.use_cache = True

    transform = build_transform(is_train=False, input_size=448, pad2square=False,
                                normalize_type="imagenet")
    gen_cfg = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                   repetition_penalty=1.0)

    results = {"infographics": [], "mapiq": []}
    for domain in ("infographics", "mapiq"):
        empty_idx = set(find_empty_indices(domain))
        jsonl_path = ROOT / f"dragon_datasets_sft2k/holdout_per_domain/{domain}.jsonl"
        samples = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        print(f"\n[{domain}] {len(empty_idx)} empty predictions out of {len(samples)}, regenerating...")

        for i, s in enumerate(samples):
            if i not in empty_idx:
                continue
            img = Image.open(IMAGE_ROOT / s["image"]).convert("RGB")
            tiles = dynamic_preprocess(img, min_num=1, max_num=MAX_TILES, image_size=448,
                                       use_thumbnail=True)
            pixel_values = torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).to(DEVICE)
            question = s["conversations"][1]["value"]
            if "<image>" not in question:
                question = "<image>\n" + question
            response = chat_keep_special_tokens(model, tokenizer, pixel_values, question,
                                                gen_cfg, DEVICE)
            n_gt = len(s["metadata"]["gt_boxes_norm"])
            bucket = classify(response)
            results[domain].append({
                "idx": i, "id": s.get("id"), "n_gt": n_gt,
                "bucket": bucket, "response_len_chars": len(response),
                "response_tail": response[-200:],
            })

        counts = {}
        for r in results[domain]:
            counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        print(f"[{domain}] classification: {counts}")

    out_path = ROOT / "dragon_grpo/empty_classification.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote details -> {out_path}")

    print("\n=== SUMMARY ===")
    for domain in ("infographics", "mapiq"):
        counts = {}
        for r in results[domain]:
            counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        total = len(results[domain])
        print(f"{domain}: total_empty={total}  {counts}")
        # a couple example tails per bucket for manual sanity check
        seen_buckets = set()
        for r in results[domain]:
            if r["bucket"] in seen_buckets:
                continue
            seen_buckets.add(r["bucket"])
            print(f"  example [{r['bucket']}] n_gt={r['n_gt']} len={r['response_len_chars']} "
                  f"tail={r['response_tail']!r}")


if __name__ == "__main__":
    main()
