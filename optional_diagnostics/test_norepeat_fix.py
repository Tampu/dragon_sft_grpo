#!/usr/bin/env python3
"""
test_norepeat_fix.py -- minutes-not-hours test of the decoding-side fix for
the runaway-repetition truncation found in the GRPO empties.

Background (what this tests):
  100% of the "empty prediction" samples on infographics/mapiq were
  TRUNCATION of a runaway repetitive coordinate loop -- every one terminated
  at ~368-374 chars regardless of n_gt (1-box and 54-box targets alike),
  which rules out token-budget shortfall and rules out abstention. The
  candidate fix is `no_repeat_ngram_size` on rollout generation, which
  mechanically blocks exact n-gram repeats without the digit-corrupting
  side effects of repetition_penalty=1.1.

What this script decides (the go/no-go):
  A) Does no_repeat_ngram_size make previously-looping samples terminate
     cleanly (proper </box>, EOS before the budget)?
  B) Is the resulting CONTENT sane -- or does the model just evade the
     n-gram filter with a jittered grid (near-duplicate boxes with slightly
     perturbed coordinates)? Evasion = the degeneracy lives in the policy,
     and only then is a reward-side self-similarity penalty warranted.
  C) Does the constraint corrupt LEGITIMATE multi-box outputs on known-good
     samples? (Coordinate strings legitimately reuse fragments like "100, ".)

Usage:
    python test_norepeat_fix.py \
        --checkpoint /path/to/grpo_or_sft_ckpt \
        --jsonl dragon_datasets_grpo/grpo5k_grounding_v4.jsonl \
        --image-root /mnt/data2/traviku2/dragon_datasets \
        --ids-file truncated_ids.txt \
        --good-ids-file known_good_ids.txt \
        --ngram-sizes 0 3 4 6 \
        --out-dir norepeat_test

  --ids-file: one sample id per line -- the ~50 previously-empty/truncated
      mapiq+infographics ids. If omitted, the script instead selects samples
      whose ORIGINAL greedy generation truncates (detected live).
  --good-ids-file: ids of samples the model previously answered correctly
      (multi-box preferred) -- the regression check for (C). If omitted,
      falls back to the first N samples not in --ids-file.
  --ngram-sizes: 0 means unconstrained (the control arm). Always include 0
      so before/after is measured in the same run, same code path.

Output: per-sample per-setting classification table + summary + a jsonl dump
of every raw generation for eyeballing, written under --out-dir.

Classification per generation:
  CLEAN      valid close tag + parseable boxes + not budget-truncated +
             passes the sanity checks below
  JITTER     parseable but boxes form a near-grid: >=6 boxes AND
             (a) high pairwise self-similarity (many box pairs with IoU>0.5
                 or near-identical width/height), or
             (b) arithmetic-progression structure in centers
             -> the evasion signature: model dodges the n-gram filter by
                perturbing coordinates instead of stopping
  LOOP       budget-truncated (hit max_new_tokens without EOS), i.e. the
             original failure still present
  EMPTY      no parseable boxes at all
  Plus, for --good-ids samples: DEGRADED if the constrained generation's
  soft-recall vs gt drops >0.15 below the unconstrained generation's.
"""

import argparse
import ast
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

NATIVE_BOX_RE = re.compile(r"<box>\s*(\[.*?\])\s*</box>", re.S)

# Plumbing only (see load_model / generate_one): this repo's scripts dir holds
# internvl_lora_checkpoint_io + sft_v4_phase0, needed to load an adapter-only
# checkpoint and to decode without stripping <ref>/<box>. No scoring logic
# below is affected.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# ---------------------------------------------------------------------------
# Parsing + geometry (self-contained; mirrors grpo_dragon6_v1.py)
# ---------------------------------------------------------------------------

def parse_pred_boxes(text: str) -> List[Box]:
    boxes: List[Box] = []
    for m in NATIVE_BOX_RE.finditer(text):
        try:
            payload = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if payload and isinstance(payload[0], (int, float)):
            payload = [payload]
        for b in payload:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                # ROBUSTNESS PATCH: len==4 does not imply 4 NUMBERS. Under an
                # n-gram constraint this model emits structurally corrupt box
                # lists -- an image text label in a coordinate slot
                # ("Weapon Law Violations per 1,00 00 People") or a nested
                # list -- which crashed the original unguarded float() and
                # killed both sweep processes at the first ngram=3 sample.
                # Skipping the malformed entry keeps the intended semantics
                # (collect the valid boxes); the malformation itself is
                # counted separately by merge_norepeat_results.py.
                try:
                    x1, y1, x2, y2 = (float(v) for v in b)
                except (TypeError, ValueError):
                    continue
                boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def soft_recall(gt: Sequence[Box], pred: Sequence[Box]) -> float:
    if not gt or not pred:
        return 0.0
    return sum(max(box_iou(g, p) for p in pred) for g in gt) / len(gt)


# ---------------------------------------------------------------------------
# Degeneracy classifiers
# ---------------------------------------------------------------------------

def _centers(boxes: Sequence[Box]) -> List[Tuple[float, float]]:
    return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]


def _is_arithmetic_progression(vals: List[float], tol: float = 6.0) -> bool:
    """True if sorted values step by a near-constant delta (the phase-0 grid
    signature: e.g. y = 373, 393, 413, ... constant +20)."""
    if len(vals) < 4:
        return False
    v = sorted(vals)
    diffs = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    diffs = [d for d in diffs if d > tol / 2]  # ignore co-located values
    if len(diffs) < 3:
        return False
    med = statistics.median(diffs)
    if med <= 0:
        return False
    close = sum(1 for d in diffs if abs(d - med) <= tol)
    return close / len(diffs) >= 0.7


def jitter_grid_score(boxes: Sequence[Box]) -> Dict[str, Any]:
    """Detects the evasion signature: many boxes that are near-duplicates or
    lie on a regular lattice with perturbed coordinates."""
    n = len(boxes)
    out = {"n": n, "dup_pair_frac": 0.0, "wh_uniform": False,
           "ap_x": False, "ap_y": False, "is_jitter_grid": False}
    if n < 6:
        return out
    # (a) near-duplicate pairs: IoU>0.5 between distinct predictions
    dup = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if box_iou(boxes[i], boxes[j]) > 0.5:
                dup += 1
    out["dup_pair_frac"] = dup / total if total else 0.0
    # (b) uniform shapes: std of width and height both tiny relative to mean
    ws = [b[2] - b[0] for b in boxes]
    hs = [b[3] - b[1] for b in boxes]
    def _uniform(vs):
        m = statistics.mean(vs)
        return m > 0 and (statistics.pstdev(vs) / m) < 0.12
    out["wh_uniform"] = _uniform(ws) and _uniform(hs)
    # (c) lattice structure in centers
    cx, cy = zip(*_centers(boxes))
    out["ap_x"] = _is_arithmetic_progression(list(cx))
    out["ap_y"] = _is_arithmetic_progression(list(cy))
    out["is_jitter_grid"] = (out["dup_pair_frac"] >= 0.15
                             or (out["wh_uniform"] and (out["ap_x"] or out["ap_y"])))
    return out


def classify(response: str, boxes: List[Box], truncated: bool,
             n_gt: int) -> Tuple[str, Dict[str, Any]]:
    if truncated:
        return "LOOP", {"note": "hit max_new_tokens without EOS"}
    if not boxes:
        return "EMPTY", {}
    g = jitter_grid_score(boxes)
    # over-length relative to gt is corroborating, not required
    if g["is_jitter_grid"] and len(boxes) >= max(6, 2 * n_gt):
        return "JITTER", g
    return "CLEAN", g


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_model(args):
    """PLUMBING PATCH (environment-specific, no scoring logic changed).

    The original body was:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, ...)
        model = InternVLChatModel.from_pretrained(args.checkpoint, ...)
    which requires a merged/full HF checkpoint. Every checkpoint in this
    repo is adapter-only (llm_lora/ + vision_lora/ + trainable_extra.
    safetensors; there is no model.safetensors), so from_pretrained would
    load random base weights or fail outright. Reuses the loader that every
    other eval script here already uses, and applies the GRPO adapter on top
    when --grpo-checkpoint is given.
    """
    import torch
    from internvl_lora_checkpoint_io import (_ensure_internvl_repo_on_path,
                                             load_lora_checkpoint)
    _ensure_internvl_repo_on_path()  # must precede any internvl.* import
    from internvl.train.constants import IMG_CONTEXT_TOKEN

    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=args.checkpoint, device=args.device,
        dtype=torch.bfloat16, hf_cache_dir=args.hf_cache_dir, merge_lora=True)

    if getattr(args, "grpo_checkpoint", None):
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, args.grpo_checkpoint)
        model.language_model = model.language_model.to(args.device)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.config.dynamic_image_size = True
    model.config.use_thumbnail = True
    model.config.max_dynamic_patch = args.max_tiles
    model.language_model.config.use_cache = True
    return model, tokenizer


def generate_one(model, tokenizer, sample: Dict[str, Any], image_root: Path,
                 args, ngram: int, sampled: bool) -> Tuple[str, bool]:
    """Returns (response_text, truncated_at_budget)."""
    import torch
    from PIL import Image
    from internvl.train.dataset import build_transform, dynamic_preprocess
    # PLUMBING PATCH: model.chat() decodes with skip_special_tokens=True
    # (modeling_internvl_chat.py:339,388), which deletes <ref>/</ref>/<box>/
    # </box> -- they were registered as special tokens during training. With
    # the stock call every generation here would parse to zero boxes and the
    # whole sweep would report EMPTY regardless of ngram. Uses the same
    # skip_special_tokens=False decode path the rest of this pipeline uses.
    from sft_v4_phase0 import chat_keep_special_tokens

    img = Image.open(image_root / sample["image"]).convert("RGB")
    transform = build_transform(is_train=False, input_size=448,
                                pad2square=False, normalize_type="imagenet")
    tiles = dynamic_preprocess(img, min_num=1, max_num=args.max_tiles,
                               image_size=448, use_thumbnail=True)
    pixel_values = torch.stack([transform(t) for t in tiles]).to(
        torch.bfloat16).to(args.device)

    question = sample["conversations"][1]["value"]
    if "<image>" not in question:
        question = "<image>\n" + question

    gen_cfg = dict(max_new_tokens=args.max_new_tokens,
                   repetition_penalty=1.0)
    if ngram > 0:
        gen_cfg["no_repeat_ngram_size"] = ngram
    if sampled:
        gen_cfg.update(do_sample=True, temperature=args.temperature,
                       top_p=1.0)
    else:
        gen_cfg.update(do_sample=False, num_beams=1)

    response = chat_keep_special_tokens(model, tokenizer, pixel_values,
                                        question, gen_cfg, args.device)
    # truncation heuristic: token count of response at/over budget
    n_tok = len(tokenizer(response, add_special_tokens=False).input_ids)
    truncated = n_tok >= args.max_new_tokens - 2
    return response, truncated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--grpo-checkpoint", default=None,
                    help="optional grpo-stepN adapter applied on top of --checkpoint")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ids-file", default=None,
                    help="ids of previously-truncated samples (one per line)")
    ap.add_argument("--good-ids-file", default=None,
                    help="ids of previously-correct samples for the regression check")
    ap.add_argument("--ngram-sizes", type=int, nargs="*", default=[0, 3, 4, 6],
                    help="0 = unconstrained control; always include 0")
    ap.add_argument("--n-auto-bad", type=int, default=30,
                    help="if no --ids-file, auto-detect up to this many looping samples")
    ap.add_argument("--n-good", type=int, default=15)
    ap.add_argument("--sampled-arm", action="store_true",
                    help="also test temperature-1.0 sampling (the GRPO rollout regime), "
                         "not just greedy -- recommended, loops can be decode-mode-dependent")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=320,
                    help="keep at the GRPO rollout value so 'truncated' means the same thing")
    ap.add_argument("--max-tiles", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_f = (out_dir / "generations.jsonl").open("w")

    samples = {s.get("id"): s for s in
               (json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip())}

    model, tokenizer = load_model(args)
    image_root = Path(args.image_root)

    # ---- pick the bad set ----
    if args.ids_file:
        bad_ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
        bad = [samples[i] for i in bad_ids if i in samples]
        missing = [i for i in bad_ids if i not in samples]
        if missing:
            print(f"[warn] {len(missing)} ids not found in jsonl: {missing[:5]}...")
    else:
        print(f"[auto] no --ids-file; probing for looping samples (greedy, ngram=0)...")
        bad = []
        for s in samples.values():
            if len(bad) >= args.n_auto_bad:
                break
            resp, trunc = generate_one(model, tokenizer, s, image_root, args,
                                       ngram=0, sampled=False)
            if trunc:
                bad.append(s)
        print(f"[auto] found {len(bad)} looping samples")

    # ---- pick the good set (regression check) ----
    if args.good_ids_file:
        good_ids = [l.strip() for l in Path(args.good_ids_file).read_text().splitlines() if l.strip()]
        good = [samples[i] for i in good_ids if i in samples]
    else:
        bad_id_set = {s.get("id") for s in bad}
        good = [s for s in samples.values() if s.get("id") not in bad_id_set][: args.n_good]

    arms = [("greedy", False)] + ([("sampled", True)] if args.sampled_arm else [])
    counts: Dict[Tuple[str, int, str], int] = {}
    degraded: Dict[int, int] = {n: 0 for n in args.ngram_sizes}
    good_baseline_sr: Dict[str, float] = {}

    def record(arm, ngram, verdict):
        counts[(arm, ngram, verdict)] = counts.get((arm, ngram, verdict), 0) + 1

    # ---- bad set: does each ngram setting stop the loop, and how? ----
    print(f"\n=== BAD SET ({len(bad)} previously-looping samples) ===")
    for s in bad:
        gt = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
        for arm_name, sampled in arms:
            for ngram in args.ngram_sizes:
                resp, trunc = generate_one(model, tokenizer, s, image_root, args,
                                           ngram=ngram, sampled=sampled)
                boxes = parse_pred_boxes(resp)
                verdict, detail = classify(resp, boxes, trunc, len(gt))
                record(arm_name, ngram, verdict)
                raw_f.write(json.dumps({
                    "set": "bad", "id": s.get("id"), "arm": arm_name,
                    "ngram": ngram, "verdict": verdict, "n_gt": len(gt),
                    "n_pred": len(boxes), "soft_recall": round(soft_recall(gt, boxes), 3),
                    "detail": detail, "response": resp}) + "\n")
                raw_f.flush()
        print(f"  done id={s.get('id')} (n_gt={len(gt)})", flush=True)

    # ---- good set: does the constraint corrupt legitimate output? ----
    print(f"\n=== GOOD SET ({len(good)} regression samples) ===")
    for s in good:
        gt = [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]]
        for ngram in args.ngram_sizes:
            resp, trunc = generate_one(model, tokenizer, s, image_root, args,
                                       ngram=ngram, sampled=False)
            boxes = parse_pred_boxes(resp)
            sr = soft_recall(gt, boxes)
            if ngram == 0:
                good_baseline_sr[s.get("id")] = sr
            else:
                base = good_baseline_sr.get(s.get("id"), 0.0)
                if base - sr > 0.15:
                    degraded[ngram] += 1
            verdict, detail = classify(resp, boxes, trunc, len(gt))
            record("good", ngram, verdict)
            raw_f.write(json.dumps({
                "set": "good", "id": s.get("id"), "arm": "greedy",
                "ngram": ngram, "verdict": verdict, "n_gt": len(gt),
                "n_pred": len(boxes), "soft_recall": round(sr, 3),
                "detail": detail, "response": resp}) + "\n")
            raw_f.flush()
        print(f"  done id={s.get('id')}", flush=True)
    raw_f.close()

    # ---- summary ----
    verdicts = ["CLEAN", "JITTER", "LOOP", "EMPTY"]
    print(f"\n{'='*72}\nSUMMARY  (bad set: want LOOP -> CLEAN, fear LOOP -> JITTER)\n{'='*72}")
    for arm_name, _ in arms:
        print(f"\n[bad set, {arm_name}]")
        print(f"  {'ngram':>6} " + " ".join(f"{v:>7}" for v in verdicts))
        for ngram in args.ngram_sizes:
            row = " ".join(f"{counts.get((arm_name, ngram, v), 0):>7}" for v in verdicts)
            print(f"  {ngram:>6} {row}")
    print(f"\n[good set, greedy]  DEGRADED = soft_recall drop >0.15 vs ngram=0")
    print(f"  {'ngram':>6} {'CLEAN':>7} {'JITTER':>7} {'LOOP':>7} {'EMPTY':>7} {'DEGRADED':>9}")
    for ngram in args.ngram_sizes:
        row = " ".join(f"{counts.get(('good', ngram, v), 0):>7}" for v in verdicts)
        print(f"  {ngram:>6} {row} {degraded.get(ngram, 0):>9}")

    print(f"""
{'='*72}
DECISION GUIDE
{'='*72}
- ngram setting with max CLEAN on bad set, ~0 JITTER, and 0 DEGRADED on
  good set -> ship that no_repeat_ngram_size into GRPO rollouts + val,
  no reward change needed.
- If JITTER dominates at every ngram: the degeneracy is in the policy, the
  decoder can only reshape it -> that is the evidence for the reward-side
  self-similarity penalty (option 2). Bring the generations.jsonl.
- If DEGRADED > 0 at the winning ngram: the filter is blocking legitimate
  coordinate fragments -> try a larger ngram before concluding anything.
- Run with --sampled-arm too before shipping: GRPO rollouts sample at
  temperature 1.0, and loop behavior can differ between greedy and sampled.
Raw generations: {out_dir/'generations.jsonl'}
""")


if __name__ == "__main__":
    main()
