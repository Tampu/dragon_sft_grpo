#!/usr/bin/env python3
"""
grpo_dragon6_v1.py -- GRPO on top of the dragon6 SFT checkpoint.

Continues from the adapter-only SFT checkpoint (LoRA + mlp1), e.g.
    /mnt/data2/traviku2/outputs/dragon6_sft2k_v4/ckpts

Design, matched to where the SFT model actually is right now
("box patterns learned, boxes in the right vicinity but too big"):

  * POLICY  = base InternVL-8B  +  SFT LoRA merged in  +  a FRESH GRPO LoRA.
    The SFT adapter is merged into the LLM (and vision backbone) weights via
    internvl_lora_checkpoint_io.load_lora_checkpoint (the same loader every
    other eval/verify script in this pipeline uses -- reused here rather
    than re-implemented, since a hand-rolled loader would need to match its
    exact PeftModel key-prefix and base-weight-then-adapter loading order
    to not silently load garbage). mlp1 weights come along with it and are
    then FROZEN (vision stack fully frozen too): GRPO only trains the fresh
    LoRA.

  * REFERENCE = the same model with the GRPO adapter disabled
    (peft `disable_adapter()`), i.e. exactly the SFT policy. No second
    model in memory -- important on a single H200.

  * REWARD is deliberately DENSE in IoU, because the failure mode is
    loose boxes, not wrong regions:
        R = w_iou   * mean 1:1-matched IoU over gt boxes      (main term;
                      a box that's 30% too big scores ~0.6 not 0/1, so the
                      gradient pressure is exactly "tighten the box")
          + w_f1    * box-level F1 at IoU 0.5                 (coverage)
          + w_fmt   * format validity (parseable <ref>/<box>)
          - w_miss  * missing-box fraction (unmatched gt at IoU>=0.5)
          - w_fp    * spurious-box fraction (unmatched pred)   (anti grid /
                      anti under-prediction, the two phase-0 residuals)
    All weights are CLI flags.

  * GRPO proper: G completions per prompt at temperature tau, group-
    normalized advantages (r - mean)/std, token-level PPO-clip objective,
    k3 KL penalty to the reference. One optimizer update per generation
    batch (mu=1) keeps it simple and stable at this scale.

Data: the reserved GRPO pool (build_grpo5k_pool.py), DISJOINT from both the
2000-sample SFT training pool and the 592-sample SFT holdout-eval pool
(checked by (image, id) against both dragon_datasets/<key>/samples.jsonl and
samples_infer_holdout.jsonl -- not just "whatever's left in samples.jsonl",
since that file was itself only a 500/300-per-dataset cap on a much larger
split_reviewed-2 raw pool; see build_grpo5k_pool.py's docstring), converted
with the SAME `sft_v4_phase0.py convert --stage A` so prompts/targets and
`metadata.gt_boxes_norm` are in the identical [0,1000] space the reward
reads -- training and reward cannot disagree about coordinates.

Usage:
    python grpo_dragon6_v1.py \
        --sft-ckpt /mnt/data2/traviku2/outputs/dragon6_sft2k_v4/ckpts \
        --jsonl /mnt/data2/traviku2/dragon_datasets_grpo/grpo5k_grounding_v4.jsonl \
        --image-root /mnt/data2/traviku2/Diagram_Attribution_Dataset \
        --output-dir /mnt/data2/traviku2/outputs/dragon6_grpo_v1 \
        --val-jsonl /mnt/data2/traviku2/dragon_datasets_sft2k/holdout600_grounding_v4.jsonl \
        --hf-cache-dir /mnt/data2/traviku2/hf_home

Throughput note (single H200, 13-tile images, G=8, max_new_tokens=320):
generation dominates; expect very roughly 20-40 prompts/hour-equivalent
optimizer progress. Start with --max-prompts 500 as a pilot before
committing the full 5k pool.
"""

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# ---------------------------------------------------------------------------
# Reward: box parsing + IoU machinery (identical parsing to sft_v4_phase0.py)
# ---------------------------------------------------------------------------

Box = Tuple[float, float, float, float]

NATIVE_BOX_RE = re.compile(r"<box>\s*(\[.*?\])\s*</box>", re.S)


def parse_pred_boxes(text: str) -> Optional[List[Box]]:
    """Returns None on unparseable output (format reward = 0), else boxes."""
    boxes: List[Box] = []
    found_tag = False
    for m in NATIVE_BOX_RE.finditer(text):
        found_tag = True
        try:
            payload = ast.literal_eval(m.group(1))
        except Exception:
            return None
        if payload and isinstance(payload[0], (int, float)):
            payload = [payload]
        for b in payload:
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                return None
            # len==4 does not imply 4 NUMBERS. Under no_repeat_ngram_size the
            # model emits structurally corrupt box lists -- an image text
            # label in a coordinate slot ("Weapon Law Violations per 1,00 00
            # People") or a nested list. The bare float() below raised
            # ValueError/TypeError OUTSIDE the literal_eval try, which killed
            # the test harness twice at the first ngram-constrained sample.
            # With the n-gram filter now enabled in rollouts this is a live
            # crash path for a multi-hour training run, so it is folded into
            # this function's documented "None == unparseable" contract
            # (-> format reward 0) rather than left to raise.
            try:
                x1, y1, x2, y2 = (float(v) for v in b)
            except (TypeError, ValueError):
                return None
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                return None
            boxes.append((x1, y1, x2, y2))
    if not found_tag:
        return None
    return boxes


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def hungarian_match(gt: Sequence[Box], pred: Sequence[Box]) -> List[Tuple[int, int, float]]:
    """1:1 matching maximizing total IoU. Hungarian via scipy, greedy fallback."""
    if not gt or not pred:
        return []
    iou = [[box_iou(g, p) for p in pred] for g in gt]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-np.array(iou))
        return [(int(r), int(c), iou[r][c]) for r, c in zip(ri, ci)]
    except Exception:
        pairs = sorted(((iou[i][j], i, j) for i in range(len(gt))
                        for j in range(len(pred))), reverse=True)
        used_g, used_p, out = set(), set(), []
        for v, i, j in pairs:
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j); out.append((i, j, v))
        return out


@dataclass
class RewardWeights:
    """Targets the three observed failure modes ASYMMETRICALLY:
      loose boxes      -> w_iou  * soft recall (mean 1:1-matched IoU per gt box;
                          dense, so a padded box scores ~0.6 not 0/1)
      missing boxes /  -> w_miss * (unmatched-gt fraction at IoU 0.5), the
      2-of-15 collapse    heaviest penalty: each additionally recovered box
                          buys back BOTH w_iou/n_gt AND w_miss/n_gt, so on a
                          15-box target going 2->3 boxes moves the reward ~0.05
                          -- large relative to within-group spread, which is
                          what GRPO's normalized advantage actually sees
      spurious boxes / -> w_fp   * (unmatched-pred fraction), lighter than
      grid spam           w_miss on purpose: at the current checkpoint,
                          over-prediction is the rarer error and we want no
                          incentive to stay conservative
    plus w_f1 * F1@0.5 as a balanced coverage term and w_fmt for parseability.
    """
    w_iou: float = 0.45
    w_f1: float = 0.15
    w_fmt: float = 0.05
    w_miss: float = 0.25    # missing-box penalty (recall side) -- heaviest
    w_fp: float = 0.10      # spurious-box penalty (precision side) -- lighter


def compute_reward(text: str, gt: Sequence[Box], w: RewardWeights) -> Dict[str, float]:
    n_gt = max(1, len(gt))
    pred = parse_pred_boxes(text)
    if pred is None or len(pred) == 0:
        # unparseable/empty = miss everything: strictly worse than any attempt
        return {"reward": w.w_fmt * 0.0 - w.w_miss * 1.0, "fmt": 0.0,
                "mean_iou": 0.0, "f1": 0.0, "miss": 1.0, "fp": 0.0, "n_pred": 0}
    matches = hungarian_match(gt, pred)
    per_gt_iou = {i: v for i, _, v in matches}
    soft_recall = sum(per_gt_iou.get(i, 0.0) for i in range(len(gt))) / n_gt
    tp = sum(1 for _, _, v in matches if v >= 0.5)
    miss = (len(gt) - tp) / n_gt                       # FN fraction
    fp = (len(pred) - tp) / len(pred)                  # FP fraction of preds
    prec, rec = tp / len(pred), tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    reward = (w.w_fmt * 1.0
              + w.w_iou * soft_recall
              + w.w_f1 * f1
              - w.w_miss * miss
              - w.w_fp * fp)
    return {"reward": reward, "fmt": 1.0, "mean_iou": soft_recall, "f1": f1,
            "miss": miss, "fp": fp, "n_pred": len(pred)}


# ---------------------------------------------------------------------------
# Model loading: reuse this pipeline's proven adapter-only loader, then
# freeze everything and add a fresh GRPO LoRA on top of the merged SFT model.
# ---------------------------------------------------------------------------

def load_policy(args) -> Tuple[Any, Any]:
    # Must run before ANY `internvl.*` import (this module can be invoked
    # standalone, so nothing else has put the InternVL repo on sys.path yet;
    # load_lora_checkpoint() does this too, but only once it's actually
    # CALLED below -- importing internvl.train.constants at module-import
    # time, before that call runs, hits ModuleNotFoundError).
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path, load_lora_checkpoint
    _ensure_internvl_repo_on_path()
    from internvl.train.constants import IMG_CONTEXT_TOKEN

    print(f"[load] SFT checkpoint (base model read from its own "
          f"lora_checkpoint_meta.json): {args.sft_ckpt}")
    # merge_lora=True: SFT's vision_lora + llm_lora are merged straight into
    # the base weights, matching this module's documented design -- GRPO
    # trains a NEW adapter on top, it does not continue optimizing SFT's.
    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=args.sft_ckpt,
        device="cpu",              # move to args.device only after LoRA wrap below
        dtype=torch.bfloat16,
        hf_cache_dir=args.hf_cache_dir,
        merge_lora=True,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Tiling config: load_lora_checkpoint already restores config.json as
    # saved by the SFT run (dynamic_image_size/use_thumbnail/max_dynamic_patch
    # all baked in from training) -- these are asserted, not silently trusted.
    assert model.config.dynamic_image_size, "SFT checkpoint config has dynamic_image_size=False"
    model.config.max_dynamic_patch = args.max_tiles

    # ---- fresh GRPO LoRA; everything else frozen ----
    for p in model.parameters():
        p.requires_grad = False
    model.wrap_llm_lora(r=args.grpo_lora_rank, lora_alpha=2 * args.grpo_lora_rank)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
    print(f"[load] fresh GRPO LoRA rank {args.grpo_lora_rank}: "
          f"{n_params // 1000}k params trainable across {len(trainable)} tensors")

    model.language_model.config.use_cache = True  # needed for generation
    if args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    model.to(args.device).eval()  # eval mode; LoRA still trains (no dropout dependence)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt building + tiling (mirrors model.chat() / training preprocessing)
# ---------------------------------------------------------------------------

def build_batch_item(sample: Dict[str, Any], image_root: Path, model, tokenizer, args):
    from internvl.train.dataset import build_transform, dynamic_preprocess
    from internvl.conversation import get_conv_template
    from internvl.train.constants import (IMG_START_TOKEN, IMG_END_TOKEN,
                                          IMG_CONTEXT_TOKEN)

    img = Image.open(image_root / sample["image"]).convert("RGB")
    transform = build_transform(is_train=False, input_size=448,
                                pad2square=False, normalize_type="imagenet")
    tiles = dynamic_preprocess(img, min_num=1, max_num=args.max_tiles,
                               image_size=448, use_thumbnail=True)
    pixel_values = torch.stack([transform(t) for t in tiles]).to(torch.bfloat16)
    num_patches = pixel_values.size(0)

    system = sample["conversations"][0]["value"]
    user = sample["conversations"][1]["value"]
    if "<image>" not in user:
        user = "<image>\n" + user

    template = get_conv_template(args.conv_style)
    template.system_message = system
    template.append_message(template.roles[0], user)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()
    image_tokens = (IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
                    + IMG_END_TOKEN)
    prompt = prompt.replace("<image>", image_tokens, 1)

    ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    gt = [tuple(map(float, b)) for b in sample["metadata"]["gt_boxes_norm"]]
    return {"pixel_values": pixel_values, "prompt_ids": ids,
            "eos_token_id": eos_token_id, "gt": gt, "id": sample.get("id")}


# ---------------------------------------------------------------------------
# Log-prob computation (policy and reference share weights; ref = adapter off)
# ---------------------------------------------------------------------------

def completion_logprobs(model, item, completion_ids: torch.Tensor,
                        device, no_grad: bool, adapter_disabled: bool) -> torch.Tensor:
    """Per-token logprobs of `completion_ids` given the prompt+image."""
    prompt_ids = item["prompt_ids"].to(device)
    input_ids = torch.cat([prompt_ids, completion_ids.to(device)]).unsqueeze(0)
    attn = torch.ones_like(input_ids)
    pv = item["pixel_values"].to(device)
    image_flags = torch.ones(pv.size(0), dtype=torch.long, device=device)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    import contextlib
    adap = (model.language_model.disable_adapter() if adapter_disabled
            else contextlib.nullcontext())
    with ctx, adap:
        out = model(pixel_values=pv, input_ids=input_ids, attention_mask=attn,
                    image_flags=image_flags, labels=None, return_dict=True,
                    use_cache=False)
    logits = out.logits[0]                       # [T, V]
    start = prompt_ids.size(0)
    # token t predicted by logits[t-1]
    sel = logits[start - 1:-1].float().log_softmax(-1)
    lp = sel.gather(1, completion_ids.to(device).unsqueeze(1)).squeeze(1)
    return lp                                    # [len(completion)]


# ---------------------------------------------------------------------------
# GRPO training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = (out_dir / "grpo_metrics.jsonl").open("a")

    model, tokenizer = load_policy(args)
    device = args.device
    w = RewardWeights(args.w_iou, args.w_f1, args.w_fmt, args.w_miss, args.w_fp)

    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if args.max_prompts:
        samples = samples[: args.max_prompts]
    random.shuffle(samples)
    print(f"[data] {len(samples)} GRPO prompts "
          f"(disjoint from SFT pool by construction -- verify upstream!)")

    val_samples = None
    if args.val_jsonl:
        val_samples = [json.loads(l) for l in Path(args.val_jsonl).read_text().splitlines()
                       if l.strip()][: args.val_size]

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    image_root = Path(args.image_root)
    G, step = args.group_size, 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for bstart in range(0, len(samples), args.prompts_per_step):
            batch = samples[bstart: bstart + args.prompts_per_step]
            rollouts = []

            # ---------------- rollout ----------------
            model.language_model.config.use_cache = True
            for s in batch:
                try:
                    item = build_batch_item(s, image_root, model, tokenizer, args)
                except Exception as e:
                    print(f"[warn] skipping {s.get('id')}: {e}")
                    continue
                pv = item["pixel_values"].to(device)
                input_ids = item["prompt_ids"].unsqueeze(0).to(device)
                attn = torch.ones_like(input_ids)
                with torch.no_grad():
                    gen = model.generate(
                        pixel_values=pv.repeat(1, 1, 1, 1),  # shared across returns
                        input_ids=input_ids, attention_mask=attn,
                        do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, num_return_sequences=G,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=item["eos_token_id"],
                        pad_token_id=tokenizer.eos_token_id or item["eos_token_id"],
                        repetition_penalty=1.0,   # NEVER 1.1 on coordinate strings
                        # Blocks runaway repeated coordinate n-grams, the cause
                        # of the truncated/empty rollouts (100% of the observed
                        # empties were repetition loops hitting max_new_tokens,
                        # not abstention). Unlike repetition_penalty=1.1 this
                        # does not rescale digit logits, so coordinate values
                        # are left intact. 0 disables it.
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        return_dict_in_generate=False)
                comp_ids = gen  # InternVL generate returns only new tokens
                texts = tokenizer.batch_decode(comp_ids, skip_special_tokens=False)
                # strip trailing eos/pad for logprob alignment
                comps, rewards, stats = [], [], []
                for row, txt in zip(comp_ids, texts):
                    row = row[row != tokenizer.pad_token_id] if tokenizer.pad_token_id \
                        is not None else row
                    txt_clean = txt.split(tokenizer.decode([item["eos_token_id"]]))[0]
                    r = compute_reward(txt_clean, item["gt"], w)
                    comps.append(row.cpu()); rewards.append(r["reward"]); stats.append(r)
                rw = torch.tensor(rewards)
                adv = (rw - rw.mean()) / (rw.std() + 1e-4)
                rollouts.append({"item": item, "comps": comps, "adv": adv,
                                 "rewards": rw, "stats": stats})

            if not rollouts:
                continue

            # ---------------- old / ref logprobs ----------------
            model.language_model.config.use_cache = False
            for ro in rollouts:
                ro["old_lp"], ro["ref_lp"] = [], []
                for c in ro["comps"]:
                    ro["old_lp"].append(completion_logprobs(
                        model, ro["item"], c, device, no_grad=True,
                        adapter_disabled=False).cpu())
                    ro["ref_lp"].append(completion_logprobs(
                        model, ro["item"], c, device, no_grad=True,
                        adapter_disabled=True).cpu())

            # ---------------- update (token-level clipped objective) --------
            opt.zero_grad(set_to_none=True)
            n_micro = sum(len(ro["comps"]) for ro in rollouts)
            tot_loss = tot_kl = 0.0
            for ro in rollouts:
                for gi, c in enumerate(ro["comps"]):
                    lp = completion_logprobs(model, ro["item"], c, device,
                                             no_grad=False, adapter_disabled=False)
                    old = ro["old_lp"][gi].to(device)
                    ref = ro["ref_lp"][gi].to(device)
                    adv = ro["adv"][gi].to(device)
                    ratio = (lp - old).exp()
                    clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                    pg = -torch.min(ratio * adv, clipped * adv)
                    kl = (ref - lp).exp() - (ref - lp) - 1.0      # k3, per token
                    loss = (pg + args.kl_beta * kl).mean() / n_micro
                    loss.backward()
                    tot_loss += loss.item() * n_micro
                    tot_kl += kl.mean().item()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
            opt.step()
            step += 1

            mean_r = torch.cat([ro["rewards"] for ro in rollouts]).mean().item()
            mean_iou = sum(s["mean_iou"] for ro in rollouts for s in ro["stats"]) / n_micro
            mean_f1 = sum(s["f1"] for ro in rollouts for s in ro["stats"]) / n_micro
            fmt_rate = sum(s["fmt"] for ro in rollouts for s in ro["stats"]) / n_micro
            mean_miss = sum(s["miss"] for ro in rollouts for s in ro["stats"]) / n_micro
            mean_fp = sum(s["fp"] for ro in rollouts for s in ro["stats"]) / n_micro
            rec = {"step": step, "epoch": epoch, "reward": round(mean_r, 4),
                   "mean_iou": round(mean_iou, 4), "f1@0.5": round(mean_f1, 4),
                   "miss": round(mean_miss, 3), "fp": round(mean_fp, 3),
                   "fmt_rate": round(fmt_rate, 3),
                   "kl": round(tot_kl / n_micro, 5),
                   "loss": round(tot_loss / n_micro, 5),
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            print(f"[step {step}] " + " ".join(f"{k}={v}" for k, v in rec.items()
                                               if k != "step"))
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

            # ---------------- checkpoint + val ----------------
            if step % args.save_every == 0 or bstart + args.prompts_per_step >= len(samples):
                ckpt = out_dir / f"grpo-step{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.language_model.save_pretrained(str(ckpt))   # GRPO adapter only
                tokenizer.save_pretrained(str(ckpt))
                print(f"[ckpt] saved GRPO adapter -> {ckpt}")
                if val_samples:
                    run_val(model, tokenizer, val_samples, image_root, args, step,
                            metrics_f)

    print("done.")


@torch.no_grad()
def run_val(model, tokenizer, val_samples, image_root, args, step, metrics_f) -> None:
    """Greedy decode on the held-out split, report mean matched IoU / recall@0.9
    / F1@0.5 -- the same trajectory metrics as the phase-0 analysis."""
    w = RewardWeights()
    ious, f1s, rec09 = [], [], []
    model.language_model.config.use_cache = True
    for s in val_samples:
        try:
            item = build_batch_item(s, image_root, model, tokenizer, args)
        except Exception:
            continue
        gen = model.generate(
            pixel_values=item["pixel_values"].to(args.device),
            input_ids=item["prompt_ids"].unsqueeze(0).to(args.device),
            attention_mask=torch.ones(1, item["prompt_ids"].size(0),
                                      device=args.device, dtype=torch.long),
            do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
            eos_token_id=item["eos_token_id"], repetition_penalty=1.0,
            # must match the rollout setting, or val measures a different
            # decoder than the one being trained against
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id or item["eos_token_id"])
        txt = tokenizer.decode(gen[0], skip_special_tokens=False)
        txt = txt.split(tokenizer.decode([item["eos_token_id"]]))[0]
        pred = parse_pred_boxes(txt) or []
        matches = hungarian_match(item["gt"], pred)
        per_gt = {i: v for i, _, v in matches}
        n = max(1, len(item["gt"]))
        ious.append(sum(per_gt.get(i, 0.0) for i in range(len(item["gt"]))) / n)
        rec09.append(sum(1 for i in range(len(item["gt"]))
                         if per_gt.get(i, 0.0) >= 0.9) / n)
        r = compute_reward(txt, item["gt"], w)
        f1s.append(r["f1"])
    rec = {"step": step, "val_mean_iou": round(sum(ious) / max(1, len(ious)), 4),
           "val_recall@0.9": round(sum(rec09) / max(1, len(rec09)), 4),
           "val_f1@0.5": round(sum(f1s) / max(1, len(f1s)), 4),
           "val_n": len(ious)}
    print(f"[val step {step}] " + " ".join(f"{k}={v}" for k, v in rec.items()))
    metrics_f.write(json.dumps({"val": rec}) + "\n"); metrics_f.flush()


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-ckpt", required=True,
                    help="adapter-only SFT dir (LoRA + mlp1), e.g. .../dragon6_sft2k_v4/ckpts. "
                         "Base model name is read from its own lora_checkpoint_meta.json.")
    ap.add_argument("--jsonl", required=True,
                    help="GRPO pool jsonl from sft_v4_phase0.py convert (MUST be disjoint from SFT)")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-jsonl", default=None)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--hf-cache-dir", type=str, default=None,
                    help="HF_HOME for from_pretrained(). Without this it falls back to "
                         "~/.cache/huggingface, which lives on the near-full root disk on "
                         "this machine -- pass /mnt/data2/traviku2/hf_home (already has the "
                         "full InternVL3-8B snapshot).")
    # model
    ap.add_argument("--grpo-lora-rank", type=int, default=16)
    ap.add_argument("--conv-style", default="internlm2-chat",
                    help="MUST match the SFT training conv_style (internlm2-chat throughout "
                         "this pipeline -- our data uses system/user/assistant roles, and "
                         "'internvl2_5' as a data-preprocessing style requires human/gpt roles "
                         "and would break; as a pure prompt TEMPLATE name here it wouldn't "
                         "crash, but would render a subtly different separator than what the "
                         "SFT checkpoint was actually trained on).")
    ap.add_argument("--max-tiles", type=int, default=12)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    # GRPO
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="pilot cap, e.g. 500 before running the full pool")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=6,
                    help="Block repeated n-grams during rollout AND val generation "
                         "(0 disables). Ships at 6 by request. Note the test-harness "
                         "evidence available at the time favoured 4: on the sampled arm "
                         "(the rollout regime) 4 scored 73%% CLEAN vs 65%% at 6 and 69%% "
                         "unconstrained, over 26 samples with the good-set DEGRADED leg "
                         "still unmeasured. Re-check merge_norepeat_results.py once the "
                         "full sweep lands and change this default if it disagrees.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    # reward weights
    ap.add_argument("--w-iou", type=float, default=0.45)
    ap.add_argument("--w-f1", type=float, default=0.15)
    ap.add_argument("--w-fmt", type=float, default=0.05)
    ap.add_argument("--w-miss", type=float, default=0.25)
    ap.add_argument("--w-fp", type=float, default=0.10)
    # misc
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


if __name__ == "__main__":
    main()
