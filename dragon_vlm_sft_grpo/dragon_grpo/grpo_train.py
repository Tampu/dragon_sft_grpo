#!/usr/bin/env python3
"""
Generic GRPO on top of an SFT LoRA checkpoint -- reads a --model-config and
is otherwise identical code regardless of which HF vision-language model
that config points at.

Design (mirrors dragon_sft_grpo's grpo_dragon6_v1.py, generalized):
  * POLICY  = base model + SFT LoRA MERGED into the weights (via
    --sft-checkpoint, optional -- omit it to run GRPO directly from the
    base model as a zero-shot-RL ablation) + a FRESH LoRA wrapped on top
    (model_config.apply_lora, target_modules="all-linear" by default --
    the vision tower, connector, and language model are all fine-tunable
    by GRPO, not just the language model, unless a model config restricts
    lora.target_modules).
  * REFERENCE = the same weights with the fresh GRPO adapter disabled
    (peft's model.disable_adapter() context) -- exactly the SFT policy, no
    second model copy in memory. This is a generic peft API and needs no
    per-architecture handling.
  * REWARD: box_matching.compute_reward -- dense in IoU (a box 30% too big
    scores ~0.6, not 0/1), asymmetric penalty (missing a gt box costs more
    than an extra spurious one).
  * GRPO proper: G completions/prompt at temperature tau, group-normalized
    advantages (r - mean)/std, token-level PPO-clip, k3 KL penalty to the
    reference. One optimizer update per generation batch.

DECODE GOTCHA (carried over as a standing warning, not rediscovered the
hard way here): always decode rollouts with skip_special_tokens=False. A
reasoning-tuned model's <think>/</think> tags are typically real special
tokens; the default tokenizer.decode(skip_special_tokens=True) would
silently strip them, which breaks grounding_prompts.strip_thinking() and
can make correctly-produced output look empty. This script decodes with
skip_special_tokens=False everywhere a rollout or val generation is scored.

NOT YET EMPIRICALLY VERIFIED (no GPU/model access when this was authored):
  - generate()'s automatic expansion of pixel_values/image_grid_thw to
    num_return_sequences=G copies. This is standard HF GenerationMixin
    behavior (any model_kwargs tensor whose leading dim matches the input
    batch size gets expanded alongside input_ids), but confirm on first run
    that G rollouts for one prompt actually see G independent samples and
    not a shape mismatch or a silently-broadcast single image.
  - per-token logprop alignment for a model whose chat template inserts
    trailing tokens after the assistant content (start with G=2,
    prompts-per-step=1, a few steps, and print rollout texts before
    committing to a long run).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from box_matching import RewardWeights, compute_reward
from grounding_prompts import build_chat_messages, parse_pred_boxes
from model_config import apply_lora, load_model_and_processor, load_model_config, render_chat_text


def load_policy(args, cfg):
    if args.sft_checkpoint:
        from lora_checkpoint_io import load_adapter_checkpoint
        print(f"[load] merging SFT adapter from {args.sft_checkpoint}")
        model, processor, _ = load_adapter_checkpoint(
            args.sft_checkpoint, device=args.device, merge=True,
            hf_cache_dir=args.hf_cache_dir, model_cfg=cfg)
    else:
        print("[load] no --sft-checkpoint given -- GRPO runs directly on the base model")
        model, processor = load_model_and_processor(cfg, device=args.device, hf_cache_dir=args.hf_cache_dir)

    for p in model.parameters():
        p.requires_grad = False
    # lora_key="grpo_lora": language-model-only by default (vision tower and
    # connector/projector stay frozen) -- deliberately NARROWER than SFT's
    # "lora" (all-linear) scope. See model_config.py's module docstring:
    # letting the vision encoder move under GRPO's sparse, noisy reward is a
    # known way to quietly degrade grounding quality even if some proxy
    # reward rises, which is why dragon_sft_grpo's own GRPO design freezes
    # the vision stack too.
    model = apply_lora(model, cfg, lora_key="grpo_lora")
    model.to(args.device)
    return model, processor


def build_prompt_item(sample: Dict[str, Any], image_root: Path, processor, cfg) -> Dict[str, Any]:
    image_abs = str((image_root / sample["image"]).resolve())
    from PIL import Image
    image = Image.open(image_abs).convert("RGB")

    messages = build_chat_messages(sample, image_abs, include_assistant=False)
    prompt_text = render_chat_text(processor, messages, cfg, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")

    gt = [tuple(map(float, b)) for b in sample["metadata"]["gt_boxes_norm"]]
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    return {"input_ids": inputs["input_ids"][0], "attention_mask": inputs["attention_mask"][0],
            "extra": extra, "gt": gt, "id": sample.get("id")}


def completion_logprobs(model, item, completion_ids: torch.Tensor, device: str,
                        no_grad: bool, adapter_disabled: bool) -> torch.Tensor:
    prompt_ids = item["input_ids"].to(device)
    input_ids = torch.cat([prompt_ids, completion_ids.to(device)]).unsqueeze(0)
    attn = torch.ones_like(input_ids)
    extra = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["extra"].items()}

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    adap = model.disable_adapter() if adapter_disabled else contextlib.nullcontext()
    with ctx, adap:
        out = model(input_ids=input_ids, attention_mask=attn, **extra, use_cache=False)
    logits = out.logits[0]
    start = prompt_ids.size(0)
    sel = logits[start - 1:-1].float().log_softmax(-1)
    return sel.gather(1, completion_ids.to(device).unsqueeze(1)).squeeze(1)


def main() -> None:
    args = parse_args()
    cfg = load_model_config(args.model_config)
    torch.manual_seed(args.seed); random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = (out_dir / "grpo_metrics.jsonl").open("a")

    model, processor = load_policy(args, cfg)
    device = args.device
    w = RewardWeights(args.w_iou, args.w_f1, args.w_fmt, args.w_miss, args.w_fp)

    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if args.max_prompts:
        samples = samples[: args.max_prompts]
    random.shuffle(samples)
    print(f"[data] {len(samples)} GRPO prompts")

    val_samples = None
    if args.val_jsonl:
        val_samples = [json.loads(l) for l in Path(args.val_jsonl).read_text().splitlines()
                       if l.strip()][: args.val_size]

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    image_root = Path(args.image_root)
    G, step = args.group_size, 0
    gen_defaults = cfg.grpo_generation
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    t0 = time.time()

    for epoch in range(args.epochs):
        for bstart in range(0, len(samples), args.prompts_per_step):
            batch = samples[bstart: bstart + args.prompts_per_step]
            rollouts = []

            # ---------------- rollout ----------------
            for s in batch:
                try:
                    item = build_prompt_item(s, image_root, processor, cfg)
                except Exception as e:
                    print(f"[warn] skipping {s.get('id')}: {e}")
                    continue
                input_ids = item["input_ids"].unsqueeze(0).to(device)
                attn = item["attention_mask"].unsqueeze(0).to(device)
                extra = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["extra"].items()}
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids, attention_mask=attn, **extra,
                        do_sample=True, temperature=gen_defaults["temperature"],
                        top_p=gen_defaults["top_p"], num_return_sequences=G,
                        max_new_tokens=gen_defaults["max_new_tokens"],
                        no_repeat_ngram_size=gen_defaults.get("no_repeat_ngram_size", 0),
                        repetition_penalty=1.0,  # never >1.0 -- corrupts digit logits, see README
                        pad_token_id=pad_id, return_dict_in_generate=False)
                comp_ids = gen[:, input_ids.shape[1]:]
                # skip_special_tokens=False: a thinking model's <think>/</think>
                # tags must survive decoding, or downstream parsing silently
                # sees empty output. See module docstring's decode gotcha.
                texts = processor.tokenizer.batch_decode(comp_ids, skip_special_tokens=False)

                comps, rewards, stats = [], [], []
                for row, txt in zip(comp_ids, texts):
                    row = row[row != pad_id] if pad_id is not None else row
                    pred = parse_pred_boxes(txt)
                    r = compute_reward(pred, item["gt"], w)
                    comps.append(row.cpu()); rewards.append(r["reward"]); stats.append(r)
                rw = torch.tensor(rewards)
                adv = (rw - rw.mean()) / (rw.std() + 1e-4)
                rollouts.append({"item": item, "comps": comps, "adv": adv,
                                 "rewards": rw, "stats": stats})

            if not rollouts:
                continue

            # ---------------- old / ref logprobs ----------------
            for ro in rollouts:
                ro["old_lp"], ro["ref_lp"] = [], []
                for c in ro["comps"]:
                    ro["old_lp"].append(completion_logprobs(
                        model, ro["item"], c, device, no_grad=True, adapter_disabled=False).cpu())
                    ro["ref_lp"].append(completion_logprobs(
                        model, ro["item"], c, device, no_grad=True, adapter_disabled=True).cpu())

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
                    kl = (ref - lp).exp() - (ref - lp) - 1.0
                    loss = (pg + args.kl_beta * kl).mean() / n_micro
                    loss.backward()
                    tot_loss += loss.item() * n_micro
                    tot_kl += kl.mean().item()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
            opt.step()
            step += 1

            mean_r = torch.cat([ro["rewards"] for ro in rollouts]).mean().item()
            mean_iou = sum(st["mean_iou"] for ro in rollouts for st in ro["stats"]) / n_micro
            mean_f1 = sum(st["f1"] for ro in rollouts for st in ro["stats"]) / n_micro
            fmt_rate = sum(st["fmt"] for ro in rollouts for st in ro["stats"]) / n_micro
            mean_miss = sum(st["miss"] for ro in rollouts for st in ro["stats"]) / n_micro
            mean_fp = sum(st["fp"] for ro in rollouts for st in ro["stats"]) / n_micro
            rec = {"step": step, "epoch": epoch, "reward": round(mean_r, 4),
                   "mean_iou": round(mean_iou, 4), "f1@0.5": round(mean_f1, 4),
                   "miss": round(mean_miss, 3), "fp": round(mean_fp, 3),
                   "fmt_rate": round(fmt_rate, 3), "kl": round(tot_kl / n_micro, 5),
                   "loss": round(tot_loss / n_micro, 5),
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            print(f"[step {step}] " + " ".join(f"{k}={v}" for k, v in rec.items() if k != "step"))
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

            if step % args.save_every == 0 or bstart + args.prompts_per_step >= len(samples):
                ckpt = out_dir / f"grpo-step{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt))       # peft: GRPO adapter only
                processor.save_pretrained(str(ckpt))
                (ckpt / "checkpoint_meta.json").write_text(json.dumps(
                    {"model_config_path": str(cfg.path), "model_id": cfg.model_id,
                     "sft_checkpoint": args.sft_checkpoint}, indent=2))
                print(f"[ckpt] saved GRPO adapter -> {ckpt}")
                if val_samples:
                    run_val(model, processor, val_samples, image_root, args, cfg, step, metrics_f, pad_id)

    print("done.")


@torch.no_grad()
def run_val(model, processor, val_samples, image_root, args, cfg, step, metrics_f, pad_id) -> None:
    """Greedy decode, but max_new_tokens / no_repeat_ngram_size are taken
    from cfg.grpo_generation -- the SAME dict rollout sampling uses -- not
    cfg.generation_defaults (that one is for the base/SFT/GRPO eval-export
    arms, a separate concern). If these two diverged, in-training val would
    silently measure a differently-decoded policy than the one actually
    being trained against; dragon_sft_grpo's README documents losing a
    training cycle to exactly this class of mismatch. do_sample=False stays
    val's own deliberate choice (a clean, deterministic read of the current
    policy), independent of rollout's sampling temperature."""
    w = RewardWeights()
    ious, f1s = [], []
    gen_defaults = cfg.grpo_generation
    for s in val_samples:
        try:
            item = build_prompt_item(s, image_root, processor, cfg)
        except Exception:
            continue
        input_ids = item["input_ids"].unsqueeze(0).to(args.device)
        attn = item["attention_mask"].unsqueeze(0).to(args.device)
        extra = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in item["extra"].items()}
        gen = model.generate(
            input_ids=input_ids, attention_mask=attn, **extra,
            do_sample=False, num_beams=1, max_new_tokens=gen_defaults["max_new_tokens"],
            repetition_penalty=1.0, no_repeat_ngram_size=gen_defaults.get("no_repeat_ngram_size", 0),
            pad_token_id=pad_id)
        txt = processor.tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=False)
        pred = parse_pred_boxes(txt) or []
        from box_matching import hungarian_match
        matches = hungarian_match(item["gt"], pred)
        per_gt = {i: v for i, _, v in matches}
        n = max(1, len(item["gt"]))
        ious.append(sum(per_gt.get(i, 0.0) for i in range(len(item["gt"]))) / n)
        r = compute_reward(pred, item["gt"], w)
        f1s.append(r["f1"])
    rec = {"step": step, "val_mean_iou": round(sum(ious) / max(1, len(ious)), 4),
           "val_f1@0.5": round(sum(f1s) / max(1, len(f1s)), 4), "val_n": len(ious)}
    print(f"[val step {step}] " + " ".join(f"{k}={v}" for k, v in rec.items()))
    metrics_f.write(json.dumps({"val": rec}) + "\n"); metrics_f.flush()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--sft-checkpoint", default=None,
                    help="adapter dir from train_sft.py (omit to GRPO the base model directly)")
    ap.add_argument("--jsonl", required=True, help="GRPO pool, from convert_to_chat_jsonl.py")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-jsonl", default=None)
    ap.add_argument("--val-size", type=int, default=120)
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    # GRPO
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    # reward weights
    ap.add_argument("--w-iou", type=float, default=0.45)
    ap.add_argument("--w-f1", type=float, default=0.15)
    ap.add_argument("--w-fmt", type=float, default=0.05)
    ap.add_argument("--w-miss", type=float, default=0.25)
    ap.add_argument("--w-fp", type=float, default=0.10)
    return ap.parse_args()


if __name__ == "__main__":
    main()
