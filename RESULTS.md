# DRAGON — QA-conditioned visual grounding: results
InternVL3-8B, LoRA r16. **SFT** = 2,000-sample supervised fine-tune. **GRPO** = 5,000-prompt GRPO (2,500 steps, ~19 h) on top of that SFT checkpoint, step-2500.

> All headline numbers are on the **official DRAGON test split**, scored with the unmodified `eval_script.py`. Earlier held-out (592) numbers are kept in §8 for context but are **not** comparable to published baselines.

---

## 1. Headline — official test split

**n = 2,432 scored** (2,445 official − 12 with no bbox annotation − 1 unconvertible). No filtering applied.

| metric | SFT | GRPO-5k @2500 | Δ | rel. |
|---|---|---|---|---|
| **MeanIoU** | 0.2668 | **0.3807** | +0.1139 | +43% |
| **GroupIoU (GRIT)** | 0.2471 | **0.3259** | +0.0787 | +32% |
| **F1@50** | 0.2968 | **0.4047** | +0.1079 | +36% |
| **F1@70** | 0.2100 | **0.3074** | +0.0974 | +46% |
| **Recall@90** | 0.0705 | **0.1391** | +0.0685 | +97% |
| **MaxIoU** | 0.5213 | **0.6614** | +0.1401 | +27% |

Macro-average over the six domains.

## 2. Per-domain — official test split

### MeanIoU

| domain | n | SFT | GRPO | Δ |
|---|---|---|---|---|
| AI2D | 483 | 0.4060 | 0.5205 | +0.1145 |
| ChartQA | 388 | 0.3597 | 0.4489 | +0.0892 |
| Circuit-VQA | 398 | 0.1941 | 0.2680 | +0.0739 |
| InfographicsVQA | 411 | 0.1799 | 0.2377 | +0.0577 |
| MapIQ | 363 | 0.1845 | 0.4301 | +0.2456 |
| MapWise | 389 | 0.2769 | 0.3792 | +0.1023 |
| **macro** | **2432** | **0.2668** | **0.3807** | **+0.1139** |

### GroupIoU

| domain | n | SFT | GRPO | Δ |
|---|---|---|---|---|
| AI2D | 483 | 0.3714 | 0.4842 | +0.1128 |
| ChartQA | 388 | 0.3917 | 0.4638 | +0.0721 |
| Circuit-VQA | 398 | 0.1556 | 0.1663 | +0.0108 |
| InfographicsVQA | 411 | 0.1464 | 0.1873 | +0.0409 |
| MapIQ | 363 | 0.1630 | 0.3512 | +0.1882 |
| MapWise | 389 | 0.2548 | 0.3023 | +0.0475 |
| **macro** | **2432** | **0.2471** | **0.3259** | **+0.0787** |

### F1@50

| domain | n | SFT | GRPO | Δ |
|---|---|---|---|---|
| AI2D | 483 | 0.5025 | 0.5674 | +0.0649 |
| ChartQA | 388 | 0.4304 | 0.5391 | +0.1086 |
| Circuit-VQA | 398 | 0.1810 | 0.2098 | +0.0288 |
| InfographicsVQA | 411 | 0.1736 | 0.2603 | +0.0867 |
| MapIQ | 363 | 0.2119 | 0.4667 | +0.2548 |
| MapWise | 389 | 0.2815 | 0.3850 | +0.1034 |
| **macro** | **2432** | **0.2968** | **0.4047** | **+0.1079** |

### F1@70

| domain | n | SFT | GRPO | Δ |
|---|---|---|---|---|
| AI2D | 483 | 0.3675 | 0.4386 | +0.0712 |
| ChartQA | 388 | 0.3681 | 0.4610 | +0.0929 |
| Circuit-VQA | 398 | 0.0709 | 0.0850 | +0.0141 |
| InfographicsVQA | 411 | 0.0841 | 0.1530 | +0.0689 |
| MapIQ | 363 | 0.1607 | 0.4319 | +0.2712 |
| MapWise | 389 | 0.2086 | 0.2750 | +0.0664 |
| **macro** | **2432** | **0.2100** | **0.3074** | **+0.0974** |

### Recall@90

| domain | n | SFT | GRPO | Δ |
|---|---|---|---|---|
| AI2D | 483 | 0.0962 | 0.1565 | +0.0603 |
| ChartQA | 388 | 0.1812 | 0.2564 | +0.0752 |
| Circuit-VQA | 398 | 0.0015 | 0.0080 | +0.0064 |
| InfographicsVQA | 411 | 0.0079 | 0.0200 | +0.0121 |
| MapIQ | 363 | 0.0882 | 0.3011 | +0.2129 |
| MapWise | 389 | 0.0482 | 0.0926 | +0.0444 |
| **macro** | **2432** | **0.0705** | **0.1391** | **+0.0685** |

## 3. Matching rule — is the gain an artefact?

`eval_script.py` matches **many-to-many**: several predictions may each match the *same* GT box and all count toward precision. COCO-style protocols forbid this. Both rules, same predictions:

| domain | SFT m2m | SFT 1:1 | GRPO m2m | GRPO 1:1 | Δ m2m | Δ 1:1 |
|---|---|---|---|---|---|---|
| AI2D | 0.5025 | 0.4979 | 0.5674 | 0.5621 | +0.0649 | +0.0642 |
| ChartQA | 0.4304 | 0.4279 | 0.5391 | 0.5373 | +0.1087 | +0.1094 |
| Circuit-VQA | 0.1816 | 0.1813 | 0.2098 | 0.2074 | +0.0282 | +0.0261 |
| InfographicsVQA | 0.1737 | 0.1715 | 0.2603 | 0.2582 | +0.0866 | +0.0867 |
| MapIQ | 0.2119 | 0.2013 | 0.4667 | 0.4607 | +0.2548 | +0.2594 |
| MapWise | 0.2822 | 0.2794 | 0.3850 | 0.3812 | +0.1028 | +0.1018 |
| **macro** | **0.2970** | **0.2932** | **0.4047** | **0.4012** | **+0.1077** | **+0.1079** |

**The gain is identical under both rules** (+0.1077 vs +0.1079), so it is not duplicate-box credit.

## 4. Contamination audit

The **released DRAGON splits are not disjoint**: `(Train ∪ Val) ∩ Test = 19 questions`. Our builders read only `Train/` and `Val/`, so any leakage is inherited, not introduced.

| pool | test questions present | % of pool | inherited |
|---|---|---|---|
| SFT-2000 | 4 | 0.20% | 4/4 |
| GRPO-5000 | 8 | 0.16% | 8/8 |
| holdout-592 | **0** | 0% | — |

Removing all 12 leaked questions (n = 2,420) changes nothing material:

| metric | SFT | GRPO | Δ (clean) | Δ (full) |
|---|---|---|---|---|
| MeanIoU | 0.2667 | 0.3800 | +0.1133 | +0.1139 |
| GroupIoU | 0.2468 | 0.3252 | +0.0785 | +0.0787 |
| F1@50 | 0.2967 | 0.4042 | +0.1075 | +0.1079 |
| F1@70 | 0.2098 | 0.3070 | +0.0972 | +0.0974 |
| Recall@90 | 0.0704 | 0.1386 | +0.0682 | +0.0685 |

IDs: `dragon_datasets_test2445/contaminated_ids.txt`.

## 5. Empty predictions

| domain | n | SFT | GRPO |
|---|---|---|---|
| AI2D | 483 | 30 | 15 |
| ChartQA | 388 | 46 | 17 |
| Circuit-VQA | 398 | 17 | 15 |
| InfographicsVQA | 411 | 132 | 89 |
| MapIQ | 363 | 116 | 29 |
| MapWise | 389 | 37 | 15 |
| **total** | **2432** | **378** | **180** |

GRPO cuts unanswerable outputs 378 → 180 (52% fewer). Driven by `no_repeat_ngram_size=6`, which stops the repetition loops that truncated rollouts.

## 6. Thin-box floor — the unsolved failure

Recall@0.9 stratified by GT box min-side (test split, 15,762 GT boxes):

| bin (min side /1000) | n boxes | SFT R@0.9 | GRPO R@0.9 | Δ |
|---|---|---|---|---|
| hairline <=10 | 1162 | 0.0000 | 0.0009 | +0.0009 |
| thin 10-20 | 1680 | 0.0018 | 0.0131 | +0.0113 |
| small 20-50 | 3742 | 0.0377 | 0.0665 | +0.0288 |
| medium 50-120 | 7701 | 0.0682 | 0.1944 | +0.1262 |
| chunky >120 | 1477 | 0.1774 | 0.2884 | +0.1110 |

**Hairline boxes are pinned at ~0 regardless of training** — 1,162 boxes, R@0.9 of 0.000 → 0.001, while chunky reaches 0.288. The gain is monotone in box size.

The cause is geometric, not a model defect. For a prediction the same size as a GT box of min-side `s`, offset by `d` along the thin axis, `IoU = (s−d)/(s+d)`, so `IoU ≥ 0.9` requires `d ≤ s/19`:

| GT min side | max offset @0.5 | @0.7 | @0.9 |
|---|---|---|---|
| 6 | 2.00 | 1.06 | 0.32 |
| 8 | 2.67 | 1.41 | 0.42 |
| 10 | 3.33 | 1.76 | 0.53 |
| 25 | 8.33 | 4.41 | 1.32 |
| 60 | 20.00 | 10.59 | 3.16 |
| 150 | 50.00 | 26.47 | 7.89 |
| 300 | 100.00 | 52.94 | 15.79 |

An 8-unit text row tolerates **0.42 units** of error at τ=0.9 — sub-pixel — and IoU's gradient is exactly zero once overlap is lost, so RL receives no signal to close a near-miss on thin evidence. This motivates a scale-adaptive reward term; **not implemented**, so the headline number and any future ablation stay independently attributable.

> Caveat: the derivation assumes the prediction is the *same size* as the GT and merely offset — the best case. Mis-sized predictions face a tighter bound, so `s/19` is a ceiling on tolerance, not a typical value.

## 7. Training dynamics (held-out 592, ai2d slice)

| step | mean_iou | recall@0.9 | f1@0.5 |
|---|---|---|---|
| SFT (0) | 0.3837 | 0.0910 | 0.4162 |
| 500 | 0.4293 | 0.1065 | 0.4351 |
| 1000 | 0.4979 | 0.1379 | 0.5014 |
| 1500 | 0.4817 | 0.1512 | 0.4440 |
| 2000 | 0.4959 | 0.1460 | 0.5050 |
| 2500 | 0.5035 | 0.1757 | 0.5043 |

**These are AI2D-only.** `run_val` takes `[:val_size]` of a domain-grouped file, so the in-training series covers one domain. Valid as a consistently-measured trend; not a multi-domain result. Use `holdout600_stratified_v4.jsonl` to fix this in future runs.

Still improving at 2,500 steps (recall@0.9 rose at every checkpoint) — the run was **not saturated**.

## 8. Held-out 592 (context only — not comparable to published baselines)

| metric | SFT | GRPO @2500 | Δ |
|---|---|---|---|
| MeanIoU | 0.2806 | 0.4018 | +0.1212 |
| GroupIoU | 0.2554 | 0.3413 | +0.0859 |
| F1@50 | 0.3162 | 0.4174 | +0.1012 |
| Recall@90 | 0.0754 | 0.1495 | +0.0742 |

Drawn from `Val/`; the test-split numbers in §1 supersede these for any comparison.

## 9. Reproduction

| artifact | path |
|---|---|
| Test predictions (SFT) | `preds_for_eval/TEST2445_sft/` |
| Test predictions (GRPO) | `preds_for_eval/TEST2445_grpo5k_step2500/` |
| Contamination-free variants | `preds_for_eval/TEST2445_*_clean/` |
| Test eval set (2,432) | `dragon_datasets_test2445/` |
| Leaked test IDs | `dragon_datasets_test2445/contaminated_ids.txt` |
| GRPO checkpoints | `outputs/dragon6_grpo_5k/grpo-step{500..2500}/` |
| SFT checkpoint | `outputs/dragon6_sft2k_v4/ckpts/` |
| Thin-box analysis | `dragon_grpo/thin_analysis_TEST2445.json` |
| Training metrics | `outputs/dragon6_grpo_5k/grpo_metrics.jsonl` |

```bash
bash dragon_grpo/run_test2445_eval.sh sft  5 6 7
bash dragon_grpo/run_test2445_eval.sh grpo 5 6 7
```

See `README.md` for the full pipeline and the non-obvious failure modes (special-token decoding, val slicing, conv_style).
