# DRAGON datasets — SFT and GRPO

Exactly what data each training stage saw, how the pools were built, every filter applied, and the measured overlaps between them. All numbers here are recomputed from the artifacts on disk, not transcribed.

---

## 1. Source data

| artifact | size | contents |
|---|---|---|
| `split_reviewed-2/` | 64 MB | 12,164 per-question annotation JSONs (+8 non-annotation files) |
| `Diagram_Attribution_Dataset/` | 2.6 GB | 40,109 source images, six domains |
| InternVL3-8B | 36 GB cached | `OpenGVLab/InternVL3-8B` from HF |

One raw record: `{image_path, q_id, question_text, choices, answers, bbox[]}`.

### Split structure

| split | files | with bbox | unique images |
|---|---|---|---|
| Train | 7,285 | 7,267 | 3,954 |
| Val | 2,434 | 2,425 | 1,326 |
| Test | 2,445 | 2,433 | 1,327 |
| **total** | **12,164** | **12,125** | |

| split | AI2D | ChartQA | Circuit-VQA | InfographicsVQA | MapIQ | MapWise |
|---|---|---|---|---|---|---|
| Train | 1400 | 1157 | 1226 | 1218 | 1122 | 1162 |
| Val | 471 | 381 | 400 | 414 | 373 | 395 |
| Test | 483 | 389 | 404 | 411 | 369 | 389 |

### Known defect: the released splits are not disjoint

Same `(image_path, q_id)` appearing in more than one split:

| pair | questions |
|---|---|
| Test + Train | 17 |
| Test + Val | 2 |
| Train + Val | 17 |
| **total** | **36** |

**19 of these touch `Test/`** — that is the leakage that can affect reported results. It is a property of the released data, not of this pipeline: no pool builder ever opens `Test/`.

Image-level overlap: {'Train&Val': 17, 'Train&Test': 17, 'Val&Test': 2}

Full file-level manifest with md5s: `dragon_datasets_manifest/split_manifest.csv` (regenerate with `python3 dragon_grpo/build_split_manifest.py`).

---

## 2. Pipeline

```
split_reviewed-2/{Train,Val}/<domain>/*.json
        |
        v  prepare_dragon6_exp2.py     (cap 500/domain, 300 MapIQ)
  dragon_datasets/<key>/samples.jsonl          2,800  <- Train
  dragon_datasets/<key>/samples_infer_holdout.jsonl  600  <- Val
        |                                   |
        v build_dragon6_sft2k.py            v build_dragon6_holdout600.py
  SFT pool  2,000                     holdout  592
        |
        v build_grpo5k_pool.py  (re-walks Train+Val, excludes both above)
  GRPO pool 5,000

split_reviewed-2/Test/ --> build_test2445.py --> test set 2,432  (no filters)
```

---

## 3. Filters, by stage

**The three pools do not use the same filters.** This is deliberate and matters for interpreting results.

| filter | Stage 1 raw | SFT pool | GRPO pool | holdout | test set |
|---|---|---|---|---|---|
| no bbox annotation | drop | drop | drop | drop | drop |
| image file missing | — | drop | drop | drop | — |
| malformed `choices` (MapIQ dict bug) | drop | — | — | — | drop |
| **>15 boxes** | — | **drop** | **KEPT** | — | **KEPT** |
| **full-image box (≥90% w AND h)** | — | **drop** | **drop** | **drop** | **KEPT** |
| **contradictory (image,question)** | — | **drop** | **drop** | **drop** | **KEPT** |
| already used by SFT / holdout | — | — | **drop** | — | — |

**Why GRPO keeps >15-box targets.** Its reward has a `w_miss` term built specifically to give gradient on dense multi-box targets; filtering those out would remove the samples that motivate the term. Consequence: **GRPO trained on denser box lists than SFT ever saw** — a confound to state alongside the data-scale difference.

**Why the test set has no filters.** Applying the training filters would score an easier subset than published baselines see. Only the 12 records with no bbox at all are dropped, since no metric can score them.

Inside `normalize_and_sort_gt_boxes()` (all pools): prefer `source=='gt'` over `pred`, dedup by box id, clamp to `[0,1000]`, drop degenerate boxes (`x2<=x1` or `y2<=y1`), sort top-to-bottom then left-to-right, collapse near-duplicates.

---

## 4. The SFT pool — 2,000

Per-domain survival through the filters (from `samples.jsonl`, 500/domain, 300 MapIQ):

| domain | raw | valid | >15 boxes | full-image | contradictory | other | **selected** |
|---|---|---|---|---|---|---|---|
| AI2D | 500 | 480 | 11 | 9 | 0 | 0 | **354** |
| ChartQA | 500 | 442 | 54 | 0 | 4 | 0 | **353** |
| Circuit-VQA | 500 | 498 | 2 | 0 | 0 | 0 | **352** |
| InfographicsVQA | 500 | 472 | 17 | 11 | 0 | 0 | **352** |
| MapIQ | 300 | 237 | 48 | 0 | 15 | 0 | **237** |
| MapWise | 500 | 487 | 12 | 0 | 0 | 1 | **352** |
| **total** | **2,800** | **2,616** | **144** | **20** | **19** | **1** | **2,000** |

Equal per-domain quota with round-robin redistribution of any shortfall. MapIQ is capacity-bound at 237 — it cannot reach the 333 an even split would give. The `>15 box` filter is by far the most aggressive (144 drops), concentrated in ChartQA (54) and MapIQ (48).

---

## 5. The GRPO pool — 5,000

Re-walks the full `split_reviewed-2` pool rather than taking leftovers, because `samples.jsonl` was itself only a 500/300-per-domain cap on a much larger source.

| domain | fresh usable | contradictory dropped | **selected** |
|---|---|---|---|
| AI2D | 1235 | 2 | **834** |
| ChartQA | 938 | 0 | **834** |
| Circuit-VQA | 1012 | 0 | **833** |
| InfographicsVQA | 1000 | 2 | **833** |
| MapIQ | 966 | 108 | **833** |
| MapWise | 955 | 0 | **833** |
| **total** | **6,106** | **112** | **5,000** |

### Why GRPO drew from `Val/`

`Train/` has 7,285 files, but only **4,408** were actually available to GRPO:

| domain | Train raw | usable after exclusions + filters |
|---|---|---|
| AI2D | 1,400 | 871 |
| ChartQA | 1,157 | 657 |
| Circuit-VQA | 1,226 | 718 |
| InfographicsVQA | 1,218 | 695 |
| MapIQ | 1,122 | 806 |
| MapWise | 1,162 | 661 |
| **total** | **7,285** | **4,408** |

Three compounding reductions: 2,800 already spent on `samples.jsonl` (excluded whole, not just the 2,000 SFT used); quality filters; and **domain balancing is the binding constraint** — at an 833/domain quota, five of six domains fall short from `Train` alone (ChartQA caps at 657). A Train-only balanced pool would cap near **6 × 657 ≈ 3,942**.

So the choice was ~3,900 balanced Train-only prompts, or 5,000 balanced using `Val/`. The builder took the second.

### Actual provenance

| pool | size | from Train | from Val | % of Train | % of Val |
|---|---|---|---|---|---|
| SFT | 2,000 | 2,000 | 0 | 27.5% | 0% |
| GRPO | 5,000 | 3,505 | 1,455 | 48.6% | 60.1% |
| combined unique | 7,000 | 5,496 | 1,462 | **75.4%** | 60.1% |

---

## 6. Evaluation sets

| set | n | source | filters |
|---|---|---|---|
| in-training val (holdout) | **592** | `Val/`, 100/domain, minus 8 contradictory | same as SFT minus >15-box |
| official test | **2,432** | `Test/` | none (2,445 − 12 no-bbox − 1 unconvertible) |

| domain | AI2D | ChartQA | Circuit-VQA | InfographicsVQA | MapIQ | MapWise |
|---|---|---|---|---|---|---|
| holdout | 100 | 98 | 100 | 100 | 94 | 100 |
| test | 483 | 388 | 398 | 411 | 363 | 389 |

---

## 7. Measured overlaps

Question-level, keyed on `(image_path, q_id)` — **not `q_id` alone**, which repeats across images within a domain (every image has its own `Q_0`).

| pair | questions | note |
|---|---|---|
| SFT ∩ GRPO | **0** | pools are fully disjoint |
| GRPO ∩ holdout | **0** | RL never trained on a val question |
| holdout ∩ Test | **0** | val set is clean w.r.t. test |
| SFT ∩ Test | **4** (0.20%) | all inherited from upstream |
| GRPO ∩ Test | **8** (0.16%) | all inherited from upstream |

Removing all 12 test-contaminated items (n = 2,420) shifts every reported metric by **≤ 0.0006**. IDs: `dragon_datasets_test2445/contaminated_ids.txt`.

### Image-level caveat

**329 of the holdout's 527 images (62.4%) also appear in the GRPO training pool** — different questions, same pictures. This follows from GRPO drawing 1,462 prompts from `Val/`, the split the holdout came from.

Consequence: **the in-training validation curve is optimistic** and is a training-progress signal, not a clean generalization measure. It likely explains the holdout-vs-test gap (F1@50 0.4174 holdout vs 0.4047 test). The official test split is unaffected — it shares only 2 images with any training data.

---

## 8. Reproducibility

All builders use `SEED = 42` with `sorted(glob)` before shuffling. **Verified empirically**: re-running `build_dragon6_sft2k.py` reproduced `raw_sft2k_records.jsonl` byte-identically (md5 `b95d4d31d9767eab0c7c8a235be5ee74` before and after).

| artifact | size | ship it? |
|---|---|---|
| `split_reviewed-2/` | 64 MB | **yes** — irreplaceable |
| images | 2.6 GB | download script (licences) |
| `dragon_datasets*/` | 54 MB | no — regenerates exactly |
| adapters (SFT + 5 GRPO ckpts) | 1.1 GB | **yes**, for exact numbers |
| `preds_for_eval/` | 41 MB | optional — CPU-only rescoring |

**Training is not bit-deterministic** (GPU nondeterminism, dataloader workers): re-running SFT/GRPO gives a similar but not identical checkpoint. Reproducing the *pipeline* needs source + code; reproducing *Table 3 exactly* needs the checkpoints.

---

## 9. Caveats to carry into the writeup

1. **GRPO used 2.5× more data than SFT** (5,000 vs 2,000) — the SFT→GRPO gain confounds objective with data scale. A clean control needs SFT@5,000 or GRPO@2,000.
2. **GRPO drew 29% of prompts from `Val/`**, SFT drew none — different source distribution.
3. **GRPO trained on >15-box targets, SFT did not** — different target density.
4. **62% image overlap** between GRPO training and the in-training val set.
5. **19 upstream questions leak into `Test/`**; 12 reached our pools, effect ≤0.0006.
6. **MapIQ is capacity-bound throughout** — 237/2,000 in SFT vs ~333 for an even split; 108 of its GRPO candidates dropped as contradictory, far more than any other domain.
