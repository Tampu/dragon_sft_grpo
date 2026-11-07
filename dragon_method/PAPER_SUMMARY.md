# Paper summary: idea, experiments, results so far (InternVL3.5-8B)

Status as of 2026-11-07. All numbers are on the **full official DRAGON Test set (2,445 items)**, scored with the
unmodified official eval_script (macro over 6 domains). One seed (17). No checkpoint selection on Test.

## 1. Task

**Answer-conditioned visual evidence attribution** on diagrams (DRAGON benchmark; 6 domains: AI2D, ChartQA,
Circuit-VQA, Infographics, MapIQ, MapWise). Input: an image, a question (with choices if any) and its **answer**.
Output: the minimal set of boxes a human needs to verify that answer, i.e. the answer region, its label, and supporting
evidence such as a legend swatch, an axis tick, a comparator or an arrow. Metrics: GroupIoU (headline), F1@50
(official, many-to-many), 1:1 F1 (Hungarian at IoU 0.5), MeanIoU.

## 2. Main idea

A grounding model trained with RL already "knows" more evidence than its greedy answer shows. **Analysis 1** (below)
shows 8 samples together hit **70% of gold boxes vs 47.5% for greedy**, and the best keep/drop over those candidates
would reach GroupIoU 0.564 vs 0.349 for greedy. The problem is **selection and correction**, not generation.

We therefore split the job into two roles on **one backbone**:

1. **Grounder** = InternVL3.5-8B → SFT (2,000 items; LoRA on LLM + vision encoder, full projector) → GRPO
   (5,266 items, 2,633 steps, LLM LoRA r16, vLLM rollouts, standard per-group z-scored advantages). Merged into the
   weights. At inference it produces a greedy set **E** and **K = 8** samples.
2. **Evidence Editor** = a **rank-16 LoRA (43.6M params, ~0.5% of the model)** on the frozen Grounder. It sees the
   diagram with all candidates (E + clustered samples) numbered, plus a **crop sheet** of each candidate. For **each
   candidate** it outputs `keep / drop / fix [box]`, and it can `add` missing boxes:
   `<verify>1 keep; 2 drop; 3 fix [x1,y1,x2,y2]</verify><add>[...]</add>`.

**Training the Editor** (recipe "v4loo"):
- **Oracle-edit warm-up (300 steps, supervised):** targets are the correct edit of each train state (keep the
  candidates that hit gold, drop the rest, add hits from the pool). It is needed because the plain Grounder
  produces only 36% parseable Editor outputs with negative edit gain (−0.119). After the warm-up, ~100% of outputs
  parse. A format-only warm-up with random targets was tried in a pilot and failed: it taught random picking.
- **GRPO (400 steps) with leave-one-out (LOO) credit:** besides the set-level reward, each verdict token gets the
  change in set reward if that single decision were flipped (keep↔drop; an add is removed). So dropping a correct box
  costs what that box was worth to the set, not 1/20 of a sequence reward. It also gets an answer-bearing recall
  term (`ab-weight 0.5`). The KL reference is the warm-up adapter.

Cost: 1 greedy + 8 samples + 1 Editor pass = 10 generations per item, against 9 for the strongest same-compute
baseline (voting_drop).

## 3. Data and protocol

| Split | Items | Use |
|---|---|---|
| Official reviewed DRAGON | Train 7,285 / Val 2,434 / Test 2,445 | |
| SFT | 2,000 (from Train) | Grounder SFT |
| GRPO pool | 5,266 (rest of Train) | Grounder GRPO; also the Editor's training states |
| Val600 | 600 (from Val) | Sanity checks + checkpoint curve only |
| Test2445 | 2,445 (all official; 13 have no usable gold, scored 0) | Reported once; 2,432 version in the appendix |

- 19 Train and 9 Val items were dropped (no bbox / no image). 1,825 Val items are unused.
- Frozen method: final SFT adapter, GRPO step 2633, Editor step 400. No selection.
- Grounder prompt: approved system prompt; SFT targets sorted by box centre.

## 4. Experiments

| ID | Row | What it tests | Status |
|---|---|---|---|
| — | SFT | Supervised Grounder | ✔ |
| A0 | SFT + GRPO | RL Grounder, greedy | ✔ |
| A1 | + voting_drop | Strongest training-free baseline: keep boxes by sample agreement (tuned on train: k_add=4, k_keep=3) | ✔ |
| A2 | Untrained Editor | Same prompt and inputs, no LoRA | ✔ |
| A3 | Warm-up only | Oracle-edit SFT, no RL | ✔ |
| A4 | Flat credit | Set reward only, no per-candidate credit | running |
| A5 | No vision | Editor trained and tested with blank images | queued |
| A6 | **Ours** | Full method | ✔ |
| Appx | Val600 checkpoint curve | Is the final GRPO step a fair choice? | ✔ |
| An. 1 | Latent evidence | Greedy vs union recall over K, oracle selection | ✔ |
| An. 2 | K curve + cost | Ours at K = 1, 2, 4, 8; seconds per item | eval ✔, timing pending |
| An. 3 | Answer conditioning | Wrong / no answer given | running |
| An. 4 | Failure analysis | Keep/drop by box area, duplicates, extents | pending (report) |
| An. 5 | Qualitative | Success / failure figures | pending (report) |

## 5. Results so far

### Main table (Test2445, GroupIoU per domain; macro GroupIoU and F1@50)

| Row | AI2D | ChartQA | Circuit | Infogr. | MapIQ | MapWise | **GroupIoU** | **F1@50** | Gen. |
|---|---|---|---|---|---|---|---|---|---|
| SFT | 0.339 | 0.405 | 0.162 | 0.196 | 0.263 | 0.287 | 0.275 | 0.368 | 1 |
| A0 SFT + GRPO | 0.449 | 0.492 | 0.178 | 0.226 | 0.419 | 0.320 | 0.347 | 0.443 | 1 |
| A1 + voting_drop | 0.468 | 0.511 | 0.183 | **0.264** | 0.447 | 0.339 | 0.369 | 0.479 | 9 |
| **A6 Ours** | **0.495** | **0.531** | **0.209** | 0.256 | **0.525** | **0.347** | **0.394** | **0.492** | 10 |

Ours is best on 5 of 6 domains. Infographics is the exception: it trails voting_drop by 0.008.

### Ablations (same Grounder and candidate pool; only the Editor changes)

| Row | P | R | Kept | GroupIoU | F1@50 |
|---|---|---|---|---|---|
| A0 SFT + GRPO | 0.418 | 0.473 | 1.000 | 0.347 | 0.443 |
| A1 + voting_drop | 0.479 | 0.479 | 0.867 | 0.369 | 0.479 |
| A2 untrained Editor | 0.361 | 0.420 | 0.858 | 0.300 | 0.391 |
| A3 warm-up only | 0.498 | 0.373 | 0.615 | 0.338 | 0.427 |
| **A6 Ours** | **0.498** | **0.488** | **0.897** | **0.394** | **0.492** |

P, R: per-item 1:1 Hungarian at IoU 0.5. Kept: share of the Grounder's correct boxes still in the final set.

- **The warm-up teaches the format; RL makes the gain.** A3 is precise but drops 38% of correct boxes and ends
  below GRPO. RL restores recall while keeping the precision.
- Ours beats voting_drop on precision **and** recall, and keeps more correct boxes (90% vs 87%).

### Paired 95% CIs (item bootstrap)

| Comparison | GroupIoU | F1@50 |
|---|---|---|
| Ours − voting_drop | **+0.025** [+0.019, +0.031] | **+0.013** [+0.007, +0.020] |
| Ours − GRPO | +0.046 [+0.040, +0.053] | +0.049 [+0.043, +0.055] |
| Ours − warm-up only | +0.056 [+0.048, +0.064] | +0.065 [+0.057, +0.074] |
| Ours − untrained Editor | +0.093 [+0.085, +0.102] | +0.101 [+0.092, +0.110] |

Unchanged without the 12 Test items that overlap with train (+0.025 / +0.014).

### Grounder checkpoint curve (Val600, greedy)

| Step | 0 (SFT) | 500 | 1000 | 1500 | 2000 | 2500 | **2633** |
|---|---|---|---|---|---|---|---|
| GroupIoU | 0.249 | 0.304 | 0.299 | 0.309 | 0.332 | 0.315 | **0.338** |
| F1@50 | 0.326 | 0.383 | 0.380 | 0.397 | 0.421 | 0.408 | **0.430** |

The final step is also the best on Val (GroupIoU, F1@50, 1:1 F1), so the fixed budget coincides with the
validation-best checkpoint. MeanIoU alone peaks at step 2500.

### Analysis 1: latent evidence (Test, n = 2,432 with gold)

| K | Greedy R | Union R (E + K samples) | Oracle-selection GroupIoU | Oracle-selection F1@50 |
|---|---|---|---|---|
| 1 | 0.475 | 0.544 | 0.479 | 0.648 |
| 4 | 0.475 | 0.642 | 0.536 | 0.712 |
| 8 | 0.475 | **0.698** | **0.564** | **0.742** |

Train shows the same pattern (union R 0.699 at K = 8). Ours (0.394) closes about 21% of the greedy-to-oracle gap
(0.349 → 0.564). There is clear headroom for better selection.

## 6. Pending

- InternVL: A4 flat (running), A5 no-vision (queued), analysis 3 (running), timing; then `ARM=eval` again and `ARM=report`.
- Second backbone: **Qwen3-VL-8B-Instruct**, same protocol (see `dragon_grpo_r1/paper/EXPERIMENT_PLAN.md`;
  porting notes in `README.md` §4).
- Optional: seeds 18 and 19 for error bars.
