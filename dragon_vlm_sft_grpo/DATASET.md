# DRAGON datasets — this pipeline's policy

What data each stage sees, and how this differs from the `dragon_sft_grpo`
(InternVL3-8B) pipeline's approach. Source data itself (`split_reviewed-2/`,
`Diagram_Attribution_Dataset/`) is identical -- see that repo's `DATASET.md`
§1 for the full per-domain, per-split file counts and the upstream
train/val/test overlap defect (36 duplicate `(image, q_id)` pairs across
splits, 19 touching `Test/` -- inherited here unchanged since this pipeline
also never opens `Test/` when building SFT/GRPO).

---

## 1. One filter policy, everywhere

Unlike the InternVL pipeline (whose SFT/GRPO/holdout/test pools each drop a
*different* subset of >15-box targets, full-image boxes, and contradictory
duplicates -- see its `DATASET.md` §3), this pipeline applies exactly the
same rule at every stage:

| filter | SFT | GRPO | Val | Test |
|---|---|---|---|---|
| no bbox annotation | drop | drop | drop | drop |
| image file missing | drop | drop | drop | drop |
| malformed `choices` (MapIQ dict bug) | sanitize | sanitize | sanitize | sanitize |
| >15 boxes | **kept** | **kept** | **kept** | **kept** |
| full-image box (≥90% w and h) | **kept** | **kept** | **kept** | **kept** |
| contradictory (image,question) | **kept** | **kept** | **kept** | **kept** |

Rationale: SFT, GRPO, validation, and test all see the same class of
examples, so no comparison across stages (or a future base/SFT/GRPO
ablation) is distorted by which curation a given stage happened to apply.
Contradictory (image,question) duplicates -- the same input mapped to two
different gold box sets -- are not a training crash risk under this
pipeline's plain-JSON-array target (`scripts/grounding_prompts.py`): two
differently-labeled copies of the same input are just two ordinary training
examples, the same as any other label noise, so there is no need for the
InternVL pipeline's hard abort-on-contradiction check.

## 2. Original splits, used for their original purpose

`build_dragon6_raw.py` walks `split_reviewed-2/{Train,Val,Test}/<domain>`
directly -- no per-domain cap (the InternVL pipeline's `prepare_dragon6_
exp2.py` capped Train at 500/domain, 300 for MapIQ, before any further
processing; this pipeline keeps every usable file). `Train` supplies both
the SFT and GRPO pools, `Val` supplies validation, `Test` supplies the
held-out test split -- matching what each split is for, rather than
re-carving custom pools that mix `Train` and `Val` together (as the
InternVL pipeline's GRPO pool does, and documents as a caveat in its own
`DATASET.md` §5, "Why GRPO drew from Val").

## 3. SFT / GRPO: domain-balanced, disjoint, drawn only from Train

`build_dragon6_pools.py` shuffles each domain's uniformly-filtered Train
pool (seed 42), quota-allocates a domain-balanced SFT pool first (default
target 2000, equal-per-domain with round-robin shortfall redistribution --
identical algorithm to the InternVL pipeline's `allocate_quotas`), then
quota-allocates GRPO (default target 5000) from what SFT left behind.
Disjointness is structural (GRPO only ever sees post-SFT-allocation
leftovers), not an after-the-fact audit.

MapIQ is expected to be the binding constraint on both quotas, the same way
it is in the InternVL pipeline -- its Train split has far fewer
box-annotated files than the other five domains. Run
`build_dragon6_raw.py` then `build_dragon6_pools.py` and read the printed
per-domain pool sizes for your own copy of the data; they are not
hand-transcribed here since, unlike the InternVL pipeline's fixed
2000/5000 pools, this pipeline's pool sizes are a `--total-sft`/
`--total-grpo` CLI default, not a fixed artifact -- rerun with different
targets and the true numbers will differ from any snapshot written here.

## 4. Validation: domain-interleaved, not domain-grouped

`build_dragon6_val.py` writes `Val` out interleaved round-robin across the
six domains (`ai2d, chartqa, circuitvqa, infographics, mapiq, mapwise,
ai2d, ...`), so that slicing `val_samples[:val_size]` during training is
domain-balanced for ANY `val_size`. This fixes, by construction, a bug in
the InternVL pipeline's design: its holdout file was grouped by domain
(ai2d first), so its `--val-size 100` in-training validation was silently
ai2d-only (see `dragon_sft_grpo/README.md` §4.3).

## 5. Test: no filters, plus a contamination audit

`dragon_grpo/build_test_split.py` applies only the two structural filters
(no bbox, missing image) -- same policy as everywhere else in this
pipeline, so unlike the InternVL pipeline there is no special "test gets
even fewer filters than training" asymmetry to reason about; it is simply
the same policy applied to the `Test/` split. It also writes
`contaminated_ids.txt`: test questions whose `(image_path, id)` also appear
in the built SFT or GRPO pool, which can only happen via the upstream
Train/Val/Test overlap defect (this pipeline never reads `Test/` when
building SFT/GRPO). Report both the full-set number and a
contamination-excluded variant, the same practice `dragon_sft_grpo/
RESULTS.md` §4 follows.

## 6. What's genuinely different from the InternVL pipeline, for the writeup

1. **Uniform filters** (§1) -- no per-stage curation asymmetry to caveat.
2. **SFT and GRPO both draw only from `Train`** -- no Val-leakage-into-
   training-pool caveat to carry (the InternVL pipeline's GRPO pool is 29%
   sourced from `Val`, which is why its in-training holdout numbers are
   flagged as optimistic in its own `RESULTS.md`).
3. **`>15`-box targets and full-image boxes are in the SFT pool too** (not
   GRPO-only) -- if SFT quality regresses relative to a curated-SFT-pool
   run, this is the first place to look; it is a deliberate trade for
   filter uniformity, not an oversight.
4. Pool **sizes are a CLI default** (`--total-sft 2000 --total-grpo 5000`),
   not fixed constants -- state the actual sizes your run produced, not
   these defaults, in any results table.
