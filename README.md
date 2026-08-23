# DRAGON: QA-conditioned visual grounding: SFT → GRPO → eval

Self-contained copy of every script needed to reproduce the SFT run, the GRPO
run, and the evaluation, for InternVL3-8B on six diagram/chart/map QA datasets.

**The task.** Given an image + question + the correct answer, predict the
bounding box(es) that *justify* that answer. Pure grounding: the model is
never shown candidate regions to choose from, it must localise them itself.

**Coordinates** are integers in `[0, 1000]`, `xyxy`, normalized to image
width/height, sorted top-to-bottom then left-to-right, near-duplicates
collapsed. This is the single coordinate convention used end to end —
training targets, the GRPO reward, and the eval exports all live in it, so
none of those three can silently disagree.

---

## 0. Layout and how to run these copies

```
dragon_sft_grpo/
  scripts/        shared core + SFT pipeline
  dragon_grpo/    GRPO training + evaluation
  optional_diagnostics/  the investigation trail (not needed to reproduce)
  configs/        base finetune config + meta
  <symlinks>      InternVL, Diagram_Attribution_Dataset, split_reviewed-2,
                  dragon_datasets*, hf_home, outputs
```

Every script resolves data as `Path(__file__).resolve().parents[1] / <name>`,
so the directory structure above is load-bearing — **the copies are runnable
in place** thanks to the symlinks, but flattening them into one folder will
break path resolution. Verified working from this directory:

```bash
cd /mnt/data2/traviku2/dragon_sft_grpo
source ../da_env_flash/bin/activate      # flash-attn built against CUDA 12.8
```

Two caveats on the copies:

- `optional_diagnostics/*` and `dragon_grpo/check_progress.py` hardcode
  `ROOT = Path("/mnt/data2/traviku2")`, and `dragon_grpo/run_export_ckpt.sh`
  starts with `cd /mnt/data2/traviku2`. They therefore operate on the
  **original** tree, not this copy. Edit those constants if you relocate.
- `da_env_flash` is the venv that matters: flash-attn there is compiled
  against CUDA 12.8 (matching system `nvcc`). The other venv, `da_env2`, has
  torch built against cu130 and flash-attn cannot be installed into it.

### Setup after cloning

This repo contains **code only** — no images, datasets, checkpoints or model
weights (the data trees it reads are ~42 GB and are deliberately excluded via
`.gitignore`). The scripts locate data as `Path(__file__).parents[1]/<name>`,
so after cloning, recreate the links (or copy/mount the real trees) at the
repo root:

```bash
cd dragon_sft_grpo
for d in InternVL Diagram_Attribution_Dataset split_reviewed-2 dragon_datasets \
         dragon_datasets_sft2k dragon_datasets_grpo hf_home outputs; do
  ln -s /path/to/your/$d $d
done
```

| link | what it must point at |
|---|---|
| `InternVL` | the InternVL repo checkout (provides the `internvl` package) |
| `Diagram_Attribution_Dataset` | source images for all six domains |
| `split_reviewed-2` | raw per-question JSON (`Train/`, `Val/` per domain) |
| `dragon_datasets` | per-domain `samples.jsonl` + holdout (built by §2.1) |
| `dragon_datasets_sft2k` / `dragon_datasets_grpo` | generated pools (§2.2, §3.1) |
| `hf_home` | HF cache holding the InternVL3-8B snapshot |
| `outputs` | where checkpoints are written |

Python deps: `torch`, `transformers`, `peft`, `deepspeed`, `timm`, `flash-attn`,
`Pillow`, `numpy`, `scipy`, `shapely` (`shapely` is needed only by
`eval_script.py`).

---

## 1. The prompts

All three pieces below are built by
`scripts/sft_v3_patches.py` (user turn) and `scripts/sft_v4_phase0.py`
(system turn + target). SFT and GRPO use the **identical** framing — that is
what makes the GRPO adapter a continuation of the SFT policy rather than a
different task.

### System turn — `sft_v4_phase0.GROUNDING_SYSTEM_PROMPT_V4_STAGE_A`

```
Given an image, a question, and its answer, identify the minimal set of regions
a human would need to justify the answer: the region depicting the answer
itself, its text label if visibly present, and any supporting visual evidence
(legend swatch, axis/tick, comparator region, arrow/connector, or linked text).
Respond with the evidence regions in this exact format:
<ref>evidence</ref><box>[[x1, y1, x2, y2], [x1, y1, x2, y2]]</box>
Coordinates are integers from 0 to 1000, normalized to the image width and
height (x1,y1 = top-left, x2,y2 = bottom-right). List boxes top-to-bottom,
then left-to-right. Do not repeat a box or include irrelevant regions.
```

"Stage A" = box line only. `GROUNDING_SYSTEM_PROMPT_V4` (the full constant)
also appends an `Explanation:` paragraph requirement; everything here was
trained Stage A.

### User turn — `sft_v3_patches.build_reasoning_prompt_v3()`

```
<image>
Question: What does the Sun provide to the plant that the plant uses in photosynthesis?
Choices: (A) Light energy; (B) A sunny disposition; (C) Companionship; (D) Warmth
Correct Answer: Light energy
```

`Choices: None` when the dataset is open-ended (only ai2d is genuinely
multiple-choice; MapIQ's "choices" are a malformed upstream artifact and are
dropped — see `prepare_dragon6_exp2.clean_choices`).

### Assistant target — `sft_v4_phase0.build_native_target()`

```
<ref>evidence</ref><box>[[182, 5, 466, 288], [137, 212, 272, 288], [312, 210, 863, 758], [90, 753, 868, 929]]</box>
```

This is InternVL's **native grounding syntax**, not free text. `<ref>`,
`</ref>`, `<box>`, `</box>` are real special tokens the tokenizer already
carries, so the format rides the base model's pretraining prior instead of
fighting it.

> **The single most important gotcha in this repo.** Because those are
> *special tokens*, `model.chat()` deletes them —
> `modeling_internvl_chat.py` decodes with `skip_special_tokens=True` (lines
> 339, 388). Any eval that calls `model.chat()` therefore parses **zero
> boxes from every sample** and reports a perfect-looking 0/N. Always decode
> via `sft_v4_phase0.chat_keep_special_tokens()`, which is `chat()`
> reimplemented with `skip_special_tokens=False`. This cost a full
> misdiagnosed training cycle.

Each record also carries `metadata.gt_boxes_norm` — the same boxes in list
form. **The GRPO reward reads that field**, which is why reward and training
target cannot drift apart.

---

## 2. SFT

### 2.1 Raw → adapter-contract samples

```bash
python3 scripts/prepare_dragon6_exp2.py
```

Walks `split_reviewed-2/{Train,Val}/<domain>/` (per-question raw JSON),
keeps box-annotated candidates in a fixed seed-42 shuffle, and writes
`dragon_datasets/<key>/samples.jsonl` (train) and `samples_infer_holdout.jsonl`
(held out). Caps at 500/domain, 300 for MapIQ — MapIQ has only ~336 usable
box-annotated examples in its whole 1122-file train pool.

Contract schema: `{id, question, choices, answer, image_path, bbox[], explanation_raw}`.

### 2.2 Select the 2000-sample SFT pool

```bash
python3 scripts/build_dragon6_sft2k.py
```

Three filters, then domain balancing:

| filter | why |
|---|---|
| drop `> 15` boxes | annotation-noise-heavy targets the model can't reproduce; teaches grid-hedging |
| drop any box covering `≥ 90 %` of the image | a whole-image box is not localized evidence |
| drop contradictory `(image, question)` groups | same input → two different gold box sets is unfittable by definition |

Then an **equal quota per domain** (not proportional to raw size, which would
let ChartQA/ai2d dominate and skew the learned format habits), with any
shortfall redistributed round-robin. Result: exactly 2000
(354/353/352/352/237/352 — MapIQ is capacity-bound at 237).

Output: `dragon_datasets_sft2k/raw_sft2k_records.jsonl`.

### 2.3 Convert to native-syntax training JSONL

```bash
python3 scripts/sft_v4_phase0.py convert \
  --in  dragon_datasets_sft2k/raw_sft2k_records.jsonl \
  --image-root Diagram_Attribution_Dataset \
  --out dragon_datasets_sft2k/sft2k_grounding_v4.jsonl \
  --stage A
```

Also re-runs the contradiction check and **aborts** rather than writing
unfittable data (`--allow-contradictions` to override).

### 2.4 Train

```bash
python3 scripts/train_dragon6_fast_v2b.py \
  --scale 300 \
  --meta-path configs/dragon6_sft2k_v4_meta.json \
  --work-dir  outputs/dragon6_sft2k_v4 \
  --gpus 2 --cuda-visible-devices 6,7 \
  --epochs 3 --lr 3e-5 --max-seq-length 4352 \
  --dynamic-image-size --use-thumbnail --max-dynamic-patch 12 \
  --no-require-choices
```

Launcher writes a merged config and `torchrun`s
`scripts/internvl_chat_finetune.py`. What matters:

- **LoRA rank 16 on both** the ViT backbone and the LLM; standard causal-LM
  cross-entropy, no bbox-specific loss. Labels masked to `-100` outside the
  assistant turn.
- **Dynamic tiling** (`max_dynamic_patch=12` + thumbnail). Single-tile
  compresses a 2000px chart to 448px and makes axis/legend text illegible;
  `max_seq_length 4352` = 13 tiles × 256 image tokens + 1024 text budget.
- **`conv_style` must be `internlm2-chat`.** `internvl2_5` routes to
  `preprocess_internvl2_5`, which requires `human`/`gpt` roles — this
  pipeline uses `system`/`user`/`assistant` and would crash.
- `--no-require-choices` because the v4 prompt says `Choices:` while the
  launcher's sanity check greps for `Options:`.
- `--scale 300` only affects default filenames; `--meta-path`/`--work-dir`
  override what it would otherwise pick.
- The launcher sets `HF_HOME` to `<repo>/hf_home`. Without that,
  `from_pretrained` falls back to `~/.cache/huggingface` on the root disk,
  which is full — training dies with `No space left on device` mid-launch.

**Checkpoints are adapter-only** (`scripts/internvl_lora_checkpoint_io.py`):
`vision_lora/` + `llm_lora/` + `trainable_extra.safetensors` (the `mlp1`
projector, trainable but *not* LoRA-wrapped, silently lost otherwise) —
~165 MB instead of the ~16 GB that stock `save_pretrained` would serialize.

Result: `train_loss 0.686`, 3 epochs / 750 steps, ~36 min on 2×H200.

### 2.5 Build the held-out eval set

```bash
python3 scripts/build_dragon6_holdout600.py
python3 scripts/sft_v4_phase0.py convert \
  --in dragon_datasets_sft2k/raw_holdout600.jsonl \
  --image-root Diagram_Attribution_Dataset \
  --out dragon_datasets_sft2k/holdout600_grounding_v4.jsonl --stage A
```

592 samples (600 minus 8 contradictory), verified disjoint from the SFT pool
by `(image_path, id)` — **not by `id` alone**: local q_ids like `Q_0` repeat
across different images within a domain, so id-only matching silently
mismatches records.

---

## 3. GRPO

### 3.1 The 5000-prompt pool

```bash
python3 dragon_grpo/build_grpo5k_pool.py
python3 scripts/sft_v4_phase0.py convert \
  --in dragon_datasets_grpo/raw_grpo5k_records.jsonl \
  --image-root Diagram_Attribution_Dataset \
  --out dragon_datasets_grpo/grpo5k_grounding_v4.jsonl --stage A
```

Excludes everything already spent on SFT-train **and** SFT-eval, by
`(image, id)`. It re-walks the full `split_reviewed-2` pool rather than
taking "what's left in samples.jsonl", because that file was itself only a
500/300-per-domain cap on a much larger raw pool (1122–1400 files/domain).
Yields 5000 balanced 833–834 per domain.

The `>15 boxes` filter is **deliberately not applied here** — the GRPO
reward's `w_miss` term exists precisely to give gradient on dense multi-box
targets, so excluding them would defeat the point of RL over SFT.

### 3.2 Train

```bash
CUDA_VISIBLE_DEVICES=5 python3 dragon_grpo/grpo_dragon6_v1.py \
  --sft-ckpt outputs/dragon6_sft2k_v4/ckpts \
  --jsonl dragon_datasets_grpo/grpo5k_grounding_v4.jsonl \
  --image-root Diagram_Attribution_Dataset \
  --val-jsonl dragon_datasets_sft2k/holdout600_grounding_v4.jsonl \
  --hf-cache-dir hf_home \
  --output-dir outputs/dragon6_grpo_5k \
  --save-every 500 --device cuda:0
```

(`CUDA_VISIBLE_DEVICES=N` + `--device cuda:0` confines it to one GPU;
without the env var it opens a ~522 MiB CUDA context on *every* visible GPU.)

**Design.**

- **Policy** = base InternVL3-8B + SFT LoRA *merged into the weights* + a
  **fresh** GRPO LoRA (rank 16, ~40 M trainable). `mlp1` and the whole
  vision stack are frozen; GRPO trains only the new adapter.
- **Reference** = the same model with the GRPO adapter disabled
  (`peft.disable_adapter()`), i.e. exactly the SFT policy — no second model
  in memory.
- **Rollouts**: G=8 samples per prompt at temperature 1.0, group-normalized
  advantages `(r − mean)/std`, token-level PPO-clip, k3 KL penalty to the
  reference, one update per generation batch.

**Reward** (all weights are CLI flags), deliberately dense in IoU because the
failure mode was loose boxes, not wrong regions:

```
R = w_fmt  * format_valid            (0.05)   parseable <ref>/<box>
  + w_iou  * soft_recall             (0.45)   mean 1:1-matched IoU per gt box
  + w_f1   * F1@0.5                  (0.15)   balanced coverage
  − w_miss * missing_fraction        (0.25)   heaviest: recall side
  − w_fp   * spurious_fraction       (0.10)   lighter: don't reward timidity
```

`soft_recall` is dense — a box 30 % too large scores ~0.6, not 0/1 — so the
gradient says "tighten this box" rather than "you failed". `w_miss > w_fp`
on purpose: under-prediction was the observed error, so nothing should
reward staying conservative.

- **`--no-repeat-ngram-size 6`** (default). 100 % of the empty predictions
  in the first pilot were runaway repetition loops truncating at
  `max_new_tokens` — every one ~370 chars regardless of whether the target
  had 1 box or 54, which rules out both budget shortfall and abstention.
  A 4-setting sweep over 76 known-bad samples picked 6: on the sampled arm
  (the rollout regime) it gives 75 % clean vs 57 % unconstrained, versus 39 %
  at ngram=4 and 32 % at ngram=3, with the least collateral damage on
  known-good multi-box outputs. It is applied to **both** rollouts and val —
  they must match, or val measures a different decoder than training.
  Note `repetition_penalty` is pinned at 1.0 and must stay there: 1.1
  rescales digit logits and corrupts coordinates.

**Known caveat.** Rollouts are generated *with* the n-gram constraint but
`completion_logprobs` scores them *without* it, so advantages come from a
constrained sampling distribution while the gradient updates an unconstrained
one. GRPO assumes completions are drawn from the policy being optimized. This
is the most likely explanation for KL sitting at ~1.0–1.4 here versus
~0.03–0.05 in the unconstrained pilot. It did not produce collapse — val rose
at every checkpoint — but it should be stated as a limitation.

Run: 2500 steps / 5000 prompts / 1 epoch, ~19 h on one H200,
5 checkpoints × 165 MB.

---

## 4. Evaluation

### 4.1 Per-domain, the paper numbers

```bash
python3 scripts/build_dragon6_holdout600.py    # if not already built
# split the 592 into per-domain files (see §4.3), then:
bash dragon_grpo/run_export_ckpt.sh 2500 0 6 7
```

`run_export_ckpt.sh <step> <gpu>...` round-robins the six domains across the
given GPUs, runs `export_for_eval_script.py` per domain to produce
`pred_<domain>.json`, then runs `eval_script.py` over the directory. It
refuses to overwrite an existing output dir.

- `dragon_grpo/export_for_eval_script.py` — checkpoint → predictions in
  `eval_script.py`'s schema. Reports `image_width/height = 1000` because our
  coordinates are already `[0,1000]`; IoU is scale-invariant, this just keeps
  `eval_script`'s own normalisation branch well-defined. Also prints a
  bijective(1:1) vs many-to-many matching comparison — `eval_script`'s own
  R/P/F1 is many-to-many, where several predictions may each match the *same*
  gt box and all count toward precision.
- `dragon_grpo/eval_script.py` — **the official metric script, unmodified**.
  MaxIoU / MeanIoU / GroupIoU (Shapely union, GRIT-style) / Soft R,P,F1 /
  R,P,F1@τ. Needs `shapely`.

> Its `pred_*.json` glob will also pick up any sidecar file starting with
> `pred_`; that is why the matching-comparison output is named
> `<domain>_matching_compare.json` with no prefix.

### 4.2 Other analyses

```bash
python3 dragon_grpo/compare_sft_vs_grpo.py --sft-ckpt ... --grpo-ckpt ...   # same run_val metric, both checkpoints
python3 dragon_grpo/thin_box_analysis.py --preds sft=... grpo=... --out x.json   # CPU only
python3 dragon_grpo/check_progress.py --out-dir outputs/dragon6_grpo_5k     # live training monitor
```

### 4.3 The val-slicing trap

`run_val` does `val_samples = [...][: args.val_size]` and
`holdout600_grounding_v4.jsonl` is **grouped by domain, ai2d first**. With
the default `--val-size 100` the in-training val is therefore **ai2d only**,
not all six domains. The same slice-before-shuffle pattern bites
`grpo_dragon6_v1.py --max-prompts N` (`samples[:N]` runs *before*
`random.shuffle`), which is why the first 500-prompt pilot was accidentally
100 % ai2d.

Fix, no code change — a round-robin interleaved copy where any prefix is
balanced:

```
dragon_datasets_sft2k/holdout600_stratified_v4.jsonl
  --val-jsonl <that file> --val-size 120     # exactly 20 per domain
```

In-training val numbers are still valid as a *consistently measured* series;
they are just single-domain. **Per-domain checkpoint eval (§4.1) is the real
multi-domain measurement.**

---

## 5. Results

Macro-average over the six domains, 592 held-out samples, via `eval_script.py`:

| metric | SFT | GRPO @1000 | GRPO @2500 |
|---|---|---|---|
| MeanIoU | 0.2806 | 0.3689 | **0.4018** |
| F1@50 | 0.3162 | 0.3786 | **0.4174** |
| Recall@90 | 0.0754 | 0.1172 | **0.1495** |

Per-domain MeanIoU at step 2500 vs SFT: ai2d 0.407→0.532, chartqa
0.408→0.499, circuitvqa 0.206→0.302, infographics 0.197→0.227, mapiq
0.209→**0.466**, mapwise 0.256→0.385. Empty predictions on mapiq fell
35 → 4.

**Open failure — the thin-box floor.** Recall@0.9 stays ≈0.01 on circuitvqa
and infographics regardless of checkpoint. `thin_box_analysis.py` shows why:
stratified by gt min-side, R@0.9 is *exactly* 0.000 in the ≤10-unit bin for
every checkpoint (206 boxes, none cleared) while the >120 bin reaches 0.218.
The mechanism is geometric, not a model defect — for a box of min-side `s`
offset by `d` along its thin axis, `IoU = (s−d)/(s+d)`, so IoU ≥ 0.9 needs
`d ≤ s/19`: an 8-unit text row tolerates **0.42 units** of error, and IoU's
gradient is exactly zero once overlap is lost. That derivation is the
motivation for a scale-adaptive reward term (`--w-scale`), which is **not
implemented** — the 5k run deliberately stays on the base reward so the
headline number and any future ablation remain independently attributable.
