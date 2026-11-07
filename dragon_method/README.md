# Paper code: InternVL3.5-8B (reference for the Qwen port)

This folder is a **snapshot of every script the paper runs use**, copied from the working repos with their original
directory layout. Nothing has been changed. Use it to read the code and to see which pieces are InternVL-specific
and must be swapped for Qwen-VL. See [PAPER_SUMMARY.md](PAPER_SUMMARY.md) for the idea, the experiments and the
current results.

> The code still contains absolute paths (`/mnt/data2/traviku2/...`, `/dev/shm/vep_merged`). It runs as-is only on the
> original machine. On another machine, set the roots in `VEP/scripts/common.py` (they read the env vars `I35_ROOT`,
> `R1_ROOT`, `VEP_MERGED_ROOT`, `FLASH_PY`) and edit the `I35=`, `R1=`, `PY=` lines at the top of the three
> `paper/run_*.sh` scripts.

## 1. Pipeline at a glance

```
DRAGON reviewed split (Train 7,285 / Val 2,434 / Test 2,445; 6 domains)
  │  build_i35_split.py → sft_v4_phase0.py (convert) → paper/build_data.py
  ▼
data_paper/: sft2000.jsonl · grpo_pool.jsonl (5,266) · val600.jsonl · test2445.jsonl (+ _wrong / _none)
  │
  │  run_main.sh  ARM=sft     SFT (LoRA on LLM + ViT, full projector), 3 epochs on sft2000
  │               ARM=export  merge SFT → /dev/shm/vep_merged/paper_sft
  │               ARM=grpo    Grounder GRPO, 2,633 steps on grpo_pool (vLLM rollouts, LLM LoRA r16)
  │               ARM=test    Test2445: SFT greedy; GRPO greedy + 8 samples → official eval
  │  run_val_sweep.sh         Val600 curve over GRPO checkpoints (appendix; final step = best)
  ▼
Grounder = paper_sft + grpo-step2633
  │
  │  run_ours.sh  ARM=grounder  merge SFT+GRPO → /dev/shm/vep_merged/paper_grounder
  │               ARM=states    greedy E + 8 samples on the 5,266 train items (Editor training input)
  │               ARM=ours      Evidence Editor: oracle warm-up 300 + GRPO 400 (LOO credit)    (A6)
  │               ARM=flat      same, set reward only, no per-candidate credit                  (A4)
  │               ARM=novis     same, blank images                                              (A5)
  │               ARM=eval      Test2445: A2 untrained, A3 warm-up, A4, A5, A6, Ours at K=1/2/4
  │               ARM=cond      analysis 3: wrong / no answer given to the Grounder + Editor
  │               ARM=timing    cost per item
  │               ARM=report    tables + analyses (CPU)
```

Hyperparameters live in the run scripts; they are the frozen recipe. Don't change them for the Qwen run unless
the backbone forces it.

## 2. Files

### Entry points: `dragon_grpo_r1/paper/`

| File | What it does |
|---|---|
| `build_data.py` | Builds `data_paper/` from the i35 conversions: applies the approved Grounder prompt and adds the 13 Test items the converter dropped (gold = [], scored 0). CPU |
| `make_cond_splits.py` | Test copies with a **wrong** answer (`test2445_wrong.jsonl` + map) or **no** answer (`test2445_none.jsonl`) for analysis 3 |
| `run_main.sh` | Main pipeline: `ARM=sft / export / grpo / test` |
| `run_val_sweep.sh` | Val600 greedy eval of SFT + GRPO steps 500–2633 (checkpoint curve) |
| `run_ours.sh` | Method + ablations + analyses: `ARM=grounder / states / ours / flat / novis / eval / cond / timing / report` |
| `score_test.py` | Official eval_script on Grounder outputs (2,445 and the scorable 2,432) |
| `report_paper.py` | Main table, ablation table A0–A6, paired bootstrap CIs (Ours vs voting_drop / GRPO / A2 / A3) |
| `analyses.py` | Analyses 1–5: `latent` (union recall, oracle selection), `kcurve`, `cond`, `fail`, `qual` |
| `audit_v4loo.py` | Leakage audit of the recipe (train/test identity = (image, question)) |
| `rep_check.py` | Reproducibility check of the vLLM Grounder path |
| `EXPERIMENT_PLAN.md` | Pending runs and GPU hours, incl. the Qwen plan |

### Core training / inference: `dragon_grpo_r1/completer_pilot/`

("completer" is the internal name of the Evidence Editor.)

| File | What it does |
|---|---|
| `train_gc.py` | **One GRPO trainer for both stages.** `--task coverage` = Grounder GRPO; `--task completer` = Editor (warm-up + GRPO, `--credit loo/label`, `--verify-credit`, `--ab-weight`, `--ablate blank_visual`). Colocated vLLM engine, LoRA synced to vLLM every step, HF model for log-probs |
| `gc_engine.py` | vLLM engine wrapper + **HF ↔ vLLM prompt/pixel parity** (`hf_pixels`, `hf_prompt_ids`), multi-image prompts (diagram + crop sheet) |
| `gc_common.py` | Candidate pool from K samples (`build_pool`), Editor prompt items (`completer_item`), rewards, parsing |
| `prompts_v4.py` | Editor prompt v4/v4t, output parser (`<verify>1 keep; 2 drop; 3 fix [..]</verify><add>..</add>`), oracle edits for warm-up, **LOO credit** (`loo_scores`) |
| `prompts_v3.py` | Shared guideline text and box formatting used by v4 |
| `crops.py` | Crop sheet (2nd image: each candidate with its surroundings, numbered), reading order |
| `generate_states.py` | Grounder inference: greedy E + K samples per item → states jsonl (input to the Editor and to Test scoring) |
| `eval_completer.py` | Editor inference on states → `sel_<tag>.jsonl` (final box sets) |
| `export_grounder.py` | Merges SFT (+ optional extra LoRA) into an HF dir that vLLM serves |
| `report.py`, `report_v4.py` | Scoring helpers (official eval_script, per-item metrics, voting_drop) |
| `completer_worker.py` | Out-of-process Editor engine (only for feedback runs; not used in the paper arms) |

### Backend: `dragon_grpo_r1/demand_rl/`, `omission_pilot/`, `VEP/scripts/`

| File | What it does |
|---|---|
| `VEP/scripts/common.py` | **Model-specific constants and I/O**: chat template (`internlm2-chat`), `<img><IMG_CONTEXT>*256*tiles</img>` expansion, tiling (max 12), `load_merged_hf`, `export_merged`, eval_script writer |
| `VEP/scripts/rollout.py` | vLLM `LLM(...)` construction, sampling, completion ids / EOS truncation |
| `VEP/scripts/vep_logitsproc.py` | vLLM logits processor (no-repeat-ngram) |
| `VEP/scripts/packed.py` | Packed sequence scoring + dynamic gradient checkpointing for the HF policy |
| `demand_rl/vllm_backend.py` | `load_policy_merged` (merged weights + fresh LLM LoRA), `sync_lora` (PEFT → vLLM LoRA) |
| `demand_rl/train_omission_vllm.py` | GRPO objective (`omission_objective`) shared by `train_gc.py` |
| `demand_rl/credit.py`, `omission_pilot/omission_grpo.py` | Per-token credit assignment, IoU/matching utilities |
| `demand_rl/precompute_elements.py` | Builds `data_count/elements_grpo_rest.json` (gold element types for the answer-bearing reward) |
| `demand_rl/branch_*.py`, `count_*.py` | Imported by shared modules; not used by the paper arms |

### SFT and data conversion: `dragon_sft_grpo_internvl35_old/`

| File | What it does |
|---|---|
| `scripts/build_i35_split.py` | Official reviewed split → SFT 2,000 / GRPO pool / Val (drops items with no bbox or image) |
| `scripts/sft_v4_phase0.py` (+ `dragon_grounding_convert_core.py`, `sft_v3_patches.py`) | Converts raw items to chat records: system prompt, `<image>` + question + choices + `Correct Answer:`, target `<ref>evidence</ref><box>[[x1,y1,x2,y2],...]</box>` with 0–1000 boxes sorted by centre |
| `scripts/train_dragon6_fast_v2b.py` → `internvl_chat_finetune.py` | InternVL SFT launcher (needs the InternVL repo's `internvl_chat` on PYTHONPATH) |
| `scripts/internvl_lora_checkpoint_io.py` | Saves/loads the SFT LoRA (LLM + ViT) and projector |
| `dragon_grpo/build_test2445.py` | Builds the full official Test set |
| `dragon_grpo/grpo_dragon6_v1.py` | Grounder reward (`compute_reward`) and box parser (`parse_boxes_r1`) |
| `dragon_grpo/qa_transfer_eval.py` | QA helpers imported by `common.py` |

### Data and eval

- `dragon_grpo_r1/dragon_grpo/eval_script.py`: **official DRAGON eval** (GroupIoU, F1@50, MeanIoU; macro over domains). Use it unchanged.
- `dragon_grpo_r1/data_count/elements_grpo_rest.json`: gold element types for the 5,266 pool items.
- Not copied (large, on the original machine): `Diagram_Attribution_Dataset/` (images), `split_reviewed-2/` (official
  split), `dragon_sft_grpo_internvl35_old/data_paper/` (built splits), checkpoints.

## 3. Data format (what the models see)

One record per line (`data_paper/*.jsonl`):

```json
{"id": "ai2d::...#i0", "image": "ai2d/images/xxx.png",
 "conversations": [
   {"from": "system", "value": "Given an image, a question, and its answer, identify the minimal set of regions ... <ref>evidence</ref><box>[[x1, y1, x2, y2], ...]</box> Coordinates are integers from 0 to 1000 ..."},
   {"from": "human",  "value": "<image>\nQuestion: ...\nChoices: (A) ...; (B) ...\nCorrect Answer: Pharynx"},
   {"from": "gpt",    "value": "<ref>evidence</ref><box>[[0, 211, 331, 250], [251, 239, 388, 253]]</box>"}],
 "metadata": {"answer": "Pharynx", "image_size": [1005, 1094], "gt_boxes_norm": [[...]]}}
```

- The answer is an **input** (answer-conditioned attribution). Gold boxes are SFT targets only.
- Boxes are integers in **0–1000**, `[x1, y1, x2, y2]`, normalized to the image. Qwen3-VL uses the same convention;
  Qwen2.5-VL uses absolute pixels, so it would need a rescale in the converter and the parser.
- States (`generate_states.py` output): `id, domain, image, question, answer, gold, E (greedy boxes), samples (K box lists)`.
- Editor input (prompt `v4t`): image 1 = diagram with numbered candidates outlined; image 2 = crop sheet; text =
  question, answer, numbered candidates tagged (current) / (proposed). Output: `<verify>1 keep; 2 drop; 3 fix [x1, y1, x2, y2]</verify><add>[...]</add>`.

## 4. Where InternVL is hard-wired (what the Qwen port must replace)

| Concern | Location | InternVL behaviour |
|---|---|---|
| Chat template | `VEP/scripts/common.py: prompt_text`, `_get_conv_template`; `grpo_dragon6_v1.py` | `internlm2-chat`, EOS `<\|im_end\|>` |
| Image tokens / tiling | `common.py` (`IMG_CONTEXT`, `NUM_IMAGE_TOKEN=256`, `MAX_TILES=12`, `expand_image`, `hf_pixel_values`) | dynamic 448px tiles + thumbnail, 256 tokens per tile |
| HF ↔ vLLM parity | `gc_engine.py: pil_pixel_values, hf_pixels, hf_prompt_ids` | must reproduce exactly the token ids vLLM conditioned on (asserted in `train_gc.py`) |
| Model load / LoRA wrap | `common.py: load_merged_hf`; `vllm_backend.py: load_policy_merged` (`model.wrap_llm_lora`) | InternVL `InternVLChatModel`, LoRA on `model.language_model` (q,k,v,o,gate,up,down; r16, alpha 32) |
| LoRA → vLLM | `vllm_backend.py: sync_lora`, `VC.save_vllm_lora / vllm_lora_state` | key renaming for vLLM's InternVL module names |
| Merge / export | `common.py: export_merged`, `export_grounder.py` | merges LLM LoRA + ViT LoRA + projector (`mlp1`) |
| vLLM engine args | `VEP/scripts/rollout.py` | `trust_remote_code`, `limit_mm_per_prompt={'image': 2}`, `max_model_len=9216` |
| SFT | `train_dragon6_fast_v2b.py` → `internvl_chat_finetune.py` | InternVL fork trainer; 3 epochs, lr 3e-5, eff. batch 8, max seq 4352 |
| Box parser | `grpo_dragon6_v1.py: parse_boxes_r1` | `<ref>evidence</ref><box>[[...]]</box>` |

Everything else (rewards, pool building, LOO credit, crop sheets, Editor prompt text, scoring, reports, analyses)
works on **box lists and strings** and should carry over unchanged.

Known pitfall: score only up to the first EOS. Padded EOS tails dominated the KL on InternVL3.5 before this fix.

## 5. Environment

- `da_env2` (vLLM 0.20.2, torch, transformers, peft): everything except SFT.
- `da_env_flash` (flash-attn): InternVL SFT.
- InternVL repo (`InternVL/internvl_chat`) on PYTHONPATH for SFT and for InternVL model classes.
- 1 GPU (~140 GB) per run; vLLM takes 0.25–0.3 of the card, the HF policy the rest.
