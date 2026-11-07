# CVPR experiment plan: pending runs and GPU hours (as of 2026-11-07)

The method is frozen: v4loo, prompt v4t, LOO credit, oracle warm-up 300 steps, verifier GRPO 400 steps, K = 8, seed 17.
There is no checkpoint selection: the Val600 sweep showed the final GRPO step (2633) is also the best on Val.
Test2445 is used once per row.

All hours are **single-GPU hours** (one H200-class card, 140 GB). The InternVL3.5-8B times are measured on this
machine. The Qwen times are those measurements scaled, so treat them as estimates (±50%). Main GRPO took about twice
its first estimate, so plan with the upper end. GPUs 4–7 only.

## 1. InternVL3.5-8B (main model)

### Done

| Run | Wall time | Output |
|---|---|---|
| SFT (sft2000, 3 epochs) | ~6.5 h | `i35/outputs/paper_sft` |
| Grounder GRPO (2,633 steps, vLLM standard) | 10.6 h | `outputs/paper/grpo/grpo-step2633` |
| Test2445 SFT + GRPO (greedy + 8 samples) | ~1 h | `outputs/paper/test2445/report_main` |
| Val600 checkpoint sweep (7 checkpoints) | ~1 h | `outputs/paper/val600_sweep` |
| Export of the merged Grounder | CPU | `/dev/shm/vep_merged/paper_grounder` |
| Train states (5,266 × greedy + 8 samples) | ~0.75 h | `outputs/paper/states/train.jsonl` |
| A6 Ours verifier (warm-up 300 + GRPO 400) | 2.4 h | `outputs/paper/verifier/ours` |

### Pending

| # | Run (`ARM=`) | Paper item | Depends on | GPU-h |
|---|---|---|---|---|
| 1 | `eval` | Main table Ours; A2 untrained; A3 warm-up only; Ours at K = 1, 2, 4 (analysis 2) | ours ✔ | 1.0 |
| 2 | `flat` | A4 flat credit (failed: out of memory on a shared GPU; rerun) | states ✔ | 2.5 |
| 3 | `novis` | A5 no vision (blank images) | states ✔ | 2.5 |
| 4 | `eval` (rerun) | Adds the A4 and A5 rows (skips results that already exist) | 2, 3 | 0.3 |
| 5 | `cond` | Analysis 3: wrong answer / no answer (states + verifier) | ours ✔ | 1.0 |
| 6 | `timing` | Analysis 2 cost table (run alone on a GPU) | ours ✔ | 0.2 |
| 7 | `report` | All tables + analyses 1–5 | all | CPU |
| | **Subtotal** | **6 GPU runs** | | **≈ 7.5 GPU-h** |

Wall time is about 3 h if runs 1, 2, 3, 5 go on separate GPUs at once, or about 8 h on one GPU.

### Optional (decide before the deadline)

| Run | Why | GPU-h |
|---|---|---|
| Seeds 18 and 19 for the A6 verifier + eval | Error bars on the headline method gain (cheap) | 2 × 3.4 ≈ 7 |
| Seeds 18 and 19 for Grounder GRPO + states + verifier + eval | Error bars on the whole pipeline | 2 × 15 ≈ 30 |
| Add the 1,825 unused Val items to the pool | More data; **changes the frozen recipe** | ≈ 15 (rerun from GRPO) |

## 2. Qwen-VL 7/8B (second backbone)

**Recommended backbone: Qwen3-VL-8B-Instruct.** It is already in `/mnt/shared/shared_hf_home`, and its grounding output is
0–1000 normalized `[x1, y1, x2, y2]`, the same box format as the InternVL targets, so the targets and the reward don't change.
The alternative is Qwen2.5-VL-7B-Instruct, which is also local but uses absolute pixel coordinates, so it needs a per-image
rescale in the converter and the parser.

Same protocol as InternVL: same splits (sft2000 / grpo_pool / val600 / test2445), same prompt text, final checkpoints,
seed 17, one Test pass.

### Code to write (nothing in the pipeline supports Qwen today)

| # | Item | Notes | Effort |
|---|---|---|---|
| C1 | Data converter `paper/build_data_qwen.py` | InternVL conversations → Qwen chat messages (system / user with image / assistant target). Same boxes, same answer-conditioned user turn | 0.5 day |
| C2 | SFT trainer | The InternVL fork can't train Qwen. Use HF `Qwen3VLForConditionalGeneration` + PEFT LoRA (or ms-swift / LLaMA-Factory). Match: 3 epochs, lr 3e-5, effective batch 8, LoRA on the LLM, frozen ViT; set `max_pixels` close to InternVL's 12 tiles | 1 day |
| C3 | Merge / export | Merge LoRA into the base for vLLM (replaces `export_grounder.py`) | 0.25 day |
| C4 | vLLM engine prompt path | `gc_engine.py` / `vllm_backend.py` build InternVL prompts (`<image>` expansion, `hf_prompt_ids`, `hf_pixels`). Add a Qwen branch: chat template + `qwen_vl_utils` processing, and keep the "HF ids == vLLM ids" check | 1 day |
| C5 | GRPO trainer (`train_gc.py --task coverage`) | HF policy forward for Qwen (pixel_values + `image_grid_thw`), LoRA sync to vLLM. Truncate at the first EOS (see the EOS-padding KL bug) | 1 day |
| C6 | States / verifier / eval | `generate_states.py`, `train_gc.py --task completer`, `eval_completer.py` get the backbone switch from C4/C5. The verifier prompt (v4t, two images) needs `limit_mm_per_prompt={'image': 2}` | 0.5 day |
| C7 | Scripts | `paper/run_main_qwen.sh`, `run_ours_qwen.sh` (copies with `BACKBONE=qwen3vl8b`); `report_paper.py --backbone` | 0.25 day |
| C8 | Smoke tests | 20-item SFT, 10 GRPO steps, 10 verifier steps, 50-item Test; check that box parsing round-trips | 0.5 day |
| | **Total** | | **≈ 5 working days** |

### Runs (main + the core ablation set)

| # | Run | GPU-h (estimate) |
|---|---|---|
| Q1 | Smoke tests (C8) | 1 |
| Q2 | SFT on sft2000 (3 epochs) | 6–8 |
| Q3 | Grounder GRPO, 2,633 steps (vLLM standard) | 10–15 |
| Q4 | Val600 sweep (for the appendix curve) | 1 |
| Q5 | Test2445 SFT + GRPO (greedy + 8 samples) | 1 |
| Q6 | Train states (5,266 × 9) | 1 |
| Q7 | A6 Ours verifier | 2.5–3.5 |
| Q8 | Test eval: Ours, A2, A3, K curve | 1 |
| Q9 | A4 flat + A5 no-vision (optional for the second backbone; main + A6 is the minimum) | 5–7 |
| | **Total** | **≈ 24 GPU-h minimum (Q1–Q8), ≈ 31 with Q9** |

Critical path: SFT → GRPO → states → verifier → eval, about **22–30 h of wall time** once the code is ready.
Only the SFT can use 2 GPUs; the other steps are sequential on 1 GPU.

## 3. Totals

| Block | GPU runs | GPU-h |
|---|---|---|
| InternVL pending (required) | 6 | ≈ 7.5 |
| Qwen3-VL-8B, main + A6 | 8 | ≈ 24 |
| Qwen A4/A5 ablations | 2 | ≈ 6 |
| **Required total** | **14–16** | **≈ 32–38** |
| Optional seeds (InternVL verifier only / whole pipeline) | | +7 / +30 |

## 4. Suggested schedule

1. **Now:** InternVL `eval` (GPU 5); `flat` and `novis` on whichever GPUs free up; then `cond` and `timing`; `report`.
2. **In parallel (CPU / coding, days 1–5):** Qwen C1–C8.
3. **Days 5–7:** Qwen Q2–Q8 on one GPU in sequence, with Q9 alongside on a second GPU.
4. **If time remains:** InternVL verifier seeds 18 and 19 (+7 GPU-h) for error bars.
