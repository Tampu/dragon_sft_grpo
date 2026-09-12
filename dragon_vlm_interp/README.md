# DRAGON interpretability: where is the model looking?

A separate project (sibling to `dragon_sft_grpo` and `dragon_vlm_sft_grpo`)
that explains a trained grounding model's own predictions: given an image,
question, and answer, which pixels actually drove the model to draw *this*
box? Built around a token-activation-map (TAM) style method, adapted from
`xmed-lab/TAM`, generalized to (a) this project's forced JSON-box-array
target instead of free-form captions, and (b) InternVL's real dynamic
multi-tile resolution instead of a single downsized tile.

**Why this is more than a nice picture for this specific project**: DRAGON
has ground-truth boxes for every example, which most VLM interpretability
work doesn't have. That means the heatmap can be checked against something
objective (does its mass fall inside the model's own predicted box?), not
just eyeballed. Two concrete uses beyond visualization:
- **Wrong-answer / no-answer ablation** — compare the heatmap for the
  model's real answer against a heatmap for a wrong or missing answer on
  the *same* image. A real shift in attention is stronger evidence of
  genuine QA-conditioning than an IoU-drop number alone.
- **Thin-box-floor mechanism** — check whether attention is measurably more
  diffuse specifically on the hairline/thin-box cases `dragon_sft_grpo`'s
  `RESULTS.md` already shows never improve, turning a geometric
  explanation into (or against) a mechanistic one.

---

## Layout (SOLID: interfaces first, vendored code untouched, model loading delegated)

```
dragon_vlm_interp/
  third_party/            TAM (MIT), cloned as-is, never edited --
                          reference/vendor only.
  interp/
    base.py               InterpretabilityBackend abstract interface,
                          InterpRequest / Heatmap dataclasses.
    tiling.py             Reproduces InternVL's public dynamic_preprocess
                          tiling decision + paints flat per-tile scores back
                          onto the ORIGINAL image. The one genuinely new
                          piece of engineering the vendored tool doesn't
                          solve on its own.
    box_spans.py          Maps each box's character span in the target JSON
                          text to a token-index span, for per-box (not
                          per-digit) relevance aggregation.
    tam_backend.py         TAMBackend(InterpretabilityBackend) -- reuses
                          TAM's rank_guassian_filter/least_squares, rewrites
                          the scoring driver for a forced target + multi-tile.
    registry.py            get_backend(name, cfg) factory -- Open/Closed
                          seam for adding IGOS++ or others later.
  configs/models/*.json    Per-model config: reuses dragon_vlm_sft_grpo's
                          ModelConfig schema (model_id, LoRA, generation)
                          PLUS a `tiling` block this project alone needs.
  run_interp.py            CLI entry point.
```

**Model loading is not reimplemented here.** `run_interp.py` imports
`load_model_and_processor` / `load_adapter_checkpoint` / `render_chat_text`
directly from `../dragon_vlm_sft_grpo/scripts/model_config.py` and
`lora_checkpoint_io.py` -- interpreting a checkpoint and training/
evaluating it go through the exact same loading code, so they can never
silently diverge on tiling config, LoRA merging, or chat templating.

## License note for eventual code release

`third_party/TAM` is **MIT** (stated in its README, no separate LICENSE
file). This project's own code (`interp/`, `run_interp.py`, `configs/`) is
original, written against TAM's MIT-licensed algorithm ideas (two small
utility functions, `rank_guassian_filter` and `least_squares`, are imported
directly from the vendored `tam.py`, not reimplemented) -- keep that import
path in mind if this repo is split out for a code release, since it's the
one place this project's own license choice is constrained by an upstream
one. (An earlier version of this project also vendored `LVLM_Interpretation`
(GPL-3.0) for a possible future IGOS++ backend; it was never wired up to
anything, so it was removed rather than carry an unused, more restrictive
license around -- re-clone it if that backend actually gets built.)

## Usage

```bash
python3 run_interp.py \
  --model-config configs/models/internvl3_5_8b.json \
  --sft-checkpoint ../dragon_vlm_sft_grpo/outputs/sft/final \
  --image path/to/chart.png \
  --question "What is the tallest bar?" --answer "Revenue" \
  --out-dir results/chartqa_001
```

Omit `--sft-checkpoint`/`--grpo-checkpoint` for the base/zero-shot arm.
Pass `--target-text '[[...]]'` explicitly (in this project's canonical
`grounding_prompts.build_target()` format) to run the wrong-answer
ablation -- explain a *counterfactual* box list on the same image, rather
than the model's own prediction.

Output: `heatmap_overlay.jpg` (JET colormap blended over the original
image), `heatmap.npy` (raw float32 array, full original resolution),
`meta.json` (tile count, thumbnail presence, grid size, box count --
inspect this before trusting the heatmap, see below).

---

## Before you trust a heatmap: what's verified vs. assumed

Authored without GPU/model access, same caveat as `dragon_vlm_sft_grpo`.
Two things are **checked at runtime and will raise loudly** rather than
silently produce a wrong-but-plausible-looking heatmap:

1. **Tiling replication matches reality** (`tiling.resolve_tile_layout`):
   the number of image tokens the real processor produced must be evenly
   divisible by the tile count `internvl_dynamic_tile_boxes` predicts from
   image size + config, and tokens-per-tile must be a perfect square. A
   mismatch means the replicated `dynamic_preprocess` doesn't match this
   checkpoint's real processor -- fix the `tiling` block in the model
   config, don't silence the error.
2. **Token boundary consistency** (`TAMBackend.explain`, two separate
   checks): prompt text must tokenize identically alone vs. as a prefix of
   the full sequence, AND target text must tokenize identically alone vs.
   as a suffix -- both are asserted, since `box_spans.py`'s token indices
   are computed on standalone tokenizations and only line up with the real
   forced sequence if both hold.

What is **not** independently verified: whether `OpenGVLab/InternVL3_5-8B`'s
specific `AutoProcessor` actually calls a `dynamic_preprocess` matching the
reproduced one in min/max tile count, image size, or thumbnail behavior
(check #1 above will catch a mismatch, but hasn't been run against the
real checkpoint yet); and whether `model.get_output_embeddings()` and the
`image_token_id`/`image_token_index`/`img_context_token_id` config-attribute
discovery in `tam_backend.py` actually resolve correctly for this specific
model -- both fail with a clear message (not a silent wrong guess) if they
don't, per the same discover-and-verify philosophy `dragon_vlm_sft_grpo`
uses throughout, but that message has not yet been seen or fixed against
a real run.

**Recommended first run**: one single image, one box, print `meta.json`,
and visually sanity-check the overlay before running anything at scale.

---

## If your checkpoint actually came from `dragon_sft_grpo` (the older, InternVL-specific repo), not `dragon_vlm_sft_grpo`

This matters because `dragon_vlm_sft_grpo` had unresolved errors for InternVL
and wasn't debugged in time, so the actual InternVL/InternVL3.5 SFT+GRPO
training for this project went through `dragon_sft_grpo` instead -- native
`<ref>evidence</ref><box>[[...]]</box>` target syntax, not this project's
plain-JSON convention. Everything below this repo's other checkpoints was
built assuming `dragon_vlm_sft_grpo`-shaped checkpoints; none of it is
guaranteed to work against a `dragon_sft_grpo` checkpoint without the fixes
described here. Check these three things, roughly in order of how likely
each is to block you and how cheap each is to check:

**1. Checkpoint file format -- will fail immediately, needs a small dedicated loader.**
`load_adapter_checkpoint` (imported from `dragon_vlm_sft_grpo`) expects one
`peft`-native adapter directory: `adapter_model.safetensors` +
`adapter_config.json` + a `checkpoint_meta.json` with `model_config_path`/
`model_id` keys. `dragon_sft_grpo`'s own `internvl_lora_checkpoint_io.py`
saves a different layout instead: separate `vision_lora/` and `llm_lora/`
peft-adapter subdirectories, a raw `trainable_extra.safetensors` for the
`mlp1` projector (not LoRA-wrapped, saved as a plain state dict), and a
differently-named `lora_checkpoint_meta.json` (`base_model_name_or_path`,
`use_backbone_lora`, `use_llm_lora` -- not the newer schema's keys).
Pointing `run_interp.py` at one of these directories today raises
`FileNotFoundError` looking for a `checkpoint_meta.json` that doesn't
exist there. Needed: a new loader (e.g. `interp/legacy_checkpoint_io.py`)
that, given a `dragon_sft_grpo`-style checkpoint dir, applies
`vision_lora/` and `llm_lora/` via `PeftModel.from_pretrained` (each
individually, since they're two separate adapters, not one) and loads
`trainable_extra.safetensors` into whichever submodule the base model
calls its projector (see #2) -- then merges and returns a plain model,
same contract `load_adapter_checkpoint` already returns, so nothing else
downstream needs to change.

**2. Base model class -- genuinely uncertain, but a one-line check settles it.**
The adapters were trained against whatever class `dragon_sft_grpo` actually
loaded (its own InternVL repo checkout, added to `sys.path` locally --
`InternVLChatModel`/`InternVLChatConfig`). Some InternVL model IDs on the HF
Hub have two different variants: a plain one (original custom code, needs
`trust_remote_code=True`) and a separately-suffixed `-HF` one (natively
reimplemented inside `transformers`, different internal module names --
this is literally what `LVLM_Interpretation`'s and `TAM`'s own InternVL demo
scripts load, `OpenGVLab/InternVL3_5-4B-HF` / `OpenGVLab/InternVL3-1B-hf`).
If the plain `OpenGVLab/InternVL3_5-8B` repo's remote code is the *same*
source as `dragon_sft_grpo`'s local checkout, loading it with
`trust_remote_code=True` should reconstruct an identical class -- same
`vision_model`/`language_model`/`mlp1` attribute names -- and the old
adapters apply correctly. This repo's current `configs/models/
internvl3_5_8b.json` has `"trust_remote_code": false`, which is very
likely wrong for this scenario. Check before assuming either way:
```python
model = AutoModelForImageTextToText.from_pretrained(
    "OpenGVLab/InternVL3_5-8B", trust_remote_code=True)
names = {n for n, _ in model.named_modules()}
print([n for n in ("vision_model", "language_model", "mlp1") if n in names])
```
All three present -> #1's loader will very likely work once written. Any
missing, or different names entirely (e.g. `vision_tower`,
`multi_modal_projector`) -> wrong class was loaded; the adapters will
either fail to attach or, worse, attach to the wrong modules silently --
do not proceed to a full run on that basis.

**3. Prompt / system-message convention -- separate from #1 and #2, and currently wrong regardless of how those resolve.**
`run_interp.py` hardcodes this project's `GROUNDING_SYSTEM_PROMPT` (plain-
JSON-instructing) and `build_user_prompt()` for whatever it sends the model.
A `dragon_sft_grpo` checkpoint was fine-tuned against that repo's own
system prompt (`sft_v4_phase0.GROUNDING_SYSTEM_PROMPT_V4_STAGE_A`) and its
own user-prompt builder (`sft_v3_patches.build_reasoning_prompt_v3`).
Scoring with the wrong system prompt means explaining the model under an
off-distribution instruction, even if the weights themselves loaded
correctly -- needed: an alternate prompt-building path (a flag or a
separate config field) that uses the old repo's exact system/user prompt
text for this checkpoint family. One piece of this needs no change:
`box_spans.py`'s box-finding regex only scans for bare `[x1, y1, x2, y2]`
digit groups anywhere in the text, so it already finds boxes correctly
*inside* the `<ref>...</ref><box>[[...]]</box>` wrapper without
modification.
