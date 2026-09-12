#!/usr/bin/env python3
"""
TAM-style token-activation-map backend, adapted from xmed-lab/TAM (MIT
licensed) for this project's specific needs. This is NOT a call into the
vendored third_party/TAM/tam.py's TAM() function -- that function's
interface bakes in three assumptions we don't want:

  1. Single fixed-size tile (`img_scores.reshape(t_h, t_w)`) -- we need
     multi-tile dynamic resolution, hence tiling.py.
  2. Explains the model's OWN free-decoded generation, discovering round
     boundaries via `special_ids['prompt_id']`/`['answer_id']` token-id
     matching -- we explain a FORCED target sequence (the model's actual
     prediction, or a counterfactual answer for the wrong-answer ablation),
     and since we construct that sequence ourselves we already know the
     exact prompt/target boundary as a plain integer index -- no per-model
     special-token discovery needed for that part at all.
  3. LaTeX (`xelatex`) for text-token visualization -- a system dependency
     we don't need for an image-side heatmap.

What IS reused, verbatim, from tam.py: `rank_guassian_filter` (denoising)
and `least_squares` (the "estimated causal inference" de-interference
step, so a visual pattern shared by two different boxes in the same image
isn't double-counted). Everything else here is a from-scratch driver
matching TAM's core idea -- project intermediate hidden states through the
LM head, read off the target token's logit as a forward-only relevance
score, clip at zero -- to our forced-target, multi-box, multi-tile case.

Cost: ONE forward pass (no backward, no generation loop) per image,
regardless of how many boxes are being explained in it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dragon_vlm_sft_grpo" / "scripts"))
from model_config import ModelConfig, render_chat_text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "TAM"))
from tam import least_squares, rank_guassian_filter  # noqa: E402

from base import Heatmap, InterpRequest, InterpretabilityBackend
from tiling import resolve_tile_layout, unflatten_vision_scores

_IMAGE_TOKEN_ID_CANDIDATES = ("image_token_id", "image_token_index", "img_context_token_id")


def _discover_image_token_id(model) -> int:
    """Generic discovery, same philosophy as model_config.py's
    _language_model_linear_names: try known HF config attribute names in
    order, fail loudly with the actual config keys if none match, rather
    than silently guessing."""
    cfg = model.config
    for name in _IMAGE_TOKEN_ID_CANDIDATES:
        val = getattr(cfg, name, None)
        if val is not None:
            return int(val)
    available = [k for k in vars(cfg).keys() if "token" in k.lower()]
    raise ValueError(
        f"_discover_image_token_id: none of {_IMAGE_TOKEN_ID_CANDIDATES} found on "
        f"model.config. Token-related config keys present: {available}. Set the image "
        f"token id explicitly for this model rather than relying on auto-discovery."
    )


def _get_lm_head(model):
    """model.get_output_embeddings() is the standard HF PreTrainedModel API
    for this -- works generically through a PeftModel's attribute
    forwarding too, unlike hardcoding `.lm_head` (which may live at
    `model.language_model.lm_head` on a composite VLM instead of the top
    level -- get_output_embeddings() abstracts that away for us)."""
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("_get_lm_head: model.get_output_embeddings() returned None -- "
                         "this model doesn't expose a standard output projection; "
                         "TAMBackend cannot score it generically.")
    return head


class TAMBackend(InterpretabilityBackend):
    name = "tam"

    def __init__(self, cfg: ModelConfig, min_tiles: int = 1, max_tiles: int = 12,
                tile_image_size: int = 448, use_thumbnail: bool = True):
        self.cfg = cfg
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.tile_image_size = tile_image_size
        self.use_thumbnail = use_thumbnail

    @torch.no_grad()
    def explain(self, model, processor, request: InterpRequest) -> Heatmap:
        device = next(model.parameters()).device
        image_token_id = _discover_image_token_id(model)
        lm_head = _get_lm_head(model)

        # ---- build the FORCED sequence: prompt (render_chat_text, generation-prompt open) + target_text ----
        prompt_text = render_chat_text(processor, request.messages, self.cfg, add_generation_prompt=True)
        full_text = prompt_text + request.target_text
        prompt_inputs = processor(text=[prompt_text], images=[request.image], return_tensors="pt")
        full_inputs = processor(text=[full_text], images=[request.image], return_tensors="pt")

        prompt_len = prompt_inputs["input_ids"].shape[1]
        full_ids = full_inputs["input_ids"][0]
        if not torch.equal(full_inputs["input_ids"][0, :prompt_len], prompt_inputs["input_ids"][0]):
            raise ValueError(
                "TAMBackend.explain: prompt_text is not a token-exact prefix of full_text after "
                "tokenization -- the prompt/target boundary this backend assumes doesn't hold for "
                "this model's tokenizer/processor. Inspect full_ids vs prompt_inputs['input_ids'] "
                "directly before trusting any heatmap from this backend on this model."
            )
        # SECOND boundary check, distinct from the one above: box_spans.compute_box_token_spans
        # tokenizes request.target_text STANDALONE and returns indices relative to that. Those
        # indices are only valid against full_ids[prompt_len:] if target_text's tokenization is
        # IDENTICAL whether tokenized alone or as a suffix of full_text -- not guaranteed for a
        # BPE-style tokenizer (a token can merge differently across the prompt/target boundary).
        # Verify it directly rather than assuming it, since a mismatch here silently attributes
        # every box's heatmap to the wrong token positions with no visible symptom.
        target_ids_standalone = processor.tokenizer(request.target_text, add_special_tokens=False,
                                                     return_tensors="pt")["input_ids"][0]
        target_ids_in_context = full_ids[prompt_len:].cpu()
        if not torch.equal(target_ids_in_context, target_ids_standalone):
            raise ValueError(
                "TAMBackend.explain: request.target_text does not tokenize identically standalone "
                "vs. as a suffix of the full prompt+target sequence -- box_spans.py's token indices "
                "(computed on the standalone tokenization) do not line up with full_ids[prompt_len:]. "
                f"standalone={target_ids_standalone.tolist()} in_context={target_ids_in_context.tolist()}. "
                "Do not trust per-box heatmaps from this backend until this is resolved for this model "
                "(e.g. a leading-space/BPE-merge difference at the boundary)."
            )

        full_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full_inputs.items()}
        out = model(**full_inputs, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1][0]                      # (seq_len, hidden) -- last layer, batch0
        logits_at_every_position = lm_head(hidden)              # (seq_len, vocab) -- one forward pass, no backward

        n_image_tokens = int((full_ids == image_token_id).sum().item())
        layout = resolve_tile_layout(request.image, n_image_tokens, self.min_tiles, self.max_tiles,
                                     self.tile_image_size, self.use_thumbnail)
        image_positions = (full_ids == image_token_id).nonzero(as_tuple=True)[0].cpu().numpy()

        target_len = full_ids.shape[0] - prompt_len
        spans = request.box_token_spans or [(0, target_len)]   # default: explain the whole target as one unit

        per_box_scores: List[np.ndarray] = []
        img_maps: List[np.ndarray] = []
        for (t_start, t_end) in spans:
            # score = target tokens' own logit for THEMSELVES, read off at every earlier position
            # (round-0-style "logit lens": if position p's representation had to predict this exact
            # token, how strongly would it? clipped at 0, matching TAM's own convention).
            target_token_ids = full_ids[prompt_len + t_start: prompt_len + t_end]
            span_scores = torch.zeros(logits_at_every_position.shape[0], device=device)
            for offset, tok_id in enumerate(target_token_ids):
                # full_ids (and so target_token_ids) is a CPU tensor; explicit int() avoids
                # indexing a device-resident tensor with a CPU 0-dim tensor -- works either way
                # for single-element indexing, but explicit is safer than relying on that.
                tok_id = int(tok_id)
                pos = prompt_len + t_start + offset - 1  # position whose logit PREDICTS this token
                span_scores[: pos + 1] += logits_at_every_position[: pos + 1, tok_id].clamp(min=0)
            span_scores = (span_scores / max(1, len(target_token_ids))).cpu().numpy()

            img_scores = span_scores[image_positions]

            # estimated causal inference: subtract the least-squares-scaled sum of PREVIOUS boxes'
            # image-score patterns, so a visual region already attributed to an earlier box in this
            # same image isn't double-counted for this one (verbatim idea from tam.py's TAM()).
            if per_box_scores:
                w = np.array([s.sum() for s in per_box_scores])
                w = w / (w.sum() + 1e-8)
                interference = (np.stack(per_box_scores, 0) * w.reshape(-1, 1)).sum(0)
                scale = least_squares(img_scores, interference)
                img_scores = np.clip(img_scores - interference * scale, 0, None)

            per_box_scores.append(img_scores)
            img_maps.append(img_scores)

        combined = np.mean(np.stack(img_maps, 0), axis=0) if len(img_maps) > 1 else img_maps[0]
        combined = _normalize01(combined)  # normalize the flat per-token scores BEFORE tiling/filtering
        # rank_guassian_filter is a pure-Python per-pixel nested loop -- MUST run on each tile's small
        # grid_side x grid_side grid (e.g. 16x16), never on the full-resolution canvas. Passed as a hook
        # so tiling.py stays filter-agnostic; see unflatten_vision_scores' docstring.
        combined = unflatten_vision_scores(
            combined, layout, pre_upsample_filter=lambda g: rank_guassian_filter(g, kernel_size=3))

        return Heatmap(
            array=combined,
            meta={"n_tiles": layout.meta["n_tiles"], "has_thumbnail": layout.meta["has_thumbnail"],
                  "grid_side": layout.grid_side, "n_boxes_explained": len(spans),
                  "backend": self.name},
        )


def _normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
