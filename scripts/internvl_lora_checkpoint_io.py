#!/usr/bin/env python3
"""
Adapter-only save/load for InternVLChatModel LoRA training runs.

Why this exists: InternVLChatModel (InternVL/internvl_chat/.../modeling_internvl_chat.py)
does not override save_pretrained, so the stock `trainer.save_model()` ->
`PreTrainedModel.save_pretrained()` path serializes the FULL merged state dict
(base weights + LoRA deltas together) -- ~16GB for InternVL3-8B in bf16, even
though wrap_backbone_lora/wrap_llm_lora + freeze logic mean only the LoRA
adapters (lora_A/lora_B in vision_model/language_model) and the mlp1
projector are actually trainable. This module saves/loads just those pieces
(tens of MB instead of ~16GB) and is shared by the trainer and both
inference scripts so save/load can't silently drift apart.

Layout written under `output_dir`:
  vision_lora/                 # PeftModel adapter (only if use_backbone_lora)
  llm_lora/                    # PeftModel adapter (only if use_llm_lora)
  trainable_extra.safetensors  # mlp1 (vision->LLM projector) state dict
  config.json                  # full InternVLChatConfig (needed to rebuild architecture/shapes)
  lora_checkpoint_meta.json    # base_model_name_or_path + LoRA ranks used
  tokenizer files              # from tokenizer.save_pretrained
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file


def _ensure_internvl_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1] / "InternVL"
    if repo_root.exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # internvl_chat/internvl/model/__init__.py does its own absolute
    # `from internvl.model... import ...`, which needs internvl_chat/ itself
    # (not just its InternVL/ parent) on sys.path -- matches the PYTHONPATH
    # the training launcher sets for internvl_chat_finetune.py.
    internvl_chat_dir = repo_root / "internvl_chat"
    if internvl_chat_dir.exists() and str(internvl_chat_dir) not in sys.path:
        sys.path.insert(0, str(internvl_chat_dir))


def save_lora_checkpoint(model, tokenizer, output_dir, model_args) -> None:
    """Adapter-only save: tens of MB instead of the ~16GB full merged state dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_backbone_lora = int(getattr(model_args, "use_backbone_lora", 0) or 0)
    use_llm_lora = int(getattr(model_args, "use_llm_lora", 0) or 0)

    if use_backbone_lora:
        model.vision_model.save_pretrained(str(output_dir / "vision_lora"))
    if use_llm_lora:
        model.language_model.save_pretrained(str(output_dir / "llm_lora"))

    # mlp1 (vision -> LLM projector) is trainable but not LoRA-wrapped by
    # wrap_backbone_lora/wrap_llm_lora -- save it explicitly or it's lost.
    save_file(model.mlp1.state_dict(), str(output_dir / "trainable_extra.safetensors"))

    model.config.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "base_model_name_or_path": model_args.model_name_or_path,
        "use_backbone_lora": use_backbone_lora,
        "use_llm_lora": use_llm_lora,
    }
    (output_dir / "lora_checkpoint_meta.json").write_text(json.dumps(meta, indent=2))

    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"[save_lora_checkpoint] wrote adapter-only checkpoint to {output_dir} "
          f"({total_bytes / 1e6:.1f} MB)")


def load_lora_checkpoint(
    checkpoint_dir,
    device: str,
    dtype: torch.dtype,
    hf_cache_dir: Optional[str] = None,
    merge_lora: bool = True,
):
    """Reconstruct a model from base weights (pristine, un-LoRA'd key names) + saved adapters.

    We deliberately build the base model with use_backbone_lora/use_llm_lora
    zeroed first so from_pretrained's state-dict key matching lines up with
    the base checkpoint's un-prefixed parameter names, then wrap with
    PeftModel.from_pretrained (the standard peft reload idiom) so the
    adapter's own saved key prefixes match what get_peft_model produces.
    Constructing straight from a config that already has use_*_lora set
    would auto-wrap LoRA *before* the base weights are loaded, and the
    resulting `base_model.model.*` key prefixes would silently fail to match
    the pristine base checkpoint's keys, loading random base weights.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    meta = json.loads((checkpoint_dir / "lora_checkpoint_meta.json").read_text())

    _ensure_internvl_repo_on_path()
    from internvl_chat.internvl.model.internvl_chat.configuration_internvl_chat import (
        InternVLChatConfig,
    )
    from internvl_chat.internvl.model.internvl_chat.modeling_internvl_chat import (
        InternVLChatModel,
    )
    from peft import PeftModel
    from transformers import AutoTokenizer

    config = InternVLChatConfig.from_pretrained(str(checkpoint_dir))
    config.use_backbone_lora = 0
    config.use_llm_lora = 0

    model = InternVLChatModel.from_pretrained(
        meta["base_model_name_or_path"],
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=hf_cache_dir,
    )

    if meta.get("use_backbone_lora"):
        model.vision_model = PeftModel.from_pretrained(model.vision_model, str(checkpoint_dir / "vision_lora"))
        if merge_lora:
            model.vision_model = model.vision_model.merge_and_unload()
    if meta.get("use_llm_lora"):
        model.language_model = PeftModel.from_pretrained(model.language_model, str(checkpoint_dir / "llm_lora"))
        if merge_lora:
            model.language_model = model.language_model.merge_and_unload()

    extra_path = checkpoint_dir / "trainable_extra.safetensors"
    if extra_path.exists():
        missing, unexpected = model.mlp1.load_state_dict(load_file(str(extra_path)), strict=True)
        if missing or unexpected:
            print(f"[load_lora_checkpoint] mlp1 load mismatch: missing={missing} unexpected={unexpected}")

    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True, use_fast=False)
    return model, tokenizer
