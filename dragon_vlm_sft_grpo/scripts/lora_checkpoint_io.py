#!/usr/bin/env python3
"""
Adapter-only save/load, generic across model_configs.

Because model_config.apply_lora() wraps the WHOLE model (vision tower +
connector + language model) in one PeftModel with target_modules=
"all-linear", there is no InternVL-style split of vision_lora/ + llm_lora/ +
a hand-saved "trainable_extra" projector weight to maintain: peft's own
save_pretrained/from_pretrained already covers every LoRA-wrapped Linear
layer, connector included, in one adapter directory. This is a real
simplification unlocked by not being tied to one bespoke model repo's
training script.

Layout written under `output_dir`:
  adapter_model.safetensors / adapter_config.json   (peft's own save format)
  processor files                                   (processor.save_pretrained)
  checkpoint_meta.json                              (which model_config this
                                                       adapter was trained
                                                       against, so a later
                                                       load doesn't need to
                                                       be told again)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from model_config import ModelConfig, apply_lora, load_model_and_processor, load_model_config


def save_adapter_checkpoint(model, processor, output_dir, model_cfg: ModelConfig) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))          # peft: adapter weights only
    processor.save_pretrained(str(output_dir))
    meta = {"model_config_path": str(model_cfg.path), "model_id": model_cfg.model_id}
    (output_dir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))

    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"[save_adapter_checkpoint] wrote adapter-only checkpoint to {output_dir} "
          f"({total_bytes / 1e6:.1f} MB)")


def load_adapter_checkpoint(
    checkpoint_dir, device: str = "cuda:0", merge: bool = True,
    hf_cache_dir: Optional[str] = None, model_cfg: Optional[ModelConfig] = None,
) -> Tuple[object, object, ModelConfig]:
    """Rebuild base model + processor from `model_cfg` (or the checkpoint's
    own recorded config path if not given), then apply the saved adapter.
    merge=True folds the adapter into the base weights (for a policy you'll
    keep training a FRESH adapter on top of, e.g. SFT -> GRPO); merge=False
    keeps it as a detachable PeftModel (for a "disable_adapter()" reference
    forward pass, or for further LoRA training of the SAME adapter)."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    meta = json.loads((checkpoint_dir / "checkpoint_meta.json").read_text())
    if model_cfg is None:
        model_cfg = load_model_config(meta["model_config_path"])

    model, processor = load_model_and_processor(model_cfg, device=device, hf_cache_dir=hf_cache_dir)

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    if merge:
        model = model.merge_and_unload()
    model = model.to(device=device, dtype=model_cfg.torch_dtype_obj).eval()
    return model, processor, model_cfg
