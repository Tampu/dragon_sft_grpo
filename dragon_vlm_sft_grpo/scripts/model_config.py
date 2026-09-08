#!/usr/bin/env python3
"""
The plug-and-play seam: every script in this repo (data prep excluded --
that part is fully model-agnostic already) takes `--model-config
configs/models/<name>.json` and calls into this module instead of importing
anything model-specific. Swapping Qwen3-VL-8B-Thinking for a different HF
vision-language model means writing one new JSON file here, not touching
train_sft.py / grpo_train.py / export_for_eval.py.

What actually varies between models, captured as config fields (everything
else -- chat templating, image handling, generation, LoRA wrapping -- goes
through the same generic transformers/peft calls regardless of which model
is loaded):
  model_id           HF repo id
  model_class /
  processor_class    name of the transformers Auto* class to instantiate.
                      AutoModelForImageTextToText + AutoProcessor is the
                      modern generic pair that covers Qwen2/2.5/3-VL,
                      InternVL (native transformers port), LLaVA-family,
                      etc. without per-model code. Override for a model
                      that predates or sits outside that generic class.
  trust_remote_code  set True for a model that ships custom modeling code
                      not yet merged into transformers proper.
  enable_thinking    passed to processor.apply_chat_template(); True lets a
                      reasoning-tuned model open a <think> block during
                      generation (base-arm eval, GRPO rollouts, val/test
                      inference). Ignored harmlessly by non-thinking models
                      (see build_generation_prompt below). NEVER affects
                      the SFT training target itself -- that stays the
                      plain JSON box array (see grounding_prompts.py) since
                      we have no gold chain-of-thought to supervise against;
                      this flag only controls how prompts are rendered for
                      generation, not what the assistant turn's label is.
  lora.target_modules "all-linear" (default) uses peft's built-in
                      auto-discovery of every nn.Linear submodule -- vision
                      tower, connector/projector, and language model alike
                      -- so there's no need to hand-guess module names per
                      architecture (the InternVL pipeline's approach, and
                      the wrong one for a plug-and-play repo since every
                      model names its internals differently). Override with
                      an explicit list to restrict LoRA to a subset (e.g.
                      language-model-only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

_DEFAULT_GENERATION = {
    "max_new_tokens": 512,
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 6,
}
_DEFAULT_GRPO_GENERATION = {
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "no_repeat_ngram_size": 6,
}
_DEFAULT_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": "all-linear",
    "modules_to_save": None,
}


@dataclass
class ModelConfig:
    name: str
    model_id: str
    revision: Optional[str] = None
    model_class: str = "AutoModelForImageTextToText"
    processor_class: str = "AutoProcessor"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    enable_thinking: bool = False
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    lora: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_LORA))
    generation_defaults: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_GENERATION))
    grpo_generation: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_GRPO_GENERATION))
    path: Optional[Path] = None

    @property
    def torch_dtype_obj(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.torch_dtype]


def load_model_config(path: Union[str, Path]) -> ModelConfig:
    path = Path(path)
    raw = json.loads(path.read_text())
    lora = {**_DEFAULT_LORA, **raw.get("lora", {})}
    gen = {**_DEFAULT_GENERATION, **raw.get("generation_defaults", {})}
    grpo_gen = {**_DEFAULT_GRPO_GENERATION, **raw.get("grpo_generation", {})}
    return ModelConfig(
        name=raw.get("name", path.stem),
        model_id=raw["model_id"],
        revision=raw.get("revision"),
        model_class=raw.get("model_class", "AutoModelForImageTextToText"),
        processor_class=raw.get("processor_class", "AutoProcessor"),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        torch_dtype=raw.get("torch_dtype", "bfloat16"),
        attn_implementation=raw.get("attn_implementation"),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        chat_template_kwargs=raw.get("chat_template_kwargs", {}) or {},
        lora=lora,
        generation_defaults=gen,
        grpo_generation=grpo_gen,
        path=path,
    )


# ---------------------------------------------------------------------------
# Generic load / LoRA-wrap. No model-specific branches: every field that
# would otherwise force a branch here is a config field instead.
# ---------------------------------------------------------------------------

def load_model_and_processor(cfg: ModelConfig, device: str = "cuda:0",
                             hf_cache_dir: Optional[str] = None):
    import transformers

    model_cls = getattr(transformers, cfg.model_class)
    processor_cls = getattr(transformers, cfg.processor_class)

    processor = processor_cls.from_pretrained(
        cfg.model_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code,
        cache_dir=hf_cache_dir,
    )
    kwargs = dict(
        revision=cfg.revision, torch_dtype=cfg.torch_dtype_obj,
        trust_remote_code=cfg.trust_remote_code, cache_dir=hf_cache_dir,
        low_cpu_mem_usage=True, device_map={"": device},
    )
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    model = model_cls.from_pretrained(cfg.model_id, **kwargs)
    return model, processor


def apply_lora(model, cfg: ModelConfig):
    """Wrap the whole model (vision tower + connector + language model) in
    one PeftModel. target_modules="all-linear" means the connector/
    projector's Linear layer(s) are LoRA-wrapped automatically too -- unlike
    the InternVL pipeline, there is no separate "trainable extra" projector
    weight to save by hand (see lora_checkpoint_io.py)."""
    from peft import LoraConfig, get_peft_model

    lora_kwargs = dict(cfg.lora)
    target_modules = lora_kwargs.pop("target_modules", "all-linear")
    modules_to_save = lora_kwargs.pop("modules_to_save", None)
    lora_config = LoraConfig(
        r=lora_kwargs.get("r", 16),
        lora_alpha=lora_kwargs.get("alpha", 32),
        lora_dropout=lora_kwargs.get("dropout", 0.05),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        task_type=None,  # a vision-language causal-LM isn't peft's CAUSAL_LM
                         # generation wrapper target; leaving it unset avoids
                         # peft trying (and failing) to re-derive a text-only
                         # forward signature for a multimodal model.
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Chat templating: the one place "enable_thinking" is consumed. Wrapped in a
# capability probe rather than a hard-coded per-model branch, since not
# every processor's apply_chat_template accepts the kwarg (non-thinking
# models simply don't) -- the try/except IS the genericity mechanism here,
# not a model name check.
# ---------------------------------------------------------------------------

def _chat_template_kwargs(cfg: ModelConfig, add_generation_prompt: bool) -> Dict[str, Any]:
    kwargs = dict(cfg.chat_template_kwargs)
    kwargs["enable_thinking"] = cfg.enable_thinking
    kwargs["add_generation_prompt"] = add_generation_prompt
    return kwargs


def render_chat_text(processor, messages: List[Dict[str, Any]], cfg: ModelConfig,
                     add_generation_prompt: bool) -> str:
    """Text-only render (no tokenization, no image processing) -- used to
    find the prompt/target token-length split for label masking, and for
    logging. Falls back to dropping enable_thinking if the model's template
    doesn't accept it (plain instruct models)."""
    kwargs = _chat_template_kwargs(cfg, add_generation_prompt)
    try:
        return processor.apply_chat_template(messages, tokenize=False, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, tokenize=False, **kwargs)
