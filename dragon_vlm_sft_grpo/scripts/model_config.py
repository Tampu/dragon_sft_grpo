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
  lora.target_modules "all-linear" (default, SFT stage) uses peft's built-in
                      auto-discovery of every nn.Linear submodule -- vision
                      tower, connector/projector, and language model alike
                      -- so there's no need to hand-guess module names per
                      architecture. Broad capacity is fine here: SFT's
                      gradient is dense and well-labeled.
  grpo_lora           SEPARATE, narrower scope for the GRPO stage's fresh
                      adapter, matching dragon_sft_grpo's proven design:
                      "mlp1 and the whole vision stack are frozen; GRPO
                      trains only the new adapter" on the language model.
                      RL's reward signal is sparse and noisy compared to
                      SFT's; letting a vision encoder drift under it is a
                      known way to quietly degrade grounding quality even
                      as some proxy reward rises. Defaults
                      target_modules to the sentinel "language-model-linear"
                      -- resolved at runtime by grepping the LOADED model's
                      named_modules() for nn.Linear layers whose dotted name
                      contains "language_model" (the attribute name HF's
                      unified VLM refactor uses across LlavaForConditional
                      Generation/Qwen*VLForConditionalGeneration/InternVL
                      ForConditionalGeneration-style ports) -- not a
                      per-architecture hardcode, but also not silently
                      falling back to "all-linear" if that name doesn't
                      match: it raises with the actual top-level submodule
                      names so you can set grpo_lora.target_modules
                      explicitly for a model that names things differently.
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
    "rank_pattern": {},
    "alpha_pattern": {},
}
_DEFAULT_GRPO_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": "language-model-linear",  # resolved by apply_lora(); see module docstring
    "modules_to_save": None,
    "rank_pattern": {},
    "alpha_pattern": {},
}
_LANGUAGE_MODEL_MARKER = "language_model"


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
    image_processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    image_processor_attrs: Dict[str, Any] = field(default_factory=dict)
    lora: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_LORA))
    grpo_lora: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_GRPO_LORA))
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
    grpo_lora = {**_DEFAULT_GRPO_LORA, **raw.get("grpo_lora", {})}
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
        image_processor_kwargs=raw.get("image_processor_kwargs", {}) or {},
        image_processor_attrs=raw.get("image_processor_attrs", {}) or {},
        lora=lora,
        grpo_lora=grpo_lora,
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
    """Image resolution/tiling is controlled here, generically, via two
    config fields -- neither hardcodes a family's knob names:

      image_processor_kwargs  passed straight into
                               `processor_cls.from_pretrained(model_id, **kwargs)`.
                               Covers processors that take sizing as
                               CONSTRUCTOR kwargs, e.g. Qwen-VL-family's
                               {"min_pixels": ..., "max_pixels": ...}.

      image_processor_attrs   applied via setattr() on `processor.
                               image_processor` AFTER loading. Covers
                               processors that expose sizing as a mutable
                               attribute instead (e.g. some models'
                               `image_processor.size = {"height":H,"width":W}`
                               or a tiling-count field).

    Both default to {} (whatever the checkpoint's own preprocessor_config.
    json ships), so setting neither reproduces prior behavior exactly."""
    import transformers

    model_cls = getattr(transformers, cfg.model_class)
    processor_cls = getattr(transformers, cfg.processor_class)

    processor = processor_cls.from_pretrained(
        cfg.model_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code,
        cache_dir=hf_cache_dir, **cfg.image_processor_kwargs,
    )
    if cfg.image_processor_attrs:
        target = getattr(processor, "image_processor", processor)
        for attr, value in cfg.image_processor_attrs.items():
            if not hasattr(target, attr):
                print(f"[load_model_and_processor] [warn] {type(target).__name__} has no "
                      f"attribute {attr!r} -- image_processor_attrs entry ignored, check spelling")
                continue
            setattr(target, attr, value)
            print(f"[load_model_and_processor] set {type(target).__name__}.{attr} = {value!r}")
    kwargs = dict(
        revision=cfg.revision, torch_dtype=cfg.torch_dtype_obj,
        trust_remote_code=cfg.trust_remote_code, cache_dir=hf_cache_dir,
        low_cpu_mem_usage=True, device_map={"": device},
    )
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    model = model_cls.from_pretrained(cfg.model_id, **kwargs)
    return model, processor


def _all_linear_names(model) -> List[str]:
    import torch.nn as nn
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _language_model_linear_names(model) -> List[str]:
    """Every nn.Linear submodule whose dotted name contains "language_model"
    -- the attribute name HF's unified VLM refactor uses across ported
    architectures (Llava/Qwen*VL/InternVL-style ForConditionalGeneration
    classes all expose .language_model, .vision_tower, .multi_modal_
    projector as top-level attributes). Generic discovery, not a
    per-architecture hardcode -- but if a given model doesn't follow that
    convention this returns empty and the caller fails loudly rather than
    silently LoRA-wrapping the wrong (or no) layers."""
    return [name for name in _all_linear_names(model) if _LANGUAGE_MODEL_MARKER in name]


def _report_pattern_matches(model, target_modules, patterns: Dict[str, Any], label: str) -> None:
    """Print how many discovered Linear layers each rank_pattern/alpha_pattern
    regex actually matches, BEFORE training starts. rank_pattern/alpha_pattern
    entries that match zero layers don't error (peft just applies the
    uniform base r/alpha everywhere in that case) -- which means a typo'd
    or wrong-for-this-model regex fails SILENTLY unless something prints
    the match count. This is exactly the failure class that caused the
    original GRPO-scope regression, so for a differential-rank config we
    verify it against the actual loaded model instead of trusting the
    regex blind."""
    import re

    candidates = _all_linear_names(model) if target_modules == "all-linear" else list(target_modules)
    for pattern, value in patterns.items():
        n_matched = sum(1 for name in candidates if re.search(pattern, name))
        tag = "OK" if n_matched > 0 else "WARNING: matched 0 layers -- regex likely wrong for this model"
        print(f"[apply_lora] {label} pattern {pattern!r} -> {value}: "
              f"matches {n_matched}/{len(candidates)} candidate Linear layers ({tag})")
    if candidates:
        sample = sorted(set(n.split(".")[0] for n in candidates))
        print(f"[apply_lora] top-level submodule names present on this model: {sample}")


def apply_lora(model, cfg: ModelConfig, lora_key: str = "lora"):
    """Wrap `model` in a PeftModel using the LoRA settings under
    `cfg.<lora_key>` -- "lora" (broad, default target_modules="all-linear":
    vision tower + connector + language model) for the SFT stage,
    "grpo_lora" (narrow, default target_modules="language-model-linear":
    language model ONLY, vision tower and connector left frozen) for the
    GRPO stage. See module docstring for why these two stages deliberately
    use different scopes.

    rank_pattern / alpha_pattern (both default {}): peft's own per-module
    rank/alpha override mechanism -- a dict of {regex: value}, matched by
    re.search against each LoRA-wrapped module's dotted name, letting ONE
    adapter use a smaller rank on some layers than others. This is how to
    run a "give vision SOME capacity, but less than language" GRPO ablation
    without going all the way to uniform all-linear: set grpo_lora.
    target_modules="all-linear" and grpo_lora.rank_pattern to a regex
    matching your model's actual vision submodule name (NOT auto-resolved
    the way "language-model-linear" is -- vision submodule naming is less
    consistent across model families than the language_model convention,
    so this is left as an explicit, verified-at-runtime regex rather than
    another guessed sentinel). The match-count report below tells you
    whether your regex actually hit anything on THIS model before you
    spend GPU time training with it."""
    from peft import LoraConfig, get_peft_model

    lora_kwargs = dict(getattr(cfg, lora_key))
    target_modules = lora_kwargs.pop("target_modules", "all-linear")
    modules_to_save = lora_kwargs.pop("modules_to_save", None)
    rank_pattern = lora_kwargs.pop("rank_pattern", {}) or {}
    alpha_pattern = lora_kwargs.pop("alpha_pattern", {}) or {}

    if target_modules == "language-model-linear":
        target_modules = _language_model_linear_names(model)
        if not target_modules:
            top_level = [n for n, _ in model.named_children()]
            raise ValueError(
                f"apply_lora(lora_key={lora_key!r}): no nn.Linear submodule name "
                f"contains {_LANGUAGE_MODEL_MARKER!r} on this model -- it doesn't "
                f"follow the .language_model naming convention this sentinel "
                f"assumes. Top-level submodules found: {top_level}. Set "
                f"model_config's {lora_key}.target_modules explicitly (a list of "
                f"module names, or 'all-linear') instead of relying on the "
                f"'language-model-linear' sentinel for this model."
            )
        print(f"[apply_lora] lora_key={lora_key!r}: LoRA-wrapping "
              f"{len(target_modules)} language-model Linear layers "
              f"(vision tower / connector frozen)")

    if rank_pattern or alpha_pattern:
        print(f"[apply_lora] lora_key={lora_key!r}: differential rank/alpha requested "
              f"(base r={lora_kwargs.get('r', 16)}) -- verifying regex matches:")
        _report_pattern_matches(model, target_modules, rank_pattern, "rank_pattern")
        _report_pattern_matches(model, target_modules, alpha_pattern, "alpha_pattern")

    lora_config = LoraConfig(
        r=lora_kwargs.get("r", 16),
        lora_alpha=lora_kwargs.get("alpha", 32),
        lora_dropout=lora_kwargs.get("dropout", 0.05),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
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
