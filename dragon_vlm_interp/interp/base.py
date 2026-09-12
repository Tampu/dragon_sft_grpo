#!/usr/bin/env python3
"""
The interpretability module's abstract seam (Dependency Inversion): every
caller (run_interp.py, and later any evaluation/ablation script) depends on
`InterpretabilityBackend`, never on a concrete backend like `TAMBackend`.
Adding a new backend (IGOS++, attention-rollout, whatever comes next) means
writing one new class implementing this interface -- Open/Closed, no
changes to run_interp.py or any existing backend.

Model/checkpoint loading is DELIBERATELY NOT part of this interface -- it
is delegated entirely to dragon_vlm_sft_grpo/scripts/model_config.py (the
same module the training/eval pipeline uses), so "interpret this checkpoint"
and "train/evaluate this checkpoint" can never silently load different
weights, tiling config, or processor behavior. This module only owns turning
a loaded (model, processor) pair + a request into a heatmap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class InterpRequest:
    """Single Responsibility: this dataclass only carries WHAT to explain,
    never HOW (that's each backend's job) or WHICH model (that's
    model_config.py's job)."""
    image: Image.Image
    messages: List[Dict[str, Any]]           # system+user turns, no assistant -- see grounding_prompts.build_chat_messages
    target_text: str                          # the exact assistant text to explain (model's own generation, or a
                                              # ground-truth/counterfactual answer for the wrong-answer ablation)
    box_token_spans: Optional[List[Tuple[int, int]]] = None
    # Optional (start, end) token-index ranges WITHIN target_text's tokenization, one per box, so relevance can be
    # aggregated per-box rather than per-digit -- see dragon_vlm_interp/README.md's "why per-digit targeting doesn't
    # work for coordinate output" note. None = explain the whole target_text span as one unit.
    max_new_tokens_hint: int = 8               # only used by backends that must generate before explaining;
                                              # ignored by backends (like ours) that force-decode target_text instead


@dataclass
class Heatmap:
    array: np.ndarray                          # H x W float32 in [0, 1], SAME size as the ORIGINAL (untiled) image
    overlay: Optional[np.ndarray] = None       # optional pre-blended BGR visualization, uint8 HxWx3
    meta: Dict[str, Any] = field(default_factory=dict)
    # diagnostic info a caller/paper-writer needs to trust the map: num_tiles, thumbnail_present, grid_side,
    # per-box sub-maps if box_token_spans was given, etc. -- populated by whichever backend produced this.


class InterpretabilityBackend(ABC):
    """Liskov substitution: any subclass must be usable wherever this type
    is expected, with the same contract -- explain() takes an already-
    loaded model+processor (never loads its own) and an InterpRequest,
    returns one Heatmap sized to the ORIGINAL image, regardless of how many
    tiles the model's own processor split it into internally."""

    name: str = "base"

    @abstractmethod
    def explain(self, model, processor, request: InterpRequest) -> Heatmap:
        ...
