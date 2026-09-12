#!/usr/bin/env python3
"""
Backend factory -- the Open/Closed seam. Adding a new backend (IGOS++,
attention-rollout, whatever comes next) means adding one entry here and
writing one new class in a new file; nothing that already works changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tam_backend import TAMBackend

if TYPE_CHECKING:
    from model_config import ModelConfig
    from base import InterpretabilityBackend

_BACKENDS = {
    "tam": TAMBackend,
}


def get_backend(name: str, cfg: "ModelConfig", **kwargs) -> "InterpretabilityBackend":
    if name not in _BACKENDS:
        raise ValueError(f"get_backend: unknown backend {name!r}. Available: {list(_BACKENDS)}")
    return _BACKENDS[name](cfg, **kwargs)
