#!/usr/bin/env python3
"""
Map each box's character span inside the target JSON text (e.g. the second
box in "[[1,2,3,4], [5,6,7,8]]") to a token-index span within that same
text's OWN tokenization -- this is what InterpRequest.box_token_spans needs
so TAMBackend can explain "this box" rather than "this single digit."

Requires a fast tokenizer (return_offsets_mapping=True) -- fails loudly
rather than silently mis-mapping spans for a slow tokenizer, since a wrong
span here means attributing relevance to the wrong box with no visible
symptom.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_ARRAY_RE = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")  # one [x1, y1, x2, y2]


def find_box_char_spans(target_text: str) -> List[Tuple[int, int]]:
    """Character (start, end) spans of each individual [x1,y1,x2,y2] box
    inside target_text, in order. Assumes target_text is exactly
    grounding_prompts.build_target()'s output -- a flat JSON list of
    4-int lists, nothing else."""
    return [m.span() for m in _ARRAY_RE.finditer(target_text)]


def char_spans_to_token_spans(tokenizer, text: str, char_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "char_spans_to_token_spans: this tokenizer doesn't return offset_mapping "
            "(needs a 'fast' tokenizer, e.g. AutoTokenizer(..., use_fast=True)). Cannot "
            "reliably map box character spans to token spans without it -- do not guess."
        )

    token_spans = []
    for (c_start, c_end) in char_spans:
        tok_start = next((i for i, (s, e) in enumerate(offsets) if e > c_start), None)
        tok_end = next((i for i, (s, e) in enumerate(offsets) if s >= c_end), len(offsets))
        if tok_start is None:
            raise ValueError(f"char_spans_to_token_spans: no token covers char offset {c_start} in {text!r}")
        token_spans.append((tok_start, tok_end))
    return token_spans


def compute_box_token_spans(tokenizer, target_text: str) -> List[Tuple[int, int]]:
    char_spans = find_box_char_spans(target_text)
    if not char_spans:
        return []
    return char_spans_to_token_spans(tokenizer, target_text, char_spans)
