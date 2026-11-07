#!/usr/bin/env python3
"""The <count> evidence-demand span: prompt rewrite, target rewrite, parsing.

One module owns the format so Count-SFT, Count-GRPO and every evaluator agree
byte-for-byte. The grounding half of the output is untouched -- parse_boxes_r1
only reads <box> segments, so adding a <count> span in front cannot change any
existing box metric.

Design notes
------------
* The count span goes FIRST, before any box. That is the whole point: the model
  must commit to how much evidence the answer needs before it starts emitting
  geometry, so the demand prediction cannot be a post-hoc description of what it
  happened to draw.
* The system prompt is rewritten surgically (two exact string replacements), so
  everything else in it stays identical to the prompt the vanilla SFT/GRPO arms
  were trained on. A paraphrased prompt would confound the count ablation with a
  prompt change.
* We ASK for |P| == N but never force it at decode time (pilot spec section 5):
  count prediction and count compliance are two separate measurements and
  constrained decoding would collapse them into one.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- the format

COUNT_RE = re.compile(r"<count>\s*(\d+)\s*</count>")

# exact fragments of the frozen DRAGON grounding system prompt
_OLD_FORMAT = ("Respond with the evidence regions in this exact format:\n"
               "<ref>evidence</ref><box>[[x1, y1, x2, y2], [x1, y1, x2, y2]]</box>")
_NEW_FORMAT = ("First state how many distinct evidence regions the answer requires, then "
               "respond with those regions, in this exact format:\n"
               "<count>N</count><ref>evidence</ref><box>[[x1, y1, x2, y2], [x1, y1, x2, y2]]</box>")
_OLD_TAIL = "Do not repeat a box or include irrelevant regions."
_NEW_TAIL = ("Do not repeat a box or include irrelevant regions. N must equal the number "
             "of boxes you list.")


def add_count_instruction(system: str) -> str:
    """Rewrite the grounding system prompt to demand a leading <count> span.

    Raises if either anchor is missing: silently training on a prompt that never
    asked for a count would produce an arm that looks like A1 and is really A0.
    """
    if "<count>" in system:
        return system
    if _OLD_FORMAT not in system:
        raise ValueError("system prompt does not contain the expected format block; "
                         "count instruction would be silently dropped")
    out = system.replace(_OLD_FORMAT, _NEW_FORMAT, 1)
    if _OLD_TAIL not in out:
        raise ValueError("system prompt does not contain the expected trailing rule")
    return out.replace(_OLD_TAIL, _NEW_TAIL, 1)


def count_span(n: int) -> str:
    return f"<count>{int(n)}</count>"


def add_count_target(assistant: str, n: int) -> str:
    """Prepend the gold count span to a grounding target."""
    if assistant.startswith("<count>"):
        return assistant
    return count_span(n) + assistant


# ---------------------------------------------------------------- parsing

def parse_count(text: str) -> Optional[int]:
    """The predicted n_hat, or None if the span is absent/malformed.

    Only the FIRST span counts: a degenerate rollout that emits several is a
    format failure, not a vote, and must not be silently averaged into a number.
    """
    m = COUNT_RE.search(text)
    return int(m.group(1)) if m else None


def strip_count(text: str) -> str:
    """Text with the count span removed, for any consumer that wants the
    grounding output exactly as the vanilla arms produce it."""
    return COUNT_RE.sub("", text, count=1)


def count_span_token_end(tokenizer, ids, max_probe: int = 32) -> int:
    """Number of leading tokens of `ids` covered by the count span.

    Returns 0 when the span is absent or does not close inside `max_probe`
    tokens, which is the signal to treat the whole completion as box tokens.
    Used to route the count reward to count tokens only (pilot spec section 12:
    the count reward is never broadcast to box tokens).

    Cost is bounded: the span is ~8 tokens, so this decodes at most `max_probe`
    short prefixes, not the whole completion.
    """
    seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    for t in range(1, min(len(seq), max_probe) + 1):
        if "</count>" in tokenizer.decode(seq[:t], skip_special_tokens=False):
            return t
    return 0


# ---------------------------------------------------------------- metrics (spec section 6)

GOLD_BANDS: List[Tuple[str, int, int]] = [
    ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8),
    ("9-15", 9, 15), ("16-25", 16, 25), ("26+", 26, 10 ** 9),
]


def gold_band(n: int) -> str:
    for name, lo, hi in GOLD_BANDS:
        if lo <= n <= hi:
            return name
    return "1"


def count_metrics(pairs) -> Dict[str, float]:
    """pairs: iterable of (n_hat or None, n_gold, n_pred_boxes).

    Reports the five spec-section-6 quantities. Count accuracy/MAE/bias are
    computed over PARSED spans only (and `parse_rate` says how many that was),
    because scoring an absent span as an arbitrary number would hide a format
    collapse behind a plausible-looking MAE.
    """
    rows = list(pairs)
    n = len(rows)
    if not n:
        return {}
    parsed = [(h, g, p) for h, g, p in rows if h is not None]
    k = len(parsed)
    out = {
        "n": n,
        "parse_rate": k / n,
        "count_exact_acc": sum(1 for h, g, _ in parsed if h == g) / k if k else 0.0,
        "count_mae": sum(abs(h - g) for h, g, _ in parsed) / k if k else float("nan"),
        "count_bias": sum(h - g for h, g, _ in parsed) / k if k else float("nan"),
        # compliance: did the model emit as many boxes as it said it would
        "compliance": sum(1 for h, _, p in parsed if p == h) / k if k else 0.0,
        # gold-count compliance is defined over ALL rows: an arm that drops the
        # span entirely still has a well-defined |P| vs n
        "gold_compliance": sum(1 for _, g, p in rows if p == g) / n,
    }
    return out
