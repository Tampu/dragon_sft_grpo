#!/usr/bin/env python3
"""
Model-agnostic prompt / target format for QA-conditioned visual grounding.

Coordinate convention: [0,1000] normalized xyxy, in plain JSON text -- no
special tokens. This is the one piece of this repo that is deliberately NOT
"whatever a given model's own pretraining prefers": different HF
vision-language models disagree with each other about native grounding
syntax (InternVL's <ref>/<box> special tokens, Qwen2.5-VL/Qwen3-VL's
bbox_2d-in-resized-pixel-space JSON, others again), and a plug-and-play repo
that re-derives a bespoke convention per model defeats its own purpose --
every script downstream of this one (the GRPO reward, eval export,
eval_script.py) is written against ONE coordinate space and ONE target
syntax, and stays that way no matter which model_config is loaded. Every
LLM's pretraining includes enormous amounts of JSON, so "emit a JSON array
of 4-integer arrays" is a safe, general SFT target regardless of whether the
loaded model has any *specific* grounding prior beyond that.

Consequence to flag for the paper: a base (zero-shot) arm is not expected to
already speak this exact format the way a grounding-pretrained model might
speak its own native syntax. A low base-arm score reflects format mismatch
as much as grounding ability -- read it as "did SFT teach the format and
GRPO improve localization within it", not as an absolute zero-shot number.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

GROUNDING_SYSTEM_PROMPT = (
    "Given an image, a question, and its answer, identify the minimal set of "
    "regions a human would need to justify the answer: the region depicting "
    "the answer itself, its text label if visibly present, and any "
    "supporting visual evidence (legend swatch, axis/tick, comparator "
    "region, arrow/connector, or linked text).\n"
    "Respond with ONLY a JSON array of boxes as your final answer:\n"
    "[[x1, y1, x2, y2], [x1, y1, x2, y2]]\n"
    "Coordinates are integers from 0 to 1000, normalized to the image width "
    "and height (x1,y1 = top-left, x2,y2 = bottom-right). List boxes "
    "top-to-bottom, then left-to-right. Do not repeat a box or include "
    "irrelevant regions."
)


def build_user_prompt(question: str, choices: List[str], answer: str) -> str:
    """Image + question + choices + answer, no candidate region list -- the
    model must localize evidence from the image itself. `Choices: None` for
    genuinely open-ended datasets (only ai2d is real multiple-choice here;
    see build_dragon6_raw.py's clean_choices() for the malformed-MapIQ-
    choices sanitization)."""
    choices_str = (
        "; ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        if choices else "None"
    )
    return (
        f"Question: {question}\n"
        f"Choices: {choices_str}\n"
        f"Correct Answer: {answer}"
    )


def build_target(boxes: Sequence[Tuple[int, int, int, int]]) -> str:
    """The SFT assistant-turn label: exactly the JSON box array, nothing
    else. No <think> content is ever added here -- see model_config.py's
    enable_thinking docstring for why."""
    return json.dumps([[int(v) for v in b] for b in boxes])


# ---------------------------------------------------------------------------
# Storage schema: {id, image, conversations: [{from, value}], metadata}.
# Model-specific chat-message-with-content-blocks translation happens at
# the point of use (build_chat_messages below), not in storage, so the same
# stored JSONL is consumed identically regardless of which model_config is
# loaded.
# ---------------------------------------------------------------------------

def build_chat_messages(sample: Dict[str, Any], image_path: str,
                        include_assistant: bool = False) -> List[Dict[str, Any]]:
    """Convert one stored record into the content-block message list every
    modern transformers VLM processor's apply_chat_template expects.
    `image_path` is an absolute filesystem path -- processors accept a
    local path directly in an {"type": "image", "image": ...} block.
    include_assistant=True appends the target as a final assistant turn
    (for building SFT training examples); False leaves the conversation
    ending on the user turn (for generation -- add_generation_prompt=True
    is passed separately at render time)."""
    system_msg = sample["conversations"][0]["value"]
    user_msg = sample["conversations"][1]["value"]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_msg},
        ]},
    ]
    if include_assistant:
        assistant_msg = sample["conversations"][2]["value"]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})
    return messages


# ---------------------------------------------------------------------------
# Tolerant box parsing -- shared by the GRPO reward and eval export. Output
# may not be clean JSON (markdown fences, a <think>...</think> block ahead
# of the answer, leading/trailing prose -- especially pre-SFT / base-arm
# generations), so this looks for the LAST top-level JSON array of 4-number
# arrays anywhere in the text (the final-answer convention: any exploratory
# lists mentioned inside reasoning come first, the committed answer last)
# rather than requiring the whole response to parse.
# ---------------------------------------------------------------------------

_ARRAY_RE = re.compile(r"\[\s*\[.*?\]\s*\]", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    """Drop a <think>...</think> block if present, for logging/inspection.
    Not required for correctness -- parse_pred_boxes already takes the LAST
    matching array in the raw text -- but keeps logs readable."""
    return _THINK_RE.sub("", text).strip()


def parse_pred_boxes(text: str) -> Optional[List[Box]]:
    """Returns None on unparseable output (format reward = 0), else a
    (possibly empty) list of validated (x1,y1,x2,y2) boxes in [0,1000]."""
    if not text:
        return None
    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.insert(0, fence.group(1))

    for cand in candidates:
        matches = list(_ARRAY_RE.finditer(cand))
        if not matches:
            continue
        for m in reversed(matches):  # last array = the committed final answer
            try:
                payload = json.loads(m.group(0))
            except Exception:
                try:
                    import ast
                    payload = ast.literal_eval(m.group(0))
                except Exception:
                    continue
            if not isinstance(payload, list):
                continue
            boxes: List[Box] = []
            ok = True
            for b in payload:
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    ok = False
                    break
                try:
                    x1, y1, x2, y2 = (float(v) for v in b)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                    ok = False
                    break
                boxes.append((x1, y1, x2, y2))
            if ok:
                return boxes
    return None
