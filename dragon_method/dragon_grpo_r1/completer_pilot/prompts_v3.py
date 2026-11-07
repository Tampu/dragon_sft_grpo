"""v3 prompts, written from the DRAGON paper's EDGE template (Appendix C.1) and annotation protocol (Appendix E).

Both roles get the same fixed evidence guideline, identical for every item; nothing in it depends on an
item's gold. The grounder keeps the output format it was trained on (<ref>/<box>, 0-1000 grid), so the
same evaluation parses it.

  GROUNDER_SYSTEM_V3   the grounder's system turn (for the v3 SFT data and the inference-only check)
  COMPLETER_SYSTEM_V3  the completer's system turn
  completer_user_v3    the completer's user turn: diagram, crop sheet, candidates in reading order,
                       no agreement counts and no current/proposed tags
"""
from __future__ import annotations

from typing import Optional, Sequence

GUIDE = (
    "This includes:\n"
    "- The visual region that directly depicts the answer\n"
    "- The textual label or annotation naming the answer\n"
    "- Any supporting visual evidence used for reasoning, such as legends, arrows, connections, neighbouring "
    "regions, chart axes, or other visual structures used to interpret the diagram\n"
    "In particular, include:\n"
    "- The answer region (visual depiction + label)\n"
    "- The question subject or anchor (e.g., a specific state, component, or object)\n"
    "- Any visual elements required for comparison, flow, or spatial reasoning\n"
    "- Supporting diagram features such as legends, axes, arrows, connectors, or colour encodings\n"
    "Only include regions that a human would reasonably inspect to answer the question. Do not give only the "
    "text label of the answer: also include the visual region corresponding to the answer."
)

GROUNDER_SYSTEM_V3 = (
    "You are given a diagram (e.g., flowchart, life cycle, bar chart, map, circuit), a question about it, and "
    "the correct answer. Identify the minimal set of visual regions required to verify the answer.\n"
    + GUIDE + "\n"
    "Respond with the evidence regions in this exact format:\n"
    "<ref>evidence</ref><box>[[x1, y1, x2, y2], [x1, y1, x2, y2]]</box>\n"
    "Coordinates are integers from 0 to 1000, normalized to the image width and height (x1,y1 = top-left, "
    "x2,y2 = bottom-right). List boxes top-to-bottom, then left-to-right. Do not repeat a box or include "
    "irrelevant regions."
)

COMPLETER_SYSTEM_V3 = (
    "You verify candidate evidence regions for a question about a diagram. You keep every candidate a human "
    "needs to verify the answer and leave out every other one."
)


def fmt_box(b) -> str:
    return "[" + ", ".join(str(int(round(c))) for c in b) + "]"


def completer_user_v3(question: str, answer: str, cands: Sequence, tags: Optional[Sequence[str]] = None) -> str:
    """cands are in display order; candidate d is shown as number d (1-based) in the text and on the crop sheet.
    tags (v3t): "current" (in the grounder's greedy set) or "proposed" (from its samples) per candidate."""
    lines = ["<image>",
             "Candidate crops: <image>",
             f"Question: {question}",
             f"Answer: {answer}",
             "Candidate evidence regions (x1, y1, x2, y2 on a 0-1000 grid). The second image shows each candidate "
             "with the same number, outlined in red, with some of its surroundings:"]
    lines += [f"[{d}] {fmt_box(b)}" + (f"  ({tags[d - 1]})" if tags else "") for d, b in enumerate(cands, 1)]
    lines.append("Select the minimal set of candidates required to verify the answer. " + GUIDE)
    lines.append("Output the numbers as <select>i, j, ...</select>. Output <select></select> if none are needed.")
    return "\n".join(lines)
