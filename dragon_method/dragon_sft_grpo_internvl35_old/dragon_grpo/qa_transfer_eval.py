#!/usr/bin/env python3
"""
qa_transfer_eval.py -- no-answer QA accuracy, to test whether grounding
training causally transferred to answering ability.

WHY THIS IS DELICATE. Every SFT/GRPO training example (a) *supplies* the
answer in the prompt ("Correct Answer: X") and (b) has a target that is
exactly `<ref>evidence</ref><box>[[...]]</box>` -- the model was never
trained to emit an answer string. Asking it to answer is therefore
off-distribution, and the result depends entirely on which prompt you use.
So this script measures BOTH arms and reports them separately:

  --mode grounding   keep the grounding system prompt, just delete the
                     "Correct Answer:" line. Measures what the deployed
                     model does when the answer is withheld. Expect box
                     syntax; `box_rate` quantifies that.
  --mode qa          replace the system prompt with a plain QA instruction.
                     Measures whether the model can still answer at all --
                     i.e. whether grounding training preserved or destroyed
                     the base model's QA ability.

Only --mode qa supports the "causal transfer" reading, and only against a
BASE arm: run with --no-adapters to score the unmodified InternVL3-8B
through the identical harness. Transfer means SFT/GRPO > base; forgetting
means base > SFT/GRPO. Without the base arm the SFT-vs-GRPO delta alone
cannot distinguish the two.

SCORING is deliberately conservative and reported three ways, because
free-form answers admit no single fair rule:
  em          normalized exact match (lowercase, strip articles/punct/space)
  contains    gold appears inside the prediction (credits verbose answers)
  mc_letter   for multiple-choice items, the predicted option letter or its
              text matches the gold option
`box_rate` = fraction of outputs containing <box>, i.e. refused to answer.
`empty_rate` = fraction with no usable text at all.

Usage:
  python qa_transfer_eval.py --mode qa \
      --checkpoint outputs/dragon6_sft2k_v4/ckpts \
      --grpo-checkpoint outputs/dragon6_grpo_5k/grpo-step2500 \
      --label grpo5k --device cuda:0 --hf-cache-dir hf_home \
      --out qa_transfer_grpo5k.json
"""
import argparse, json, re, string, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

QA_SYSTEM = ("Answer the question about the image. Reply with the answer only, "
             "as few words as possible. Do not explain.")

_ART = re.compile(r"\b(a|an|the)\b")
_PUNC = str.maketrans("", "", string.punctuation)


def norm(s: str) -> str:
    s = (s or "").lower().translate(_PUNC)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def strip_answer_line(user_turn: str) -> str:
    """Remove the 'Correct Answer: ...' line the model was trained to see."""
    return "\n".join(l for l in user_turn.splitlines()
                     if not l.strip().lower().startswith("correct answer:"))


def parse_choices(user_turn: str):
    """-> {'a': 'Light energy', ...} from the 'Choices: (A) x; (B) y' line."""
    for l in user_turn.splitlines():
        if l.strip().lower().startswith("choices:"):
            body = l.split(":", 1)[1].strip()
            if body.lower() == "none":
                return {}
            out = {}
            for part in body.split(";"):
                m = re.match(r"\s*\(([A-Za-z])\)\s*(.+)", part)
                if m:
                    out[m.group(1).lower()] = m.group(2).strip()
            return out
    return {}


def score(pred: str, gold: str, choices: dict) -> dict:
    # strip any box syntax before scoring the textual answer
    text = re.sub(r"<ref>.*?</ref>|<box>.*?</box>", " ", pred, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text).strip()
    p, g = norm(text), norm(gold)
    em = float(bool(g) and p == g)
    contains = float(bool(g) and g in p)
    mc = 0.0
    if choices:
        gold_letter = next((k for k, v in choices.items() if norm(v) == g), None)
        if gold_letter:
            m = re.match(r"^\(?([a-z])\)?\b", p)
            if m and m.group(1) == gold_letter:
                mc = 1.0
            elif norm(choices[gold_letter]) in p:
                mc = 1.0
    return {"em": em, "contains": contains, "mc_letter": mc,
            "has_box": float("<box>" in pred), "empty": float(not text)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="SFT adapter-only dir")
    ap.add_argument("--grpo-checkpoint", default=None)
    ap.add_argument("--no-adapters", action="store_true",
                    help="score the BASE model (SFT LoRA NOT merged) -- the control arm")
    ap.add_argument("--mode", choices=["qa", "grounding"], default="qa")
    ap.add_argument("--label", default="model")
    ap.add_argument("--holdout-dir", default="dragon_datasets_sft2k/holdout_per_domain")
    ap.add_argument("--domains", nargs="*",
                    default=["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"])
    ap.add_argument("--per-domain", type=int, default=0, help="0 = all")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-tiles", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dump", type=int, default=6, help="print N raw outputs per domain")
    ap.add_argument("--out", default="qa_transfer_eval.json")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path, load_lora_checkpoint
    _ensure_internvl_repo_on_path()
    from internvl.train.dataset import build_transform, dynamic_preprocess
    from internvl.train.constants import IMG_CONTEXT_TOKEN
    from sft_v4_phase0 import chat_keep_special_tokens

    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=a.checkpoint, device=a.device, dtype=torch.bfloat16,
        hf_cache_dir=a.hf_cache_dir, merge_lora=not a.no_adapters)
    if a.no_adapters:
        # load_lora_checkpoint with merge_lora=False leaves PeftModel wrappers;
        # disable them so this arm is the pristine base model.
        try:
            model.language_model.disable_adapter_layers()
            model.vision_model.disable_adapter_layers()
        except Exception as e:
            print(f"[warn] could not disable adapters cleanly: {e}")
    elif a.grpo_checkpoint:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, a.grpo_checkpoint)
        model.language_model = model.language_model.to(a.device)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.config.max_dynamic_patch = a.max_tiles
    model.language_model.config.use_cache = True

    tf = build_transform(is_train=False, input_size=448, pad2square=False,
                         normalize_type="imagenet")
    gen = dict(max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
               repetition_penalty=1.0)
    image_root = ROOT / "Diagram_Attribution_Dataset"

    agg = defaultdict(lambda: defaultdict(list))
    dumps = defaultdict(list)
    for dom in a.domains:
        rows = [json.loads(l) for l in
                (ROOT / a.holdout_dir / f"{dom}.jsonl").read_text().splitlines() if l.strip()]
        if a.per_domain:
            rows = rows[: a.per_domain]
        for s in rows:
            gold = s["metadata"].get("answer", "")
            if not gold:
                continue
            user = s["conversations"][1]["value"]
            choices = parse_choices(user)
            q = strip_answer_line(user)
            if a.mode == "qa":
                model.system_message = QA_SYSTEM
            else:
                model.system_message = s["conversations"][0]["value"]
            if "<image>" not in q:
                q = "<image>\n" + q
            img = Image.open(image_root / s["image"]).convert("RGB")
            tiles = dynamic_preprocess(img, min_num=1, max_num=a.max_tiles,
                                       image_size=448, use_thumbnail=True)
            pv = torch.stack([tf(t) for t in tiles]).to(torch.bfloat16).to(a.device)
            resp = chat_keep_special_tokens(model, tokenizer, pv, q, gen, a.device)
            sc = score(resp, gold, choices)
            for k, v in sc.items():
                agg[dom][k].append(v)
            if len(dumps[dom]) < a.dump:
                dumps[dom].append({"gold": gold, "pred": resp[:160]})
        n = len(agg[dom]["em"])
        print(f"[{dom}] n={n}  em={sum(agg[dom]['em'])/max(1,n):.3f}  "
              f"contains={sum(agg[dom]['contains'])/max(1,n):.3f}  "
              f"mc={sum(agg[dom]['mc_letter'])/max(1,n):.3f}  "
              f"box_rate={sum(agg[dom]['has_box'])/max(1,n):.3f}", flush=True)

    keys = ["em", "contains", "mc_letter", "has_box", "empty"]
    res = {d: {k: round(sum(v[k]) / max(1, len(v[k])), 4) for k in keys} for d, v in agg.items()}
    macro = {k: round(sum(res[d][k] for d in res) / max(1, len(res)), 4) for k in keys}
    print(f"\n=== {a.label} | mode={a.mode} | MACRO ===")
    for k in keys:
        print(f"  {k:<10} {macro[k]:.4f}")
    print("\n--- sample raw outputs ---")
    for d, ex in dumps.items():
        for e in ex[:2]:
            print(f"  [{d}] gold={e['gold']!r}\n        pred={e['pred']!r}")
    Path(a.out).write_text(json.dumps(
        {"label": a.label, "mode": a.mode, "per_domain": res, "macro": macro,
         "samples": {d: v for d, v in dumps.items()}}, indent=2))
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
