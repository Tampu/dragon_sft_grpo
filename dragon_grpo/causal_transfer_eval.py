#!/usr/bin/env python3
"""
causal_transfer_eval.py -- THE bridge experiment.

CLAIM UNDER TEST: grounding-only RL (the model never receives answer supervision
at any stage -- the answer is an INPUT during training, never a target) improves
the model's ability to ANSWER diagram questions unaided.

If yes  -> "teaching evidence attribution teaches reasoning"; the paper's spine.
If flat -> "faithfulness and answering are separately trainable"; still a result,
           but a weaker one. Either way the number decides the framing, so the
           design below is built to make the null interpretable rather than
           ambiguous.

------------------------------------------------------------------------------
DESIGN (each element exists to kill a specific confound)
------------------------------------------------------------------------------
THREE ARMS -- base / sft / grpo.
  The base (untrained InternVL3-8B) arm is NOT optional. Without it, SFT<base<GRPO
  and base<SFT<GRPO are indistinguishable, yet they support opposite claims:
  the first is "GRPO recovers QA ability that SFT destroyed", the second is
  "grounding training improves answering". Report all three or report nothing.

TWO CONDITIONS -- identical prompts across arms.
  direct   : image + question (+ choices) -> answer.
  grounded : image + question -> evidence boxes, THEN answer, one generation.
  Both arms see byte-identical prompts; only the checkpoint changes.

ANSWER-EMISSION RATE is logged separately from accuracy.
DOMAIN-APPROPRIATE METRICS (ChartQA relaxed acc, InfoVQA ANLS, else EM);
token-F1 reported for every domain as the common comparable number.
PAIRED BOOTSTRAP CIs on the deltas.
CONTAMINATION: --exclude-ids dragon_datasets_test2445/contaminated_ids.txt

------------------------------------------------------------------------------
PLUMBING FIXES vs the original draft (design/metrics untouched):
  1. system prompt: chat() reads `self.system_message`, which is assigned in
     __init__ (modeling_internvl_chat.py:99). Setting `model.config.system_message`
     AFTER construction is a NO-OP -- the model would silently keep the
     checkpoint's training prompt (which states the answer is given) and the
     whole experiment would answer the wrong question while appearing to work.
     We set `model.system_message` directly and assert it took.
  2. adapter-only checkpoints: SFT/GRPO ckpts hold LoRA adapters, not a merged
     model, so InternVLChatModel.from_pretrained() cannot load them. Uses this
     repo's load_lora_checkpoint(); --arm base loads the pristine base model.
  3. decode: model.chat() strips <ref>/<box> (skip_special_tokens=True), which
     erases the evidence chain in the grounded condition. Uses
     sft_v4_phase0.chat_keep_special_tokens().
  4. internvl import path setup before any internvl.* import.
  5. defaults corrected for this repo (image root, combined jsonl w/ dataset).

USAGE
  python causal_transfer_eval.py generate --arm base --condition direct \
      --out-dir causal_transfer/ --device cuda:0
  # arms: base | sft | grpo ; conditions: direct | grounded  (6 runs total)
  python causal_transfer_eval.py score --pred-dir causal_transfer/
"""

import argparse
import json
import random
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Prompts -- IDENTICAL across arms. Only the checkpoint varies.
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = (
    "You are given a diagram and a question about it. Answer the question using "
    "only the information visible in the diagram.\n"
    "Respond with exactly one line:\n"
    "Answer: <your answer>\n"
    "Give the answer only -- a value, name, or option -- with no explanation."
)

GROUNDED_SYSTEM = (
    "You are given a diagram and a question about it. First identify the visual "
    "evidence regions needed to answer it, then answer using that evidence.\n"
    "Respond with exactly two lines:\n"
    "<ref>evidence</ref><box>[[x1, y1, x2, y2], ...]</box>\n"
    "Answer: <your answer>\n"
    "Coordinates are integers from 0 to 1000, normalized to the image width and "
    "height. Give the answer only -- a value, name, or option -- with no explanation."
)


def build_user_prompt(sample: Dict[str, Any]) -> str:
    md = sample.get("metadata", {})
    question = md.get("question") or sample.get("question") or ""
    if not question:
        raw = sample["conversations"][1]["value"]
        m = re.search(r"Question:\s*(.*?)(?:\nChoices:|\nCorrect Answer:|$)", raw, re.S)
        question = m.group(1).strip() if m else raw
    choices = md.get("choices") or sample.get("choices") or []
    if not choices:
        raw = sample["conversations"][1]["value"]
        m = re.search(r"Choices:\s*(.*?)(?:\nCorrect Answer:|$)", raw, re.S)
        if m and m.group(1).strip().lower() not in ("none", ""):
            choices = [c.strip() for c in m.group(1).split(";") if c.strip()]
    choices_str = "; ".join(choices) if choices else "None"
    return f"<image>\nQuestion: {question}\nChoices: {choices_str}"


def get_gold_answer(sample: Dict[str, Any]) -> str:
    md = sample.get("metadata", {})
    if md.get("answer"):
        return str(md["answer"])
    raw = sample["conversations"][1]["value"]
    m = re.search(r"Correct Answer:\s*(.*)", raw)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

BOX_TAG_RE = re.compile(r"<ref>.*?</ref>\s*<box>.*?</box>", re.S)
ANSWER_LINE_RE = re.compile(r"Answer\s*:\s*(.+)", re.I)


def extract_answer(text: str) -> Optional[str]:
    stripped = BOX_TAG_RE.sub(" ", text or "")
    m = ANSWER_LINE_RE.search(stripped)
    if m:
        ans = m.group(1).strip().split("\n")[0].strip()
        return ans if ans else None
    leftover = re.sub(r"<[^>]+>|\[\[.*?\]\]", " ", stripped).strip()
    leftover = leftover.split("\n")[0].strip()
    if leftover and len(leftover) <= 200 and not leftover.startswith("["):
        return leftover
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation or ch in ".-%")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    gc = defaultdict(int)
    for t in g:
        gc[t] += 1
    overlap = 0
    for t in p:
        if gc[t] > 0:
            gc[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


NUM_RE = re.compile(r"-?\d+\.?\d*")


def _to_num(s: str) -> Optional[float]:
    s = (s or "").replace(",", "").replace("%", "").strip()
    m = NUM_RE.fullmatch(s) or NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def relaxed_accuracy(pred: str, gold: str, tol: float = 0.05) -> float:
    gp, gg = _to_num(pred), _to_num(gold)
    if gp is None or gg is None:
        return exact_match(pred, gold)
    if gg == 0:
        return float(abs(gp) < 1e-6)
    return float(abs(gp - gg) / abs(gg) <= tol)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gold: str, tau: float = 0.5) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    d = _levenshtein(p, g) / max(len(p), len(g))
    s = 1.0 - d
    return s if s >= tau else 0.0


PRIMARY_METRIC = {
    "chartqa": "relaxed_acc",
    "infographics": "anls",
    "infographicsvqa": "anls",
    "infovqa": "anls",
    "ai2d": "exact_match",
}


def score_one(pred: Optional[str], gold: str, domain: str) -> Dict[str, float]:
    if pred is None:
        return {"answered": 0.0, "exact_match": 0.0, "token_f1": 0.0,
                "relaxed_acc": 0.0, "anls": 0.0, "primary": 0.0}
    d = {"answered": 1.0,
         "exact_match": exact_match(pred, gold),
         "token_f1": token_f1(pred, gold),
         "relaxed_acc": relaxed_accuracy(pred, gold),
         "anls": anls(pred, gold)}
    d["primary"] = d[PRIMARY_METRIC.get(domain.lower(), "exact_match")]
    return d


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float, float]:
    assert len(a) == len(b) and len(a) > 0
    rng = random.Random(seed)
    n = len(a)
    obs = sum(b) / n - sum(a) / n
    deltas = []
    for _ in range(n_boot):
        pick = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in pick) / n
        db = sum(b[i] for i in pick) / n
        deltas.append(db - da)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    if obs >= 0:
        p = 2.0 * sum(1 for d in deltas if d <= 0) / n_boot
    else:
        p = 2.0 * sum(1 for d in deltas if d >= 0) / n_boot
    return obs, lo, hi, min(1.0, p)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SFT_CKPT = "outputs/dragon6_sft2k_v4/ckpts"
GRPO_CKPT = "outputs/dragon6_grpo_5k/grpo-step2500"


def load_arm(args):
    """PLUMBING FIX 2+4: adapter-aware loading. base = pristine InternVL3-8B;
    sft = SFT LoRA merged; grpo = SFT merged + GRPO adapter applied."""
    import torch
    from internvl_lora_checkpoint_io import (_ensure_internvl_repo_on_path,
                                             load_lora_checkpoint)
    _ensure_internvl_repo_on_path()
    from internvl.train.constants import IMG_CONTEXT_TOKEN

    merge = args.arm != "base"
    model, tokenizer = load_lora_checkpoint(
        checkpoint_dir=ROOT / SFT_CKPT, device=args.device, dtype=torch.bfloat16,
        hf_cache_dir=args.hf_cache_dir, merge_lora=merge)
    if args.arm == "base":
        for m in (model.language_model, model.vision_model):
            try:
                m.disable_adapter_layers()
            except Exception:
                pass
    elif args.arm == "grpo":
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model,
                                                         str(ROOT / GRPO_CKPT))
        model.language_model = model.language_model.to(args.device)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.config.max_dynamic_patch = args.max_tiles
    model.language_model.config.use_cache = True
    return model, tokenizer


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path
    _ensure_internvl_repo_on_path()
    from internvl.train.dataset import build_transform, dynamic_preprocess
    from sft_v4_phase0 import chat_keep_special_tokens   # PLUMBING FIX 3

    samples = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if args.exclude_ids:
        bad = {l.strip() for l in Path(args.exclude_ids).read_text().splitlines() if l.strip()}
        before = len(samples)
        samples = [s for s in samples if str(s.get("id")) not in bad]
        print(f"[data] excluded {before - len(samples)} contaminated ids")
    if args.per_domain:
        by = defaultdict(list)
        for s in samples:
            by[s.get("metadata", {}).get("dataset", "?")].append(s)
        samples = [s for d in sorted(by) for s in by[d][: args.per_domain]]
        print(f"[data] stratified to {args.per_domain}/domain")
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"[data] {len(samples)} samples | arm={args.arm} condition={args.condition}")

    model, tokenizer = load_arm(args)

    # PLUMBING FIX 1: chat() reads self.system_message (set in __init__).
    # Assigning model.config.system_message here would be a silent no-op.
    want = GROUNDED_SYSTEM if args.condition == "grounded" else DIRECT_SYSTEM
    model.system_message = want
    assert model.system_message == want, "system prompt did not take"
    print(f"[prompt] system message set ({len(want)} chars, {args.condition})")

    transform = build_transform(is_train=False, input_size=448,
                                pad2square=False, normalize_type="imagenet")
    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                   repetition_penalty=1.0, no_repeat_ngram_size=6)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gen_{args.arm}_{args.condition}.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"REFUSING to overwrite {out_path} (pass --overwrite)")
    image_root = Path(args.image_root)
    with out_path.open("w") as f:
        for k, s in enumerate(samples, 1):
            img = Image.open(image_root / s["image"]).convert("RGB")
            tiles = dynamic_preprocess(img, min_num=1, max_num=args.max_tiles,
                                       image_size=448, use_thumbnail=True)
            pv = torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).to(args.device)
            resp = chat_keep_special_tokens(model, tokenizer, pv,
                                            build_user_prompt(s), gen_cfg, args.device)
            f.write(json.dumps({
                "id": s.get("id"),
                "dataset": s.get("metadata", {}).get("dataset") or args.dataset_hint or "",
                "gold": get_gold_answer(s),
                "response": resp,
            }) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"  [{k}/{len(samples)}]", flush=True)
    print(f"Wrote -> {out_path}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def cmd_score(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    files = sorted(pred_dir.glob("gen_*.jsonl"))
    if not files:
        raise SystemExit(f"No gen_*.jsonl in {pred_dir}")

    runs: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    domains: Dict[str, str] = {}
    for f in files:
        m = re.match(r"gen_(.+?)_(direct|grounded)\.jsonl$", f.name)
        if not m:
            print(f"  [skip] {f.name}")
            continue
        arm, cond = m.group(1), m.group(2)
        per_id = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            dom = (r.get("dataset") or "").lower()
            domains[str(r["id"])] = dom
            pred = extract_answer(r.get("response", ""))
            per_id[str(r["id"])] = score_one(pred, r.get("gold", ""), dom)
        runs[(arm, cond)] = per_id
        print(f"  loaded {f.name}: {len(per_id)} samples")

    metrics = ["primary", "token_f1", "exact_match", "answered"]
    arms_present = sorted({a for a, _ in runs})
    conds_present = sorted({c for _, c in runs})

    for cond in conds_present:
        keysets = [set(runs[(a, cond)]) for a in arms_present if (a, cond) in runs]
        if not keysets:
            continue
        common = set.intersection(*keysets)
        print(f"\n{'='*78}\nCONDITION: {cond}   (paired on {len(common)} common ids)\n{'='*78}")
        print(f"\n{'arm':<8} " + " ".join(f"{m:>12}" for m in metrics))
        for a in arms_present:
            if (a, cond) not in runs:
                continue
            d = runs[(a, cond)]
            row = [sum(d[i][m] for i in common) / len(common) for m in metrics]
            print(f"{a:<8} " + " ".join(f"{v:>12.4f}" for v in row))

        print(f"\nPAIRED DELTAS (bootstrap 95% CI, n={len(common)}):")
        pairs = []
        if "base" in arms_present and "sft" in arms_present:
            pairs.append(("base", "sft"))
        if "sft" in arms_present and "grpo" in arms_present:
            pairs.append(("sft", "grpo"))
        if "base" in arms_present and "grpo" in arms_present:
            pairs.append(("base", "grpo"))
        for lo_arm, hi_arm in pairs:
            if (lo_arm, cond) not in runs or (hi_arm, cond) not in runs:
                continue
            print(f"\n  {hi_arm} - {lo_arm}:")
            for m in ["primary", "token_f1", "answered"]:
                a = [runs[(lo_arm, cond)][i][m] for i in sorted(common)]
                b = [runs[(hi_arm, cond)][i][m] for i in sorted(common)]
                obs, lo, hi, p = paired_bootstrap(a, b, args.n_boot, args.seed)
                sig = "*" if (lo > 0 or hi < 0) else " "
                print(f"    {m:<12} {obs:>+8.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  p={p:.4f} {sig}")

        doms = sorted({domains[i] for i in common if domains.get(i)})
        if doms:
            print(f"\nPER-DOMAIN (primary metric):")
            print(f"  {'domain':<18} {'n':>5} " + " ".join(f"{a:>10}" for a in arms_present))
            for dom in doms:
                ids = [i for i in common if domains.get(i) == dom]
                if not ids:
                    continue
                row = []
                for a in arms_present:
                    d = runs.get((a, cond))
                    row.append(sum(d[i]["primary"] for i in ids) / len(ids) if d else float("nan"))
                print(f"  {dom:<18} {len(ids):>5} " + " ".join(f"{v:>10.4f}" for v in row))

    print(f"""
{'='*78}
HOW TO READ THIS
{'='*78}
* marks a delta whose 95% CI excludes zero.

The claim "grounding-only RL improves answering" requires:
  (a) grpo - sft  POSITIVE and significant on 'primary', AND
  (b) 'answered' rates comparable across arms -- if grpo answers far less
      often, an accuracy gain may just be selective answering, not reasoning.

Check the base arm before believing any story:
  base < sft < grpo        -> grounding training improves answering. The spine.
  sft < base, grpo > sft   -> GRPO RECOVERS ability SFT destroyed. Report it
                              that way; it is NOT evidence for the thesis.
  base > grpo              -> fine-tuning cost QA ability overall. Honest
                              limitation; the paper falls back to "faithfulness
                              is separately trainable".

Compare conditions too: grounded > direct within an arm means emitting the
evidence chain scaffolds the answer -- that is the paper's thesis in its
purest form, and it is a within-arm comparison so it is immune to the
catastrophic-forgetting confound.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run one arm x condition")
    g.add_argument("--arm", required=True, choices=["base", "sft", "grpo"])
    g.add_argument("--condition", required=True, choices=["direct", "grounded"])
    g.add_argument("--jsonl", default=str(ROOT / "dragon_datasets_test2445/test2445_combined.jsonl"))
    g.add_argument("--image-root", default=str(ROOT / "Diagram_Attribution_Dataset"))
    g.add_argument("--out-dir", required=True)
    g.add_argument("--hf-cache-dir", default=str(ROOT / "hf_home"))
    g.add_argument("--exclude-ids",
                   default=str(ROOT / "dragon_datasets_test2445/contaminated_ids.txt"))
    g.add_argument("--dataset-hint", default=None)
    g.add_argument("--per-domain", type=int, default=0,
                   help="stratified subsample, N per domain (0 = all 2432)")
    g.add_argument("--max-samples", type=int, default=None)
    g.add_argument("--max-tiles", type=int, default=12)
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("score", help="score all gen_*.jsonl in a dir (CPU only)")
    s.add_argument("--pred-dir", required=True)
    s.add_argument("--n-boot", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
