"""Shared paths, splits, prompts, scoring and model I/O for the VEP pipeline.

Everything that defines the frozen comparison (R1 box parser, QA scoring, prompt template,
tiling) is imported from the GRPO-R / i35 code rather than re-implemented.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I35 = Path(os.environ.get("I35_ROOT", "/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old"))
R1 = Path(os.environ.get("R1_ROOT", "/mnt/data2/traviku2/dragon_grpo_r1"))
MERGED_ROOT = Path(os.environ.get("VEP_MERGED_ROOT", "/dev/shm/vep_merged"))
FLASH_PY = os.environ.get("FLASH_PY", "/mnt/data2/traviku2/da_env_flash/bin/python")

SFT_CKPT = I35 / "outputs/i35_sft2k/ckpts"
ORIG_GRPO = I35 / "outputs/i35_grpo5k/grpo-step2633"
GRPOR_RUN = R1 / "outputs/i35_grpo5k_r1"
IMAGE_ROOT = I35 / "Diagram_Attribution_Dataset"
HF_HOME = I35 / "hf_home"
EVAL_SCRIPT = R1 / "dragon_grpo/eval_script.py"
DOMAINS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]
TAUS = ["0.1", "0.3", "0.5", "0.7", "0.9"]

EOS_TOKEN, IMG_CONTEXT, IMG_START, IMG_END = "<|im_end|>", "<IMG_CONTEXT>", "<img>", "</img>"
NUM_IMAGE_TOKEN = 256
MAX_TILES = 12
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

for _p in (I35 / "scripts", R1 / "dragon_grpo"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from grpo_dragon6_v1 import RewardWeights, compute_reward, parse_boxes_r1  # noqa: E402,F401
from qa_transfer_eval import QA_SYSTEM, parse_choices, strip_answer_line  # noqa: E402,F401
from qa_transfer_eval import score as qa_score  # noqa: E402

VEP_SYSTEM = (
    "Given an image, a question, and a candidate answer, collect the visual evidence needed to verify the "
    "answer, one region at a time, and then commit. Evidence is the minimal set of regions a "
    "human would need to justify the answer: the region depicting the answer itself, its text label if "
    "visibly present, and any supporting visual evidence (legend swatch, axis/tick, comparator region, "
    "arrow/connector, or linked text).\n"
    "Write each evidence region on its own numbered line in this exact format:\n"
    "1<box>[x1, y1, x2, y2]</box>\n"
    "2<box>[x1, y1, x2, y2]</box>\n"
    "Coordinates are integers from 0 to 1000, normalized to the image width and height "
    "(x1,y1 = top-left, x2,y2 = bottom-right). Do not repeat a region or include irrelevant regions. "
    "When the evidence is sufficient, stop and write on the last line:\n"
    "<COMMIT>")
GRPOR_MAX_NEW_TOKENS = 640     # GRPO-R budget; fits <= 32 boxes in its format
VEP_MAX_NEW_TOKENS = 800       # smallest budget fitting as many boxes (32) in the VEP format


# ---------------------------------------------------------------- data

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_files(name):
    if name == "val600":
        return [I35 / f"data_i35/val_by_domain/{d}.jsonl" for d in DOMAINS]
    if name == "test":
        return [I35 / f"dragon_datasets_test2445/{d}.jsonl" for d in DOMAINS]
    if name == "val300":
        return [R1 / "data_r1/val300.jsonl"]
    if name == "grpo_pool":
        return [I35 / "data_i35/i35_grpo_rest_v4.jsonl"]
    if name == "sft_pool":
        return [I35 / "data_i35/i35_sft2000_v4.jsonl"]
    return [Path(name)]


def to_item(s):
    conv = s["conversations"]
    return {"id": s["id"], "domain": s["id"].split("::")[0], "image": s["image"],
            "system": conv[0]["value"], "user": conv[1]["value"],
            "gold": [tuple(map(float, b)) for b in s["metadata"]["gt_boxes_norm"]],
            "answer": s["metadata"].get("answer", "") or "",
            "choices": parse_choices(conv[1]["value"]),
            "target": conv[2]["value"] if len(conv) > 2 else None}


def load_split(name, limit_per_domain=None):
    items = [to_item(s) for f in split_files(name) for s in load_jsonl(f)]
    if limit_per_domain:
        cnt, keep = {}, []
        for it in items:
            cnt[it["domain"]] = cnt.get(it["domain"], 0) + 1
            if cnt[it["domain"]] <= limit_per_domain:
                keep.append(it)
        items = keep
    return items


# ---------------------------------------------------------------- prompts

_CONV = None


def _get_conv_template(name):
    global _CONV
    if _CONV is None:
        from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path
        _ensure_internvl_repo_on_path()
        from internvl.conversation import get_conv_template
        _CONV = get_conv_template
    return _CONV(name)


def user_turn(item, mode):
    """grounding and vep keep the 'Correct Answer:' line (answer-conditioned, the DRAGON protocol);
    only the unaided QA protocol removes it."""
    u = strip_answer_line(item["user"]) if mode == "qa" else item["user"]
    return u if "<image>" in u else "<image>\n" + u


def system_turn(item, mode):
    return {"grounding": item["system"], "vep": VEP_SYSTEM, "qa": QA_SYSTEM}[mode]


def prompt_text(item, mode):
    """internlm2-chat prompt with a single '<image>' placeholder (GRPO-R / R1 eval template)."""
    t = _get_conv_template("internlm2-chat")
    t.system_message = system_turn(item, mode)
    t.append_message(t.roles[0], user_turn(item, mode))
    t.append_message(t.roles[1], None)
    return t.get_prompt()


def expand_image(prompt, num_patches):
    return prompt.replace("<image>", IMG_START + IMG_CONTEXT * NUM_IMAGE_TOKEN * num_patches + IMG_END, 1)



# ---------------------------------------------------------------- images (HF path, as GRPO-R)

_TF = None


def hf_pixel_values(image_rel, max_tiles=MAX_TILES):
    import torch
    from PIL import Image
    global _TF
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path
    _ensure_internvl_repo_on_path()
    from internvl.train.dataset import build_transform, dynamic_preprocess
    if _TF is None:
        _TF = build_transform(is_train=False, input_size=448, pad2square=False, normalize_type="imagenet")
    img = Image.open(IMAGE_ROOT / image_rel).convert("RGB")
    tiles = dynamic_preprocess(img, min_num=1, max_num=max_tiles, image_size=448, use_thumbnail=True)
    return torch.stack([_TF(t) for t in tiles]).to(torch.bfloat16)


def load_pil(image_rel):
    from PIL import Image
    return Image.open(IMAGE_ROOT / image_rel).convert("RGB")


# ---------------------------------------------------------------- model I/O

def base_snapshot():
    snaps = sorted(glob.glob(str(HF_HOME / "hub/models--OpenGVLab--InternVL3_5-8B/snapshots/*")))
    if not snaps:
        snaps = sorted(glob.glob(str(HF_HOME / "models--OpenGVLab--InternVL3_5-8B/snapshots/*")))
    return Path(snaps[-1])


def merged_dir(name):
    return MERGED_ROOT / name


def _fork_model_classes():
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path
    _ensure_internvl_repo_on_path()
    from internvl_chat.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl_chat.internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
    return InternVLChatConfig, InternVLChatModel


def _fork_config(path, attn):
    Cfg, Model = _fork_model_classes()
    cfg = Cfg.from_pretrained(str(path))
    cfg.use_backbone_lora = 0
    cfg.use_llm_lora = 0
    cfg.llm_config._attn_implementation = attn  # the SFT config says flash_attention_2 (absent in da_env2)
    return cfg, Model


def load_merged_hf(mdir, device="cpu", attn="sdpa"):
    """Load an exported merged model with the i35 fork's InternVLChatModel (same weights vLLM serves)."""
    import torch
    from transformers import AutoTokenizer
    cfg, Model = _fork_config(mdir, attn)
    model = Model.from_pretrained(str(mdir), config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.language_model.config._attn_implementation = attn
    tok = AutoTokenizer.from_pretrained(str(mdir), trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT)
    model.config.max_dynamic_patch = MAX_TILES
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval(), tok


def export_merged(out_dir, extra_lora=None, sft_ckpt=SFT_CKPT, shard_gb=4.0):
    """base InternVL3.5-8B + SFT LoRA (vision+LLM) merged + SFT mlp1 [+ extra LLM LoRA merged],
    written as an HF InternVLChatModel directory that both vLLM and load_merged_hf() read."""
    import torch
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    if (out_dir / "DONE").exists():
        return out_dir
    from peft import PeftModel
    from safetensors.torch import load_file
    ck = Path(sft_ckpt).resolve()
    meta = json.load(open(ck / "lora_checkpoint_meta.json"))
    # = internvl_lora_checkpoint_io.load_lora_checkpoint(merge_lora=True, load_extra=True), with SDPA attention
    cfg, Model = _fork_config(ck, "sdpa")
    model = Model.from_pretrained(meta["base_model_name_or_path"], config=cfg, torch_dtype=torch.bfloat16,
                                  low_cpu_mem_usage=True, cache_dir=str(HF_HOME))
    if meta.get("use_backbone_lora"):
        model.vision_model = PeftModel.from_pretrained(model.vision_model, str(ck / "vision_lora")).merge_and_unload()
    if meta.get("use_llm_lora"):
        model.language_model = PeftModel.from_pretrained(model.language_model, str(ck / "llm_lora")).merge_and_unload()
    model.mlp1.load_state_dict(load_file(str(ck / "trainable_extra.safetensors")), strict=True)
    if extra_lora:
        model.language_model = PeftModel.from_pretrained(model.language_model, str(extra_lora)).merge_and_unload()
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    snap = base_snapshot()
    ref = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    if set(sd) != set(ref):
        raise RuntimeError(f"merged keys differ from base: missing={sorted(set(ref) - set(sd))[:5]} "
                           f"extra={sorted(set(sd) - set(ref))[:5]}")
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shards, cur, size, lim = [], {}, 0, shard_gb * 2 ** 30
    for k in sorted(sd):
        nb = sd[k].numel() * 2
        if cur and size + nb > lim:
            shards.append(cur); cur, size = {}, 0
        cur[k] = sd[k]; size += nb
    shards.append(cur)
    wmap = {}
    for i, sh in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(tmp / fn), metadata={"format": "pt"})
        wmap.update({k: fn for k in sh})
    json.dump({"metadata": {"total_size": sum(v.numel() * 2 for v in sd.values())}, "weight_map": wmap},
              open(tmp / "model.safetensors.index.json", "w"), indent=1)
    cfgj = json.load(open(ck / "config.json"))
    cfgj["use_backbone_lora"], cfgj["use_llm_lora"] = 0, 0
    cfgj["torch_dtype"] = "bfloat16"
    cfgj["llm_config"]["attn_implementation"] = "sdpa"
    json.dump(cfgj, open(tmp / "config.json", "w"), indent=2)
    for f in list(snap.glob("*.py")) + [snap / "generation_config.json"]:
        shutil.copy(f, tmp / f.name)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (Path(sft_ckpt) / f).exists():
            shutil.copy(Path(sft_ckpt) / f, tmp / f)
    json.dump({"sft_ckpt": str(sft_ckpt), "extra_lora": str(extra_lora) if extra_lora else None},
              open(tmp / "vep_export.json", "w"), indent=2)
    (tmp / "DONE").write_text("ok\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return out_dir


def vllm_lora_state(peft_state):
    """PEFT keys of an LLM-only adapter -> vLLM InternVLChatModel module names, bf16."""
    import torch
    out = {}
    for k, v in peft_state.items():
        k = k.replace(".default.", ".")
        if not k.startswith("base_model.model.model.layers."):
            raise ValueError(f"unexpected LoRA key {k}")
        out["base_model.model.language_model." + k[len("base_model.model."):]] = \
            v.detach().to(torch.bfloat16).contiguous().cpu()
    return out


def save_vllm_lora(state, out_dir, r=16, alpha=32):
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": r, "lora_alpha": alpha, "target_modules": LORA_TARGETS,
               "bias": "none", "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
               "lora_dropout": 0.0, "modules_to_save": None, "task_type": "CAUSAL_LM",
               "base_model_name_or_path": "vep-merged"}, open(out_dir / "adapter_config.json", "w"))
    return out_dir


def peft_dir_to_vllm(peft_dir, out_dir=None):
    """Convert a saved PEFT adapter dir (e.g. grpo-stepN) to the vLLM layout (cached next to it in /dev/shm)."""
    from safetensors.torch import load_file
    peft_dir = Path(peft_dir).resolve()
    out_dir = Path(out_dir) if out_dir else MERGED_ROOT / "lora_cache" / (
        peft_dir.parent.name + "__" + peft_dir.name)
    if (out_dir / "adapter_model.safetensors").exists():
        return out_dir
    cfg = json.load(open(peft_dir / "adapter_config.json"))
    return save_vllm_lora(vllm_lora_state(load_file(str(peft_dir / "adapter_model.safetensors"))),
                          out_dir, r=cfg["r"], alpha=cfg["lora_alpha"])


# ---------------------------------------------------------------- eval files

def xywh(b):
    return {"x": b[0], "y": b[1], "w": max(0.0, b[2] - b[0]), "h": max(0.0, b[3] - b[1])}


def write_pred_dir(out_dir, records):
    """records: dicts with id, domain, gold, pred (xyxy lists) + raw fields. Writes eval_script schema
    (pred_<domain>.json; coords in [0,1000] with image size 1000 as in export_for_eval_script) + raw.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in records:
        by.setdefault(r["domain"], []).append({
            "sample_id": r["id"], "dataset": r["domain"], "image_width": 1000, "image_height": 1000,
            "gt_boxes_raw": [{**xywh(b), "source": "gt", "kind": "bbox"} for b in r["gold"]],
            "pred_boxes_parsed": [xywh(b) for b in r["pred"]],
            "response": r["response"], "n_resp_tokens": r["n_tokens"]})
    for d, rows in by.items():
        json.dump(rows, open(out_dir / f"pred_{d}.json", "w"))
    with open(out_dir / "raw.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_official_eval(pred_dir):
    """Unmodified eval_script.py (many-to-many R/P/F1, MeanIoU, GroupIoU), macro over domains."""
    pred_dir = Path(pred_dir)
    subprocess.run([FLASH_PY, str(EVAL_SCRIPT), "--pred_dir", str(pred_dir), "--tau", *TAUS,
                    "--out_dir", str(pred_dir / "eval")], check=True, stdout=subprocess.DEVNULL)
    ds = json.load(open(pred_dir / "eval" / "all_datasets_summary.json"))["datasets"]
    keys = ["MeanIoU", "GroupIoU", "IoU", "F1@50", "Recall@50", "Precision@50", "F1@70", "Recall@90"]
    macro = {k: sum(v[k] for v in ds.values()) / len(ds) for k in keys}
    return {"macro": macro, "per_domain": {d: {k: v[k] for k in keys} for d, v in ds.items()}}


def read_raw(pred_dir):
    return load_jsonl(Path(pred_dir) / "raw.jsonl")


def setup_caches():
    cache = Path(os.environ.get("VEP_CACHE", "/mnt/data2/traviku2/qwen_vllm_cache"))
    for k, sub in (("VLLM_CACHE_ROOT", "vllm"), ("TRITON_CACHE_DIR", "triton"),
                   ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("XDG_CACHE_HOME", "xdg")):
        os.environ.setdefault(k, str(cache / sub))
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
