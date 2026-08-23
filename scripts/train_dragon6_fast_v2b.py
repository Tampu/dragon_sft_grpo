#!/usr/bin/env python3
"""
Fast SFT launcher for InternVL3-8B on the combined 6-dataset DRAGON grounding
mix (ai2d, infographics, mapwise, mapIQ, circuitvqa, chartqa), at one of three
comparison scales (50/150/300 samples per dataset).

Adapted from train_ai2d_2k_fast_v2b.py: same CLI shape and hyperparameters
(LoRA rank 16 on both backbone and LLM, matching the requested rank), plus
--scale to select which configs/dragon6_meta_n{scale}.json (built via
build_dragon6_meta.py) to train against.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast SFT launcher for InternVL3-8B on the combined 6-dataset DRAGON mix.")
    parser.add_argument("--scale", type=int, required=True, choices=[50, 150, 300], help="Samples per dataset.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for torchrun.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/internvl3_8b_ai2d_sft.json"),
        help="Base config JSON to merge overrides into.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Meta JSON to train against. Default: configs/dragon6_meta_n{scale}.json "
             "(built automatically via build_dragon6_meta.py if it doesn't exist yet).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated config + checkpoints. Default: outputs/dragon6_sft_n{scale}.",
    )
    parser.add_argument("--master-port", type=str, default=os.environ.get("MASTER_PORT", "29600"))
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Trade ~30%% more compute time for much lower activation memory. Needed at "
             "max_seq_length >= ~2048 with per_device_train_batch_size=4 -- without it, "
             "4096 OOMs on a single H200 (observed: 139.49 GiB used of 139.80 GiB capacity).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing (only safe at short max_seq_length).",
    )
    parser.add_argument(
        "--dynamic-image-size",
        action="store_true",
        help="Split each image into up to --max-dynamic-patch tiles (+ thumbnail if "
             "--use-thumbnail) instead of a single downsampled 448x448 tile. Needed for "
             "small text (chart axes, legends, infographic labels) to be legible to the ViT "
             "-- single-tile compresses e.g. a 2000px-wide chart down to 448px.",
    )
    parser.add_argument("--use-thumbnail", action="store_true")
    parser.add_argument("--max-dynamic-patch", type=int, default=12)
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES"),
        help="Comma-separated GPU ids to expose, e.g. '7' or '4,6'.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HF_HOME for the training subprocess. Default: <repo>/hf_home, which already "
             "holds the full InternVL3-8B cache -- without this, from_pretrained() falls back "
             "to ~/.cache/huggingface, which lives on the near-full root disk (observed: "
             "training died with 'No space left on device' trying to re-download shards there).",
    )
    parser.add_argument(
        "--check-choices-lines",
        type=int,
        default=200,
        help="How many lines to sample from each annotation for verifying 'Options:' in user prompt.",
    )
    parser.add_argument(
        "--require-choices",
        action="store_true",
        default=True,
        help="Fail fast if sampled prompts do not contain options/choices.",
    )
    parser.add_argument(
        "--no-require-choices",
        action="store_false",
        dest="require_choices",
        help="Disable the choices presence check.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of dataset keys to include when building the meta.json "
             "(default: all 6). Forwarded to build_dragon6_meta.py -- useful for dry-running "
             "against ai2d_grounding alone before the other 5 datasets exist.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=str,
        default="",
        help="Comma-separated global steps to save an adapter-only checkpoint at, "
             "e.g. '100,250,500,750,1000'. Independent of save_strategy.",
    )
    parser.add_argument(
        "--eval-after-checkpoint",
        action="store_true",
        help="After each --checkpoint-steps save, block training and run holdout eval "
             "against it (writes to <checkpoint-dir>/eval).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    base_config = (root / args.base_config).resolve() if not args.base_config.is_absolute() else args.base_config
    meta_path = args.meta_path or Path(f"configs/dragon6_meta_n{args.scale}.json")
    meta_path = (root / meta_path).resolve() if not meta_path.is_absolute() else meta_path
    work_dir = args.work_dir or Path(f"outputs/dragon6_sft_n{args.scale}")
    work_dir = (root / work_dir).resolve() if not work_dir.is_absolute() else work_dir
    train_script = root / "scripts" / "internvl_chat_finetune.py"
    torchrun = Path(sys.executable).with_name("torchrun")

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    if not meta_path.exists():
        build_cmd = [
            sys.executable,
            str(root / "scripts" / "build_dragon6_meta.py"),
            "--scale", str(args.scale),
            "--output", str(meta_path),
        ]
        if args.only:
            build_cmd += ["--only", args.only]
        print("Meta file not found, building it:", " ".join(build_cmd))
        subprocess.run(build_cmd, check=True, cwd=str(root))

    meta = json.loads(meta_path.read_text())
    total_samples = 0
    for ds_name, ds_meta in meta.items():
        ann = Path(ds_meta["annotation"])
        if not ann.is_absolute():
            ann = root / ann
        if not ann.exists():
            raise FileNotFoundError(f"Annotation file not found for dataset '{ds_name}': {ann}")
        with ann.open("r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        total_samples += count
        print(f"Dataset '{ds_name}': {count} samples from {ann}")
        if args.require_choices:
            checked = 0
            bad = 0
            with ann.open("r", encoding="utf-8") as f:
                for line in f:
                    if checked >= args.check_choices_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    checked += 1
                    try:
                        obj = json.loads(line)
                        convs = obj.get("conversations", [])
                        user_text = convs[0].get("value", "") if convs else ""
                    except Exception:
                        bad += 1
                        continue
                    if "Options:" not in user_text:
                        bad += 1
            print(f"Choices check '{ds_name}': {checked - bad}/{checked} lines include 'Options:'")
            if checked > 0 and bad > 0:
                raise RuntimeError(
                    f"Choices check failed for dataset '{ds_name}' ({bad}/{checked} sampled lines missing 'Options:'). "
                    "Regenerate its annotation with prepare_dragon_grounding_sft.py."
                )
    print(f"Target scale: {args.scale}/dataset x {len(meta)} dataset(s) = {total_samples} total samples")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "ckpts"
    config_out = work_dir / f"internvl3_8b_dragon6_n{args.scale}.json"

    cfg = json.loads(base_config.read_text())
    cfg.update(
        {
            "model_name_or_path": "OpenGVLab/InternVL3-8B",
            "freeze_llm": False,
            "freeze_backbone": False,
            "use_llm_lora": 16,
            "use_backbone_lora": 16,
            "max_seq_length": args.max_seq_length,
            "force_image_size": 448,
            "dynamic_image_size": args.dynamic_image_size,
            "use_thumbnail": args.use_thumbnail,
            "max_dynamic_patch": args.max_dynamic_patch,
            "conv_style": "internlm2-chat",
            "meta_path": str(meta_path),
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "num_train_epochs": args.epochs,
            "gradient_checkpointing": args.gradient_checkpointing,
            "grad_checkpoint": args.gradient_checkpointing,
            "bf16": True,
            "tf32": True,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "report_to": "none",
            "logging_steps": 20,
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "save_strategy": "no",
            "checkpoint_steps": args.checkpoint_steps,
            "eval_after_checkpoint": args.eval_after_checkpoint,
        }
    )
    config_out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"Saved config: {config_out}")

    internvl_chat_dir = root / "InternVL" / "internvl_chat"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = str(internvl_chat_dir) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    hf_cache_dir = Path(args.hf_cache_dir).resolve() if args.hf_cache_dir else (root / "hf_home")
    if not (hf_cache_dir / "hub" / "models--OpenGVLab--InternVL3-8B").exists():
        print(f"[warn] {hf_cache_dir} has no cached InternVL3-8B snapshot -- "
              f"from_pretrained() will try to download it there (needs ~16GB free).")
    print(f"Using HF_HOME={hf_cache_dir}")

    env = dict(
        os.environ,
        LAUNCHER="pytorch",
        MASTER_ADDR=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        MASTER_PORT=args.master_port,
        PYTHONPATH=pythonpath,
        HF_HOME=str(hf_cache_dir),
    )
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    cmd = [
        str(torchrun),
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        str(train_script),
        str(config_out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(root))


if __name__ == "__main__":
    main()
