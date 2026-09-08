#!/usr/bin/env python3
"""
Generic LoRA SFT trainer -- reads a converted chat JSONL (see
convert_to_chat_jsonl.py) and a --model-config, and is otherwise identical
code regardless of which HF vision-language model that config points at.

Multi-GPU: this is a plain `transformers.Trainer` script, so launch it the
standard way, e.g.:
    accelerate launch train_sft.py --model-config ... --train-jsonl ...
    # or: torchrun --nproc_per_node=N train_sft.py ...
No model-specific launcher/merged-config step is needed (unlike a bespoke
repo's own finetune.py) -- Trainer picks up distributed state from the
launcher's environment variables.

Label masking (prompt tokens ignored, only the assistant's JSON-array turn
contributes to the loss) is computed via LONGEST-COMMON-PREFIX matching
between the full conversation's tokenized ids and the prompt-only (system+
user, generation-prompt open) tokenized ids, rather than assuming the two
renders share an exact-length prefix. This is deliberately defensive: for
most chat templates the two are byte-identical up to the assistant content
and this degenerates to a plain length cut, but a thinking-capable model
with enable_thinking=False can inject an empty <think></think> block into
the generation-prompt render that the real (non-empty-think) assistant turn
never contained -- LCP matching finds the true shared boundary either way
instead of silently mis-masking. A large gap between the LCP length and the
prompt-only length is logged so a mismatch is visible rather than silent.

NOT YET EMPIRICALLY VERIFIED: this repo was authored without GPU/model
access. The multimodal batching path (padding input_ids/attention_mask/
labels, concatenating every other processor output -- pixel_values,
image_grid_thw, etc. -- along dim 0) follows the standard pattern used by
HF's own multimodal SFT examples for the Qwen-VL family, but confirm the
first training step's loss is finite and `trainable_extra`-free adapter
size looks sane before trusting a long run. See README.md's "Before you
trust a long run" checklist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grounding_prompts import build_chat_messages
from model_config import apply_lora, load_model_and_processor, load_model_config, render_chat_text
from lora_checkpoint_io import save_adapter_checkpoint

IGNORE_INDEX = -100


class GroundingSFTDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_root: Path, processor, cfg):
        self.samples = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        self.image_root = Path(image_root)
        self.processor = processor
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image

        sample = self.samples[idx]
        image_abs = str((self.image_root / sample["image"]).resolve())
        image = Image.open(image_abs).convert("RGB")

        full_messages = build_chat_messages(sample, image_abs, include_assistant=True)
        prompt_messages = build_chat_messages(sample, image_abs, include_assistant=False)

        full_text = render_chat_text(self.processor, full_messages, self.cfg, add_generation_prompt=False)
        prompt_text = render_chat_text(self.processor, prompt_messages, self.cfg, add_generation_prompt=True)

        full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")

        full_ids = full_inputs["input_ids"][0]
        prompt_ids = prompt_inputs["input_ids"][0]

        lcp = 0
        max_check = min(len(full_ids), len(prompt_ids))
        while lcp < max_check and full_ids[lcp] == prompt_ids[lcp]:
            lcp += 1
        if lcp < 0.9 * len(prompt_ids):
            print(f"[warn] sample {sample.get('id')}: prompt/full common-prefix length "
                  f"{lcp} is well short of the prompt's own length {len(prompt_ids)} -- "
                  f"label masking may be off for this model's chat template; inspect.")

        labels = full_ids.clone()
        labels[:lcp] = IGNORE_INDEX

        item: Dict[str, torch.Tensor] = {"input_ids": full_ids,
                                         "attention_mask": full_inputs["attention_mask"][0],
                                         "labels": labels}
        for k, v in full_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            item[k] = v[0] if torch.is_tensor(v) and v.shape[0] == 1 else v
        return item


class GroundingCollator:
    """Pads input_ids/attention_mask/labels to the batch max length;
    concatenates every other processor output (pixel_values, image_grid_thw,
    or whatever a given model's processor emits) along dim 0 unchanged --
    the standard flattened-multi-image batching convention the Qwen-VL
    family and similar processors expect, and a generic default that needs
    no per-model special-casing."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            n = item["input_ids"].shape[0]
            pad = max_len - n
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=self.pad_token_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad), value=IGNORE_INDEX))

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        extra_keys = [k for k in batch[0] if k not in ("input_ids", "attention_mask", "labels")]
        for k in extra_keys:
            vals = [item[k] for item in batch]
            out[k] = torch.cat(vals, dim=0) if torch.is_tensor(vals[0]) else vals
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--deepspeed", default=None, help="optional deepspeed config json")
    args = ap.parse_args()

    import os

    from transformers import Trainer, TrainingArguments

    # Under torchrun/accelerate multi-GPU launch, each process must place its
    # model on ITS OWN local-rank device -- a hardcoded "cuda:0" here would
    # put every rank's model on GPU 0 and silently break DDP. LOCAL_RANK is
    # set by both launchers.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    cfg = load_model_config(args.model_config)
    model, processor = load_model_and_processor(cfg, device=device, hf_cache_dir=args.hf_cache_dir)
    model = apply_lora(model, cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    dataset = GroundingSFTDataset(args.train_jsonl, args.image_root, processor, cfg)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    collator = GroundingCollator(pad_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_checkpointing=False,  # enabled manually above (needs enable_input_require_grads with peft)
        deepspeed=args.deepspeed,
        report_to=[],
        # MULTIMODAL TRAINER GOTCHA: without this, Trainer inspects the
        # model's forward signature and silently drops any batch column
        # (pixel_values, image_grid_thw, ...) it doesn't recognize as a
        # named forward argument on the *outer* PeftModel wrapper.
        remove_unused_columns=False,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    save_adapter_checkpoint(model, processor, Path(args.output_dir) / "final", cfg)
    print(f"Done. Adapter checkpoint -> {Path(args.output_dir) / 'final'}")


if __name__ == "__main__":
    main()
