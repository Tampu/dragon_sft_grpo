#!/usr/bin/env bash
# Evaluate one arm (base / sft / grpo) on the official test split, all six
# domains, then run eval_script.py unmodified. Generic across model_configs
# -- pass the same --model-config you trained with.
#
#   usage: run_eval_arms.sh <model-config.json> <base|sft|grpo> <out-dir> \
#              [<sft-ckpt>] [<grpo-ckpt>]
#
# Example:
#   ./run_eval_arms.sh ../configs/models/qwen3vl_8b_thinking.json sft \
#       preds_for_eval/sft outputs/sft/final
#   ./run_eval_arms.sh ../configs/models/qwen3vl_8b_thinking.json grpo \
#       preds_for_eval/grpo outputs/sft/final outputs/grpo/grpo-step2500
set -euo pipefail
cd "$(dirname "$0")"

MODEL_CFG=${1:?usage: run_eval_arms.sh <model-config.json> <base|sft|grpo> <out-dir> [<sft-ckpt>] [<grpo-ckpt>]}
ARM=${2:?}
OUT=${3:?}
SFT_CKPT=${4:-}
GRPO_CKPT=${5:-}

EXTRA=()
case "$ARM" in
  base) : ;;
  sft)  EXTRA=(--sft-checkpoint "$SFT_CKPT") ;;
  grpo) EXTRA=(--sft-checkpoint "$SFT_CKPT" --grpo-checkpoint "$GRPO_CKPT") ;;
  *) echo "arm must be 'base', 'sft' or 'grpo'"; exit 1 ;;
esac

[ -e "$OUT" ] && { echo "REFUSING: $OUT already exists"; exit 1; }
DOMAINS=(ai2d chartqa circuitvqa infographics mapiq mapwise)

for d in "${DOMAINS[@]}"; do
  python3 export_for_eval.py --model-config "$MODEL_CFG" "${EXTRA[@]}" \
    --jsonl "../dragon_datasets/test_split/${d}.jsonl" \
    --dataset-name "$d" --image-root ../Diagram_Attribution_Dataset \
    --out-dir "$OUT" --device cuda:0 \
    2>&1 | tee "logs_eval_${ARM}_${d}.log"
  echo "  [$ARM] done $d"
done

python3 eval_script.py --pred_dir "$OUT" --tau 0.1 0.3 0.5 0.7 0.9 \
  --out_dir "$OUT/eval"
