#!/usr/bin/env bash
# Per-domain eval of any 5k-run checkpoint. Parameterized so it isn't another
# one-off script.   usage:  bash run_export_ckpt.sh <step> <gpu> [<gpu> ...]
# Writes to preds_for_eval/grpo_5k_step<step>/ -- refuses to clobber.
set -uo pipefail
cd /mnt/data2/traviku2
source da_env_flash/bin/activate

STEP=${1:?usage: run_export_ckpt.sh <step> <gpu> [<gpu> ...]}; shift
GPUS=("$@"); [ ${#GPUS[@]} -gt 0 ] || { echo "need >=1 gpu"; exit 1; }

SFT=outputs/dragon6_sft2k_v4/ckpts
GRPO=outputs/dragon6_grpo_5k/grpo-step${STEP}
OUT=preds_for_eval/grpo_5k_step${STEP}
DOMAINS=(ai2d chartqa circuitvqa infographics mapiq mapwise)

[ -d "$GRPO" ] || { echo "no such checkpoint: $GRPO"; exit 1; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT already exists"; exit 1; }

run_set () {
  local gpu=$1; shift
  for d in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu python3 dragon_grpo/export_for_eval_script.py export \
      --checkpoint "$SFT" --grpo-checkpoint "$GRPO" --hf-cache-dir hf_home \
      --jsonl "dragon_datasets_sft2k/holdout_per_domain/${d}.jsonl" \
      --dataset-name "$d" --image-root Diagram_Attribution_Dataset --out-dir "$OUT" \
      --device cuda:0 --skip-matching-compare \
      > "logs_export_${d}_step${STEP}.log" 2>&1
    echo "  done $d on gpu$gpu (exit $?)"
  done
}

# round-robin the 6 domains across the given GPUs
i=0
for g in "${GPUS[@]}"; do
  set -- ; assigned=()
  for ((j=i; j<${#DOMAINS[@]}; j+=${#GPUS[@]})); do assigned+=("${DOMAINS[$j]}"); done
  run_set "$g" "${assigned[@]}" &
  i=$((i+1))
done
wait
echo "=== ALL EXPORTS DONE (step ${STEP}) ==="
python3 dragon_grpo/eval_script.py --pred_dir "$OUT" --tau 0.1 0.3 0.5 0.7 0.9 \
  --out_dir "$OUT/eval_tau9"
