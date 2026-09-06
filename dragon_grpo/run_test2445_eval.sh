#!/usr/bin/env bash
# Evaluate on the OFFICIAL DRAGON test split (2,432 scoreable of 2,445) --
# the only numbers that belong in Table 3.
#
#   usage: bash run_test2445_eval.sh <base|sft|grpo> <gpu> [<gpu> ...]
#
#   sft   = SFT checkpoint alone (your Table 3 baseline row)
#   grpo  = SFT + GRPO-5k step-2500 adapter (the contribution row)
#
# Round-robins the 6 domains across the given GPUs, then runs eval_script.py
# unmodified. Refuses to clobber an existing output dir.
set -uo pipefail
cd /mnt/data2/traviku2
source da_env_flash/bin/activate

ARM=${1:?usage: run_test2445_eval.sh <base|sft|grpo> <gpu> [<gpu> ...]}; shift
GPUS=("$@"); [ ${#GPUS[@]} -gt 0 ] || { echo "need >=1 gpu"; exit 1; }

SFT=outputs/dragon6_sft2k_v4/ckpts
case "$ARM" in
  base) EXTRA="--no-adapters";                                      OUT=preds_for_eval/TEST2445_base ;;
  sft)  EXTRA="";                                                   OUT=preds_for_eval/TEST2445_sft ;;
  grpo) EXTRA="--grpo-checkpoint outputs/dragon6_grpo_5k/grpo-step2500"; OUT=preds_for_eval/TEST2445_grpo5k_step2500 ;;
  *) echo "arm must be 'base', 'sft' or 'grpo'"; exit 1 ;;
esac
DOMAINS=(ai2d chartqa circuitvqa infographics mapiq mapwise)
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 1; }

run_set () {
  local gpu=$1; shift
  for d in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu python3 dragon_grpo/export_for_eval_script.py export \
      --checkpoint "$SFT" $EXTRA --hf-cache-dir hf_home \
      --jsonl "dragon_datasets_test2445/${d}.jsonl" \
      --dataset-name "$d" --image-root Diagram_Attribution_Dataset \
      --out-dir "$OUT" --device cuda:0 --skip-matching-compare \
      > "logs_test2445_${ARM}_${d}.log" 2>&1
    echo "  [$ARM] done $d on gpu$gpu (exit $?)"
  done
}

i=0
for g in "${GPUS[@]}"; do
  assigned=()
  for ((j=i; j<${#DOMAINS[@]}; j+=${#GPUS[@]})); do assigned+=("${DOMAINS[$j]}"); done
  run_set "$g" "${assigned[@]}" &
  i=$((i+1))
done
wait
echo "=== ALL EXPORTS DONE ($ARM) ==="

# both matchings, on the same predictions
for d in "${DOMAINS[@]}"; do
  python3 dragon_grpo/export_for_eval_script.py rescore \
    --pred-json "$OUT/pred_${d}.json" > /dev/null 2>&1
done
python3 dragon_grpo/eval_script.py --pred_dir "$OUT" --tau 0.1 0.3 0.5 0.7 0.9 \
  --out_dir "$OUT/eval_tau9"
