#!/usr/bin/env bash
# Val600 checkpoint curve for the main GRPO run (appendix; decides whether step 2633 stays).
# Greedy at 640 tokens through vLLM, the same path as run_main.sh ARM=test, on SFT (step 0) and every saved
# GRPO checkpoint. Then the official eval_script via score_test.py. Test2445 is not touched.
#
#   GPU=5 bash dragon_grpo_r1/paper/run_val_sweep.sh        (~1 h; re-running skips finished checkpoints)
#
# score_test.py prints "Test2445"/"Test2432" headers and a 2,445-item warning: ignore them, both tables are
# Val600 (every Val item has gold, so the two tables are identical).
set -euo pipefail
I35=/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old
R1=/mnt/data2/traviku2/dragon_grpo_r1
DATA=$I35/data_paper
PY=/mnt/data2/traviku2/da_env2/bin/python
GRPO=$R1/outputs/paper/grpo
OUT=$R1/outputs/paper/val600_sweep
mkdir -p $OUT
exec 9>$OUT/.lock; flock -n 9 || { echo "another sweep holds the lock"; exit 1; }
cd $R1
N=$(wc -l < $DATA/val600.jsonl)
$PY -c "import json; print('\n'.join(json.loads(l)['id'] for l in open('$DATA/val600.jsonl')))" > $OUT/val600_ids.txt

gen() {  # $1 = name, rest = generate_states args
  local O=$OUT/$1.jsonl; shift
  [ -f $O ] && [ "$(wc -l < $O)" = "$N" ] && { echo "$O done"; return; }
  rm -f $O
  CUDA_VISIBLE_DEVICES=${GPU:?set GPU (4-7)} $PY completer_pilot/generate_states.py --base paper_sft \
      --split $DATA/val600.jsonl --K 0 --out $O "$@"
}

ROWS=(--states step0=$OUT/step0.jsonl)
gen step0
for s in 500 1000 1500 2000 2500 2633; do
  gen step$s --lora $GRPO/grpo-step$s
  ROWS+=(--states step$s=$OUT/step$s.jsonl)
done
CUDA_VISIBLE_DEVICES= $PY paper/score_test.py "${ROWS[@]}" --scorable $OUT/val600_ids.txt --out-dir $OUT/report
echo "curve: $OUT/report/report.txt"
