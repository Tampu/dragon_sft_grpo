#!/usr/bin/env bash
# Paper: Ours + ablations + analyses on the new InternVL Grounder (after run_main.sh). One seed (17) for now.
# Method frozen = the audited v4loo recipe (paper/audit_v4loo.py): prompt v4t, LOO credit, oracle warm-up 300,
# GRPO 400, K = 8. No checkpoint selection: every verifier is reported at its final step (400).
#
#   ARM=grounder         bash dragon_grpo_r1/paper/run_ours.sh  # CPU: merged SFT + GRPO -> /dev/shm/vep_merged/paper_grounder
#   ARM=states    GPU=4  bash ...   # train states: greedy + 8 samples on the 5,266 GRPO-pool items (~1.5 h)
#   ARM=ours      GPU=4  bash ...   # A6 Ours                                  (~5-6 h)
#   ARM=flat      GPU=5  bash ...   # A4 flat credit: set reward only, no LOO  (~5-6 h)
#   ARM=novis     GPU=7  bash ...   # A5 no vision: both images blank, trained (~5-6 h)
#   ARM=eval      GPU=4  bash ...   # Test2445: A2 untrained, A3 warm-up, A4, A5, A6 + Ours at K = 1, 2, 4 (~1 h)
#   ARM=cond      GPU=4  bash ...   # analysis 3: Grounder + Ours with a wrong / no answer (~1 h)
#   ARM=timing    GPU=4  bash ...   # analysis 2 cost: seconds/item for greedy, 8 samples, verifier (200 items, ~10 min)
#   ARM=report           bash ...   # CPU: main + ablation tables, analyses 1-5
# ours / flat / novis are independent: run them side by side once ARM=states is done.
set -euo pipefail
ARM=${ARM:?set ARM=grounder, states, ours, flat, novis, eval, cond or report}
I35=/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old
R1=/mnt/data2/traviku2/dragon_grpo_r1
DATA=$I35/data_paper
PY=/mnt/data2/traviku2/da_env2/bin/python
OUT=$R1/outputs/paper
GLORA=$OUT/grpo/grpo-step2633
TRAIN=$OUT/states/train.jsonl
TEST=$OUT/test2445
V=$OUT/verifier
EV=$TEST/eval
mkdir -p $OUT/states $V $EV
exec 9>$OUT/.lock_ours_$ARM; flock -n 9 || { echo "another ARM=$ARM run holds the lock"; exit 1; }
cd $R1
gpu() { CUDA_VISIBLE_DEVICES=${GPU:?set GPU (4-7)} $PY "$@"; }
cpu() { CUDA_VISIBLE_DEVICES= $PY "$@"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done; }

COMMON=(--task completer --base paper_grounder --states $TRAIN --K 8 --group-size 4 --prompts-per-step 8
        --curriculum-frac 0 --seed 17 --prompt v4t --warmup-mode oracle --oracle-steps 300 --max-steps 400
        --save-every 100 --ab-weight 0.5 --types data_count/elements_grpo_rest.json)
train() {   # $1 = run name, rest = arm-specific args
  local N=$V/$1; shift
  need /dev/shm/vep_merged/paper_grounder/DONE $TRAIN
  [ -f $N/DONE ] && { echo "$N done"; return; }
  local R_=(); [ -f $N/trainer_state.pt ] && R_=(--resume)
  gpu -u completer_pilot/train_gc.py "${COMMON[@]}" "$@" --output-dir $N "${R_[@]}" 2>&1 | tee -a $N.log
}
ev() {      # $1 = out name, $2 = states, $3 = K, $4 = TAG=PATH, rest = extra eval args
  local O=$1 S=$2 K=$3 L=$4; shift 4
  gpu completer_pilot/eval_completer.py --base paper_grounder --prompt v4t --states $S --K $K \
      --out-dir $EV/$O --skip-existing --lora $L "$@"
}

case $ARM in
grounder)
  need $GLORA/adapter_model.safetensors $I35/outputs/paper_sft/ckpts/lora_checkpoint_meta.json
  cpu completer_pilot/export_grounder.py --sft-ckpt $I35/outputs/paper_sft/ckpts --lora $GLORA --name paper_grounder
  ;;
states)
  # same path as the Test states (run_main.sh ARM=test): LoRA on the SFT-merged base, greedy + 8 at T = 1, 640 tokens
  need $GLORA/adapter_model.safetensors
  gpu completer_pilot/generate_states.py --base paper_sft --lora $GLORA --split $DATA/grpo_pool.jsonl --K 8 --out $TRAIN
  [ "$(wc -l < $TRAIN)" = 5266 ] || { echo "train states incomplete: $(wc -l < $TRAIN)"; exit 1; }
  ;;
ours)  train ours  --credit loo   --verify-credit 0.5 ;;
flat)  train flat  --credit label --verify-credit 0   ;;    # sequence-level set reward only: no per-candidate credit
novis) train novis --credit loo   --verify-credit 0.5 --ablate blank_visual ;;
eval)
  need $TEST/grpo.jsonl $V/ours/DONE
  S=$TEST/grpo.jsonl
  ev untrained   $S 8 step0=none                                    # A2: same prompt and inputs, no LoRA
  ev ours        $S 8 warmup=$V/ours/warmup                         # A3: oracle-edit warm-up only (step 300 of SFT)
  ev ours        $S 8 step400=$V/ours/grpo-step400                  # A6: Ours
  for k in 1 2 4; do ev ours_K$k $S $k step400=$V/ours/grpo-step400; done      # analysis 2
  [ -f $V/flat/DONE ]  && ev flat  $S 8 step400=$V/flat/grpo-step400
  [ -f $V/novis/DONE ] && ev novis $S 8 step400=$V/novis/grpo-step400 --ablate blank_visual
  [ -f $V/flat/DONE ] && [ -f $V/novis/DONE ] || echo "NOTE: rerun ARM=eval after flat / novis finish (skip-existing keeps the rest)"
  ;;
cond)
  need $V/ours/DONE $DATA/test2445_wrong.jsonl
  for c in wrong none; do
    gpu completer_pilot/generate_states.py --base paper_sft --lora $GLORA --split $DATA/test2445_$c.jsonl --K 8 \
        --out $TEST/grpo_$c.jsonl
    ev ours_$c $TEST/grpo_$c.jsonl 8 step400=$V/ours/grpo-step400
  done
  ;;
timing)
  # wall-clock per item on one GPU, 200 Test items (first 34 per domain), engine start-up excluded by subtraction
  need $V/ours/DONE
  TD=$TEST/timing; mkdir -p $TD
  $PY -c "
import json, collections
c = collections.Counter(); out = open('$TD/sub.jsonl', 'w')
for l in open('$DATA/test2445.jsonl'):
    d = json.loads(l)['id'].split('::')[0]
    if c[d] < 34: c[d] += 1; out.write(l)"
  t() { local s=$(date +%s.%N); "$@" > /dev/null 2>&1; echo "$(date +%s.%N) - $s" | bc; }
  rm -f $TD/k0.jsonl $TD/k8.jsonl $TD/k0b.jsonl; rm -rf $TD/ev $TD/ev0
  T0=$(t gpu completer_pilot/generate_states.py --base paper_sft --lora $GLORA --split $TD/sub.jsonl --K 0 --out $TD/k0.jsonl)
  T8=$(t gpu completer_pilot/generate_states.py --base paper_sft --lora $GLORA --split $TD/sub.jsonl --K 8 --out $TD/k8.jsonl)
  # engine start-up: the same call on 1 item
  head -1 $TD/sub.jsonl > $TD/one.jsonl
  TS=$(t gpu completer_pilot/generate_states.py --base paper_sft --lora $GLORA --split $TD/one.jsonl --K 0 --out $TD/k0b.jsonl)
  TV=$(t gpu completer_pilot/eval_completer.py --base paper_grounder --prompt v4t --states $TD/k8.jsonl --K 8 --out-dir $TD/ev --lora s=$V/ours/grpo-step400)
  head -1 $TD/k8.jsonl > $TD/k8one.jsonl
  TV0=$(t gpu completer_pilot/eval_completer.py --base paper_grounder --prompt v4t --states $TD/k8one.jsonl --K 8 --out-dir $TD/ev0 --lora s=$V/ours/grpo-step400)
  $PY -c "
n = $(wc -l < $TD/sub.jsonl)
g = ($T0 - $TS) / n; s8 = ($T8 - $T0) / n; v = ($TV - $TV0) / n
import json; json.dump({'greedy': g, '8 samples': s8, 'verifier': v, 'GRPO total': g, 'voting_drop total': g + s8,
                        'Ours total': g + s8 + v}, open('$TEST/timing.json', 'w'), indent=1); print(open('$TEST/timing.json').read())"
  ;;
report)
  cpu paper/report_paper.py --out-dir $TEST/report_paper
  cpu paper/analyses.py all --out-dir $TEST/analyses
  ;;
*) echo "unknown ARM=$ARM"; exit 1 ;;
esac
