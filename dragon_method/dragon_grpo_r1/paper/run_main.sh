#!/usr/bin/env bash
# Paper main experiment, InternVL3.5-8B (method frozen, Nov 6 2026). Approved prompt: data_paper/ (build_data.py).
# No checkpoint selection anywhere: SFT = the final adapter (3 epochs), GRPO = the final step (2633).
#
#   ARM=sft     GPUS=4,7 bash dragon_grpo_r1/paper/run_main.sh   # ~1 h 40 min on 2 GPUs (3 h 15 min on 1)
#   ARM=export           bash dragon_grpo_r1/paper/run_main.sh   # CPU: merge base + SFT -> /dev/shm/vep_merged/paper_sft
#   ARM=grpo    GPU=4    bash dragon_grpo_r1/paper/run_main.sh   # vanilla GRPO, vLLM rollouts, standard GRPO, ~5 h
#   ARM=test    GPU=4    bash dragon_grpo_r1/paper/run_main.sh   # Test2445: SFT greedy, GRPO greedy + 8 samples, scores
#
# Outputs: SFT  dragon_sft_grpo_internvl35_old/outputs/paper_sft
#          GRPO dragon_grpo_r1/outputs/paper/grpo
#          Test dragon_grpo_r1/outputs/paper/test2445/{sft,grpo}.jsonl + report_main.txt
set -euo pipefail
ARM=${ARM:?set ARM=sft, export, grpo or test}
I35=/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old
R1=/mnt/data2/traviku2/dragon_grpo_r1
DATA=$I35/data_paper
PY=/mnt/data2/traviku2/da_env2/bin/python
FLASH=/mnt/data2/traviku2/da_env_flash/bin/python
SFT=$I35/outputs/paper_sft
OUT=$R1/outputs/paper
GRPO=$OUT/grpo
TEST=$OUT/test2445
mkdir -p $OUT
exec 9>$OUT/.lock_$ARM; flock -n 9 || { echo "another ARM=$ARM run holds the lock"; exit 1; }
[ -f $DATA/test2445.jsonl ] || { echo "run dragon_grpo_r1/paper/build_data.py first"; exit 1; }

case $ARM in
sft)
  # Same recipe as run_i35_sft.sh: 3 epochs, lr 3e-5, effective batch 8 (4 per GPU), 12 tiles, 4352 tokens.
  GPUS=${GPUS:?set GPUS, e.g. 4 or 4,7}
  NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
  GA=$(( 2 / NGPU ))
  [ -e $SFT ] && { echo "REFUSING: $SFT exists (the trainer overwrites its output dir)"; exit 1; }
  cd $I35
  HF_HOME=$I35/hf_home $FLASH -u scripts/train_dragon6_fast_v2b.py \
    --model-path OpenGVLab/InternVL3_5-8B --scale 300 \
    --meta-path $DATA/sft2000_meta.json --work-dir outputs/paper_sft \
    --gpus $NGPU --cuda-visible-devices $GPUS --grad-accum $GA \
    --epochs 3 --lr 3e-5 --max-seq-length 4352 \
    --dynamic-image-size --use-thumbnail --max-dynamic-patch 12 \
    --no-require-choices --checkpoint-steps 250,500 \
    2>&1 | tee $OUT/sft.log
  ;;
export)
  [ -f $SFT/ckpts/lora_checkpoint_meta.json ] || { echo "no final SFT adapter at $SFT/ckpts"; exit 1; }
  CUDA_VISIBLE_DEVICES= $PY $R1/completer_pilot/export_grounder.py --sft-ckpt $SFT/ckpts --lora none --name paper_sft
  ;;
grpo)
  # Vanilla GRPO with the original recipe; only the rollout backend is vLLM (colocated, LoRA synced each step).
  # standard GRPO: per-group z-scored advantages, per-rollout token-mean loss; 320-token rollouts as the original.
  [ -f /dev/shm/vep_merged/paper_sft/DONE ] || { echo "run ARM=export first"; exit 1; }
  cd $R1
  CUDA_VISIBLE_DEVICES=${GPU:?set GPU} $PY -u completer_pilot/train_gc.py --task coverage --reward-version vanilla \
    --grpo-norm standard --base paper_sft --jsonl $DATA/grpo_pool.jsonl \
    --group-size 8 --prompts-per-step 2 --max-new-tokens 320 --no-repeat-ngram-size 6 --temperature 1.0 \
    --lr 1e-5 --kl-beta 0.02 --clip-eps 0.2 --max-grad-norm 1.0 --lora-rank 16 \
    --curriculum-frac 0 --div-every 0 --max-steps 2633 --save-every 500 --seed 17 \
    --output-dir $GRPO $( [ -f $GRPO/trainer_state.pt ] && echo --resume ) \
    2>&1 | tee -a $OUT/grpo.log
  ;;
test)
  # Greedy at 640 tokens (every Test number in this project); GRPO also draws the 8 samples the Assembler and
  # voting_drop use, in the same pass. Then the official eval_script on 2,445 (and the scorable 2,432).
  LAST=$(ls -d $GRPO/grpo-step* | sed 's/.*step//' | sort -n | tail -1)
  [ "$LAST" = 2633 ] || { echo "GRPO not finished (last checkpoint step $LAST)"; exit 1; }
  cd $R1
  CUDA_VISIBLE_DEVICES=${GPU:?set GPU} $PY completer_pilot/generate_states.py --base paper_sft \
    --split $DATA/test2445.jsonl --K 0 --out $TEST/sft.jsonl
  CUDA_VISIBLE_DEVICES=$GPU $PY completer_pilot/generate_states.py --base paper_sft --lora $GRPO/grpo-step2633 \
    --split $DATA/test2445.jsonl --K 8 --out $TEST/grpo.jsonl
  CUDA_VISIBLE_DEVICES= $PY paper/score_test.py --states sft=$TEST/sft.jsonl --states grpo=$TEST/grpo.jsonl \
    --scorable $DATA/test2432_ids.txt --out-dir $TEST/report_main
  ;;
*) echo "ARM must be sft, export, grpo or test"; exit 1 ;;
esac
