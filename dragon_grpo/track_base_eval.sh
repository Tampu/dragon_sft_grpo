#!/usr/bin/env bash
# Progress tracker for run_test2445_eval.sh. usage: bash track_base_eval.sh [arm]
cd /mnt/data2/traviku2
ARM=${1:-base}
declare -A TOT=( [ai2d]=483 [chartqa]=388 [circuitvqa]=398 [infographics]=411 [mapiq]=363 [mapwise]=389 )
case "$ARM" in
  base) OUT=preds_for_eval/TEST2445_base ;;
  sft)  OUT=preds_for_eval/TEST2445_sft ;;
  grpo) OUT=preds_for_eval/TEST2445_grpo5k_step2500 ;;
esac
done_n=0; total=0; running=0
printf "  %-15s %>0s\n" "" "" 2>/dev/null
printf "  %-15s %11s %9s  %s\n" "domain" "progress" "pct" "state"
printf "  %s\n" "------------------------------------------------------"
for d in ai2d chartqa circuitvqa infographics mapiq mapwise; do
  t=${TOT[$d]}; total=$((total+t))
  if [ -f "$OUT/pred_$d.json" ]; then
    n=$t; state="DONE"
  else
    log="logs_test2445_${ARM}_${d}.log"
    # one "pad_token_id" line per generate() call = finest-grained progress
    n=$(grep -c "pad_token_id" "$log" 2>/dev/null); [ -z "$n" ] && n=0
    m=$(grep -oE '^\s+\[[0-9]+/' "$log" 2>/dev/null | tail -1 | tr -dc 0-9)
    [ -n "$m" ] && [ "$m" -gt "$n" ] && n=$m
    if [ -f "$log" ]; then state="running"; running=$((running+1)); else state="queued"; fi
  fi
  done_n=$((done_n+n))
  printf "  %-15s %5s/%-5s %8s%%  %s\n" "$d" "$n" "$t" "$(( n*100/t ))" "$state"
done
printf "  %s\n" "------------------------------------------------------"
pct=$(( done_n*100/total ))
printf "  %-15s %5s/%-5s %8s%%\n" "TOTAL" "$done_n" "$total" "$pct"
# eta from process elapsed time
p=$(pgrep -f "export_for_eval_script.py export" | head -1)
if [ -n "$p" ]; then
  es=$(ps -o etimes= -p $p | xargs)
  if [ "$done_n" -gt 0 ] && [ -n "$es" ]; then
    rate=$(echo "$es $done_n" | awk '{printf "%.2f", $1/$2}')
    rem=$(( total - done_n ))
    eta=$(echo "$rem $rate $running" | awk '{printf "%.0f", $1*$2/($3>0?$3:1)/60}')
    echo "  elapsed ${es}s | ~${rate}s/sample/gpu | ETA ~${eta} min"
  fi
else
  echo "  no export process running"
fi
[ -f "$OUT/eval_tau9/all_datasets_summary.json" ] && echo "  eval_script.py: COMPLETE -> run three_arm_table.py" || echo "  eval_script.py: pending (auto-runs when all 6 domains finish)"
