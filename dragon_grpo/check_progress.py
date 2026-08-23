#!/usr/bin/env python3
"""Progress/health check for the 5k GRPO run. Safe to run any time -- reads
only the metrics file, touches nothing the training process is using.

    python3 dragon_grpo/check_progress.py
    python3 dragon_grpo/check_progress.py --out-dir outputs/dragon6_grpo_v1   # the pilot
"""
import argparse, datetime, json, statistics, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="outputs/dragon6_grpo_5k")
ap.add_argument("--total-steps", type=int, default=2500)
ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--block", type=int, default=100, help="trend block size")
a = ap.parse_args()

root = Path("/mnt/data2/traviku2")
mp = root / a.out_dir / "grpo_metrics.jsonl"
if not mp.exists():
    raise SystemExit(f"no metrics yet at {mp}")

rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
steps = [r for r in rows if "step" in r and "val" not in r]
vals = [r["val"] for r in rows if "val" in r]
if not steps:
    raise SystemExit("metrics file has no step rows yet")

r, TOT = steps[-1], a.total_steps
rate = ((steps[-1]["elapsed_min"] - steps[-21]["elapsed_min"]) / 20
        if len(steps) > 20 else r["elapsed_min"] / r["step"])
nval = len([s for s in range(r["step"] + 1, TOT + 1) if s % a.save_every == 0])
eta = (TOT - r["step"]) * rate / 60 + nval * 12 / 60
done = datetime.datetime.now() + datetime.timedelta(hours=eta)

alive = subprocess.run(["pgrep", "-f", "grpo_dragon6_v1.py"], capture_output=True,
                       text=True).stdout.split()
print(f"PROGRESS  step {r['step']}/{TOT} ({r['step']/TOT*100:.1f}%)   "
      f"{'RUNNING pid ' + alive[0] if alive else '*** NOT RUNNING ***'}")
print(f"  elapsed {r['elapsed_min']/60:.2f} h | {rate*60:.1f} s/step | "
      f"ETA ~{eta:.1f} h -> {done:%a %H:%M}")

print(f"\nTREND (means per {a.block}-step block)")
print(f"  {'block':>13} {'reward':>8} {'mean_iou':>9} {'f1@0.5':>8} "
      f"{'miss':>7} {'fp':>7} {'fmt':>6} {'kl':>9}")
for i in range(0, len(steps), a.block):
    b = steps[i:i + a.block]
    if len(b) < 5:
        continue
    m = lambda k: statistics.mean(x[k] for x in b)
    print(f"  {b[0]['step']:>5}-{b[-1]['step']:<7} {m('reward'):>8.4f} {m('mean_iou'):>9.4f} "
          f"{m('f1@0.5'):>8.4f} {m('miss'):>7.3f} {m('fp'):>7.3f} {m('fmt_rate'):>6.3f} "
          f"{m('kl'):>9.4f}")

print("\nVAL  (SFT baseline: mean_iou 0.3837 | recall@0.9 0.0910 | f1@0.5 0.4162)")
if vals:
    for v in vals:
        print(f"  step {v['step']:>5}: mean_iou={v['val_mean_iou']:.4f}  "
              f"recall@0.9={v['val_recall@0.9']:.4f}  f1@0.5={v['val_f1@0.5']:.4f}")
else:
    nxt = ((r["step"] // a.save_every) + 1) * a.save_every
    print(f"  none yet - first at step {nxt} (~{(nxt-r['step'])*rate/60:.1f} h away)")

ck = sorted((root / a.out_dir).glob("grpo-step*"), key=lambda p: int(p.name.split("step")[1]))
print(f"\nCHECKPOINTS: {', '.join(c.name for c in ck) if ck else 'none yet'}")
