#!/usr/bin/env bash
# Queue 2: the TRUE beta=0 round that queue 1 failed to run.
#
# Queue 1's step 4 was meant to be constant beta=0 but ran beta annealing 0 -> 0.30
# instead, because run_concurrent_dagger.sh used ${BETA_FINAL:-0.3} -- the colon form
# substitutes the default for an EMPTY value, so BETA_FINAL="" re-enabled annealing.
# Fixed to ${BETA_FINAL-0.3} plus a "none" sentinel. This queue re-runs the intended
# experiment.
#
# Controlled against run 2: same warm start (run 1's best), same PBVS driving, same
# guards, same shared feature cache. The ONLY difference is beta held at 0, so the
# comparison isolates the expert share.
#
#   setsid nohup bash scripts/run_queue2.sh > logs/queue2_boot.log 2>&1 &
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-$HOME/miniconda3/envs/cnsv2/bin/python}
Q=logs/queue2.log
mkdir -p logs
say() { echo "[queue2 $(date +%H:%M:%S)] $*" | tee -a "$Q"; }

ITERS=40000
FEAT=checkpoints/cnsv2_feat
R1_BEST=checkpoints/run1_beta_annealed/cnsv2_best.pth
OUT3=checkpoints/cnsv2_b0true.pth
R3_BEST=checkpoints/cnsv2_b0true_best.pth

say "waiting for queue 1 to finish before touching the GPUs"
while pgrep -f "run_queue.sh" >/dev/null 2>&1; do sleep 60; done
while pgrep -f "train_cnsv2.py --data|eval_servo_bproc.py" >/dev/null 2>&1; do sleep 60; done
say "queue 1 done: $(grep -c '^\[queue' logs/queue.log 2>/dev/null) step lines"
sleep 20
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_eval_" 2>/dev/null
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_" 2>/dev/null
sleep 5

# ---- archive run 2 so its artifacts are not overwritten -------------------
say "archiving run 2 (beta 0->0.3, the mis-specified round)"
mkdir -p checkpoints/run2_beta_up_to_03
for f in cnsv2_b0_best.pth cnsv2_b0_last.pth cnsv2_b0.pth cnsv2_b0_sync.pth; do
  [ -e "checkpoints/$f" ] && mv "checkpoints/$f" checkpoints/run2_beta_up_to_03/
done
rm -f checkpoints/cnsv2_b0_iter*.pth
rm -rf checkpoints/cnsv2_b0_dagger
cat > checkpoints/run2_beta_up_to_03/WHAT_THIS_IS.txt <<'EOF'
beta was intended to be constant 0 but annealed 0 -> 0.30 (expert share RISING),
because run_concurrent_dagger.sh used ${BETA_FINAL:-0.3} and was passed
BETA_FINAL="" -- the colon form takes the default for an empty value too. The
pre-seeded sync checkpoint also carried iters_done=32200, so the first beta read
0.24 until the trainer's first publish. Still a valid mostly-on-policy DAgger
round 2 (PBVS driving, guards, textured renders, ~15% expert on average), just not
the beta=0 experiment. cnsv2_b0true_* is the corrected run.
EOF

if [ ! -e "$R1_BEST" ]; then say "ABORT: $R1_BEST missing"; exit 1; fi

# ---- the true beta=0 round -----------------------------------------------
say "step A: TRUE beta=0 round (constant, PBVS driving, guards, warm start from run 1)"
rm -rf data/dagger_live; mkdir -p data/dagger_live
rm -rf checkpoints/cnsv2_b0true_dagger checkpoints/cnsv2_b0true_iter*.pth
# Seed the sync checkpoint so the beta<1 collectors have a policy at t=0, but ZERO
# its iters_done: the collector anneals on that field, and run 1's 32200 made the
# first beta read 0.24 last time. Harmless with constant beta, wrong in general.
"$PY" - "$R1_BEST" checkpoints/cnsv2_b0true_sync.pth <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
ck["iters_done"] = 0
torch.save({"state_dict": ck["state_dict"], "config": ck["config"], "iters_done": 0},
           sys.argv[2])
print(f"[seed] wrote {sys.argv[2]} with iters_done=0")
PYEOF
say "step A: seeded checkpoints/cnsv2_b0true_sync.pth (iters_done zeroed)"
PYTHON="$PY" DRIVE=pbvs BETA=0.0 BETA_FINAL=none \
  TRAIN_EXTRA="--init $R1_BEST --feat-cache $FEAT" \
  bash scripts/run_concurrent_dagger.sh data/isaac_train "$OUT3" "$ITERS" 100000 \
  >> "$Q" 2>&1
say "step A: done rc=$?; $(grep '\[done\]' logs/train.log 2>/dev/null | tail -1 || echo 'no [done] line')"
say "step A: beta values seen = $(grep -oE 'beta [0-9.]+' logs/collect_0.log 2>/dev/null | sort -u | tr '\n' ' ')"
sleep 30
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_" 2>/dev/null
pkill -KILL -f "collect_dagger.py" 2>/dev/null
sleep 5
mkdir -p logs/run3_b0true
for f in train.log collect_0.log collect_1.log server_0.log server_1.log; do
  [ -e "logs/$f" ] && cp "logs/$f" logs/run3_b0true/
done

if [ ! -e "$R3_BEST" ]; then say "step B SKIPPED: no $R3_BEST"; exit 1; fi
say "step B: gate 6 on the true beta=0 checkpoint"
PYTHON="$PY" bash scripts/run_isaac_eval.sh gate6_b0true "$R3_BEST" 20 \
  --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05 \
  >> "$Q" 2>&1
say "step B: done rc=$?"

say "================ QUEUE 2 COMPLETE ================"
say "beta=0 val: $(grep -oE 'BEST val l_dir [0-9.]+.*' logs/run3_b0true/train.log 2>/dev/null | tail -1)"
say "gate6 SR:   $(grep 'SR:' logs/eval_gate6_b0true.log 2>/dev/null | tail -1)"
say "gate6 ratio: $(grep 'median TE ratio' logs/eval_gate6_b0true.log 2>/dev/null | tail -1)"
