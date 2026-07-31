#!/usr/bin/env bash
# Sequential unattended queue, 2026-07-30. Runs one step at a time, logging to
# logs/queue.log, and survives the launching session ending (start with setsid).
#
#   1. wait for the in-flight beta-annealed run to finish
#   2. archive it (checkpoints/run1_beta_annealed/, logs/run1/)
#   3. gate 6 closed-loop eval on run 1
#   4. beta=0 round: PBVS driving, divergence guards, warm-started from run 1
#   5. gate 6 closed-loop eval on run 2
#
# Steps 4-5 are skipped if the preceding training produced no checkpoint -- there
# is no point evaluating or building on a failed run.
#
#   setsid nohup bash scripts/run_queue.sh > logs/queue_boot.log 2>&1 &
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-$HOME/miniconda3/envs/cnsv2/bin/python}
Q=logs/queue.log
mkdir -p logs

say() { echo "[queue $(date +%H:%M:%S)] $*" | tee -a "$Q"; }

ITERS=40000
FEAT=checkpoints/cnsv2_feat
R1=checkpoints/run1_beta_annealed
R1_BEST=$R1/cnsv2_best.pth
OUT2=checkpoints/cnsv2_b0.pth
R2_BEST=checkpoints/cnsv2_b0_best.pth

say "queue starting; python=$PY"

# ---------------------------------------------------------------- 1. wait
say "step 1: waiting for the in-flight run to finish"
while pgrep -f "train_cnsv2.py --data" >/dev/null 2>&1; do sleep 60; done
say "step 1: trainer exited. $(grep -m1 'training finished rc=' logs/run_concurrent.log 2>/dev/null || echo 'no rc line found')"
say "step 1: $(grep '\[done\]' logs/train.log 2>/dev/null | tail -1 || echo 'no [done] line')"
# The launcher's own teardown handles the servers, but it has failed before.
sleep 30
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_" 2>/dev/null
pkill -KILL -f "collect_dagger.py" 2>/dev/null
sleep 5

# ---------------------------------------------------------------- 2. archive
say "step 2: archiving run 1"
mkdir -p "$R1" logs/run1
for f in cnsv2_best.pth cnsv2_last.pth cnsv2.pth cnsv2_sync.pth; do
  [ -e "checkpoints/$f" ] && mv "checkpoints/$f" "$R1/"
done
# Intermediates are only useful for resuming, and run 1 is done. Their presence
# would ALSO make a re-launch resume run 1 instead of starting run 2.
rm -f checkpoints/cnsv2_iter*.pth
rm -rf checkpoints/cnsv2_dagger            # ~22GB of run-1 pool memmaps
for f in train.log collect_0.log collect_1.log server_0.log server_1.log run_concurrent.log; do
  [ -e "logs/$f" ] && mv "logs/$f" logs/run1/
done
say "step 2: run 1 archived -> $R1 ($(du -sh "$R1" 2>/dev/null | cut -f1)); disk $(df -h / | tail -1 | awk '{print $4}') free"

if [ ! -e "$R1_BEST" ]; then
  say "ABORT: no $R1_BEST -- run 1 produced no checkpoint, so there is nothing to"
  say "ABORT: evaluate and nothing to warm-start run 2 from."
  exit 1
fi

# ---------------------------------------------------------------- 3. gate 6, run 1
say "step 3: gate 6 on run 1 (hybrid deploy, 20 paired episodes, tight thresholds)"
PYTHON="$PY" bash scripts/run_isaac_eval.sh gate6_run1 "$R1_BEST" 20 \
  --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05 \
  >> "$Q" 2>&1
say "step 3: done rc=$?; $(grep -A 5 'SERVO EVAL' logs/eval_gate6_run1.log 2>/dev/null | tr '\n' ' ' | tr -s ' ')"

# ---------------------------------------------------------------- 4. beta=0 round
say "step 4: beta=0 round -- PBVS driving, guards on, warm start from run 1"
rm -rf data/dagger_live; mkdir -p data/dagger_live
rm -rf checkpoints/cnsv2_b0_dagger checkpoints/cnsv2_b0_iter*.pth
# collect_dagger refuses to start with beta<1 and no policy, and the launcher starts
# collectors BEFORE the trainer publishes its first sync checkpoint. Seed it.
cp "$R1_BEST" checkpoints/cnsv2_b0_sync.pth
say "step 4: seeded checkpoints/cnsv2_b0_sync.pth from run 1 so the beta=0 collectors have a policy at t=0"
PYTHON="$PY" DRIVE=pbvs BETA=0.0 BETA_FINAL="" \
  TRAIN_EXTRA="--init $R1_BEST --feat-cache $FEAT" \
  bash scripts/run_concurrent_dagger.sh data/isaac_train "$OUT2" "$ITERS" 100000 \
  >> "$Q" 2>&1
RC2=$?
say "step 4: done rc=$RC2; $(grep '\[done\]' logs/train.log 2>/dev/null | tail -1 || echo 'no [done] line')"
sleep 30
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_" 2>/dev/null
pkill -KILL -f "collect_dagger.py" 2>/dev/null
sleep 5
mkdir -p logs/run2
for f in train.log collect_0.log collect_1.log server_0.log server_1.log; do
  [ -e "logs/$f" ] && cp "logs/$f" logs/run2/
done

# ---------------------------------------------------------------- 5. gate 6, run 2
if [ ! -e "$R2_BEST" ]; then
  say "step 5 SKIPPED: no $R2_BEST -- the beta=0 run produced no checkpoint"
  say "queue finished with run 2 incomplete"
  exit 1
fi
say "step 5: gate 6 on run 2 (beta=0 checkpoint)"
PYTHON="$PY" bash scripts/run_isaac_eval.sh gate6_run2 "$R2_BEST" 20 \
  --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05 \
  >> "$Q" 2>&1
say "step 5: done rc=$?; $(grep -A 5 'SERVO EVAL' logs/eval_gate6_run2.log 2>/dev/null | tr '\n' ' ' | tr -s ' ')"

say "================ QUEUE COMPLETE ================"
say "run 1 (beta 1.0->0.3, hybrid driving): $(grep -oE 'BEST val l_dir [0-9.]+.*' logs/run1/train.log 2>/dev/null | tail -1)"
say "run 2 (beta 0, PBVS driving, guards):  $(grep -oE 'BEST val l_dir [0-9.]+.*' logs/run2/train.log 2>/dev/null | tail -1)"
say "gate6 run1: $(grep 'median TE ratio' logs/eval_gate6_run1.log 2>/dev/null | tail -1)"
say "gate6 run2: $(grep 'median TE ratio' logs/eval_gate6_run2.log 2>/dev/null | tail -1)"
say "SR run1: $(grep 'SR:' logs/eval_gate6_run1.log 2>/dev/null | tail -1)"
say "SR run2: $(grep 'SR:' logs/eval_gate6_run2.log 2>/dev/null | tail -1)"
