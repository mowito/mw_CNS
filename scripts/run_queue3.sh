#!/usr/bin/env bash
# Queue 3: train with the four Sec. 6.23 precision fixes, then gate 6.
#
#   1. split min_vel gate (per-DOF tv/rw)         -> admits fine-translation pairs
#   2. --min-in-frame out-of-view guard           -> both collector and eval
#   3. --near-frac 0.25 near-goal seeding         -> collection servers only
#   4. postprocess max_mag=3.0 magnitude clamp    -> everywhere
#
# Warm-started from run 2's best (the best closed-loop so far: gate 6 median TE
# ratio 0.172), so this is a DAgger round 3 and the question it answers is narrow:
# does near-goal supervision close any of the 17-92 mm stall?
#
# beta is left on the annealed default. Sec. 6.22 measured beta as a NULL RESULT
# (0 vs 0->0.30 indistinguishable at n=20), and near-goal coverage now comes from
# --near-frac rather than from a high expert share, so there is nothing to tune.
#
#   setsid nohup bash scripts/run_queue3.sh > logs/queue3_boot.log 2>&1 &
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-$HOME/miniconda3/envs/cnsv2/bin/python}
Q=logs/queue3.log
mkdir -p logs
say() { echo "[queue3 $(date +%H:%M:%S)] $*" | tee -a "$Q"; }

ITERS=40000
FEAT=checkpoints/cnsv2_feat
INIT=checkpoints/run2_beta_up_to_03/cnsv2_b0_best.pth
OUT=checkpoints/cnsv2_r4.pth
BEST=checkpoints/cnsv2_r4_best.pth

[ -e "$INIT" ] || { say "ABORT: warm-start checkpoint $INIT missing"; exit 1; }
[ -d "$FEAT" ] || { say "ABORT: feature cache $FEAT missing"; exit 1; }

say "waiting for the GPUs to be idle"
while pgrep -f "train_cnsv2.py --data|eval_servo_bproc.py|run_queue2.sh" >/dev/null 2>&1; do sleep 60; done
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_" 2>/dev/null
sleep 5

say "step A: training with the Sec. 6.23 fixes, warm start from $INIT"
rm -rf data/dagger_live; mkdir -p data/dagger_live
rm -rf checkpoints/cnsv2_r4_dagger checkpoints/cnsv2_r4_iter*.pth checkpoints/cnsv2_r4_sync.pth
PYTHON="$PY" DRIVE=pbvs NEAR_FRAC=0.25 \
  TRAIN_EXTRA="--init $INIT --feat-cache $FEAT" \
  bash scripts/run_concurrent_dagger.sh data/isaac_train "$OUT" "$ITERS" 100000 \
  >> "$Q" 2>&1
say "step A: done rc=$?; $(grep '\[done\]' logs/train.log 2>/dev/null | tail -1 || echo 'no [done] line')"

# The Sec. 6.12 regression watch: if the DAgger half's near-goal share climbs toward
# 50% the extra near-goal admission has gone too far, and that is the number to read.
say "step A: final pool = $(grep '\[dagger\] +' logs/train.log 2>/dev/null | tail -1 | tr -s ' ')"
say "step A: guards fired = DIVERGED $(grep -c DIVERGED logs/collect_0.log 2>/dev/null), LOST-VIEW $(grep -c LOST-VIEW logs/collect_0.log 2>/dev/null) of $(grep -c '^ep ' logs/collect_0.log 2>/dev/null) episodes"
say "step A: beta seen = $(grep -oE 'beta [0-9.]+' logs/collect_0.log 2>/dev/null | sort -u | tr '\n' ' ')"
say "step A: episodes reaching 2mm = $(grep -oE 'last te +[0-9.]+ mm' logs/collect_0.log 2>/dev/null | grep -oE '[0-9.]+' | awk '$1<2.0' | wc -l)"

sleep 30
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_" 2>/dev/null
pkill -KILL -f "collect_dagger.py" 2>/dev/null
sleep 5
mkdir -p logs/run4_precision
for f in train.log collect_0.log collect_1.log server_0.log server_1.log; do
  [ -e "logs/$f" ] && cp "logs/$f" logs/run4_precision/
done

[ -e "$BEST" ] || { say "step B SKIPPED: no $BEST"; exit 1; }
say "step B: gate 6 (full initial distribution -- near-frac deliberately NOT set here)"
PYTHON="$PY" bash scripts/run_isaac_eval.sh gate6_r4 "$BEST" 20 \
  --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05 \
  >> "$Q" 2>&1
say "step B: done rc=$?"

say "================ QUEUE 3 COMPLETE ================"
say "val:   $(grep -oE 'BEST val l_dir [0-9.]+.*' logs/run4_precision/train.log 2>/dev/null | tail -1)"
say "SR:    $(grep 'SR:' logs/eval_gate6_r4.log 2>/dev/null | tail -1)"
say "ratio: $(grep 'median TE ratio' logs/eval_gate6_r4.log 2>/dev/null | tail -1)"
say "TE:    $(grep 'median final TE' logs/eval_gate6_r4.log 2>/dev/null | tail -1)"
say "prior best (run 2): ratio 0.172, median final TE 91.8 mm, SR 0/20"
