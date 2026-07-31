#!/usr/bin/env bash
# Queue 4: the spec's loss weighting. cnsv2_implementation_spec.md Sec. 1 and Sec. 5
# both give L = L_dir + 0.1*L_norm (inherited from CNS v1, which states it; v2 does
# not). Every run so far used norm_weight = 1.0.
#
# This is not a guess. train_cnsv2.py:49 independently measured the failure mode:
# "once near-goal data is present, sigma_inv targets reach -3.9, so l_norm runs ~1.09
# against l_dir ~0.69 -- ~60% of the gradient goes to magnitude while direction, which
# is what actually drives servo convergence, stalls." Run 4 shows the signature:
# l_norm 0.269 > l_dir 0.193, i.e. most of the loss is magnitude, in the very run
# where near-goal data was deliberately added.
#
# Spec Sec. 5: "Convergence quality lives in the direction term; precision lives in
# the norm term. They can be traded independently." Everything else is held identical
# to run 4 so the comparison isolates the weight.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PYTHON:-$HOME/miniconda3/envs/cnsv2/bin/python}
Q=logs/queue4.log; mkdir -p logs
say() { echo "[queue4 $(date +%H:%M:%S)] $*" | tee -a "$Q"; }

ITERS=40000
INIT=checkpoints/run2_beta_up_to_03/cnsv2_b0_best.pth   # same parent as run 4
OUT=checkpoints/cnsv2_nw01.pth
BEST=checkpoints/cnsv2_nw01_best.pth

[ -e "$INIT" ] || { say "ABORT: $INIT missing"; exit 1; }
say "waiting for idle GPUs"
while pgrep -f "train_cnsv2.py --data|eval_servo_bproc.py" >/dev/null 2>&1; do sleep 60; done
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_" 2>/dev/null; sleep 5

say "step A: --norm-weight 0.1 (spec Sec. 5); all else identical to run 4"
rm -rf data/dagger_live; mkdir -p data/dagger_live
rm -rf checkpoints/cnsv2_nw01_dagger checkpoints/cnsv2_nw01_iter*.pth checkpoints/cnsv2_nw01_sync.pth
PYTHON="$PY" DRIVE=pbvs NEAR_FRAC=0.25 \
  TRAIN_EXTRA="--init $INIT --feat-cache checkpoints/cnsv2_feat --norm-weight 0.1" \
  bash scripts/run_concurrent_dagger.sh data/isaac_train "$OUT" "$ITERS" 100000 >> "$Q" 2>&1
say "step A: done rc=$?; $(grep '\[done\]' logs/train.log 2>/dev/null | tail -1)"
say "step A: l_dir vs l_norm at the end = $(grep 'iter ' logs/train.log | tail -1 | tr -s ' ')"
sleep 30
pkill -KILL -f "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_" 2>/dev/null
pkill -KILL -f "collect_dagger.py" 2>/dev/null; sleep 5
mkdir -p logs/run5_nw01
for f in train.log collect_0.log collect_1.log server_0.log server_1.log; do
  [ -e "logs/$f" ] && cp "logs/$f" logs/run5_nw01/; done

[ -e "$BEST" ] || { say "step B SKIPPED: no $BEST"; exit 1; }
say "step B: gate 6"
PYTHON="$PY" bash scripts/run_isaac_eval.sh gate6_nw01 "$BEST" 20 \
  --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05 >> "$Q" 2>&1
say "step B: done rc=$?"
say "================ QUEUE 4 COMPLETE ================"
say "val:   $(grep -oE 'BEST val l_dir [0-9.]+.*' logs/run5_nw01/train.log 2>/dev/null | tail -1)"
say "SR:    $(grep 'SR:' logs/eval_gate6_nw01.log 2>/dev/null | tail -1)"
say "ratio: $(grep 'median TE ratio' logs/eval_gate6_nw01.log 2>/dev/null | tail -1)"
say "TE:    $(grep 'median final TE' logs/eval_gate6_nw01.log 2>/dev/null | tail -1)"
say "baselines -- run2: ratio 0.172 / 91.8mm | run4 (nw=1.0, same parent): ratio 0.220 / 100.8mm"
