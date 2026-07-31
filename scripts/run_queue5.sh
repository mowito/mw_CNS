#!/usr/bin/env bash
# Queue 5: the ARCHITECTURAL fix. Adds 2D axial RoPE to the controller's self-attention
# (spec Sec. 3.6 specifies it; the code had no positional encoding at all).
#
# Sec. 6.24: without it the controller was EXACTLY permutation invariant over the 1024
# tokens -- shuffling all three streams together gave cos(base, shuffled) = 1.000000 on
# a trained checkpoint. Position could only reach the head through token CONTENT (RADIO
# absolute embeddings inside F_c, P's sparsity pattern), and the fine CNN branch has
# neither, so the branch the paper adds "to capture the pixel-wise error to improve the
# servo precision" had its spatial signal discarded by construction.
#
# This is the leading hypothesis for the ~100 mm precision floor that no data-side fix
# moved (Sec. 6.22/6.23: adding the missing sub-20mm supervision did not help and may
# have hurt). Verified the fix works: permutation equivariance error goes 4.8e-07
# (position blind) -> 2.4e-02.
#
# Warm start carries 321/345 tensors; only controller.self_blocks is re-initialized,
# so the refine transformer and fine CNN keep their learned features. norm-weight is
# left at the DEFAULT 1.0 -- the spec's 0.1 is a separate variable and queue 4 was
# killed to get here, so it stays untested for now.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PYTHON:-$HOME/miniconda3/envs/cnsv2/bin/python}
Q=logs/queue5.log; mkdir -p logs
say() { echo "[queue5 $(date +%H:%M:%S)] $*" | tee -a "$Q"; }

ITERS=40000
INIT=checkpoints/run2_beta_up_to_03/cnsv2_b0_best.pth   # same parent as run 4
OUT=checkpoints/cnsv2_rope.pth
BEST=checkpoints/cnsv2_rope_best.pth
[ -e "$INIT" ] || { say "ABORT: $INIT missing"; exit 1; }

say "waiting for idle GPUs"
while pgrep -f "train_cnsv2.py --data" >/dev/null 2>&1; do sleep 60; done
sleep 5
say "step A: controller RoPE, warm start from $INIT (only self_blocks re-init)"
rm -rf data/dagger_live; mkdir -p data/dagger_live
rm -rf checkpoints/cnsv2_rope_dagger checkpoints/cnsv2_rope_iter*.pth checkpoints/cnsv2_rope_sync.pth
PYTHON="$PY" DRIVE=pbvs NEAR_FRAC=0.25 \
  TRAIN_EXTRA="--init $INIT --feat-cache checkpoints/cnsv2_feat" \
  bash scripts/run_concurrent_dagger.sh data/isaac_train "$OUT" "$ITERS" 100000 >> "$Q" 2>&1
say "step A: done rc=$?; $(grep '\[done\]' logs/train.log 2>/dev/null | tail -1)"
say "step A: init report = $(grep -m1 '\[init\] warm-started' logs/train.log 2>/dev/null)"
say "step A: re-init = $(grep -m1 'RE-INITIALIZED' logs/train.log 2>/dev/null)"
sleep 30
for pat in "collect_dagger.py" "isaac_eval_server.py --workdir /tmp/cnsv2_dagger_ipc_"; do
  pkill -KILL -f "$pat" 2>/dev/null; done
sleep 5
mkdir -p logs/run6_rope
for f in train.log collect_0.log collect_1.log server_0.log server_1.log; do
  [ -e "logs/$f" ] && cp "logs/$f" logs/run6_rope/; done

[ -e "$BEST" ] || { say "step B SKIPPED: no $BEST"; exit 1; }
say "step B: gate 6"
PYTHON="$PY" bash scripts/run_isaac_eval.sh gate6_rope "$BEST" 20 \
  --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05 >> "$Q" 2>&1
say "step B: done rc=$?"
say "================ QUEUE 5 COMPLETE ================"
say "val:   $(grep -oE 'BEST val l_dir [0-9.]+.*' logs/run6_rope/train.log 2>/dev/null | tail -1)"
say "SR:    $(grep 'SR:' logs/eval_gate6_rope.log 2>/dev/null | tail -1)"
say "ratio: $(grep 'median TE ratio' logs/eval_gate6_rope.log 2>/dev/null | tail -1)"
say "TE:    $(grep 'median final TE' logs/eval_gate6_rope.log 2>/dev/null | tail -1)"
say "baselines -- run2 0.172/91.8mm | run4 (same parent, no RoPE) 0.220/100.8mm"
