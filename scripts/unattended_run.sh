#!/usr/bin/env bash
# Unattended: wait for dataset generation, verify it, then run the full
# concurrent-DAgger training to completion. Written to be safe to leave alone.
#
#   setsid nohup bash scripts/unattended_run.sh > logs/unattended.log 2>&1 &
#
# Order matters: gate 2 (scripts/verify_isaac_data.py) runs BEFORE training so an
# 80-minute run is never spent on a bad dataset. Sec. 6.11's shared-texture bug is
# exactly the kind of thing that produces a plausible-looking but useless dataset,
# so the run aborts rather than proceeds on a failed gate.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

GEN_LOG=${GEN_LOG:-/tmp/claude-1000/-home-mowito-mw-CNS/befdb4fc-7218-4c63-a607-2f23566ee14b/scratchpad/gen_full.log}
DATA=${DATA:-data/isaac_train}
OUT=${OUT:-checkpoints/cnsv2.pth}
ITERS=${ITERS:-40000}
LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"

source /home/mowito/miniconda3/etc/profile.d/conda.sh
conda activate cnsv2

say() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- 1. wait
say "waiting for ISAAC_GEN_DONE in $GEN_LOG"
while ! grep -q "ISAAC_GEN_DONE" "$GEN_LOG" 2>/dev/null; do
  if ! pgrep -f "isaac_scene.py --out $REPO/$DATA" > /dev/null 2>&1 \
     && ! grep -q "ISAAC_GEN_DONE" "$GEN_LOG" 2>/dev/null; then
    say "WARNING: generator no longer running and no DONE marker; proceeding with what exists"
    break
  fi
  sleep 30
done
N_SCENES=$(ls "$DATA"/scene_*.npz 2>/dev/null | wc -l)
say "generation finished: $N_SCENES scenes"
if [ "$N_SCENES" -lt 100 ]; then
  say "ABORT: only $N_SCENES scenes present"; exit 1
fi

# ---------------------------------------------------------------- 2. gate 2
say "gate 2: verifying renderer contract + pose distribution"
python scripts/verify_isaac_data.py --data "$DATA" \
  --sheet "$LOGDIR/contact_sheet.png" > "$LOGDIR/gate2.log" 2>&1
GATE2=$?
tail -32 "$LOGDIR/gate2.log"
if [ $GATE2 -ne 0 ]; then
  say "ABORT: gate 2 FAILED (see $LOGDIR/gate2.log). Not training on this dataset."
  exit 1
fi
say "gate 2 PASSED"

# ---------------------------------------------------------------- 3. train
# --dagger-frac 0.35 rather than 0.5: with 2 collector envs over an ~80 min run the
# pool reaches only a few thousand pairs, so a 50% share would revisit each DAgger
# sample ~160 times and overfit that half. 0.35 keeps fresh on-policy data
# influential without letting a small pool dominate.
# INSTANCES=2 rather than 3: each render server holds ~7GB and each collector's
# policy ~2GB on GPU 1, so 3 pairs would sit at ~27GB of 32GB. Unattended, headroom
# beats throughput.
# --patience 0: Sec. 6.6 -- val l_dir went 0.589 -> 0.645 -> 0.227 once, and a run
# was killed on that misreading. Genuine overfitting does not reverse. Best
# checkpoint is tracked separately, so nothing is lost by running to the end.
say "starting concurrent DAgger training: $ITERS iters on $N_SCENES scenes"
INSTANCES=2 DAGGER_DIR=data/dagger_live LOGDIR="$LOGDIR" \
  bash scripts/run_concurrent_dagger.sh "$DATA" "$OUT" "$ITERS" 100000
RC=$?
say "training exited rc=$RC"

# ---------------------------------------------------------------- 4. gates 5/6
if [ $RC -eq 0 ] && [ -f "$OUT" ]; then
  # ONE SERVER PER CLIENT. The IPC protocol has the client count episodes from 0,
  # so a second client against a still-running server reads the stale ep0_meta.npz
  # left on disk while the server waits on ep20_req0 -- a silent deadlock until
  # timeout. Each eval therefore gets its own server and its own workdir.
  run_eval() {   # $1 = label, $2 = extra flags
    local label=$1 extra=${2:-}
    local wd=/tmp/cnsv2_eval_$label
    rm -rf "$wd"; mkdir -p "$wd"
    ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES "$HOME/isaacsim/python.sh" \
      cns/render/isaac_eval_server.py --workdir "$wd" --episodes 25 --gpu 1 \
      --usd data/gso_usd --hdri data/hdri --tex data/cc_textures \
      > "$LOGDIR/eval_server_$label.log" 2>&1 &
    local srv=$!
    for t in $(seq 1 900); do [ -f "$wd/READY" ] && break; sleep 1; done
    if [ ! -f "$wd/READY" ]; then
      say "eval server ($label) never became READY"; kill $srv 2>/dev/null; return 1
    fi
    # shellcheck disable=SC2086
    python eval_servo_bproc.py --ckpt "$OUT" --workdir "$wd" --episodes 20 $extra \
      > "$LOGDIR/eval_$label.log" 2>&1
    local rc=$?
    tail -14 "$LOGDIR/eval_$label.log"
    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    return $rc
  }

  # Sec. 6.9: the oracle uses ground-truth poses and never looks at an image, so
  # 100% oracle SR does NOT validate the image path. Run it to isolate
  # loop/integrator/pose-convention bugs, then read it narrowly.
  say "gate 5: closed-loop ORACLE in the IsaacSim training domain"
  run_eval oracle "--oracle"

  say "gate 6: closed-loop POLICY -> Table I row 1 comparison"
  run_eval policy ""
  say "target: SR 20/20, TE 0.948+-0.606 mm, RE 0.075+-0.048 deg"
  say "report SR *and* the median final/initial TE ratio: SR alone hides a policy"
  say "that halves the error but misses the gate (Sec. 7 gate 5)"
fi

say "UNATTENDED_RUN_COMPLETE rc=$RC"
