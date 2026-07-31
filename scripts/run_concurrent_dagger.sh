#!/usr/bin/env bash
# Paper Fig. 3, run for real: three processes at once, not a phase 2.
#
#   GPU 1 : IsaacSim render server  (cns/render/isaac_eval_server.py)
#   CPU/1 : DAgger collector        (collect_dagger.py --watch)
#   GPU 0 : training                (train_cnsv2.py --dagger-dir ...)
#
# The collector re-reads the weights the trainer publishes to <out>_sync.pth, and
# the trainer ingests the collector's rollouts every --ingest-every iters and mixes
# them into every batch at --dagger-frac. Nothing is frozen and nothing waits for a
# "round" to end.
#
# Both 5090s are used, which is why isaac_scene.launch() disables multi_gpu -- with
# it on, Isaac copies render targets between devices through main memory.
#
#   bash scripts/run_concurrent_dagger.sh data/isaac_train checkpoints/cnsv2.pth 40000
set -uo pipefail

DATA=${1:-data/isaac_train}
OUT=${2:-checkpoints/cnsv2.pth}
ITERS=${3:-40000}
EPISODES=${4:-100000}          # effectively unbounded; killed when training ends
# Fig. 3's Simulation Process #2 contains "Env 1  Env 2  ...  Env N". One
# collector managed ~1 episode per 150 s against a trainer doing 8 it/s, which is
# far too little fresh on-policy data for a 40k-iter run; N pairs of
# (render server, collector) fix that, all feeding one DAGGER_DIR. Each pair gets
# its own IPC workdir, GPU assignment, seed and filename --tag.
INSTANCES=${INSTANCES:-2}
WORKDIR=${WORKDIR:-/tmp/cnsv2_dagger_ipc}
DAGGER_DIR=${DAGGER_DIR:-data/dagger_live}
LOGDIR=${LOGDIR:-logs}

# Rollout policy mix and driving law, overridable so a beta=0 round can reuse this
# launcher unchanged. BETA_FINAL empty => no annealing (and no horizon needed).
DRIVE=${DRIVE:-pbvs}
BETA=${BETA:-1.0}
# NO COLON on this one. ${BETA_FINAL:-0.3} substitutes the default for an EMPTY
# value as well as an unset one, so BETA_FINAL="" silently re-enabled annealing and
# turned an intended constant-beta=0 round into beta annealing 0 -> 0.3 (expert
# share RISING). "none" is accepted as an explicit, unmissable way to say no.
BETA_FINAL=${BETA_FINAL-0.3}
[ "$BETA_FINAL" = "none" ] && BETA_FINAL=""
# Extra trainer flags, e.g. warm start + shared feature cache for a second round.
TRAIN_EXTRA=${TRAIN_EXTRA:-}
# Near-goal quota for the COLLECTION render servers (Sec. 6.22). run_isaac_eval.sh
# deliberately does NOT set this -- gate 6 must sample the paper's full initial
# distribution or the SR is not comparable to Table I.
NEAR_FRAC=${NEAR_FRAC:-0.25}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p "$LOGDIR" "$DAGGER_DIR"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

SYNC="${OUT%.pth}_sync.pth"
ISAAC=${ISAAC:-$HOME/isaacsim}

# The trainer and collectors were invoked as bare `python`, which resolves only
# inside an activated cnsv2 env. Launched detached (nohup/setsid/cron) it does
# not resolve, and because the check would otherwise come AFTER the render
# servers, the failure surfaced as `python: command not found` four minutes of
# IsaacSim startup later. Overridable, and verified before anything is spawned.
PYTHON=${PYTHON:-python}
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "[run] '$PYTHON' not on PATH -- activate the cnsv2 env or pass PYTHON=/path/to/python"
  exit 127
}
"$PYTHON" -c "import torch" 2>/dev/null || {
  echo "[run] '$PYTHON' cannot import torch -- wrong interpreter?"
  exit 127
}
echo "[run] python: $(command -v "$PYTHON")"
echo "[run] rollout: drive=$DRIVE beta=$BETA beta_final=${BETA_FINAL:-<none>} iters=$ITERS near_frac=$NEAR_FRAC"

PIDS=()
cleanup() {
  echo "[run] shutting down ${#PIDS[@]} helper processes..."
  for p in "${PIDS[@]:-}"; do
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  done
  # TERM alone is not enough: python.sh execs a kit python that ignores it, so
  # every run so far has left two render servers alive after the trainer exited,
  # each holding GPU memory and ~300W until noticed by hand. Escalate.
  for t in $(seq 1 15); do
    alive=""
    for p in "${PIDS[@]:-}"; do
      [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && alive="$alive $p"
    done
    [ -z "$alive" ] && break
    sleep 2
  done
  for p in ${alive:-}; do
    echo "[run] $p ignored TERM; KILLing"
    pkill -KILL -P "$p" 2>/dev/null      # python.sh's kit-python child
    kill -KILL "$p" 2>/dev/null
  done
  # The servers are grandchildren via python.sh, so a PID-only sweep can still
  # miss them; match on the command line this run started.
  pkill -KILL -f "isaac_eval_server.py --workdir ${WORKDIR}_" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

for i in $(seq 0 $((INSTANCES - 1))); do
  # Renderers go on GPU 1 so GPU 0 is left to training. Both cards are 32GB and a
  # render server sits at ~7GB, so several fit alongside each other.
  GPU=1
  WD="${WORKDIR}_$i"
  rm -rf "$WD"; mkdir -p "$WD"

  echo "[run] env $i: IsaacSim render server (GPU $GPU) -> $LOGDIR/server_$i.log"
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES "$ISAAC/python.sh" \
    cns/render/isaac_eval_server.py \
    --workdir "$WD" --episodes "$EPISODES" --gpu "$GPU" --seed $((1000 + i)) \
    --near-frac "$NEAR_FRAC" \
    --usd data/gso_usd --hdri data/hdri --tex data/cc_textures \
    > "$LOGDIR/server_$i.log" 2>&1 &
  PIDS+=($!)
  SPID=${PIDS[-1]}

  echo -n "[run] env $i: waiting for READY"
  for t in $(seq 1 900); do
    [ -f "$WD/READY" ] && break
    kill -0 "$SPID" 2>/dev/null || { echo; echo "[run] SERVER $i DIED:"; tail -25 "$LOGDIR/server_$i.log"; exit 1; }
    sleep 1; echo -n "."
  done
  echo
  [ -f "$WD/READY" ] || { echo "[run] server $i never became READY"; tail -25 "$LOGDIR/server_$i.log"; exit 1; }

  echo "[run] env $i: DAgger collector -> $LOGDIR/collect_$i.log"
  # --beta-iters must be the trainer's --iters: beta then anneals with TRAINING
  # progress. Annealing over --episodes (which is the 100000 "unbounded" sentinel
  # above) is what pinned beta at 1.00 for two entire 40k runs, so every rollout
  # was expert-driven and no DAgger data was ever on-policy.
  BETA_ARGS=(--beta "$BETA")
  if [ -n "$BETA_FINAL" ]; then
    BETA_ARGS+=(--beta-final "$BETA_FINAL" --beta-iters "$ITERS")
  fi
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" collect_dagger.py \
    --ckpt "$SYNC" --workdir "$WD" --out "$DAGGER_DIR" \
    --episodes "$EPISODES" --watch "${BETA_ARGS[@]}" --drive "$DRIVE" \
    --seed $((2000 + i)) --tag "i$i" \
    > "$LOGDIR/collect_$i.log" 2>&1 &
  PIDS+=($!)
done

echo "[run] 3/3 training on GPU 0 -> $LOGDIR/train.log"
# shellcheck disable=SC2086  # TRAIN_EXTRA is intentionally word-split
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_cnsv2.py \
  --data "$DATA" --out "$OUT" --iters "$ITERS" --batch 16 \
  --dagger-dir "$DAGGER_DIR" --sync-path "$SYNC" \
  --sync-every 500 --ingest-every 200 --patience 0 \
  --dagger-frac 0.35 --dagger-reserve 8000 $TRAIN_EXTRA \
  2>&1 | tee "$LOGDIR/train.log"
# set -o pipefail makes $? the trainer's exit code, not tee's -- and it must be
# PROPAGATED. The first 40k run died in its final torch.save (disk full) yet the
# wrapper reported rc=0, so unattended_run.sh believed training had succeeded.
TRAIN_RC=$?

echo "[run] training finished rc=$TRAIN_RC; collector/server will be torn down"
exit $TRAIN_RC
