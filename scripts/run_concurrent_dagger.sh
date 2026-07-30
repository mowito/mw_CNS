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

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p "$LOGDIR" "$DAGGER_DIR"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

SYNC="${OUT%.pth}_sync.pth"
ISAAC=${ISAAC:-$HOME/isaacsim}

PIDS=()
cleanup() {
  echo "[run] shutting down ${#PIDS[@]} helper processes..."
  for p in "${PIDS[@]:-}"; do
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  done
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
  CUDA_VISIBLE_DEVICES=1 python collect_dagger.py \
    --ckpt "$SYNC" --workdir "$WD" --out "$DAGGER_DIR" \
    --episodes "$EPISODES" --watch --beta 1.0 --beta-final 0.3 \
    --seed $((2000 + i)) --tag "i$i" \
    > "$LOGDIR/collect_$i.log" 2>&1 &
  PIDS+=($!)
done

echo "[run] 3/3 training on GPU 0 -> $LOGDIR/train.log"
CUDA_VISIBLE_DEVICES=0 python train_cnsv2.py \
  --data "$DATA" --out "$OUT" --iters "$ITERS" --batch 16 \
  --dagger-dir "$DAGGER_DIR" --sync-path "$SYNC" \
  --sync-every 500 --ingest-every 200 --patience 0 \
  --dagger-frac 0.35 --dagger-reserve 8000 \
  2>&1 | tee "$LOGDIR/train.log"
# set -o pipefail makes $? the trainer's exit code, not tee's -- and it must be
# PROPAGATED. The first 40k run died in its final torch.save (disk full) yet the
# wrapper reported rc=0, so unattended_run.sh believed training had succeeded.
TRAIN_RC=$?

echo "[run] training finished rc=$TRAIN_RC; collector/server will be torn down"
exit $TRAIN_RC
