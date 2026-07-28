#!/bin/bash
# DAgger rollout collection in the BlenderProc domain.
#   bash scripts/run_dagger_collect.sh [CKPT] [EPISODES] [BETA] [OUTDIR]
# Starts the persistent Blender render server + the CUDA-side rollout client.
# BETA is P(expert action)/step -- see collect_dagger.py: beta=0 cannot reach the
# near-goal states this is meant to collect.
set -u
cd "$(dirname "$0")/.." || exit 1
CKPT=${1:-checkpoints/cnsv2.pth}
EPISODES=${2:-50}
BETA=${3:-1.0}
OUT=${4:-data/dagger_r0}
MESHES=data/gso/models/google_scanned_objects/models_normalized
W=$(mktemp -d /tmp/dagger_ipc_XXXXXX)
SLOG=$W/server.log

cleanup() { [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null; }
trap cleanup EXIT

echo "[dagger] workdir $W  out $OUT  beta $BETA  episodes $EPISODES"
# Seed differs from the eval server's 777 so collection scenes are not the same
# scenes we certify on.
blenderproc run cns/render/bproc_eval_server.py -- \
    --workdir "$W" --episodes "$EPISODES" --seed 31337 \
    --meshes "$MESHES" --hdri data/hdri > "$SLOG" 2>&1 &
SPID=$!

for _ in $(seq 1 600); do
    [ -f "$W/READY" ] && break
    if ! kill -0 "$SPID" 2>/dev/null; then
        echo "[dagger] SERVER DIED before READY:"; tail -40 "$SLOG"; exit 1
    fi
    sleep 1
done
[ -f "$W/READY" ] || { echo "[dagger] server never READY"; tail -40 "$SLOG"; exit 1; }

PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 collect_dagger.py --ckpt "$CKPT" --workdir "$W" --out "$OUT" \
    --episodes "$EPISODES" --beta "$BETA"
RC=$?
echo "[dagger] client exit $RC"
wait "$SPID" 2>/dev/null
rm -rf "$W"
exit $RC
