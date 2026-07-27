#!/bin/bash
# Closed-loop servo eval in the BlenderProc (training) domain.
# Starts the persistent render server + the CUDA-side policy client, waits for
# both, then tears down. Usage:
#   bash scripts/run_bproc_eval.sh [CKPT] [EPISODES] [EXTRA_CLIENT_ARGS...]
set -u
cd "$(dirname "$0")/.." || exit 1
CKPT=${1:-checkpoints/cnsv2.pth}
EPISODES=${2:-20}
shift 2 2>/dev/null || true
MESHES=data/gso/models/google_scanned_objects/models_normalized
W=$(mktemp -d /tmp/servo_ipc_XXXXXX)
SLOG=$W/server.log

cleanup() { [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null; }
trap cleanup EXIT

echo "[run] workdir $W"
blenderproc run cns/render/bproc_eval_server.py -- \
    --workdir "$W" --episodes "$EPISODES" --seed 777 \
    --meshes "$MESHES" --hdri data/hdri > "$SLOG" 2>&1 &
SPID=$!

# Server writes READY once Blender is up; surface its log if it dies first.
for _ in $(seq 1 600); do
    [ -f "$W/READY" ] && break
    if ! kill -0 "$SPID" 2>/dev/null; then
        echo "[run] SERVER DIED before READY -- last 40 lines:"; tail -40 "$SLOG"; exit 1
    fi
    sleep 1
done
[ -f "$W/READY" ] || { echo "[run] server never became READY"; tail -40 "$SLOG"; exit 1; }
echo "[run] server READY"

PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 eval_servo_bproc.py --ckpt "$CKPT" --workdir "$W" \
    --episodes "$EPISODES" "$@"
RC=$?
echo "[run] client exit $RC"
wait "$SPID" 2>/dev/null
echo "[run] server log tail:"; tail -5 "$SLOG"
rm -rf "$W"
exit $RC
