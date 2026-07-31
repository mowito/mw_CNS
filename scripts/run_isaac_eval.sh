#!/usr/bin/env bash
# Closed-loop servo eval in the ISAAC (training) domain -- gates 5 and 6, Sec. 7.
#
# The Blender-domain equivalent is run_bproc_eval.sh; this is the Isaac one, which
# had been launched by hand as two processes. That is how every session so far
# ended with an orphaned render server holding a GPU: the kit python behind
# python.sh ignores SIGTERM, so the teardown here escalates to KILL.
#
#   bash scripts/run_isaac_eval.sh <TAG> <CKPT> <EPISODES> [extra client args...]
#
# Gate 6 (Table I row 1 comparable -- thresholds must be TIGHT or the reported
# final TE just reflects the stop gate, see Sec. 6 eval note):
#   bash scripts/run_isaac_eval.sh gate6 checkpoints/cnsv2.pth 20 \
#       --deploy hybrid --max-steps 1500 --te-thresh 0.0005 --re-thresh 0.05
#
# Seed is left at the server default so runs are PAIRED with the oracle gate on
# identical scenes and initial poses -- a failure is then the policy's, not the draw.
set -uo pipefail

TAG=${1:?usage: run_isaac_eval.sh TAG CKPT EPISODES [client args...]}
CKPT=${2:-checkpoints/cnsv2.pth}
EPISODES=${3:-20}
shift 3 2>/dev/null || true

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"

ISAAC=${ISAAC:-$HOME/isaacsim}
SERVER_GPU=${SERVER_GPU:-1}          # renderer on 1, policy on 0
CLIENT_GPU=${CLIENT_GPU:-0}
W=${WORKDIR:-/tmp/cnsv2_eval_$TAG}

PYTHON=${PYTHON:-python}
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "[eval] '$PYTHON' not on PATH -- pass PYTHON=/path/to/python"; exit 127; }
"$PYTHON" -c "import torch" 2>/dev/null || {
  echo "[eval] '$PYTHON' cannot import torch"; exit 127; }
[ -f "$CKPT" ] || { echo "[eval] no checkpoint at $CKPT"; exit 1; }

SLOG="$LOGDIR/eval_server_$TAG.log"
CLOG="$LOGDIR/eval_$TAG.log"
rm -rf "$W"; mkdir -p "$W"

SPID=""
cleanup() {
  [ -n "$SPID" ] || return 0
  kill "$SPID" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$SPID" 2>/dev/null; then
    echo "[eval] server ignored TERM; KILLing"
    pkill -KILL -P "$SPID" 2>/dev/null
    kill -KILL "$SPID" 2>/dev/null
  fi
  # Grandchild via python.sh, so sweep by cmdline too.
  pkill -KILL -f "isaac_eval_server.py --workdir $W" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[eval] $TAG | ckpt $CKPT | $EPISODES episodes | workdir $W"
echo "[eval] client args: $*"
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES "$ISAAC/python.sh" \
  cns/render/isaac_eval_server.py \
  --workdir "$W" --episodes "$EPISODES" --gpu "$SERVER_GPU" \
  --usd data/gso_usd --hdri data/hdri --tex data/cc_textures \
  > "$SLOG" 2>&1 &
SPID=$!

echo -n "[eval] waiting for READY"
for _ in $(seq 1 900); do
  [ -f "$W/READY" ] && break
  kill -0 "$SPID" 2>/dev/null || { echo; echo "[eval] SERVER DIED:"; tail -30 "$SLOG"; exit 1; }
  sleep 1; echo -n "."
done
echo
[ -f "$W/READY" ] || { echo "[eval] server never became READY"; tail -30 "$SLOG"; exit 1; }

CUDA_VISIBLE_DEVICES=$CLIENT_GPU PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" eval_servo_bproc.py --ckpt "$CKPT" --workdir "$W" \
    --episodes "$EPISODES" "$@" 2>&1 | tee "$CLOG"
RC=$?
echo "[eval] client exit $RC -> $CLOG"
exit $RC
