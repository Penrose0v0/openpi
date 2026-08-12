#!/usr/bin/env bash
# One-shot Oracle-without-wrist (object set) experiment:
#   1. starts the pi05 server for the no-wrist oracle (config pi05_libero_oracle_no_wrist_lora,
#      trained on libero_all_views with the wrist view dropped -> served with use_wrist=False,
#      so the wrist slot is zero-padded and masked to match training),
#   2. waits until it's serving,
#   3. runs the libero_object eval at azimuths 0 / 45 / 90 in sequence,
#   4. aggregates results, then ALWAYS stops the server (even on error / Ctrl-C).
#
# The eval side is unchanged: main.py still sends observation/wrist_image; the SERVER's
# use_wrist=False transform masks it out. Only the served config differs from the with-wrist
# oracle.
#
# Usage (defaults shown):
#   examples/libero/run_oracle_no_wrist_object.sh
#   CKPT_DIR=/path/to/ckpt AZIMUTHS="0 45 90" SERVER_GPU=7 SIM_GPU=7 examples/libero/run_oracle_no_wrist_object.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ---- config (override via env) ----------------------------------------------
CKPT_DIR="${CKPT_DIR:-/root/workspace/checkpoints/pi05_libero_oracle_lora/libero_oracle_without_wrist/29999}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_oracle_no_wrist_lora}"
TASK_SUITE="${TASK_SUITE:-libero_object}"
AZIMUTHS="${AZIMUTHS:-0 45 90}"
SERVER_GPU="${SERVER_GPU:-1}"          # GPU for the policy server
SIM_GPU="${SIM_GPU:-1}"                # GPU for MuJoCo/EGL rendering
MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}" # seconds to wait for the server to come up
SERVER_LOG="$REPO/oracle_no_wrist_server.log"
LIBERO_VENV="${LIBERO_VENV:-$HERE/.venv}"   # eval env (separate from the uv-managed server env)

[ -d "$CKPT_DIR/params" ] || { echo "ERROR: no 'params' under CKPT_DIR=$CKPT_DIR"; exit 1; }

# ---- server lifecycle -------------------------------------------------------
SERVER_PID=""
cleanup() {
  echo "[orchestrator] stopping server..."
  if [ -n "$SERVER_PID" ]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true   # kill the whole process group
    sleep 3
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[orchestrator] starting Oracle-no-wrist server"
echo "    config : $POLICY_CONFIG"
echo "    dir    : $CKPT_DIR"
echo "    gpu    : $SERVER_GPU  (mem_fraction=$MEM_FRACTION)  port=$PORT"
cd "$REPO"
# setsid -> new process group, so the trap can kill uv + python + children together.
setsid env CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
  uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config="$POLICY_CONFIG" --policy.dir="$CKPT_DIR" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[orchestrator] server process group $SERVER_PID -> log: $SERVER_LOG"

echo "[orchestrator] waiting for server on port $PORT (timeout ${READY_TIMEOUT}s)..."
waited=0
until (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
  exec 3>&- 3<&- 2>/dev/null || true
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[orchestrator] server exited before serving. Last log lines:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  if [ "$waited" -ge "$READY_TIMEOUT" ]; then
    echo "[orchestrator] timed out waiting for server. Last log lines:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  sleep 5; waited=$((waited + 5))
done
exec 3>&- 3<&- 2>/dev/null || true
echo "[orchestrator] server is up after ${waited}s. Giving it 3s to settle."
sleep 3

# ---- activate the libero eval env (the server already runs in its own uv env) ---
[ -f "$LIBERO_VENV/bin/activate" ] || { echo "ERROR: eval venv not found at $LIBERO_VENV"; exit 1; }
echo "[orchestrator] activating eval env: $LIBERO_VENV"
set +u
# shellcheck disable=SC1091
source "$LIBERO_VENV/bin/activate"
set -u
export PYTHONPATH="${PYTHONPATH:-}:$REPO/third_party/libero"

# ---- run the sweep ----------------------------------------------------------
echo "[orchestrator] eval: model=oracle_nowrist suite=$TASK_SUITE azimuths='$AZIMUTHS' (sim on GPU $SIM_GPU)"
MODEL_TAG=oracle_nowrist TASK_SUITE="$TASK_SUITE" AZIMUTHS="$AZIMUTHS" PORT="$PORT" \
  MUJOCO_EGL_DEVICE_ID="$SIM_GPU" \
  bash "$HERE/sweep_angles.sh"

# ---- aggregate --------------------------------------------------------------
SWEEP_DIR="$REPO/data/sweep/$TASK_SUITE"
echo "[orchestrator] aggregating results in $SWEEP_DIR ..."
python "$HERE/plot_sweep.py" --sweep_dir "$SWEEP_DIR" \
  --out_csv "$SWEEP_DIR/summary.csv" \
  --out_png "$SWEEP_DIR/summary.png" || true

echo "[orchestrator] DONE. Server will be stopped by the trap now."
