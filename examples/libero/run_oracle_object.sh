#!/usr/bin/env bash
# One-shot Oracle (object set) experiment:
#   1. starts the pi05 policy server for the Oracle model (config pi05_libero_oracle_lora),
#   2. waits until it's serving,
#   3. runs the libero_object eval at azimuths 0 / 45 / 90 in sequence,
#   4. aggregates results, then ALWAYS stops the server (even on error / Ctrl-C).
#
# The server and the eval run as separate process groups; a trap kills the server on exit
# so nothing is left holding the GPU.
#
# Usage (defaults shown):
#   examples/libero/run_oracle_object.sh
#   ORACLE_DIR=/path/to/ckpt SERVER_GPU=1 SIM_GPU=0 examples/libero/run_oracle_object.sh
#
# Watch progress in another shell:
#   tail -f oracle_server.log                      # model loading / server
#   tail -f examples/libero/data/sweep/oracle_libero_object_az45.log   # a given angle
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ---- config (override via env) ----------------------------------------------
ORACLE_DIR="${ORACLE_DIR:-/root/workspace/checkpoints/pi05_libero_oracle_lora/libero_oracle/29999}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_oracle_lora}"
TASK_SUITE="${TASK_SUITE:-libero_object}"
AZIMUTHS="${AZIMUTHS:-0 45 90}"
SERVER_GPU="${SERVER_GPU:-0}"          # GPU for the policy server
SIM_GPU="${SIM_GPU:-0}"                # GPU for MuJoCo/EGL rendering
MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}" # seconds to wait for the server to come up
SERVER_LOG="$REPO/oracle_server.log"
LIBERO_VENV="${LIBERO_VENV:-$HERE/.venv}"   # eval env (separate from the uv-managed server env)

[ -d "$ORACLE_DIR/params" ] || { echo "ERROR: no 'params' under ORACLE_DIR=$ORACLE_DIR"; exit 1; }

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

echo "[orchestrator] starting Oracle server"
echo "    config : $POLICY_CONFIG"
echo "    dir    : $ORACLE_DIR"
echo "    gpu    : $SERVER_GPU  (mem_fraction=$MEM_FRACTION)  port=$PORT"
cd "$REPO"
# setsid -> new process group, so the trap can kill uv + python + children together.
setsid env CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
  uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config="$POLICY_CONFIG" --policy.dir="$ORACLE_DIR" \
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

# ---- run the three-angle sweep ----------------------------------------------
echo "[orchestrator] eval: model=oracle suite=$TASK_SUITE azimuths='$AZIMUTHS' (sim on GPU $SIM_GPU)"
MODEL_TAG=oracle TASK_SUITE="$TASK_SUITE" AZIMUTHS="$AZIMUTHS" PORT="$PORT" \
  MUJOCO_EGL_DEVICE_ID="$SIM_GPU" \
  bash "$HERE/sweep_angles.sh"

# ---- aggregate --------------------------------------------------------------
echo "[orchestrator] aggregating results..."
python "$HERE/plot_sweep.py" --sweep_dir "$HERE/data/sweep" \
  --out_csv "$HERE/data/sweep/oracle_summary.csv" \
  --out_png "$HERE/data/sweep/oracle_summary.png" || true

echo "[orchestrator] DONE. Server will be stopped by the trap now."
