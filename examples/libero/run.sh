#!/usr/bin/env bash
# Background (nohup) wrapper around sweep_angles.sh: sweeps the eval camera azimuth for
# ONE already-running policy server and logs to a file you can tail.
#
# Prereqs:
#   * A policy server for the model under test is serving on $PORT.
#   * configs/relocate.yaml has radius_delta/elevation_delta set from calibrate_camera.py
#     and policy_image_flip: vflip.
#
# Usage:
#   MODEL_TAG=view45   examples/libero/run.sh
#   MODEL_TAG=allviews TASK_SUITE=libero_object AZIMUTHS="-15 0 15 30 45 60 75 90 105" examples/libero/run.sh
#
# Env vars (forwarded to sweep_angles.sh): MODEL_TAG (required), TASK_SUITE, AZIMUTHS, PORT.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

MODEL_TAG="${MODEL_TAG:?set MODEL_TAG, e.g. view45 or allviews}"
TASK_SUITE="${TASK_SUITE:-libero_object}"
log_file="sweep_${MODEL_TAG}_${TASK_SUITE}.log"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  MODEL_TAG="$MODEL_TAG" TASK_SUITE="$TASK_SUITE" \
  AZIMUTHS="${AZIMUTHS:-0 15 30 45 60 75 90}" PORT="${PORT:-8000}" \
  nohup bash examples/libero/sweep_angles.sh > "$log_file" 2>&1 &

pid=$!
echo "Started sweep PID $pid (model=$MODEL_TAG suite=$TASK_SUITE)"
echo "  log: $log_file"
echo "  per-angle logs + videos: data/sweep/"
echo "  tail -f $log_file"
echo "When done, aggregate: python examples/libero/plot_sweep.py --sweep_dir data/sweep"
