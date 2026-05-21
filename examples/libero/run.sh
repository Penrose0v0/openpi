#!/usr/bin/env bash
# Wrapper for examples/libero/main.py — runs in background with nohup.
#
# Usage:
#   examples/libero/run.sh              # full run with defaults below
#   examples/libero/run.sh trial        # quick smoke test (1 trial per task, 1 task per scene type)
#
# Env var overrides (all optional):
#   TASK_SUITE    default: libero_10
#   ORBIT_DEG     default: -30       (deg around world +Z; +CCW, -CW viewed from +Z)
#   OFFSET_BACK   default: 0.5       (meters along camera local +Z, away from look dir)
#   OFFSET_UP     default: 0.4       (meters along world +Z)
#   OFFSET_RIGHT  default: 0.0       (meters along camera local +X, image right; negative = left)
#   PORT          default: 8000

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

TASK_SUITE="${TASK_SUITE:-libero_goal}"
ORBIT_DEG="${ORBIT_DEG:-"30"}"
OFFSET_BACK="${OFFSET_BACK:-0.0}"
OFFSET_UP="${OFFSET_UP:-0.0}"
OFFSET_RIGHT="${OFFSET_RIGHT:-0.0}"
PORT="${PORT:-8000}"

extra_args=()
mode="full"
if [[ "${1:-}" == "trial" ]]; then
    extra_args+=(--args.num_trials_per_task 1 --args.one_task_per_scene_type)
    mode="trial"
fi

deg_int="${ORBIT_DEG%.*}"
name="pi05_${TASK_SUITE}_rot${deg_int}_back${OFFSET_BACK}_up${OFFSET_UP}_right${OFFSET_RIGHT}_${mode}"
video_dir="data/${name}/videos"
log_file="${name}.log"

mkdir -p "$(dirname "$video_dir")"

MUJOCO_GL=osmesa nohup python examples/libero/main.py \
    --args.port "$PORT" \
    --args.task-suite-name "$TASK_SUITE" \
    --args.video-out-path "$video_dir" \
    --args.agentview_orbit_deg "$ORBIT_DEG" \
    --args.agentview_offset_back "$OFFSET_BACK" \
    --args.agentview_offset_up "$OFFSET_UP" \
    --args.agentview_offset_right "$OFFSET_RIGHT" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1 &

pid=$!
echo "Started PID $pid (mode=$mode)"
echo "  log:    $log_file"
echo "  videos: $video_dir"
echo "  tail -f $log_file"
