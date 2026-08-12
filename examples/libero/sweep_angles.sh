#!/usr/bin/env bash
# Sweep the eval camera azimuth for ONE already-running policy server, across the
# training angles (+ optional extrapolation). Run once per model (view45 / all_views),
# pointing MODEL_TAG at that model so outputs don't collide.
#
# Prereqs:
#   * A policy server for the model under test is serving on $PORT.
#   * examples/libero/configs/relocate.yaml has radius_delta/elevation_delta set from
#     calibrate_camera.py, and policy_image_flip: vflip.
#
# Usage:
#   MODEL_TAG=view45  TASK_SUITE=libero_object bash examples/libero/sweep_angles.sh
#   MODEL_TAG=allviews TASK_SUITE=libero_object AZIMUTHS="-15 0 15 30 45 60 75 90 105" bash examples/libero/sweep_angles.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"

MODEL_TAG="${MODEL_TAG:?set MODEL_TAG, e.g. view45 or allviews}"
TASK_SUITE="${TASK_SUITE:-libero_object}"
AZIMUTHS="${AZIMUTHS:-0 15 30 45 60 75 90}"
PORT="${PORT:-8000}"
EGL_DEVICE="${MUJOCO_EGL_DEVICE_ID:-0}"
BASE="configs/relocate.yaml"
OUT_ROOT="${SWEEP_OUT:-$REPO/data/sweep/$TASK_SUITE}"   # per-suite: /root/workspace/data/sweep/<suite>

mkdir -p "$OUT_ROOT"  # tee targets $OUT_ROOT/<name>.log; must exist before the pipeline

for az in $AZIMUTHS; do
  name="${MODEL_TAG}_${TASK_SUITE}_az${az}"
  cfg="configs/_gen_${name}.yaml"
  video_dir="$OUT_ROOT/${name}"

  # Patch azimuth_delta, task suite, port, and output dir into a generated config.
  python - "$BASE" "$cfg" "$az" "$TASK_SUITE" "$PORT" "$video_dir" <<'PY'
import sys, yaml
base, out, az, suite, port, vdir = sys.argv[1:7]
c = yaml.safe_load(open(base))
c["task_suite_name"] = suite
c["port"] = int(port)
c["video_out_path"] = vdir
for cam in c["cameras"]:
    if cam.get("orbit_object_centroid"):
        cam["azimuth_delta"] = float(az)
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print(f"wrote {out}: azimuth={az} suite={suite}")
PY

  echo ">>> [$name] running eval"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE" \
    python main.py --args.config "$cfg" 2>&1 | tee "$OUT_ROOT/${name}.log"
  # Success rate is logged as "Total success rate: ..." at the end of each run.
done

echo "=== summary (${MODEL_TAG}, ${TASK_SUITE}) -> $OUT_ROOT ==="
for az in $AZIMUTHS; do
  log="$OUT_ROOT/${MODEL_TAG}_${TASK_SUITE}_az${az}.log"
  sr=$(grep -oE "Total success rate: [0-9.]+" "$log" | tail -1 || echo "n/a")
  printf "  az=%-4s %s\n" "$az" "$sr"
done
