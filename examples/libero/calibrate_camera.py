"""Calibrate the eval object-centroid camera against a relocate_camera training dataset.

Reproduces a single training frame from `libero_view45` (or any azimuth view) by replaying
the SAME demo initial state through main.py's `_apply_orbit_object_centroid`, then pixel-diffs
the render against the stored dataset image. If the eval camera math + flip match the
data-collection pipeline, the two images are near-identical.

Two checks:
  1. INVARIANT: azimuth/radius/elevation deltas of (0,0,0) must reproduce the ORIGINAL
     agentview exactly (MAE ~ 0). This validates the port end-to-end in the eval env.
  2. MATCH: sweep candidate (radius, elevation) deltas at the dataset's azimuth and report
     the one that best matches the training frame. The winner is the (radius, elevation)
     you must set in the eval config; if the best MAE is still large, the flip or azimuth
     convention is off.

Run inside the `libero` conda env with EGL:
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
    cd examples/libero
    python calibrate_camera.py \
        --demo_file /root/share/datasets/libero_related/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5 \
        --view_root /root/share/datasets/libero_related/libero_view45 \
        --azimuth 45
"""
import argparse
import io
import json
import os

import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Reuse the EXACT camera math the eval loop uses.
from main import CameraConfig, _apply_orbit_object_centroid


def resolve_bddl(demo_file):
    with h5py.File(demo_file, "r") as f:
        bddl_attr = f["data"].attrs["bddl_file_name"]
        language = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
    suite = os.path.basename(os.path.dirname(bddl_attr))
    local = os.path.join(get_libero_path("bddl_files"), suite, os.path.basename(bddl_attr))
    assert os.path.exists(local), f"bddl not found: {local}"
    return local, language


def load_train_frame(view_root, task_lang, demo_idx, frame):
    """Training image for (task, demo_idx). Episodes for a task are ordered by demo index,
    so the demo_idx-th episode of this task is demo_{demo_idx}."""
    eps = [json.loads(l) for l in open(os.path.join(view_root, "meta", "episodes.jsonl"))]
    task_eps = [e for e in eps if e["tasks"][0] == task_lang]
    if not task_eps:
        raise SystemExit(f"task not found in {view_root}: {task_lang!r}")
    ep = task_eps[demo_idx]
    ei = ep["episode_index"]
    info = json.load(open(os.path.join(view_root, "meta", "info.json")))
    cs = info["chunks_size"]
    p = os.path.join(view_root, "data", f"chunk-{ei // cs:03d}", f"episode_{ei:06d}.parquet")
    df = pq.read_table(p).to_pandas()
    img = Image.open(io.BytesIO(df.iloc[frame]["image"]["bytes"])).convert("RGB")
    return np.asarray(img), ei, ep["length"]


def mae(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def render_agentview(env, states, cam):
    """Reset (restore original camera), set state, relocate camera, re-render. Returns the
    vertically-flipped agentview (matching the dataset's write convention)."""
    env.reset()
    env.set_init_state(states)          # place objects so the centroid is correct
    if cam is not None:
        _apply_orbit_object_centroid(env, cam)
    obs = env.set_init_state(states)    # re-render with the relocated camera
    return obs["agentview_image"][::-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_file", required=True)
    p.add_argument("--view_root", required=True, help="e.g. .../libero_view45")
    p.add_argument("--azimuth", type=float, default=45.0)
    p.add_argument("--demo_idx", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--elevations", type=float, nargs="+", default=[0.0])
    p.add_argument("--out", default="calib.png")
    args = p.parse_args()

    bddl, language = resolve_bddl(args.demo_file)
    print(f"task: {language!r}")

    train, ei, length = load_train_frame(args.view_root, language, args.demo_idx, args.frame)
    print(f"train frame: episode {ei}, len {length}, demo {args.demo_idx}, frame {args.frame}")

    with h5py.File(args.demo_file, "r") as f:
        states = f["data"][f"demo_{args.demo_idx}"]["states"][()]
    print(f"demo states: {states.shape}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )

    st = states[args.frame]

    # --- Check 1: invariant. (0,0,0) deltas must reproduce the original agentview. ---
    orig = render_agentview(env, st, None)
    zero = render_agentview(
        env, st, CameraConfig(name="agentview", orbit_object_centroid=True)
    )
    inv = mae(orig, zero)
    print(f"\n[invariant] MAE(original_agentview, deltas=0) = {inv:.3f}  "
          f"({'OK' if inv < 1.0 else 'FAIL -> port is wrong'})")

    # --- Check 2: match. Sweep (radius, elevation) at the dataset azimuth. ---
    print(f"\n[match] azimuth_delta = {args.azimuth}")
    print(f"{'radius':>8} {'elev':>8} {'MAE_vs_train':>14}")
    best = None
    for r in args.radii:
        for e in args.elevations:
            cam = CameraConfig(
                name="agentview", orbit_object_centroid=True,
                azimuth_delta=args.azimuth, radius_delta=r, elevation_delta=e,
            )
            rend = render_agentview(env, st, cam)
            m = mae(rend, train)
            print(f"{r:8.2f} {e:8.2f} {m:14.3f}")
            if best is None or m < best[0]:
                best = (m, r, e, rend)
    env.close()

    m, r, e, rend = best
    print(f"\nBEST: radius_delta={r}, elevation_delta={e}, MAE={m:.3f}")
    if m < 3.0:
        print("  -> strong match. Use these deltas (and policy_image_flip: vflip) in the eval config.")
    else:
        print("  -> weak match. Check azimuth sign/convention and policy_image_flip.")

    diff = np.abs(rend.astype(np.int32) - train.astype(np.int32)).clip(0, 255).astype(np.uint8)
    panel = np.concatenate([train, rend, diff], axis=1)  # train | render | |diff|
    Image.fromarray(panel).save(args.out)
    print(f"saved side-by-side (train | render | diff): {args.out}")


if __name__ == "__main__":
    main()
