import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional, Set, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import yaml

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class CameraConfig:
    # MuJoCo camera name. Also the prefix of the obs key ({name}_image).
    name: str
    # Subfolder under video_out_path. Defaults to `name`.
    save_subdir: Optional[str] = None
    # If False, the camera is still rendered (and may feed the model) but no video is saved.
    save: bool = True
    # Optional absolute pose override (world frame). Applied before orbit/translate.
    # quat is (w, x, y, z), MuJoCo convention.
    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    # Orbit around world +Z through orbit_center (XY), CCW viewed from +Z.
    orbit_deg: float = 0.0
    orbit_center: Tuple[float, float] = (0.0, 0.0)
    # Translation in meters: along camera-local +Z (away from look dir), world +Z, camera-local +X.
    offset_back: float = 0.0
    offset_up: float = 0.0
    offset_right: float = 0.0
    # Apply 180-degree rotation when storing the rendered image (matches LIBERO training convention).
    rotate_180: bool = True

    # --- Object-centroid orbit mode (matches the data-collection renderer) ---------------
    # When True, ignore the world-Z orbit/offset fields above and instead reposition the
    # camera exactly like relocate_camera/scripts/render_relocated.py: orbit on a sphere
    # around a per-scene look-at center (the objects-of-interest centroid, projected onto
    # the ORIGINAL agentview sightline), always re-aiming at that center. radius/azimuth/
    # elevation are DELTAS relative to the original agentview pose, so (0,0,0) reproduces
    # the original view. This is the ONLY way to reproduce the libero_various_view /
    # libero_all_views / libero_view45 training viewpoints.
    orbit_object_centroid: bool = False
    radius_delta: float = 0.0
    azimuth_delta: float = 0.0
    elevation_delta: float = 0.0
    # Optional fixed look-at center (x, y, z). None -> objects-of-interest centroid.
    center: Optional[Tuple[float, float, float]] = None


@dataclasses.dataclass
class ExperimentConfig:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Task suite
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    one_task_per_scene_type: bool = False

    # Rendering / output
    camera_resolution: int = 256
    # If None, _load_config derives it from the config filename: data/libero/<config_stem>.
    video_out_path: Optional[str] = None
    seed: int = 7

    # Cameras that feed the model. Both must appear in `cameras`.
    policy_image_camera: str = "agentview"
    policy_wrist_camera: str = "robot0_eye_in_hand"

    # How to flip the raw MuJoCo render before feeding it to the policy. Must match how the
    # TRAINING data was written. Standard openpi LIBERO data uses "rot180" (raw[::-1, ::-1]).
    # The relocate_camera datasets (libero_view45 / libero_all_views) were written with a
    # VERTICAL flip only (raw[::-1]) -> use "vflip" when evaluating models trained on them.
    policy_image_flip: str = "rot180"

    # All cameras to render. Order matters only for readability.
    cameras: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Args:
    config: pathlib.Path


def _load_config(path: pathlib.Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cameras_raw = raw.pop("cameras", [])
    cams = []
    for entry in cameras_raw:
        # Normalize tuple-typed fields (yaml gives lists).
        for key in ("pos", "quat", "orbit_center", "center"):
            if key in entry and entry[key] is not None:
                entry[key] = tuple(entry[key])
        cams.append(CameraConfig(**entry))
    cfg = ExperimentConfig(cameras=cams, **raw)
    if cfg.video_out_path is None:
        cfg.video_out_path = f"data/{path.stem}"
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    names = [c.name for c in cfg.cameras]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate camera names in config: {names}")
    if cfg.policy_image_camera not in names:
        raise ValueError(
            f"policy_image_camera={cfg.policy_image_camera!r} must appear in cameras list ({names})."
        )
    if cfg.policy_wrist_camera not in names:
        raise ValueError(
            f"policy_wrist_camera={cfg.policy_wrist_camera!r} must appear in cameras list ({names})."
        )
    if cfg.policy_image_flip not in ("rot180", "vflip"):
        raise ValueError(
            f"policy_image_flip must be 'rot180' or 'vflip', got {cfg.policy_image_flip!r}."
        )
    for c in cfg.cameras:
        if (c.pos is None) != (c.quat is None):
            raise ValueError(
                f"Camera {c.name!r}: pos and quat must be set together (got pos={c.pos}, quat={c.quat})."
            )
        if c.orbit_object_centroid and (c.pos is not None or c.orbit_deg != 0.0):
            raise ValueError(
                f"Camera {c.name!r}: orbit_object_centroid is mutually exclusive with "
                f"pos/quat and world-Z orbit_deg. Use radius/azimuth/elevation_delta instead."
            )


def eval_libero(args: Args) -> None:
    cfg = _load_config(args.config)
    np.random.seed(cfg.seed)

    # Register any cameras with explicit (pos, quat) with the arena. For built-in names
    # (agentview, frontview, ...) this overwrites pose; for new names (e.g. "cam1") it
    # adds the camera to the model so it can be rendered.
    _install_custom_cameras(cfg.cameras)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {cfg.task_suite_name}")

    out_root = pathlib.Path(cfg.video_out_path)
    out_root.mkdir(parents=True, exist_ok=True)
    for cam in cfg.cameras:
        if cam.save:
            (out_root / (cam.save_subdir or cam.name)).mkdir(parents=True, exist_ok=True)
    (out_root / "observations").mkdir(parents=True, exist_ok=True)

    if cfg.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif cfg.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif cfg.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif cfg.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {cfg.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(cfg.host, cfg.port)

    camera_names = [c.name for c in cfg.cameras]

    total_episodes, total_successes = 0, 0
    seen_scene_types: Set[str] = set()
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)

        if cfg.one_task_per_scene_type:
            scene_type = _scene_type_from_task_name(task.name)
            if scene_type in seen_scene_types:
                logging.info(f"Skipping {task.name}: already ran a {scene_type} task")
                continue
            seen_scene_types.add(scene_type)

        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, cfg.camera_resolution, camera_names, cfg.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            # robosuite's reset rebuilds the model, restoring cam_pos to XML values.
            # Re-apply per-camera pose every reset. World-Z orbit/offset cameras are
            # scene-independent, so they can be placed now. Object-centroid cameras need
            # the object positions, so they're placed AFTER set_init_state below.
            for cam in cfg.cameras:
                _apply_camera_pose(env, cam)
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            # Object-centroid orbit cameras depend on this episode's object layout (matching
            # the data-collection renderer, which places the camera from states[0]). Place
            # them now, then re-render so `obs` reflects the relocated camera.
            obj_cams = [c for c in cfg.cameras if c.orbit_object_centroid]
            if obj_cams:
                for cam in obj_cams:
                    _apply_orbit_object_centroid(env, cam)
                obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_per_cam: dict[str, list] = {c.name: [] for c in cfg.cameras if c.save}
            episode_states = []

            logging.info(f"Starting episode {task_episodes+1}...")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Policy input images: flip (per training convention) + resize.
                    policy_img = _prepare_policy_image(
                        obs[f"{cfg.policy_image_camera}_image"], cfg.resize_size, cfg.policy_image_flip
                    )
                    wrist_img = _prepare_policy_image(
                        obs[f"{cfg.policy_wrist_camera}_image"], cfg.resize_size, cfg.policy_image_flip
                    )

                    # Append each requested camera's frame to its replay buffer.
                    for cam in cfg.cameras:
                        if not cam.save:
                            continue
                        frame = obs[f"{cam.name}_image"]
                        if cam.rotate_180:
                            frame = frame[::-1, ::-1]
                        else:
                            # MuJoCo offscreen renders are upside-down; flip vertically for natural viewing.
                            frame = frame[::-1]
                        replay_per_cam[cam.name].append(np.ascontiguousarray(frame))

                    if not action_plan:
                        element = {
                            "observation/image": policy_img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= cfg.replan_steps
                        ), f"We want to replan every {cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: cfg.replan_steps])

                    action = action_plan.popleft()

                    episode_states.append(
                        np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                    )

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            stem = f"rollout_{task_segment}_ep{episode_idx+1:03d}_{suffix}"

            for cam in cfg.cameras:
                if not cam.save:
                    continue
                frames = replay_per_cam[cam.name]
                if not frames:
                    continue
                subdir = out_root / (cam.save_subdir or cam.name)
                imageio.mimwrite(subdir / f"{stem}.mp4", frames, fps=10)

            states_arr = np.stack(episode_states) if episode_states else np.empty((0,), dtype=np.float32)
            np.save(out_root / "observations" / f"{stem}_states.npy", states_arr)

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _prepare_policy_image(raw, resize_size: int, flip: str = "rot180"):
    # "rot180": raw[::-1, ::-1] (standard openpi LIBERO convention).
    # "vflip":  raw[::-1]       (relocate_camera datasets: view45 / all_views).
    flipped = raw[::-1, ::-1] if flip == "rot180" else raw[::-1]
    img = np.ascontiguousarray(flipped)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))


# Cameras present by default in every LIBERO scene. Names outside this set are treated
# as user-defined: they need to be registered with the arena before model compilation.
_BUILTIN_CAMERAS = frozenset({
    "agentview",
    "canonical_agentview",
    "frontview",
    "birdview",
    "sideview",
    "robot0_eye_in_hand",
    "robot0_robotview",
})


def _install_custom_cameras(cameras: list) -> None:
    """Patch every `_setup_camera` along the BDDLBaseDomain hierarchy so each scene
    rebuild also registers our extra cameras.

    For each user camera we decide a registration pose:
      * If `pos` and `quat` are given -> use them directly.
      * Otherwise (new camera with only orbit/offset) -> inherit the **LIBERO default**
        agentview pose for the current scene. We snapshot it immediately after the
        original `_setup_camera` runs (per-scene defaults differ — see
        libero_living_room_tabletop_manipulation.py and friends), before applying any
        user override that targets agentview itself.

    Each LIBERO problem subclass overrides `_setup_camera` without calling super(),
    so patching only the base class would be a no-op. We walk all subclasses that
    define their own `_setup_camera` and patch each.
    """
    from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
    from robosuite.utils.mjcf_utils import find_elements, string_to_array

    # Anything not in BUILTIN needs registration; same goes for builtin entries that
    # provide an explicit pose override.
    needs_register = [
        c for c in cameras
        if c.name not in _BUILTIN_CAMERAS or (c.pos is not None and c.quat is not None)
    ]
    if not needs_register:
        return

    # Validate up front: a new (non-builtin) camera with no pose can only inherit
    # the agentview default. If `agentview` is missing from the arena XML for some
    # exotic scene, we'll error at patch-call time below — but we can still flag
    # an obviously-broken combo here.
    for c in needs_register:
        if c.name not in _BUILTIN_CAMERAS and c.pos is None:
            # Fine — will inherit agentview default at scene-build time.
            pass

    def make_patched(original):
        def patched(self, mujoco_arena):
            original(self, mujoco_arena)
            # Snapshot the LIBERO default agentview pose for this scene BEFORE
            # applying any user overrides.
            av = find_elements(
                root=mujoco_arena.worldbody,
                tags="camera",
                attribs={"name": "agentview"},
                return_first=True,
            )
            av_default_pos = list(string_to_array(av.get("pos"))) if av is not None else None
            av_default_quat = list(string_to_array(av.get("quat"))) if av is not None else None

            for cam in needs_register:
                if cam.pos is not None and cam.quat is not None:
                    pos, quat = list(cam.pos), list(cam.quat)
                else:
                    if av_default_pos is None:
                        raise RuntimeError(
                            f"Camera {cam.name!r} has no pos/quat and the current "
                            f"scene has no agentview to inherit from."
                        )
                    pos, quat = av_default_pos, av_default_quat
                mujoco_arena.set_camera(camera_name=cam.name, pos=pos, quat=quat)
        return patched

    # Collect all classes (including BDDLBaseDomain itself) that define their own
    # _setup_camera. Subclasses that inherit unchanged from a base we already patched
    # don't need patching.
    seen, stack = set(), [BDDLBaseDomain]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())

    for cls in seen:
        if "_setup_camera" not in cls.__dict__:
            continue
        if "_openpi_setup_camera_orig" in cls.__dict__:
            # Already patched in this process; reuse the saved original.
            continue
        cls._openpi_setup_camera_orig = cls.__dict__["_setup_camera"]
        cls._setup_camera = make_patched(cls._openpi_setup_camera_orig)


def _apply_camera_pose(env, cam: CameraConfig) -> None:
    """Apply absolute pose override (if any) then orbit + translate to the named camera."""
    if cam.orbit_object_centroid:
        return  # placed after set_init_state via _apply_orbit_object_centroid
    if (
        cam.pos is None
        and cam.quat is None
        and cam.orbit_deg == 0.0
        and cam.offset_back == 0.0
        and cam.offset_up == 0.0
        and cam.offset_right == 0.0
    ):
        return
    cam_id = env.sim.model.camera_name2id(cam.name)
    if cam.pos is not None:
        env.sim.model.cam_pos[cam_id] = np.asarray(cam.pos, dtype=np.float64)
        env.sim.model.cam_quat[cam_id] = np.asarray(cam.quat, dtype=np.float64)
    if cam.orbit_deg != 0.0:
        _orbit_camera(env, cam_id, cam.orbit_deg, cam.orbit_center)
    if cam.offset_back != 0.0 or cam.offset_up != 0.0 or cam.offset_right != 0.0:
        _translate_camera(env, cam_id, cam.offset_back, cam.offset_up, cam.offset_right)
    env.sim.forward()


def _orbit_camera(env, cam_id: int, angle_deg: float, center_xy) -> None:
    """Rotate the camera around the world +Z axis through center_xy by angle_deg (CCW viewed from +Z)."""
    pos = env.sim.model.cam_pos[cam_id].copy()
    quat = env.sim.model.cam_quat[cam_id].copy()  # (w, x, y, z)
    cx, cy = center_xy
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = pos[0] - cx, pos[1] - cy
    pos[0] = cx + c * dx - s * dy
    pos[1] = cy + s * dx + c * dy
    qw, qx, qy, qz = quat
    qrw, qrz = math.cos(theta / 2), math.sin(theta / 2)
    env.sim.model.cam_pos[cam_id] = pos
    env.sim.model.cam_quat[cam_id] = np.array([
        qrw * qw - qrz * qz,
        qrw * qx - qrz * qy,
        qrw * qy + qrz * qx,
        qrw * qz + qrz * qw,
    ])


def _translate_camera(env, cam_id: int, back: float, up: float, right: float) -> None:
    """Translate the camera: `back` along camera-local +Z (away from look dir),
    `up` along world +Z, `right` along camera-local +X (image right)."""
    pos = env.sim.model.cam_pos[cam_id].copy()
    qw, qx, qy, qz = env.sim.model.cam_quat[cam_id]
    back_dir = np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ])
    right_dir = np.array([
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qw * qz),
        2.0 * (qx * qz - qw * qy),
    ])
    pos = pos + back * back_dir + right * right_dir + up * np.array([0.0, 0.0, 1.0])
    env.sim.model.cam_pos[cam_id] = pos


# ----------------------------------------------------------------------------- #
# Object-centroid orbit camera. Ported verbatim from the data-collection renderer
# (relocate_camera/scripts/render_relocated.py) so eval reproduces the exact
# training viewpoints. MuJoCo: camera looks along local -z, +x right, +y up;
# quaternion stored wxyz.
# ----------------------------------------------------------------------------- #
def _mat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def _look_at_quat(pos, center, world_up=np.array([0.0, 0.0, 1.0])):
    """Quaternion (wxyz) orienting a MuJoCo camera at `pos` to look at `center`."""
    z = pos - center  # camera +z points from the target back toward the camera
    z = z / np.linalg.norm(z)
    x = np.cross(world_up, z)
    if np.linalg.norm(x) < 1e-6:  # looking straight down/up: pick an arbitrary right
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return _mat_to_quat_wxyz(R)


def _orbit_pose(center, radius, azimuth_deg, elevation_deg):
    """Camera (pos, quat) on an orbit of `radius` around `center`."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    offset = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.asarray(center, dtype=float) + offset
    return pos, _look_at_quat(pos, center)


def _quat_to_mat(q):
    """Rotation matrix from a MuJoCo quaternion (wxyz)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _axis_pivot(orig_pos, orig_quat, target):
    """Pivot on the original camera's optical axis nearest the target, so deltas of
    (0,0,0) recover the original camera pose exactly on every task suite."""
    fwd = -_quat_to_mat(orig_quat)[:, 2]
    d = float(np.dot(np.asarray(target, dtype=float) - orig_pos, fwd))
    return orig_pos + d * fwd


def _spherical_of(pos, center):
    """Inverse of _orbit_pose's position: (radius, azimuth_deg, elevation_deg)."""
    v = np.asarray(pos, dtype=float) - np.asarray(center, dtype=float)
    r = np.linalg.norm(v)
    az = np.degrees(np.arctan2(v[1], v[0]))
    el = np.degrees(np.arcsin(np.clip(v[2] / r, -1.0, 1.0)))
    return r, az, el


def _resolve_obj_pos(sim, name):
    """Best-effort world position for an object-of-interest name."""
    bodies = set(sim.model.body_names)
    try:
        sites = set(sim.model.site_names)
    except Exception:
        sites = set()
    for cand in (name + "_main", name):
        if cand in bodies:
            return np.array(sim.data.body_xpos[sim.model.body_name2id(cand)])
    if name in sites:
        return np.array(sim.data.site_xpos[sim.model.site_name2id(name)])
    toks = name.split("_")
    for k in range(len(toks) - 1, 0, -1):  # strip trailing tokens -> parent object
        base = "_".join(toks[:k])
        for cand in (base + "_main", base):
            if cand in bodies:
                return np.array(sim.data.body_xpos[sim.model.body_name2id(cand)])
    return None


def _compute_center(env, override):
    """Look-at target: CLI override, else centroid of objects-of-interest."""
    if override is not None:
        return np.asarray(override, dtype=float)
    sim = env.sim
    pts = [p for obj in env.env.obj_of_interest if (p := _resolve_obj_pos(sim, obj)) is not None]
    if not pts:  # fall back to all rigid object bodies ('*_main', minus robot/mount)
        pts = [
            np.array(sim.data.body_xpos[sim.model.body_name2id(b)])
            for b in sim.model.body_names
            if b.endswith("_main") and not b.startswith(("robot", "gripper", "mount"))
        ]
    center = np.mean(pts, axis=0)
    center[2] += 0.05  # lift slightly off the floor toward object bodies
    return center


def _apply_orbit_object_centroid(env, cam: CameraConfig) -> None:
    """Reposition `cam` exactly like render_relocated.py: orbit around the objects-of-interest
    centroid (projected onto the original sightline), re-aiming at it, with radius/azimuth/
    elevation applied as DELTAS from the original camera pose. Must run AFTER set_init_state
    so object positions are set."""
    cid = env.sim.model.camera_name2id(cam.name)
    # The current pose (post-reset, pre-relocation) IS LIBERO's original camera pose.
    orig_pos = np.array(env.sim.model.cam_pos[cid], dtype=float)
    orig_quat = np.array(env.sim.model.cam_quat[cid], dtype=float)

    target = _compute_center(env, cam.center)
    if cam.center is not None:
        center = np.asarray(cam.center, dtype=float)
    else:
        center = _axis_pivot(orig_pos, orig_quat, target)

    r0, az0, el0 = _spherical_of(orig_pos, center)
    cam_pos, cam_quat = _orbit_pose(
        center,
        r0 + cam.radius_delta,
        az0 + cam.azimuth_delta,
        el0 + cam.elevation_delta,
    )
    env.sim.model.cam_pos[cid] = cam_pos
    env.sim.model.cam_quat[cid] = cam_quat
    env.sim.forward()


def _scene_type_from_task_name(name: str) -> str:
    if name.startswith("KITCHEN_"):
        return "kitchen"
    if name.startswith("LIVING_ROOM_"):
        return "living_room"
    if name.startswith("STUDY_"):
        return "study"
    return "default"


def _get_libero_env(task, resolution, camera_names, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
