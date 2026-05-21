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
        for key in ("pos", "quat", "orbit_center"):
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
    for c in cfg.cameras:
        if (c.pos is None) != (c.quat is None):
            raise ValueError(
                f"Camera {c.name!r}: pos and quat must be set together (got pos={c.pos}, quat={c.quat})."
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
            # Re-apply per-camera pose every reset.
            for cam in cfg.cameras:
                _apply_camera_pose(env, cam)
            action_plan = collections.deque()

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

                    # Policy input images: rotate 180 + resize, matching training preprocessing.
                    policy_img = _prepare_policy_image(
                        obs[f"{cfg.policy_image_camera}_image"], cfg.resize_size
                    )
                    wrist_img = _prepare_policy_image(
                        obs[f"{cfg.policy_wrist_camera}_image"], cfg.resize_size
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


def _prepare_policy_image(raw, resize_size: int):
    img = np.ascontiguousarray(raw[::-1, ::-1])
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
