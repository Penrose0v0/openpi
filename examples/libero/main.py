import collections
import dataclasses
import logging
import math
import pathlib
from typing import Set

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    one_task_per_scene_type: bool = False  # If True, run only the first task encountered per scene type (kitchen/living_room/study)
    agentview_orbit_deg: float = 0.0  # Orbit agentview around world +Z by this angle (deg, CCW viewed from +Z). 0 = no change.
    agentview_orbit_center_x: float = 0.0  # XY center for the orbit (world frame).
    agentview_orbit_center_y: float = 0.0
    agentview_offset_back: float = 0.0  # Translate agentview along its local +Z (away from look direction), in meters.
    agentview_offset_up: float = 0.0  # Translate agentview along world +Z, in meters.
    agentview_offset_right: float = 0.0  # Translate agentview along its local +X (image right), in meters.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    seen_scene_types: Set[str] = set()
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        if args.one_task_per_scene_type:
            scene_type = _scene_type_from_task_name(task.name)
            if scene_type in seen_scene_types:
                logging.info(f"Skipping {task.name}: already ran a {scene_type} task")
                continue
            seen_scene_types.add(scene_type)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            # robosuite's reset rebuilds the model, restoring cam_pos to XML values.
            # Re-apply the agentview orbit after every reset.
            _orbit_agentview(
                env,
                args.agentview_orbit_deg,
                (args.agentview_orbit_center_x, args.agentview_orbit_center_y),
            )
            _translate_agentview(
                env,
                args.agentview_offset_back,
                args.agentview_offset_up,
                args.agentview_offset_right,
            )
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_states = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
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

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Record state at the moment the action is issued (pre-step)
                    episode_states.append(
                        np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                    )

                    # Execute action in environment
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

            # # Save a replay video of the episode
            # suffix = "success" if done else "failure"
            # task_segment = task_description.replace(" ", "_")
            # imageio.mimwrite(
            #     pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
            #     [np.asarray(x) for x in replay_images],
            #     fps=10,
            # )

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx+1:03d}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Save state trajectory alongside the video
            states_arr = np.stack(episode_states) if episode_states else np.empty((0,), dtype=np.float32)
            np.save(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx+1:03d}_{suffix}_states.npy",
                states_arr,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _orbit_agentview(env, angle_deg: float, center_xy) -> None:
    """Rotate agentview around the world +Z axis through center_xy by angle_deg (CCW viewed from +Z)."""
    if angle_deg == 0.0:
        return
    cam_id = env.sim.model.camera_name2id("agentview")
    pos = env.sim.model.cam_pos[cam_id].copy()
    quat = env.sim.model.cam_quat[cam_id].copy()  # MuJoCo convention: (w, x, y, z)
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
    env.sim.forward()


def _translate_agentview(env, back: float, up: float, right: float = 0.0) -> None:
    """Translate agentview in meters: `back` along camera's local +Z (away from look dir),
    `up` along world +Z, `right` along camera's local +X (image right). Applied after any orbit."""
    if back == 0.0 and up == 0.0 and right == 0.0:
        return
    cam_id = env.sim.model.camera_name2id("agentview")
    pos = env.sim.model.cam_pos[cam_id].copy()
    qw, qx, qy, qz = env.sim.model.cam_quat[cam_id]
    # Camera's local +Z in world coords (look dir is local -Z, so +Z is backward).
    back_dir = np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ])
    # Camera's local +X in world coords (image right).
    right_dir = np.array([
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qw * qz),
        2.0 * (qx * qz - qw * qy),
    ])
    pos = pos + back * back_dir + right * right_dir + up * np.array([0.0, 0.0, 1.0])
    env.sim.model.cam_pos[cam_id] = pos
    env.sim.forward()


def _scene_type_from_task_name(name: str) -> str:
    if name.startswith("KITCHEN_"):
        return "kitchen"
    if name.startswith("LIVING_ROOM_"):
        return "living_room"
    if name.startswith("STUDY_"):
        return "study"
    return "default"


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
