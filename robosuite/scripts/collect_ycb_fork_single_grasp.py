"""Collect a table-pick, single-grasp Panda demo for ``YCBForkInRack``.

The fork starts flat on the table. A scripted OSC policy grasps its handle
once, lifts it, rotates it 90 degrees without releasing, pre-inserts the
handle, then opens the gripper so gravity seats the fork in the rack.

Example:
    NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
        python robosuite/scripts/collect_ycb_fork_single_grasp.py
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.scripts.collect_ycb_fork_scripted import (
    RolloutRecorder,
    configure_panda_grasp,
    eef_pose,
    make_osc_config,
)


TABLE_FORK_POS = np.array((0.12, -0.10, 0.814))
HANDLE_GRASP_LOCAL_X = -0.055
ROTATION_PIVOT = np.array((0.30, 0.0, 1.0))


def initialize_table_fork(env):
    """Set a stable, reachable table pose before the recorded rollout."""
    # Starting upside-down reverses the long axis. The reachable +90-degree
    # Panda wrist motion then produces the task's exact -90-degree fork pose.
    quat_xyzw = Rotation.from_euler("y", 180.0, degrees=True).as_quat()
    env._set_free_object_pose(env.fork, TABLE_FORK_POS, quat_xyzw[[3, 0, 1, 2]])
    env.sim.data.set_joint_qvel(env.fork.joints[0], np.zeros(6))
    env.sim.forward()


def object_space_servo(env, recorder, desired_pos, desired_quat, steps, phase):
    """Apply the measured fork pose error to the EEF target."""
    fork_pos, fork_quat = env._body_pose(env.fork)
    current_eef_pos, current_eef_quat = eef_pose(env)
    delta_rotation = T.quat2mat(desired_quat) @ T.quat2mat(fork_quat).T
    target_eef_pos = current_eef_pos + (np.asarray(desired_pos) - fork_pos)
    target_eef_quat = T.mat2quat(delta_rotation @ T.quat2mat(current_eef_quat))
    recorder.servo(target_eef_pos, target_eef_quat, 1.0, steps, phase)


def run_single_grasp_policy(env, recorder):
    gripper = env.robots[0].gripper["right"]
    _, initial_eef_quat = eef_pose(env)
    initial_eef_rotation = T.quat2mat(initial_eef_quat)
    recorder.hold_video(20, "table reset")

    grasp = TABLE_FORK_POS + np.array((0.055, 0.0, 0.002))
    recorder.servo(grasp + np.array((0.0, 0.0, 0.12)), initial_eef_quat, -1.0, 75, "approach")
    recorder.servo(grasp, initial_eef_quat, -1.0, 55, "descend")
    recorder.servo(grasp, initial_eef_quat, 1.0, 45, "close")
    if not env._check_grasp(gripper, env.fork.contact_geoms):
        raise RuntimeError("Panda failed to establish the initial handle grasp")

    for target, phase in (
        ((0.18, -0.10, 0.95), "lift"),
        ((0.24, -0.05, 1.00), "transfer 1/2"),
        ((0.30, 0.00, 1.00), "transfer 2/2"),
    ):
        recorder.servo(np.asarray(target), initial_eef_quat, 1.0, 100, phase)
        if not env._check_grasp(gripper, env.fork.contact_geoms):
            raise RuntimeError(f"Fork slipped during {phase}")

    rotated_eef_quat = None
    for index in range(18):
        angle = 90.0 * (index + 1) / 18.0
        delta_rotation = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        rotated_eef_quat = T.mat2quat(delta_rotation @ initial_eef_rotation)
        recorder.servo(
            ROTATION_PIVOT,
            rotated_eef_quat,
            1.0,
            40,
            f"rotate {int(angle):02d} deg",
        )
        if not env._check_grasp(gripper, env.fork.contact_geoms):
            raise RuntimeError(f"Fork slipped during rotation at {angle:.1f} degrees")
    recorder.servo(ROTATION_PIVOT, rotated_eef_quat, 1.0, 120, "rotation settle")

    goal_pos, goal_quat = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
    for offset, repeats in ((0.20, 5), (0.12, 5), (0.075, 5), (0.055, 5), (0.045, 5)):
        desired_pos = goal_pos + np.array((0.0, 0.0, offset))
        for correction in range(repeats):
            object_space_servo(
                env,
                recorder,
                desired_pos,
                goal_quat,
                40,
                f"preinsert {offset:.3f} m {correction + 1}/{repeats}",
            )
        if not env._check_grasp(gripper, env.fork.contact_geoms):
            raise RuntimeError(f"Fork slipped while approaching the {offset:.3f} m waypoint")

    release_pos, release_quat = eef_pose(env)
    recorder.servo(release_pos, release_quat, -1.0, 70, "release")
    for _ in range(180):
        action = np.zeros(env.action_dim)
        action[6] = -1.0
        recorder.step(action, "gravity seat")
    if not env._check_success():
        raise RuntimeError(f"Fork did not seat after release: error={env._fork_error()}")

    retreat = release_pos + np.array((0.12, -0.08, 0.15))
    recorder.servo(retreat, release_quat, -1.0, 120, "retreat")
    recorder.hold_video(30, "complete")
    if not env._check_success():
        raise RuntimeError(f"Fork left the goal after retreat: error={env._fork_error()}")


def write_outputs(output_dir, env, recorder, fps, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "YCBForkInRack_single_grasp_agent_wrist.mp4"
    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        pixelformat="yuv420p",
    ) as writer:
        for frame in recorder.frames:
            writer.append_data(frame)

    trajectory_path = output_dir / "YCBForkInRack_single_grasp_demo.npz"
    np.savez_compressed(
        trajectory_path,
        initial_state=recorder.initial_state,
        actions=np.asarray(recorder.actions),
        states=np.asarray(recorder.states),
        rewards=np.asarray(recorder.rewards),
        successes=np.asarray(recorder.successes),
        phases=np.asarray(recorder.phases),
    )

    sheet_path = output_dir / "YCBForkInRack_single_grasp_contact_sheet.png"
    indices = (0, len(recorder.frames) // 3, 2 * len(recorder.frames) // 3, len(recorder.frames) - 1)
    frame_height, frame_width = recorder.frames[0].shape[:2]
    sheet = Image.new("RGB", (frame_width, frame_height * len(indices)))
    for row, index in enumerate(indices):
        sheet.paste(Image.fromarray(recorder.frames[index]), (0, row * frame_height))
    sheet.save(sheet_path)

    metadata = {
        "task": "YCBForkInRack",
        "robot": "Panda",
        "policy": "scripted_osc_pose_single_grasp",
        "initialization": "fork_flat_on_table",
        "recorded_transitions_are_action_driven": True,
        "grasp_count": 1,
        "release_count": 1,
        "cameras": ["agentview", "robot0_eye_in_hand"],
        "control_frequency_hz": env.control_freq,
        "video_fps": fps,
        "seed": seed,
        "steps": len(recorder.actions),
        "success": bool(env._check_success()),
        "final_pose_error": list(env._fork_error()),
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "contact_sheet": str(sheet_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return video_path, trajectory_path, sheet_path, metadata_path


def collect(output_dir, seed=7, fps=30, size=320):
    control_freq = 60
    env = suite.make(
        "YCBForkInRack",
        robots="Panda",
        controller_configs=make_osc_config("Panda"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        hard_reset=False,
        reward_shaping=True,
        control_freq=control_freq,
        horizon=4000,
        ignore_done=True,
        seed=seed,
    )
    try:
        env.reset()
        configure_panda_grasp(env)
        initialize_table_fork(env)
        recorder = RolloutRecorder(env, size=size, fps=fps, control_freq=control_freq)
        run_single_grasp_policy(env, recorder)
        return write_outputs(output_dir, env, recorder, fps, seed)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ycb_fork_single_grasp"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()
    collect(args.output_dir.resolve(), seed=args.seed, fps=args.fps, size=args.size)


if __name__ == "__main__":
    main()
