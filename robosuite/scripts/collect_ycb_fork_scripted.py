"""Collect a Panda scripted demonstration for ``YCBForkInRack``.

The original Play2Perfect task uses a dexterous hand. This Panda adaptation
starts each recorded episode from a pre-grasped fork: setup moves the arm,
places the fork between nearly closed fingers, and is excluded from the saved
trajectory. Every recorded transition after that setup is produced by a
normal robosuite OSC action (close, lift, transport, insert, release, retreat).

The script writes a compressed action / state trajectory and a side-by-side
agent-view + wrist-view MP4.

Example:
    NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
        python robosuite/scripts/collect_ycb_fork_scripted.py
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import robosuite as suite
import robosuite.utils.transform_utils as T


CAMERAS = ("agentview", "robot0_eye_in_hand")
CAMERA_LABELS = ("agentview", "wrist view")
GRASP_POINT_LOCAL = np.array((0.055, 0.0, 0.0))
STAGING_EEF_POS = np.array((0.10, -0.10, 1.04))


def make_osc_config(robot="Panda"):
    config = suite.load_composite_controller_config(robot=robot)
    arm_names = [
        name for name, part in config["body_parts"].items() if part.get("type", "").startswith("OSC")
    ]
    if len(arm_names) != 1:
        raise ValueError(f"Expected one OSC arm, found {arm_names}")
    arm = config["body_parts"][arm_names[0]]
    arm.update(
        {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "world",
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
    )
    return config


def eef_pose(env):
    site_id = env.robots[0].eef_site_id["right"]
    pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=float).copy()
    mat = np.asarray(env.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()
    return pos, T.mat2quat(mat)


def shortest_axisangle(target_quat, current_quat):
    error = T.quat_distance(np.asarray(target_quat).copy(), np.asarray(current_quat).copy())
    if error[3] < 0.0:
        error = -error
    return T.quat2axisangle(error)


def osc_action(env, target_pos, target_quat, gripper):
    current_pos, current_quat = eef_pose(env)
    action = np.zeros(env.action_dim, dtype=float)
    action[:3] = np.clip((np.asarray(target_pos) - current_pos) / 0.05, -1.0, 1.0)
    action[3:6] = np.clip(shortest_axisangle(target_quat, current_quat) / 0.5, -1.0, 1.0)
    action[6] = gripper
    return action


def configure_panda_grasp(env):
    """Use high-friction pads and the Panda's physical grasp-force range."""
    for geom_id, name in enumerate(env.sim.model.geom_names):
        if name and ("finger" in name or "pad" in name):
            env.sim.model.geom_friction[geom_id] = (5.0, 0.05, 0.01)
            env.sim.model.geom_condim[geom_id] = 6
    for actuator_id, name in enumerate(env.sim.model.actuator_names):
        if name and ("gripper" in name or "finger" in name):
            env.sim.model.actuator_forcerange[actuator_id] = (-80.0, 80.0)
            env.sim.model.actuator_forcelimited[actuator_id] = 1


def _fingerpad_midpoint(env):
    positions = []
    for name in env.sim.model.geom_names:
        if name and "pad_collision" in name:
            geom_id = env.sim.model.geom_name2id(name)
            positions.append(np.asarray(env.sim.data.geom_xpos[geom_id]).copy())
    if len(positions) != 2:
        raise RuntimeError(f"Expected two Panda finger pads, found {len(positions)}")
    return np.mean(positions, axis=0)


def run_unrecorded_servo(env, target_pos, target_quat, gripper, steps):
    for _ in range(steps):
        env.step(osc_action(env, target_pos, target_quat, gripper))


def initialize_pregrasp(env):
    """Create the declared pre-grasped episode initial state."""
    _, initial_quat = eef_pose(env)
    run_unrecorded_servo(env, STAGING_EEF_POS, initial_quat, -1.0, 140)

    gripper = env.robots[0].gripper["right"]
    for joint, qpos in zip(gripper.joints, (0.011, -0.011)):
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.data.set_joint_qvel(joint, 0.0)
    # The preceding open command leaves the binary gripper integrator at -1.
    # Synchronize it with the nearly closed qpos before recording begins.
    gripper.current_action = np.ones(gripper.dof)
    env.sim.forward()

    _, fork_goal_quat = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
    fork_pos = _fingerpad_midpoint(env) - T.quat2mat(fork_goal_quat) @ GRASP_POINT_LOCAL
    env._set_free_object_pose(env.fork, fork_pos, fork_goal_quat[[3, 0, 1, 2]])
    env.sim.data.set_joint_qvel(env.fork.joints[0], np.zeros(6))
    env.sim.forward()
    if not env._check_grasp(gripper, env.fork.contact_geoms):
        raise RuntimeError("Pre-grasp initialization did not put the fork between both finger pads")
    return initial_quat


def render_pair(env, size, phase, success=False):
    panels = []
    for camera, label in zip(CAMERAS, CAMERA_LABELS):
        pixels = env.sim.render(height=size, width=size, camera_name=camera)
        panel = Image.fromarray(np.flipud(pixels))
        draw = ImageDraw.Draw(panel, "RGBA")
        draw.rectangle((0, 0, size, 28), fill=(12, 18, 25, 220))
        draw.text((10, 8), label, fill=(255, 255, 255, 255))
        panels.append(np.asarray(panel))
    frame = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, frame.height - 30, frame.width, frame.height), fill=(0, 0, 0, 205))
    text = f"scripted Panda policy | {phase}"
    if success:
        text += " | success=True"
    draw.text((10, frame.height - 22), text, fill=(255, 215, 80, 255))
    return np.asarray(frame)


class RolloutRecorder:
    def __init__(self, env, size, fps, control_freq):
        self.env = env
        self.size = size
        self.stride = max(1, round(control_freq / fps))
        self.initial_state = np.asarray(env.sim.get_state().flatten()).copy()
        self.frames = []
        self.actions = []
        self.states = []
        self.rewards = []
        self.successes = []
        self.phases = []
        self.step_count = 0

    def hold_video(self, count, phase):
        frame = render_pair(self.env, self.size, phase, bool(self.env._check_success()))
        self.frames.extend(frame.copy() for _ in range(count))

    def step(self, action, phase):
        _, reward, _, _ = self.env.step(action)
        success = bool(self.env._check_success())
        self.actions.append(np.asarray(action).copy())
        self.states.append(np.asarray(self.env.sim.get_state().flatten()).copy())
        self.rewards.append(float(reward))
        self.successes.append(success)
        self.phases.append(phase)
        if self.step_count % self.stride == 0:
            self.frames.append(render_pair(self.env, self.size, phase, success))
        self.step_count += 1

    def servo(self, target_pos, target_quat, gripper, steps, phase):
        for _ in range(steps):
            self.step(osc_action(self.env, target_pos, target_quat, gripper), phase)


def run_policy(env, recorder, eef_quat):
    gripper = env.robots[0].gripper["right"]
    recorder.hold_video(20, "pre-grasped reset")

    recorder.servo(STAGING_EEF_POS, eef_quat, 1.0, 35, "close")
    if not env._check_grasp(gripper, env.fork.contact_geoms):
        raise RuntimeError("Fork was not grasped after the close stage")

    lift = STAGING_EEF_POS + np.array((0.0, 0.0, 0.10))
    recorder.servo(lift, eef_quat, 1.0, 60, "lift")
    if not env._check_grasp(gripper, env.fork.contact_geoms):
        raise RuntimeError("Fork slipped during lift")

    goal_pos, _ = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
    above = np.array((goal_pos[0], goal_pos[1], lift[2]))
    recorder.servo(above, eef_quat, 1.0, 100, "transport")

    fork_pos, _ = env._body_pose(env.fork)
    descend_start, _ = eef_pose(env)
    descend_goal = descend_start + (goal_pos - fork_pos)
    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.0), start=1):
        waypoint = descend_start + fraction * (descend_goal - descend_start)
        recorder.servo(waypoint, eef_quat, 1.0, 50, f"insert {index}/4")

    if not env._check_success():
        raise RuntimeError(f"Insertion did not reach the goal: error={env._fork_error()}")

    release_pos, _ = eef_pose(env)
    recorder.servo(release_pos, eef_quat, -1.0, 45, "release")
    retreat = release_pos + np.array((0.15, -0.05, 0.15))
    recorder.servo(retreat, eef_quat, -1.0, 100, "retreat")
    recorder.hold_video(30, "complete")

    if not env._check_success():
        raise RuntimeError(f"Fork did not remain seated after release: error={env._fork_error()}")


def write_outputs(output_dir, env, recorder, fps, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "YCBForkInRack_scripted_agent_wrist.mp4"
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

    trajectory_path = output_dir / "YCBForkInRack_scripted_demo.npz"
    np.savez_compressed(
        trajectory_path,
        initial_state=recorder.initial_state,
        actions=np.asarray(recorder.actions),
        states=np.asarray(recorder.states),
        rewards=np.asarray(recorder.rewards),
        successes=np.asarray(recorder.successes),
        phases=np.asarray(recorder.phases),
    )

    sheet_path = output_dir / "YCBForkInRack_scripted_contact_sheet.png"
    indices = (0, len(recorder.frames) // 2, len(recorder.frames) - 1)
    sheet = Image.new("RGB", (recorder.frames[0].shape[1], recorder.frames[0].shape[0] * 3))
    for row, index in enumerate(indices):
        sheet.paste(Image.fromarray(recorder.frames[index]), (0, row * recorder.frames[0].shape[0]))
    sheet.save(sheet_path)

    metadata = {
        "task": "YCBForkInRack",
        "robot": "Panda",
        "policy": "scripted_osc_pose",
        "initialization": "pregrasped_fork_excluded_from_trajectory",
        "recorded_transitions_are_action_driven": True,
        "cameras": list(CAMERAS),
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
    return video_path, trajectory_path, sheet_path, metadata_path, metadata


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
        horizon=1200,
        ignore_done=True,
        seed=seed,
    )
    try:
        env.reset()
        configure_panda_grasp(env)
        initial_quat = initialize_pregrasp(env)
        recorder = RolloutRecorder(env, size=size, fps=fps, control_freq=control_freq)
        run_policy(env, recorder, initial_quat)
        outputs = write_outputs(output_dir, env, recorder, fps, seed)
        print(json.dumps(outputs[-1], indent=2))
        return outputs
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/ycb_fork_scripted"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, default=320, help="Width and height of each camera panel")
    args = parser.parse_args()
    collect(args.output_dir.resolve(), seed=args.seed, fps=args.fps, size=args.size)


if __name__ == "__main__":
    main()
