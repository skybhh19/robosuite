"""Collect production-style joint-position demos for ``YCBForkInRack``.

Each episode randomizes a flat, approximately crosswise fork pose and the rack
pose. A Panda picks the fork once, follows a continuous cubic transfer arc,
inserts it, releases once, and retreats. Both regimes grasp the same continuous
neck region with identical timing. ``full_visible`` and ``partial_hidden`` use
opposed 0 / 180-degree grasps. At the shared 115 mm alignment pose, full must
expose all nine rack-aperture samples and partial must occlude all nine.

The output matches the action convention used by the Threading and ToolHang
collectors: 7 absolute Panda joint targets plus one gripper command. Raw
MuJoCo states, joint-position training labels, policy diagnostics, and a
side-by-side agent / wrist video are saved for every accepted episode.

Example:
    NUMBA_DISABLE_JIT=1 MUJOCO_GL=glfw \
      python robosuite/scripts/collect_ycb_fork_joint.py \
        --regime both --num-demos-per-regime 1 --video-count 2
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.scripts.collect_ycb_fork_scripted import (
    configure_panda_grasp,
    make_osc_config,
    osc_action,
)
from robosuite.utils.ik_utils import IKSolver
from robosuite.wrappers import DataCollectionWrapper


CAMERAS = ("agentview", "robot0_eye_in_hand")
CAMERA_LABELS = ("agentview", "wrist view")
REGIMES = ("full_visible", "partial_hidden")
MOTION_STYLES = (
    "direct_low",
    "high_arc",
    "side_sweep",
    "early_approach",
    "delayed_approach",
    "low_s_curve",
    "vertical_first",
    "shallow_sweep",
    "over_then_back",
    "short_direct",
)
STYLE_VARIANTS = ("plain", "early_bend", "late_bend", "wide_bend", "soft_bend")
TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "ycb_fork_threading_quality_templates.npz"
)

# Both regimes sample the same continuous neck region. Crosswise left/right
# fork resets make the exact opposed 180-degree grasp reachable without the
# long table-level wrist loop required by the earlier longitudinal reset.
COMMON_GRASP_RANGE = (-0.0052, -0.0048)
FULL_VISIBLE_GRASP_RANGE = COMMON_GRASP_RANGE
PARTIAL_HIDDEN_GRASP_RANGE = COMMON_GRASP_RANGE
TEMPLATE_GRASP_LOCAL_X = -0.005
GRASP_YAW_DEG = {"full_visible": 0.0, "partial_hidden": 180.0}
GRASP_LOCAL_Y_M = {"full_visible": 0.0, "partial_hidden": 0.0}
FORK_FACE_ROLL_DEG = {"full_visible": 0.0, "partial_hidden": 0.0}
CANONICAL_GRASP_LOCAL_X = -0.05685087527831503
HANDLE_LOCAL_X_BOUNDS = (-0.099, -0.027)
HANDLE_EDGE_MARGIN = 0.010

# Nominal centers for the bounded collision-screened fork / rack reset ranges.
TABLE_FORK_CENTER = np.array((0.11835732002149688, -0.09975530595515765, 0.814))
# The absolute joint controller reaches this posture with substantially more
# joint-limit margin than the old OSC-only x=0.30 pivot.
ROTATION_PIVOT = np.array((0.26, -0.02, 1.0))
CONTROL_FREQUENCY = 20
PLANNER_FREQUENCY = 60
JOINT_DELTA_SCALE = 0.05
RACK_NOMINAL_POS = np.array((-0.08, 0.10, 0.8425))
FORK_NOMINAL_POS = np.array((0.12, -0.10, 0.814))
FULL_FORK_YAW_CENTER_DEG = 90.0
PARTIAL_FORK_YAW_CENTER_DEG = -90.0


def make_joint_position_config(robot="Panda"):
    """Use the same absolute JOINT_POSITION semantics as Threading / ToolHang."""
    config = suite.load_composite_controller_config(robot=robot)
    arm_names = [
        name for name, part in config["body_parts"].items()
        if part.get("type", "").startswith("OSC")
    ]
    if len(arm_names) != 1:
        raise ValueError(f"Expected one Panda OSC arm to replace, found {arm_names}")
    arm_name = arm_names[0]
    gripper = config["body_parts"][arm_name].get("gripper", {"type": "GRIP"})
    # Threading and ToolHang use the same absolute-joint action convention at
    # kp=100. This millimetre-clearance rack needs kp=300 to track the seated
    # contact path (kp=100 misses the goal by about 9.5 cm); smoothness is
    # therefore enforced explicitly on both commanded and measured joints.
    config["body_parts"][arm_name] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 300,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper,
    }
    return config


def controller_metadata():
    return {
        "backend": "joint_position",
        "action_dim": 8,
        "action_space": "absolute_joint_position_plus_gripper",
        "arm_controller": "JOINT_POSITION",
        "input_type": "absolute",
        "kp": 300,
        "control_frequency_hz": CONTROL_FREQUENCY,
        "joint_delta_scale_rad": JOINT_DELTA_SCALE,
    }


def eef_pose(env):
    site_id = env.robots[0].eef_site_id["right"]
    pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=float).copy()
    mat = np.asarray(env.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()
    return pos, T.mat2quat(mat)


def unit(vector, fallback=(1.0, 0.0, 0.0)):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return np.asarray(fallback, dtype=float) if norm < 1e-9 else vector / norm


def shortest_axisangle(target_quat, current_quat):
    error = T.quat_distance(np.asarray(target_quat).copy(), np.asarray(current_quat).copy())
    if error[3] < 0.0:
        error = -error
    return T.quat2axisangle(error)


class JointPositionPoseAdapter:
    """Convert world-frame EEF goals to smooth absolute Panda joint targets."""

    def __init__(self, env, damping=0.12, integration_dt=0.05, max_dq=1.2):
        robot = env.robots[0]
        arm = robot.arms[0]
        controller = robot.composite_controller.part_controllers[arm]
        if controller.name != "JOINT_POSITION" or controller.input_type != "absolute":
            raise ValueError("absolute JOINT_POSITION controller required")
        self.arm_dim = len(controller.joint_names)
        self.qpos_indexes = np.asarray(controller.qpos_index, dtype=int)
        self.max_target_step = 0.04
        self.ik = IKSolver(
            model=env.sim.model._model,
            data=env.sim.data._data,
            robot_config={
                "joint_names": list(controller.joint_names),
                "end_effector_sites": [controller.ref_name],
                "nullspace_gains": np.zeros(self.arm_dim),
            },
            damping=damping,
            integration_dt=integration_dt,
            max_dq=max_dq,
            input_action_repr="absolute",
            input_rotation_repr="axis_angle",
            input_ref_frame="world",
        )
        self.ik.q0 = self.current_qpos(env)
        self.previous_target = None
        self.global_ik_rng = np.random.RandomState(0)

    def current_qpos(self, env):
        return np.asarray(env.sim.data.qpos[self.qpos_indexes], dtype=float).copy()

    def action(self, env, target_pos, target_quat, gripper):
        axis_angle = T.quat2axisangle(np.asarray(target_quat, dtype=float))
        q_des = self.ik.solve(np.r_[target_pos, axis_angle])
        previous = self.current_qpos(env) if self.previous_target is None else self.previous_target
        q_des = 0.45 * previous + 0.55 * q_des
        q_des = np.clip(q_des, previous - self.max_target_step, previous + self.max_target_step)
        joint_ranges = env.sim.model.jnt_range[self.ik.dof_ids]
        q_des = np.clip(q_des, joint_ranges[:, 0], joint_ranges[:, 1])
        self.previous_target = q_des.copy()
        return np.r_[q_des, np.clip(gripper, -1.0, 1.0)]

    def global_solution(self, env, target_pos, target_quat, restarts=64):
        """Solve a precise waypoint with deterministic multi-start IK."""
        model = env.sim.model._model
        site_id = env.robots[0].eef_site_id["right"]
        lower, upper = model.jnt_range[self.qpos_indexes].T
        reference = self.current_qpos(env)
        work = mujoco.MjData(model)

        def residual(qpos):
            work.qpos[:] = env.sim.data.qpos
            work.qpos[self.qpos_indexes] = qpos
            mujoco.mj_forward(model, work)
            position_error = np.asarray(work.site(site_id).xpos) - np.asarray(target_pos)
            matrix = np.asarray(work.site(site_id).xmat).reshape(3, 3)
            rotation_error = shortest_axisangle(target_quat, T.mat2quat(matrix))
            return np.r_[100.0 * position_error, 5.0 * rotation_error, 0.002 * (qpos - reference)]

        starts = [np.clip(reference, lower + 1e-8, upper - 1e-8)]
        local_restarts = min(restarts - 1, max(16, restarts // 3))
        for _ in range(local_restarts):
            starts.append(
                np.clip(reference + self.global_ik_rng.normal(0.0, 0.45, self.arm_dim), lower, upper)
            )
        for _ in range(restarts - 1 - local_restarts):
            starts.append(self.global_ik_rng.uniform(lower, upper))
        solutions = []
        for start in starts:
            result = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=200,
                ftol=1e-8,
                xtol=1e-8,
            )
            solutions.append((float(np.linalg.norm(residual(result.x)[:6])), result.x.copy()))
        solutions.sort(key=lambda item: item[0])
        best_error = solutions[0][0]
        if best_error > 1.3:
            raise RuntimeError(f"global IK residual too large: {best_error:.6f}")
        return solutions[0][1], solutions[0][0]

    def joint_target_action(self, env, joint_target, gripper):
        previous = self.current_qpos(env) if self.previous_target is None else self.previous_target
        q_des = 0.45 * previous + 0.55 * np.asarray(joint_target, dtype=float)
        q_des = np.clip(q_des, previous - self.max_target_step, previous + self.max_target_step)
        self.previous_target = q_des.copy()
        return np.r_[q_des, np.clip(gripper, -1.0, 1.0)]


def render_pair(env, size, phase, regime, success=False):
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
    text = f"joint position | {regime} | {phase}"
    if success:
        text += " | success=True"
    # Keep the lower wrist pixels unobstructed: the rack aperture naturally
    # projects near the bottom of the eye-in-hand image during pre-insertion.
    draw.rectangle((size - 118, 0, size + 246, 28), fill=(12, 18, 25, 220))
    draw.text((size - 108, 8), text, fill=(255, 215, 80, 255))
    return np.asarray(frame)


class EpisodeRecorder:
    """Record videos and quality metrics while DataCollectionWrapper stores states."""

    def __init__(self, env, adapter, regime, size, video_fps, capture_video=True):
        self.env = env
        self.adapter = adapter
        self.regime = regime
        self.size = size
        self.stride = max(1, round(CONTROL_FREQUENCY / video_fps))
        self.capture_video = capture_video
        self.frames = []
        self.phases = []
        self.actions = []
        self.eef_position_errors = []
        self.eef_orientation_errors = []
        self.action_delta_norms = []
        self.action_jerk_norms = []
        self.previous_delta = None
        self.actual_joint_positions = []
        self.step_count = 0

    def hold_video(self, count, phase):
        if not self.capture_video:
            return
        frame = render_pair(self.env, self.size, phase, self.regime, self.env._check_success())
        self.frames.extend(frame.copy() for _ in range(count))

    def step(self, target_pos, target_quat, gripper, phase, joint_target=None):
        current_pos, current_quat = eef_pose(self.env)
        orientation_error = shortest_axisangle(target_quat, current_quat)
        self.eef_position_errors.append(float(np.linalg.norm(np.asarray(target_pos) - current_pos)))
        self.eef_orientation_errors.append(float(np.linalg.norm(orientation_error)))
        action = (
            self.adapter.action(self.env, target_pos, target_quat, gripper)
            if joint_target is None
            else self.adapter.joint_target_action(self.env, joint_target, gripper)
        )
        if self.actions:
            delta = action[:-1] - self.actions[-1][:-1]
            self.action_delta_norms.append(float(np.linalg.norm(delta)))
            if self.previous_delta is not None:
                self.action_jerk_norms.append(float(np.linalg.norm(delta - self.previous_delta)))
            self.previous_delta = delta
        self.env.step(action)
        self.actions.append(action.copy())
        self.phases.append(phase)
        self.actual_joint_positions.append(
            np.asarray(self.env.sim.data.qpos[self.adapter.qpos_indexes], dtype=float).copy()
        )
        if self.capture_video and self.step_count % self.stride == 0:
            self.frames.append(
                render_pair(self.env, self.size, phase, self.regime, self.env._check_success())
            )
        self.step_count += 1

    def step_action(self, action, phase):
        """Record and execute one already-retimed absolute joint command."""
        action = np.asarray(action, dtype=float).copy()
        if self.actions:
            delta = action[:-1] - self.actions[-1][:-1]
            self.action_delta_norms.append(float(np.linalg.norm(delta)))
            if self.previous_delta is not None:
                self.action_jerk_norms.append(float(np.linalg.norm(delta - self.previous_delta)))
            self.previous_delta = delta
        self.env.step(action)
        self.actions.append(action)
        self.phases.append(phase)
        self.actual_joint_positions.append(
            np.asarray(self.env.sim.data.qpos[self.adapter.qpos_indexes], dtype=float).copy()
        )
        if self.capture_video and self.step_count % self.stride == 0:
            self.frames.append(
                render_pair(self.env, self.size, phase, self.regime, self.env._check_success())
            )
        self.step_count += 1

    def servo(self, target_pos, target_quat, gripper, steps, phase, global_ik=False):
        joint_target = None
        if global_ik:
            joint_target, _ = self.adapter.global_solution(self.env, target_pos, target_quat)
        for _ in range(steps):
            self.step(target_pos, target_quat, gripper, phase, joint_target=joint_target)

    def servo_until(
        self,
        target_pos,
        target_quat,
        gripper,
        phase,
        *,
        minimum_steps=8,
        maximum_steps=120,
        position_tolerance=0.004,
        orientation_tolerance=np.deg2rad(4.0),
        stable_steps=3,
        global_ik=False,
    ):
        """Track a Cartesian waypoint to a measured, consecutive-step gate."""
        joint_target = None
        if global_ik:
            joint_target, _ = self.adapter.global_solution(self.env, target_pos, target_quat)
        stable = 0
        final_position_error = float("inf")
        final_orientation_error = float("inf")
        for index in range(maximum_steps):
            self.step(target_pos, target_quat, gripper, phase, joint_target=joint_target)
            current_pos, current_quat = eef_pose(self.env)
            final_position_error = float(np.linalg.norm(np.asarray(target_pos) - current_pos))
            final_orientation_error = float(
                np.linalg.norm(shortest_axisangle(target_quat, current_quat))
            )
            if (
                index + 1 >= minimum_steps
                and final_position_error <= position_tolerance
                and final_orientation_error <= orientation_tolerance
            ):
                stable += 1
                if stable >= stable_steps:
                    return {
                        "steps": index + 1,
                        "position_error_m": final_position_error,
                        "orientation_error_rad": final_orientation_error,
                    }
            else:
                stable = 0
        raise RuntimeError(
            f"waypoint {phase!r} did not converge in {maximum_steps} steps: "
            f"position_error={final_position_error:.6f} m, "
            f"orientation_error={np.rad2deg(final_orientation_error):.3f} deg"
        )

    def metrics(self):
        def summarize(values):
            values = np.asarray(values, dtype=float)
            return {
                "max": float(np.max(values)) if len(values) else 0.0,
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "p95": float(np.quantile(values, 0.95)) if len(values) else 0.0,
            }

        phase_counts = dict(Counter(self.phases))
        actual = np.asarray(self.actual_joint_positions, dtype=float)
        actual_delta = np.linalg.norm(np.diff(actual, axis=0), axis=1) if len(actual) > 1 else []
        actual_jerk = np.linalg.norm(np.diff(actual, n=2, axis=0), axis=1) if len(actual) > 2 else []
        return {
            "recorded_steps": len(self.actions),
            "phase_step_counts": phase_counts,
            "action_delta_norm": summarize(self.action_delta_norms),
            "action_jerk_norm": summarize(self.action_jerk_norms),
            "actual_joint_step_norm": summarize(actual_delta),
            "actual_joint_second_difference_norm": summarize(actual_jerk),
            "eef_position_error_m": summarize(self.eef_position_errors),
            "eef_orientation_error_rad": summarize(self.eef_orientation_errors),
        }


def wrist_line_of_sight(env, target):
    """Project a point into the wrist image and test first-hit occlusion."""
    camera_id = env.sim.model.camera_name2id("robot0_eye_in_hand")
    camera_position = np.asarray(env.sim.data.cam_xpos[camera_id], dtype=float).copy()
    camera_matrix = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    vector = np.asarray(target, dtype=float) - camera_position
    distance = float(np.linalg.norm(vector))
    camera_vector = camera_matrix.T.dot(vector)
    depth = float(-camera_vector[2])
    tan_half_fovy = float(np.tan(np.deg2rad(env.sim.model.cam_fovy[camera_id]) / 2.0))
    nx = float(camera_vector[0] / (depth * tan_half_fovy)) if depth > 0 else float("inf")
    ny = float(camera_vector[1] / (depth * tan_half_fovy)) if depth > 0 else float("inf")
    in_frame = bool(depth > 0 and abs(nx) <= 1.0 and abs(ny) <= 1.0)
    geom_id = np.array([-1], dtype=np.int32)
    hit_distance = float(
        mujoco.mj_ray(
            env.sim.model._model,
            env.sim.data._data,
            camera_position,
            unit(vector),
            np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8),
            1,
            -1,
            geom_id,
        )
    )
    # A collision hit even 1 mm before the center point blocks the direct
    # optical ray. The old 3 mm tolerance mislabeled shallow fork occlusion as
    # visible in faster-but-still-smooth replays.
    occluded = bool(0.0 <= hit_distance < distance - 0.001)
    hit_name = env.sim.model.geom_id2name(int(geom_id[0])) if geom_id[0] >= 0 else None
    return {
        "target_distance_m": distance,
        "first_hit_distance_m": hit_distance,
        "first_hit_geom": hit_name,
        "in_frame": in_frame,
        "normalized_image_xy": [nx, ny],
        "center_ray_occluded": occluded,
        "center_ray_visible": bool(in_frame and not occluded),
        "hidden_from_wrist": bool(not in_frame or occluded),
    }


def rack_hole_center(env):
    """Return the task-relevant center of the rack opening just above its rim."""
    rack_pos, rack_quat = env._body_pose(env.rack)
    rack_pose = T.pose2mat((rack_pos, rack_quat))
    # Use the centerline immediately above the 42.5 mm rim. This is the same
    # task-space height as the seated fork COM and avoids calling a rack-wall
    # grazing ray a visible opening.
    return rack_pose[:3, :3] @ np.array((0.0, 0.0, 0.063277)) + rack_pose[:3, 3]


def wrist_slot_visibility(env):
    """Measure center visibility and retain a grid only as a diagnostic."""
    rack_pos, rack_quat = env._body_pose(env.rack)
    rack_pose = T.pose2mat((rack_pos, rack_quat))
    center = wrist_line_of_sight(env, rack_hole_center(env))
    rays = []
    for local_x in (-0.006, 0.0, 0.006):
        for local_y in (-0.008, 0.0, 0.008):
            world = rack_pose[:3, :3] @ np.array((local_x, local_y, 0.063277)) + rack_pose[:3, 3]
            rays.append({
                "rack_local_xy_m": [local_x, local_y],
                **wrist_line_of_sight(env, world),
            })
    visible_count = sum(ray["center_ray_visible"] for ray in rays)
    return {
        "hole_center": center,
        "visible_rays": int(visible_count),
        "total_rays": len(rays),
        "visible_fraction": float(visible_count / len(rays)),
        "rays": rays,
    }


def initialize_table_fork(env, regime, reset_offset_xy):
    """Apply a conservative, replayable table-pose variation."""
    position = TABLE_FORK_CENTER.copy()
    position[:2] += np.asarray(reset_offset_xy, dtype=float)
    rotation = (
        Rotation.from_euler("y", 180.0, degrees=True)
        * Rotation.from_euler("x", FORK_FACE_ROLL_DEG[regime], degrees=True)
    )
    quat_xyzw = rotation.as_quat()
    env._set_free_object_pose(env.fork, position, quat_xyzw[[3, 0, 1, 2]])
    env.sim.data.set_joint_qvel(env.fork.joints[0], np.zeros(6))
    env.sim.forward()
    return position


def configure_high_friction_contacts(env):
    """Use stable table / finger contacts without changing visual geometry."""
    configure_panda_grasp(env)
    for geom_id, name in enumerate(env.sim.model.geom_names):
        if not name:
            continue
        if name.startswith("fork_") and (
            "handle_collision" in name or "neck_collision" in name
        ):
            env.sim.model.geom_friction[geom_id] = (4.0, 0.08, 0.02)
            env.sim.model.geom_condim[geom_id] = 6
        elif "finger" in name or "pad" in name:
            env.sim.model.geom_friction[geom_id] = (8.0, 0.08, 0.02)
            env.sim.model.geom_condim[geom_id] = 6
    for actuator_id, name in enumerate(env.sim.model.actuator_names):
        if name and "gripper" in name:
            env.sim.model.actuator_forcerange[actuator_id] = (-100.0, 100.0)
            env.sim.model.actuator_forcelimited[actuator_id] = 1


def apply_scene_randomization(env, regime, variation):
    """Apply the paired fork / rack sample used by planning and recording."""
    center = (
        FULL_FORK_YAW_CENTER_DEG
        if regime == "full_visible"
        else PARTIAL_FORK_YAW_CENTER_DEG
    )
    fork_yaw_deg = center + float(variation["fork_yaw_jitter_deg"])
    fork_rotation = (
        Rotation.from_euler("z", fork_yaw_deg, degrees=True)
        * Rotation.from_euler("y", 180.0, degrees=True)
    )
    fork_quat_xyzw = fork_rotation.as_quat()
    fork_position = FORK_NOMINAL_POS.copy()
    fork_position[:2] += np.asarray(variation["fork_offset_xy_m"], dtype=float)
    env._set_free_object_pose(
        env.fork,
        fork_position,
        fork_quat_xyzw[[3, 0, 1, 2]],
    )
    env.sim.data.set_joint_qvel(env.fork.joints[0], np.zeros(6))

    rack_body_id = env.object_body_ids[env.rack.name]
    rack_position = RACK_NOMINAL_POS.copy()
    rack_position[:2] += np.asarray(variation["rack_offset_xy_m"], dtype=float)
    env.sim.model.body_pos[rack_body_id] = rack_position
    rack_quat_xyzw = Rotation.from_euler(
        "z", float(variation["rack_yaw_deg"]), degrees=True
    ).as_quat()
    env.sim.model.body_quat[rack_body_id] = rack_quat_xyzw[[3, 0, 1, 2]]
    env.sim.forward()
    return {
        "fork_position_m": fork_position.tolist(),
        "fork_yaw_deg": float(fork_yaw_deg),
        "rack_position_m": rack_position.tolist(),
        "rack_yaw_deg": float(variation["rack_yaw_deg"]),
    }


def refresh_collector_initial_state(env):
    """Make post-reset fork initialization the declared dataset frame zero."""
    env._current_task_instance_xml = env.sim.model.get_xml()
    env._current_task_instance_state = np.asarray(env.sim.get_state().flatten()).copy()


def object_space_servo(
    env,
    recorder,
    desired_pos,
    desired_quat,
    steps,
    phase,
    *,
    precision=False,
    correct_orientation=True,
    global_ik=True,
):
    fork_pos, fork_quat = env._body_pose(env.fork)
    current_eef_pos, current_eef_quat = eef_pose(env)
    target_eef_pos = current_eef_pos + (np.asarray(desired_pos) - fork_pos)
    if correct_orientation:
        delta_rotation = T.quat2mat(desired_quat) @ T.quat2mat(fork_quat).T
        target_eef_quat = T.mat2quat(delta_rotation @ T.quat2mat(current_eef_quat))
    else:
        # Once the fork is aligned over the 17 mm slot, preserve the measured
        # grasp attitude. Repeated orientation corrections against contact
        # torque the narrow handle out of a parallel-jaw grasp.
        target_eef_quat = current_eef_quat
    # In contact, tracking error is expected compliance rather than an IK
    # failure. Apply a bounded target for a fixed duration and gate measured
    # fork geometry after the complete alignment / insertion segment.
    recorder.servo(
        target_eef_pos,
        target_eef_quat,
        1.0,
        max(steps, 22 if precision else 30),
        phase,
        global_ik=global_ik,
    )


def require_grasp(env, phase):
    gripper = env.robots[0].gripper["right"]
    if not env._check_grasp(gripper, env.fork.contact_geoms):
        raise RuntimeError(f"fork slipped during {phase}")


def run_policy(env, recorder, grasp_local_x, grasp_yaw_deg, variation):
    """Run one common one-grasp policy and return preinsert visibility diagnostics."""
    fork_pos, fork_quat = env._body_pose(env.fork)
    fork_rotation = T.quat2mat(fork_quat)
    _, reset_eef_quat = eef_pose(env)
    reset_eef_rotation = T.quat2mat(reset_eef_quat)
    grasp_roll_deg = float(variation.get("grasp_roll_deg", 0.0))
    handle_roll = Rotation.from_rotvec(
        np.deg2rad(grasp_roll_deg) * fork_rotation[:, 0]
    ).as_matrix()
    initial_eef_rotation = (
        Rotation.from_euler("z", grasp_yaw_deg, degrees=True).as_matrix()
        @ handle_roll
        @ reset_eef_rotation
    )
    initial_eef_quat = T.mat2quat(initial_eef_rotation)
    grasp_local_y = float(variation.get("grasp_local_y_m", 0.0))
    grasp_local_z = float(variation.get("grasp_local_z_m", 0.002))
    grasp = fork_pos + fork_rotation @ np.array((grasp_local_x, grasp_local_y, grasp_local_z))
    untwist_end = np.array((0.18, -0.10, 0.95))
    precomputed_untwist_joint_end = None
    if variation.get("untwist_after_grasp", False):
        precomputed_untwist_joint_end, _ = recorder.adapter.global_solution(
            env,
            untwist_end,
            reset_eef_quat,
        )
        recorder.adapter.global_ik_rng = np.random.RandomState(0)
    recorder.hold_video(12, "table reset")

    if variation.get("already_at_grasp", False):
        pass
    elif variation.get("direct_grasp_ik", False):
        recorder.servo_until(
            grasp,
            initial_eef_quat,
            -1.0,
            "direct joint-space pregrasp",
            maximum_steps=220,
            position_tolerance=0.004,
            orientation_tolerance=np.deg2rad(15.0),
            global_ik=True,
        )
    else:
        recorder.servo_until(
            grasp + np.array((0.0, 0.0, 0.12)),
            initial_eef_quat,
            -1.0,
            "approach",
            maximum_steps=140,
            position_tolerance=0.035,
            global_ik=bool(variation.get("global_grasp_ik", False)),
        )
        recorder.servo_until(
            grasp,
            initial_eef_quat,
            -1.0,
            "descend",
            maximum_steps=120,
            position_tolerance=float(variation.get("grasp_position_tolerance_m", 0.004)),
            global_ik=bool(variation.get("global_grasp_ik", False)),
        )
    recorder.servo(grasp, initial_eef_quat, 1.0, 35, "close once")
    require_grasp(env, "initial grasp")

    if variation.get("untwist_after_grasp", False):
        # A reversed table grasp places the wrist camera on the opposite side
        # of the fork, but that wrist branch cannot lift vertically very far.
        # Clear the table first, then untwist in world space while rising. The
        # rigid gripper--fork transform (and therefore camera parallax) is
        # preserved throughout this single grasp.
        lift_start, lift_quat = eef_pose(env)
        recorder.servo_until(
            lift_start + np.array((0.0, 0.0, 0.028)),
            lift_quat,
            1.0,
            "edge clearance lift",
            maximum_steps=100,
            position_tolerance=0.006,
            orientation_tolerance=np.deg2rad(16.0),
            global_ik=True,
        )
        require_grasp(env, "edge clearance lift")
        untwist_start, _ = eef_pose(env)
        untwist_joint_start = recorder.adapter.current_qpos(env)
        untwist_joint_end = precomputed_untwist_joint_end
        untwist_steps = 360
        for index in range(untwist_steps):
            progress = (index + 1) / float(untwist_steps)
            smooth_progress = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            target_pos = untwist_start + smooth_progress * (untwist_end - untwist_start)
            joint_target = untwist_joint_start + smooth_progress * (
                untwist_joint_end - untwist_joint_start
            )
            recorder.step(
                target_pos,
                reset_eef_quat,
                1.0,
                "lift and untwist",
                joint_target=joint_target,
            )
            if (index + 1) % 20 == 0:
                require_grasp(env, f"lift and untwist step {index + 1}")
        initial_eef_rotation = reset_eef_rotation
        initial_eef_quat = reset_eef_quat

    early_vertical_rotation = bool(variation.get("early_vertical_rotation", False))
    if early_vertical_rotation:
        clearance_pos, clearance_quat = eef_pose(env)
        recorder.servo(
            clearance_pos + np.array((0.0, 0.0, 0.022)),
            clearance_quat,
            1.0,
            80,
            "tail clearance lift",
        )
        require_grasp(env, "tail clearance lift")
        rotation_pos, rotation_quat = eef_pose(env)
        rotation_matrix = T.quat2mat(rotation_quat)
        rotation_end_pos = rotation_pos + np.array((0.0, 0.0, 0.030))
        rotation_end_quat = T.mat2quat(
            Rotation.from_euler("y", 90.0, degrees=True).as_matrix() @ rotation_matrix
        )
        rotation_joint_start = recorder.adapter.current_qpos(env)
        rotation_joint_end, _ = recorder.adapter.global_solution(
            env,
            rotation_end_pos,
            rotation_end_quat,
        )
        for index in range(160):
            progress = (index + 1) / 160.0
            smooth_progress = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            angle = 90.0 * smooth_progress
            target_pos = rotation_pos + np.array((0.0, 0.0, 0.030 * smooth_progress))
            target_quat = T.mat2quat(
                Rotation.from_euler("y", angle, degrees=True).as_matrix() @ rotation_matrix
            )
            joint_target = rotation_joint_start + smooth_progress * (
                rotation_joint_end - rotation_joint_start
            )
            recorder.step(
                target_pos,
                target_quat,
                1.0,
                "early upward rotation",
                joint_target=joint_target,
            )
            if (index + 1) % 10 == 0:
                require_grasp(env, f"early upward rotation {angle:.1f} deg")
    else:
        style = variation["motion_style"]
        lateral_y = variation.get("transfer_lateral_y_m")
        lateral = (
            np.array((0.0, float(lateral_y), 0.008))
            if lateral_y is not None
            else {
                "left_arc": np.array((0.0, 0.025, 0.008)),
                "right_arc": np.array((0.0, -0.025, 0.008)),
            }[style]
        )
        transfer_targets = (
            ()
            if variation.get("untwist_after_grasp", False)
            else (((0.18, -0.10, 0.95), "lift"),)
        ) + (
            (tuple(np.array((0.24, -0.05, 1.00)) + lateral), f"transfer {style}"),
            (tuple(ROTATION_PIVOT), "transfer 2/2"),
        )
        for target, phase in transfer_targets:
            recorder.servo_until(
                np.asarray(target),
                initial_eef_quat,
                1.0,
                phase,
                maximum_steps=120,
                position_tolerance=0.008,
                global_ik=bool(variation.get("global_transfer_ik", False)),
            )
            require_grasp(env, phase)

        _, fork_before_rotation_quat = env._body_pose(env.fork)
        _, fork_goal_rotation_quat = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
        rotation_eef_pos, rotation_eef_quat = eef_pose(env)
        if variation.get("fixed_quarter_turn", False):
            rotated_eef_quat = T.mat2quat(
                Rotation.from_euler("y", 90.0, degrees=True).as_matrix()
                @ T.quat2mat(rotation_eef_quat)
            )
        else:
            fork_to_goal_rotation = (
                T.quat2mat(fork_goal_rotation_quat)
                @ T.quat2mat(fork_before_rotation_quat).T
            )
            rotated_eef_quat = T.mat2quat(
                fork_to_goal_rotation @ T.quat2mat(rotation_eef_quat)
            )
        if variation.get("debug_rotation_target", False):
            print("ROTATION_TARGET", rotation_eef_pos.tolist(), rotated_eef_quat.tolist())
        rotation_slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_quat(np.vstack((rotation_eef_quat, rotated_eef_quat))),
        )
        use_local_rotation_ik = bool(variation.get("local_rotation_ik", False))
        rotation_joint_start = recorder.adapter.current_qpos(env)
        rotation_joint_end = None
        if not use_local_rotation_ik:
            rotation_joint_end, _ = recorder.adapter.global_solution(
                env,
                rotation_eef_pos,
                rotated_eef_quat,
            )
        rotation_steps = 100
        for index in range(rotation_steps):
            progress = (index + 1) / rotation_steps
            smooth_progress = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            target_quat = rotation_slerp([smooth_progress]).as_quat()[0]
            joint_target = None
            if rotation_joint_end is not None:
                joint_target = rotation_joint_start + smooth_progress * (
                    rotation_joint_end - rotation_joint_start
                )
            recorder.step(
                rotation_eef_pos,
                target_quat,
                1.0,
                "continuous goal rotation",
                joint_target=joint_target,
            )
            if (index + 1) % 10 == 0:
                require_grasp(env, f"goal rotation {100.0 * smooth_progress:.1f}%")
        recorder.servo(rotation_eef_pos, rotated_eef_quat, 1.0, 15, "rotation settle")

    goal_pos, goal_quat = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
    if early_vertical_rotation:
        fork_transfer_pos, fork_transfer_quat = env._body_pose(env.fork)
        eef_transfer_pos, eef_transfer_quat = eef_pose(env)
        desired_fork_pos = goal_pos + np.array((0.0, 0.0, 0.20))
        target_eef_pos = eef_transfer_pos + (desired_fork_pos - fork_transfer_pos)
        # Preserve the established grasp attitude during the long translation;
        # orientation is corrected only after the fork is above the rack.
        target_eef_quat = eef_transfer_quat
        transfer_slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_quat(np.vstack((eef_transfer_quat, target_eef_quat))),
        )
        vertical_transfer_steps = 480
        for index in range(vertical_transfer_steps):
            progress = (index + 1) / float(vertical_transfer_steps)
            smooth_progress = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            target_pos = eef_transfer_pos + smooth_progress * (
                target_eef_pos - eef_transfer_pos
            )
            target_quat = transfer_slerp([smooth_progress]).as_quat()[0]
            recorder.step(target_pos, target_quat, 1.0, "vertical transfer to rack")
            if (index + 1) % 20 == 0:
                require_grasp(env, f"vertical transfer step {index + 1}")
    visibility = None
    contact_preload_xy = np.asarray(
        variation.get("contact_preload_xy_m", (-0.004, 0.0)), dtype=float
    )
    observation_offset_xy = np.asarray(
        variation.get("observation_offset_xy_m", (0.0, 0.0)), dtype=float
    )
    observation_min_height = float(variation.get("observation_min_height_m", 0.12))
    observation_fade_end_height = float(
        variation.get("observation_fade_end_height_m", observation_min_height)
    )
    for offset, repeats in (
        (0.20, 1),
        (0.16, 2),
        (0.12, 6),
        (0.10, 3),
        (0.09, 2),
        (0.08, 2),
        (0.075, 5),
        (0.070, 1),
        (0.065, 1),
        (0.060, 1),
        (0.055, 1),
        (0.050, 1),
        (0.045, 1),
        (0.042, 1),
    ):
        if np.isclose(offset, 0.075) and np.any(observation_offset_xy):
            repeats = max(repeats, 7)
        ideal_desired_pos = goal_pos + np.array((0.0, 0.0, offset))
        # The held handle deflects about 2--3 mm toward +X under rack contact.
        # A fixed -X preload is shared by both regimes; acceptance is always
        # evaluated against the unbiased rack centerline below.
        contact_preload = (
            np.r_[contact_preload_xy, 0.0] if offset <= 0.075 else np.zeros(3)
        )
        # Canonicalize the EEF insertion path across grasp regimes. A grasp
        # farther down the now-vertical handle would otherwise perturb the IK
        # target and trigger a different Panda branch; the corresponding
        # millimetre-scale fork-height shift is removed by gravity on release.
        grasp_height_compensation = np.array(
            (0.0, 0.0, CANONICAL_GRASP_LOCAL_X - grasp_local_x)
        )
        if offset >= observation_min_height:
            observation_scale = 1.0
        elif offset > observation_fade_end_height:
            observation_scale = (
                (offset - observation_fade_end_height)
                / (observation_min_height - observation_fade_end_height)
            )
        else:
            observation_scale = 0.0
        observation_offset = np.r_[observation_scale * observation_offset_xy, 0.0]
        desired_pos = (
            ideal_desired_pos
            + contact_preload
            + observation_offset
            + grasp_height_compensation
        )
        for correction in range(repeats):
            object_space_servo(
                env,
                recorder,
                desired_pos,
                goal_quat,
                22,
                f"preinsert {offset:.3f} m {correction + 1}/{repeats}",
                precision=offset <= 0.075,
                correct_orientation=(
                    not early_vertical_rotation
                    and not bool(variation.get("preserve_contact_orientation", False))
                    and (
                        bool(variation.get("correct_orientation_in_contact", False))
                        or offset > 0.075
                        or (offset == 0.075 and correction == 0)
                    )
                ),
                global_ik=(
                    not bool(variation.get("local_insertion_ik", False))
                    and not np.any(observation_offset)
                ),
            )
        if np.isclose(offset, float(variation.get("visibility_offset_m", 0.055))):
            visibility = wrist_slot_visibility(env)
            visibility["fork_position_world_m"] = env._body_pose(env.fork)[0].tolist()
            visibility["ideal_fork_position_world_m"] = ideal_desired_pos.tolist()
            if variation.get("return_at_visibility", False):
                return visibility
        require_grasp(env, f"preinsert {offset:.3f} m")
        if (
            offset <= 0.12
            and not np.any(observation_offset)
            and not variation.get("return_at_visibility", False)
            and not variation.get("skip_alignment_gates", False)
        ):
            actual_pos, actual_quat = env._body_pose(env.fork)
            lateral_error = float(np.linalg.norm((actual_pos - ideal_desired_pos)[:2]))
            orientation_error = float(
                np.linalg.norm(
                    shortest_axisangle(goal_quat, actual_quat)
                )
            )
            lateral_limit = 0.005 if np.isclose(offset, 0.12) else 0.007
            orientation_limit = np.deg2rad(11.0)
            if lateral_error > lateral_limit or orientation_error > orientation_limit:
                raise RuntimeError(
                    f"precision alignment failed at {offset:.3f} m: "
                    f"lateral_error={lateral_error:.6f} m, "
                    f"orientation_error={np.rad2deg(orientation_error):.3f} deg, "
                    f"actual={actual_pos.tolist()}, ideal={ideal_desired_pos.tolist()}"
                )

    release_pos, release_quat = eef_pose(env)
    pre_release_error = env._fork_error()
    if pre_release_error[0] > 0.055 or pre_release_error[1] > np.deg2rad(8.0):
        actual_fork_pos, _ = env._body_pose(env.fork)
        raise RuntimeError(
            "pre-release fork pose is outside the mechanical seat gate: "
            f"position_error={pre_release_error[0]:.6f} m, "
            f"orientation_error={np.rad2deg(pre_release_error[1]):.3f} deg, "
            f"actual={actual_fork_pos.tolist()}, goal={goal_pos.tolist()}, "
            f"eef={release_pos.tolist()}"
        )
    recorder.servo(release_pos, release_quat, -1.0, 28, "release once")
    for _ in range(65):
        recorder.step(release_pos, release_quat, -1.0, "gravity seat")
    if not env._check_success() and not variation.get("allow_unseated", False):
        raise RuntimeError(f"fork did not seat after release: error={env._fork_error()}")

    retreat = release_pos + np.array((0.12, -0.08, 0.15))
    recorder.servo(retreat, release_quat, -1.0, 45, "retreat")
    recorder.hold_video(18, "complete")
    if not env._check_success() and not variation.get("allow_unseated", False):
        raise RuntimeError(f"fork left goal after retreat: error={env._fork_error()}")
    return visibility


def template_phase(regime, source_index):
    """Human-readable phases for the continuous Threading-style trajectory."""
    del regime
    boundaries = (
        (120, "continuous approach"),
        (155, "grasp and initial lift"),
        (455, "direct lift rotate transfer"),
        (485, "continuous alignment"),
        (725, "insert"),
        (785, "release and gravity seat"),
    )
    for boundary, name in boundaries:
        if source_index < boundary:
            return name
    return "retreat"


def minimum_jerk(progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def cubic_bezier(start, control1, control2, end, progress):
    return (
        (1.0 - progress) ** 3 * start
        + 3.0 * (1.0 - progress) ** 2 * progress * control1
        + 3.0 * (1.0 - progress) * progress**2 * control2
        + progress**3 * end
    )


def randomized_transfer_controls(variation, arc_start_pos, arc_end_pos):
    """Build one of Threading's ten path families with episode-level bends."""
    side = float(variation["side_sign"])
    presets = {
        "direct_low": ((0.000, 0.000, 0.145), (0.045, -0.035, 0.055)),
        "high_arc": ((0.000, 0.000, 0.190), (0.025, -0.035, 0.095)),
        "side_sweep": ((-0.005, 0.024 * side, 0.160), (0.035, -0.040 + 0.012 * side, 0.075)),
        "early_approach": ((-0.014, 0.010 * side, 0.140), (0.060, -0.026, 0.055)),
        "delayed_approach": ((0.005, 0.006 * side, 0.190), (0.020, -0.052, 0.090)),
        "low_s_curve": ((-0.005, 0.018 * side, 0.150), (0.052, -0.040 - 0.012 * side, 0.060)),
        "vertical_first": ((0.000, 0.000, 0.200), (0.030, -0.040, 0.100)),
        "shallow_sweep": ((0.000, 0.016 * side, 0.135), (0.058, -0.030, 0.045)),
        "over_then_back": ((-0.018, 0.012 * side, 0.175), (0.068, -0.050, 0.075)),
        "short_direct": ((0.000, 0.000, 0.150), (0.035, -0.035, 0.065)),
    }
    first, second = presets[variation["motion_style"]]
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    variant = variation["style_variant"]
    if variant == "early_bend":
        first[1] += 0.008 * side
        second[1] -= 0.004 * side
    elif variant == "late_bend":
        first[1] += 0.003 * side
        second[1] += 0.010 * side
    elif variant == "wide_bend":
        first[1] += 0.012 * side
        second[1] -= 0.012 * side
    elif variant == "soft_bend":
        first[1] *= 0.7
        second[1] *= 0.8
        first[2] += 0.005
    first += np.asarray(variation["control1_jitter_m"], dtype=float)
    second += np.asarray(variation["control2_jitter_m"], dtype=float)
    return arc_start_pos + first, arc_end_pos + second


def plan_randomized_joint_trajectory(env, regime, variation):
    """Plan in task space, then return only absolute joint-position actions.

    The 60 Hz OSC environment is a deterministic trajectory optimizer, not a
    data source. Its measured Panda joint path is sampled at 20 Hz and replayed
    through the same absolute JOINT_POSITION controller used by Threading.
    """
    arm = env.robots[0].arms[0]
    qpos_indexes = np.asarray(
        env.robots[0].composite_controller.part_controllers[arm].qpos_index,
        dtype=int,
    )
    joint_positions = []
    gripper_commands = []

    def step(target_pos, target_quat, gripper):
        env.step(osc_action(env, target_pos, target_quat, gripper))
        joint_positions.append(np.asarray(env.sim.data.qpos[qpos_indexes]).copy())
        gripper_commands.append(float(gripper))

    fork_pos, fork_quat = env._body_pose(env.fork)
    start_eef_pos, start_eef_quat = eef_pose(env)
    fork_yaw_deg = float(variation["fork_yaw_deg"])
    grasp_yaw_deg = GRASP_YAW_DEG[regime]
    grasp_local_x = float(variation["grasp_local_x_m"])
    grasp_pos = fork_pos + T.quat2mat(fork_quat) @ np.array(
        (grasp_local_x, 0.0, 0.0)
    )
    # The crosswise table pose requires a slightly deeper center height; high
    # friction prevents this low approach from sliding the flat fork.
    grasp_pos[2] = 0.812
    grasp_quat = T.mat2quat(
        Rotation.from_euler(
            "z", fork_yaw_deg + grasp_yaw_deg, degrees=True
        ).as_matrix()
        @ T.quat2mat(start_eef_quat)
    )

    # One continuous approach. Orientation finishes early, while position
    # reaches zero speed only at the physical grasp point.
    for index in range(120):
        progress = (index + 1) / 120.0
        path_t = minimum_jerk(min(1.0, progress / 0.92))
        target_pos = (
            (1.0 - path_t) * start_eef_pos
            + path_t * grasp_pos
            + 4.0
            * path_t
            * (1.0 - path_t)
            * np.array((0.0, 0.0, float(variation["approach_arc_height_m"])))
        )
        target_quat = T.quat_slerp(
            start_eef_quat,
            grasp_quat,
            minimum_jerk(
                min(1.0, progress / float(variation["orientation_finish_fraction"]))
            ),
        )
        step(target_pos, target_quat, -1.0)

    # Close while beginning the first 8 mm of lift. Only the first 0.29 s is
    # stationary so the pads can establish opposing contact.
    for index in range(35):
        progress = (index + 1) / 35.0
        lift_t = minimum_jerk(max(0.0, (progress - 0.50) / 0.50))
        step(grasp_pos + np.array((0.0, 0.0, 0.008 * lift_t)), grasp_quat, 1.0)
    require_grasp(env, "planned initial grasp")

    grasped_fork_pos, grasped_fork_quat = env._body_pose(env.fork)
    arc_start_pos, arc_start_quat = eef_pose(env)
    fork_to_eef = np.linalg.inv(
        T.pose2mat((grasped_fork_pos, grasped_fork_quat))
    ) @ T.pose2mat((arc_start_pos, arc_start_quat))
    goal_pos, goal_quat = env._goal_pose(env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)
    desired_fork_end = goal_pos + np.array((0.0, 0.0, 0.13))
    desired_fork_pose = T.pose2mat((desired_fork_end, goal_quat))
    desired_eef_pose = desired_fork_pose @ fork_to_eef
    arc_end_pos = desired_eef_pose[:3, 3]
    control1, control2 = randomized_transfer_controls(
        variation, arc_start_pos, arc_end_pos
    )
    arc_end_quat = T.mat2quat(desired_eef_pose[:3, :3])

    # Lift, rotate, and translate in one direct cubic arc. There is no distant
    # pivot and no zero-velocity intermediate waypoint.
    first_arc_grasp_loss = None
    first_arc_grasp_loss_detail = None
    for index in range(300):
        progress = (index + 1) / 300.0
        path_t = minimum_jerk(progress ** float(variation["arc_progress_power"]))
        target_pos = cubic_bezier(
            arc_start_pos, control1, control2, arc_end_pos, path_t
        )
        # Establish vertical clearance before rotating the off-center utensil;
        # finish rotation before contact alignment, without a hold at either end.
        rotation_t = minimum_jerk(
            np.clip(
                (progress - float(variation["rotation_delay"]))
                / float(variation["rotation_span"]),
                0.0,
                1.0,
            )
        )
        target_quat = T.quat_slerp(arc_start_quat, arc_end_quat, rotation_t)
        step(target_pos, target_quat, 1.0)
        if first_arc_grasp_loss is None:
            gripper = env.robots[0].gripper["right"]
            if not env._check_grasp(gripper, env.fork.contact_geoms):
                first_arc_grasp_loss = index
                current_fork_pos, current_fork_quat = env._body_pose(env.fork)
                current_eef_pos, current_eef_quat = eef_pose(env)
                current_relative = np.linalg.inv(
                    T.pose2mat((current_fork_pos, current_fork_quat))
                ) @ T.pose2mat((current_eef_pos, current_eef_quat))
                relative_delta = np.linalg.inv(fork_to_eef) @ current_relative
                first_arc_grasp_loss_detail = (
                    float(np.linalg.norm(relative_delta[:3, 3])),
                    float(Rotation.from_matrix(relative_delta[:3, :3]).magnitude()),
                )
    gripper = env.robots[0].gripper["right"]
    if not env._check_grasp(gripper, env.fork.contact_geoms):
        current_fork_pos, current_fork_quat = env._body_pose(env.fork)
        current_eef_pos, current_eef_quat = eef_pose(env)
        current_relative = np.linalg.inv(
            T.pose2mat((current_fork_pos, current_fork_quat))
        ) @ T.pose2mat((current_eef_pos, current_eef_quat))
        relative_delta = np.linalg.inv(fork_to_eef) @ current_relative
        final_arc_drift = (
            float(np.linalg.norm(relative_delta[:3, 3])),
            float(Rotation.from_matrix(relative_delta[:3, :3]).magnitude()),
        )
        if final_arc_drift[0] > 0.006 or final_arc_drift[1] > np.deg2rad(5.0):
            raise RuntimeError(
                "fork slipped during direct lift-rotate-transfer arc "
                f"at planner step {first_arc_grasp_loss + 1}/300; "
                f"first drift={first_arc_grasp_loss_detail}; final drift={final_arc_drift}"
            )

    rack_rotation = T.quat2mat(env._body_pose(env.rack)[1])

    def insertion_step(offset, progress, curve=True):
        lateral = 0.0
        if curve:
            lateral = float(variation["insertion_lateral_curve_m"]) * 4.0 * progress * (1.0 - progress)
        desired_fork_pos = goal_pos + rack_rotation @ np.array((0.003, lateral, offset))
        desired_pose = T.pose2mat((desired_fork_pos, goal_quat)) @ fork_to_eef
        actual_fork_pos, _ = env._body_pose(env.fork)
        fork_position_error = desired_fork_pos - actual_fork_pos
        step(
            desired_pose[:3, 3] + 0.8 * fork_position_error,
            T.mat2quat(desired_pose[:3, :3]),
            1.0,
        )

    # A short moving alignment lead-in replaces the old two-second hold.
    for index in range(30):
        progress = minimum_jerk((index + 1) / 30.0)
        insertion_step(0.13 + (0.115 - 0.13) * progress, progress, curve=False)
    planner_visibility = wrist_slot_visibility(env)
    for index in range(240):
        raw_progress = (index + 1) / 240.0
        progress = minimum_jerk(
            raw_progress ** float(variation["insertion_progress_power"])
        )
        insertion_step(0.115 + (0.025 - 0.115) * progress, progress)
    insertion_fork_pos, insertion_fork_quat = env._body_pose(env.fork)
    insertion_eef_pos, insertion_eef_quat = eef_pose(env)
    insertion_relative = np.linalg.inv(
        T.pose2mat((insertion_fork_pos, insertion_fork_quat))
    ) @ T.pose2mat((insertion_eef_pos, insertion_eef_quat))
    insertion_delta = np.linalg.inv(fork_to_eef) @ insertion_relative
    insertion_drift = (
        float(np.linalg.norm(insertion_delta[:3, 3])),
        float(Rotation.from_matrix(insertion_delta[:3, :3]).magnitude()),
    )
    if insertion_drift[0] > 0.008 or insertion_drift[1] > np.deg2rad(7.0):
        raise RuntimeError(f"fork slipped during planned insertion: drift={insertion_drift}")

    release_pos, release_quat = eef_pose(env)
    for _ in range(30):
        step(release_pos, release_quat, -1.0)
    for _ in range(30):
        step(release_pos, release_quat, -1.0)
    retreat_pos = release_pos + np.array((0.12, -0.08, 0.15))
    for index in range(90):
        progress = minimum_jerk((index + 1) / 90.0)
        step(
            (1.0 - progress) * release_pos + progress * retreat_pos,
            release_quat,
            -1.0,
        )
    if not env._check_success():
        raise RuntimeError(f"planner did not seat fork: error={env._fork_error()}")

    raw_actions = np.c_[np.asarray(joint_positions), np.asarray(gripper_commands)]
    sample_indices = np.unique(
        np.r_[np.arange(0, len(raw_actions), PLANNER_FREQUENCY // CONTROL_FREQUENCY), len(raw_actions) - 1]
    )
    return raw_actions[sample_indices], sample_indices.astype(float), {
        "planner_control_frequency_hz": PLANNER_FREQUENCY,
        "raw_planner_steps": int(len(raw_actions)),
        "joint_replay_steps": int(len(sample_indices)),
        "visibility": planner_visibility,
        "planner_final_pose_error": list(env._fork_error()),
    }


def run_generated_joint_policy(env, recorder, regime, actions, source_indices):
    """Replay an episode-specific absolute-joint plan and apply hard gates."""
    recorder.hold_video(3, "table reset")
    visibility = None
    grasp_confirmed = False
    for action, source_index in zip(actions, source_indices):
        recorder.step_action(action, template_phase(regime, source_index))
        if 120 <= source_index <= 665:
            grasp_confirmed |= bool(
                env._check_grasp(
                    env.robots[0].gripper["right"], env.fork.contact_geoms
                )
            )
        if visibility is None and source_index >= 485:
            visibility = wrist_slot_visibility(env)
    recorder.hold_video(4, "complete")
    if not grasp_confirmed:
        raise RuntimeError("generated plan never established the single fork grasp")
    if not env._check_success():
        raise RuntimeError(f"generated joint plan did not seat fork: error={env._fork_error()}")
    return visibility


def retimed_template(regime, variation):
    """Load the validated 20 Hz path and add a C1 free-space variation."""
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"missing YCB fork trajectory asset: {TEMPLATE_PATH}")
    with np.load(TEMPLATE_PATH) as data:
        actions = np.asarray(data[f"{regime}_actions"], dtype=float)
        source = np.asarray(data[f"{regime}_source_indices"], dtype=float)
    actions = actions.copy()
    if variation["motion_style"] == "offset_arc":
        # A compact C1 bump changes the free-space transfer path, then returns
        # exactly to the calibrated trajectory before the 90-degree rotation.
        # Contact alignment and the full/partial observation geometry remain
        # invariant across styles.
        begin, end = 200.0, 430.0
        phase = np.clip((source - begin) / (end - begin), 0.0, 1.0)
        bump = np.sin(np.pi * phase) ** 2
        actions[:, 0] += 0.018 * bump
        actions[:, 2] -= 0.010 * bump
    return actions, source


def run_template_policy(env, recorder, regime, variation):
    """Replay one short, validated absolute-joint trajectory."""
    actions, source_indices = retimed_template(regime, variation)
    recorder.hold_video(6, "table reset")
    visibility = None
    grasp_confirmed = False
    for action, source_index in zip(actions, source_indices):
        recorder.step_action(action, template_phase(regime, source_index))
        if 210 <= source_index <= 700:
            grasp_confirmed |= bool(
                env._check_grasp(env.robots[0].gripper["right"], env.fork.contact_geoms)
            )
        # Evaluate both regimes at the identical 130 mm alignment pose. The
        # label therefore depends only on the opposed grasp / camera frame.
        visibility_index = 595
        if visibility is None and source_index >= visibility_index:
            visibility = wrist_slot_visibility(env)
    recorder.hold_video(8, "complete")
    if not grasp_confirmed:
        raise RuntimeError("template never established the single fork grasp")
    if not env._check_success():
        raise RuntimeError(f"retimed template did not seat fork: error={env._fork_error()}")
    return visibility


def flush_episode(env, accepted, stats, keep_failed):
    # A planner can fail before the recording environment takes its first
    # action. In that case ep_directory still names the preceding episode;
    # never overwrite or delete that accepted demonstration.
    if not env.has_interaction:
        return None
    episode_directory = env.ep_directory
    env._flush()
    env.has_interaction = False
    if episode_directory and os.path.isdir(episode_directory):
        Path(episode_directory, "policy_stats.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        if not accepted and not keep_failed:
            shutil.rmtree(episode_directory)
            return None
    return Path(episode_directory) if episode_directory else None


def save_video(recorder, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        pixelformat="yuv420p",
    ) as writer:
        for frame in recorder.frames:
            writer.append_data(frame)


def make_variation_plan(seed, count, enabled=True):
    """Return a balanced plan whose diversity does not depend on lucky RNG draws."""
    rng = np.random.RandomState(seed)
    if not enabled:
        return [
            {
                "enabled": False,
                "variation_id": 0,
                "motion_style": "direct_low",
                "style_variant": "plain",
                "side_sign": 1.0,
                "fork_offset_xy_m": [0.0, 0.0],
                "fork_yaw_jitter_deg": 0.0,
                "rack_offset_xy_m": [0.0, 0.0],
                "rack_yaw_deg": 0.0,
                "grasp_local_x_m": TEMPLATE_GRASP_LOCAL_X,
                "control1_jitter_m": [0.0, 0.0, 0.0],
                "control2_jitter_m": [0.0, 0.0, 0.0],
                "arc_progress_power": 1.0,
                "rotation_delay": 0.25,
                "rotation_span": 0.65,
                "approach_arc_height_m": 0.055,
                "orientation_finish_fraction": 0.65,
                "insertion_lateral_curve_m": 0.0,
                "insertion_progress_power": 1.0,
            }
            for _ in range(count)
        ]
    # As in Threading, cycle through every coarse path family before reuse;
    # variants, side, control points, speed profile, and insertion curve are
    # sampled per episode, so repeated styles are not repeated trajectories.
    styles = list(MOTION_STYLES)
    return [
        {
            "enabled": True,
            "variation_id": index,
            "motion_style": styles[index % len(styles)],
            # Latin-style ordering: even a ten-demo collection sees all ten
            # coarse styles, all five bend variants, and both sides. Across
            # 100 demos every style x variant x side combination appears once.
            "style_variant": STYLE_VARIANTS[
                ((index % len(styles)) + (index // len(styles)))
                % len(STYLE_VARIANTS)
            ],
            "side_sign": float(
                -1.0
                if ((index % len(styles)) + (index // len(styles))) % 2 == 0
                else 1.0
            ),
            "fork_offset_xy_m": rng.uniform(
                (-0.002, -0.002), (0.002, 0.002)
            ).tolist(),
            "fork_yaw_jitter_deg": float(rng.uniform(-3.0, 3.0)),
            "rack_offset_xy_m": rng.uniform(
                (-0.002, -0.002), (0.002, 0.002)
            ).tolist(),
            "rack_yaw_deg": float(rng.uniform(-1.0, 1.0)),
            "grasp_local_x_m": float(rng.uniform(*COMMON_GRASP_RANGE)),
            "control1_jitter_m": rng.uniform(
                (-0.004, -0.004, -0.006), (0.004, 0.004, 0.006)
            ).tolist(),
            "control2_jitter_m": rng.uniform(
                (-0.004, -0.004, -0.006), (0.004, 0.004, 0.006)
            ).tolist(),
            "arc_progress_power": float(rng.uniform(0.90, 1.10)),
            "rotation_delay": float(rng.uniform(0.22, 0.30)),
            "rotation_span": float(rng.uniform(0.60, 0.68)),
            "approach_arc_height_m": float(rng.uniform(0.050, 0.064)),
            "orientation_finish_fraction": float(rng.uniform(0.58, 0.72)),
            "insertion_lateral_curve_m": float(rng.uniform(-0.0012, 0.0012)),
            "insertion_progress_power": float(rng.uniform(0.92, 1.08)),
        }
        for index in range(count)
    ]


def collect(args):
    requested_regimes = list(REGIMES) if args.regime == "both" else [args.regime]
    targets = {regime: args.num_demos_per_regime for regime in requested_regimes}
    accepted = {regime: 0 for regime in requested_regimes}
    attempts = 0
    videos_written = 0
    variation_plan = make_variation_plan(
        args.seed,
        args.num_demos_per_regime,
        enabled=not args.no_variation,
    )
    accepted_variations = {regime: [] for regime in requested_regimes}
    slot_retries = {}
    args.directory.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    planner_env = suite.make(
        "YCBForkInRack",
        robots="Panda",
        controller_configs=make_osc_config("Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        reward_shaping=True,
        initialization_noise=None,
        control_freq=PLANNER_FREQUENCY,
        horizon=1200,
        ignore_done=True,
        seed=args.seed,
    )
    base_env = suite.make(
        "YCBForkInRack",
        robots="Panda",
        controller_configs=make_joint_position_config("Panda"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        hard_reset=False,
        reward_shaping=True,
        initialization_noise=None,
        control_freq=CONTROL_FREQUENCY,
        horizon=args.horizon,
        ignore_done=True,
        seed=args.seed,
    )
    env = DataCollectionWrapper(
        base_env,
        str(args.directory),
        collect_freq=1,
        flush_freq=args.horizon + 1,
        record_joint_position_fields=True,
        joint_delta_scale=JOINT_DELTA_SCALE,
        joint_position_label_source="sim_qpos",
        reload_from_xml_on_episode_start=False,
    )
    manifest_path = args.directory / "manifest.jsonl"
    try:
        while any(accepted[name] < targets[name] for name in requested_regimes):
            if attempts >= args.max_attempts:
                raise RuntimeError(f"collection exhausted {args.max_attempts} attempts: {accepted}")
            regime = next(name for name in requested_regimes if accepted[name] < targets[name])
            variation_index = accepted[regime]
            slot_key = (regime, variation_index)
            retry_index = slot_retries.get(slot_key, 0)
            if retry_index == 0:
                variation = dict(variation_plan[variation_index])
            else:
                retry_plan = make_variation_plan(
                    args.seed + 1009 * retry_index,
                    args.num_demos_per_regime,
                    enabled=not args.no_variation,
                )
                variation = dict(retry_plan[variation_index])
                variation["retry_index"] = retry_index
            attempts += 1
            grasp_local_x = (
                float(args.grasp_offset_override)
                if args.grasp_offset_override is not None
                else float(variation["grasp_local_x_m"])
            )
            variation["grasp_local_x_m"] = grasp_local_x
            fork_yaw_center = (
                FULL_FORK_YAW_CENTER_DEG
                if regime == "full_visible"
                else PARTIAL_FORK_YAW_CENTER_DEG
            )
            variation["fork_yaw_deg"] = (
                fork_yaw_center + float(variation["fork_yaw_jitter_deg"])
            )
            grasp_yaw_deg = GRASP_YAW_DEG[regime]

            planner_error = None
            planner_stats = None
            planned_actions = None
            planned_source = None
            try:
                planner_env.reset()
                configure_high_friction_contacts(planner_env)
                apply_scene_randomization(planner_env, regime, variation)
                planned_actions, planned_source, planner_stats = (
                    plan_randomized_joint_trajectory(
                        planner_env,
                        regime,
                        variation,
                    )
                )
            except Exception as exc:
                planner_error = f"{type(exc).__name__}: {exc}"

            env.reset()
            configure_high_friction_contacts(env)
            scene = apply_scene_randomization(env, regime, variation)
            refresh_collector_initial_state(env)
            recorder = EpisodeRecorder(
                env,
                JointPositionPoseAdapter(env),
                regime,
                args.size,
                args.video_fps,
                capture_video=(videos_written < args.video_count or args.keep_failed),
            )
            error = planner_error
            visibility = None
            try:
                if error is not None:
                    raise RuntimeError(error)
                visibility = run_generated_joint_policy(
                    env,
                    recorder,
                    regime,
                    planned_actions,
                    planned_source,
                )
                success = bool(env._check_success())
            except Exception as exc:  # preserve diagnostics for rejected physical attempts
                error = f"{type(exc).__name__}: {exc}"
                success = False

            final_pos_error, final_rot_error = env._fork_error()
            visibility_contract = bool(
                visibility is not None
                and (
                    visibility["visible_rays"] == visibility["total_rays"]
                    if regime == "full_visible"
                    else visibility["visible_rays"] == 0
                )
            )
            quality = recorder.metrics()
            pose_contract = bool(
                final_pos_error <= 0.005
                and final_rot_error <= np.deg2rad(2.0)
            )
            # Match Threading's production action gates (0.18 delta / 0.095
            # jerk / 0.04 mean delta), and add measured-joint safety gates.
            smoothness_contract = bool(
                quality["action_delta_norm"]["max"] <= 0.18
                and quality["action_jerk_norm"]["max"] <= 0.095
                and quality["action_delta_norm"]["mean"] <= 0.040
                and quality["actual_joint_step_norm"]["max"] <= 0.12
                and quality["actual_joint_second_difference_norm"]["max"] <= 0.050
            )
            if success and error is None and not visibility_contract:
                error = f"{regime} wrist-visibility contract failed"
            if success and error is None and not smoothness_contract:
                error = f"{regime} Threading/ToolHang smoothness contract failed"
            if success and error is None and not pose_contract:
                error = (
                    f"{regime} final-pose quality contract failed: "
                    f"position={final_pos_error:.6f}, rotation={final_rot_error:.6f}"
                )
            accepted_episode = bool(
                success
                and error is None
                and visibility_contract
                and smoothness_contract
                and pose_contract
            )
            stats = {
                "task": "YCBForkInRack",
                "robot": "Panda",
                "controller": controller_metadata(),
                "policy": "threading_quality_continuous_joint_position_single_grasp",
                "regime": regime,
                "regime_definition": {
                    "assignment_source": "opposed_grasp_frame_at_identical_alignment_pose",
                    "full_visible_grasp_range_local_x_m": list(FULL_VISIBLE_GRASP_RANGE),
                    "partial_hidden_grasp_range_local_x_m": list(PARTIAL_HIDDEN_GRASP_RANGE),
                    "full_visible_grasp_yaw_deg": GRASP_YAW_DEG["full_visible"],
                    "partial_hidden_grasp_yaw_deg": GRASP_YAW_DEG["partial_hidden"],
                    "full_visible_grasp_local_y_m": GRASP_LOCAL_Y_M["full_visible"],
                    "partial_hidden_grasp_local_y_m": GRASP_LOCAL_Y_M["partial_hidden"],
                    "full_visible_fork_face_roll_deg": FORK_FACE_ROLL_DEG["full_visible"],
                    "partial_hidden_fork_face_roll_deg": FORK_FACE_ROLL_DEG["partial_hidden"],
                    "same_phase_structure_across_regimes": True,
                    "distinct_grasp_trajectory_per_regime": True,
                    "full_requires_complete_3x3_aperture_visibility": True,
                    "visibility_contract_is_acceptance_gate": True,
                    "calibration_status": {
                        "partial_hidden": "opposed neck grasp occludes all aperture samples at the shared 115 mm alignment pose",
                        "full_visible": "nominal neck grasp exposes all aperture samples at the shared 115 mm alignment pose",
                    },
                },
                "grasp_offset_local_x_m": grasp_local_x,
                "grasp_yaw_deg": grasp_yaw_deg,
                "grasp_offset_local_y_m": GRASP_LOCAL_Y_M[regime],
                "fork_face_roll_deg": FORK_FACE_ROLL_DEG[regime],
                "initial_fork_position_m": scene["fork_position_m"],
                "initialization": "fork_flat_crosswise_randomized_left_or_right",
                "variation": variation,
                "scene": scene,
                "planner": planner_stats,
                "recorded_transitions_are_action_driven": True,
                "grasp_count": 1,
                "release_count": 1,
                "preinsert_wrist_visibility": visibility,
                "visibility_contract_passed": visibility_contract,
                "smoothness_contract_passed": smoothness_contract,
                "pose_contract_passed": pose_contract,
                "success": success,
                "accepted": accepted_episode,
                "error": error,
                "final_pose_error": {
                    "position_m": float(final_pos_error),
                    "rotation_rad": float(final_rot_error),
                },
                "quality": quality,
                "seed": args.seed,
                "attempt": attempts,
            }
            episode_directory = flush_episode(env, accepted_episode, stats, args.keep_failed)
            if accepted_episode:
                episode_index = accepted[regime]
                accepted[regime] += 1
                accepted_variations[regime].append(variation)
                video_path = None
                if videos_written < args.video_count:
                    video_path = args.video_dir / f"YCBForkInRack_{regime}_{episode_index:03d}.mp4"
                    save_video(recorder, video_path, args.video_fps)
                    videos_written += 1
                record = {
                    "episode_directory": str(episode_directory),
                    "video": str(video_path) if video_path else None,
                    **stats,
                }
                with manifest_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({
                    "accepted": accepted,
                    "regime": regime,
                    "episode_directory": str(episode_directory),
                    "video": str(video_path) if video_path else None,
                    "final_pose_error": stats["final_pose_error"],
                    "preinsert_wrist_visibility": visibility,
                }, indent=2))
            else:
                slot_retries[slot_key] = retry_index + 1
                if args.keep_failed and recorder.frames:
                    failed_video = args.video_dir / f"rejected_{regime}_attempt_{attempts:03d}.mp4"
                    save_video(recorder, failed_video, args.video_fps)
                print(f"rejected attempt={attempts} regime={regime}: {error}")
    finally:
        env.close()
        planner_env.close()

    if not args.no_variation:
        for regime in requested_regimes:
            expected_unique = min(args.num_demos_per_regime, len(MOTION_STYLES))
            actual_unique = len(
                {item["motion_style"] for item in accepted_variations[regime]}
            )
            if actual_unique != expected_unique:
                raise RuntimeError(
                    f"variation coverage failed for {regime}: "
                    f"expected {expected_unique} unique styles, got {actual_unique}"
                )

    summary = {
        "task": "YCBForkInRack",
        "accepted": accepted,
        "attempts": attempts,
        "controller": controller_metadata(),
        "variation": {
            "enabled": not args.no_variation,
            "bounds": {
                "fork_offset_xy_m": [-0.002, 0.002],
                "fork_yaw_jitter_deg": [-3.0, 3.0],
                "rack_offset_xy_m": [-0.002, 0.002],
                "rack_yaw_deg": [-1.0, 1.0],
                "grasp_local_x_m": list(COMMON_GRASP_RANGE),
                "control_point_jitter_m": [-0.004, 0.004],
                "arc_progress_power": [0.90, 1.10],
                "rotation_delay_fraction": [0.22, 0.30],
                "rotation_span_fraction": [0.60, 0.68],
                "approach_arc_height_m": [0.050, 0.064],
                "orientation_finish_fraction": [0.58, 0.72],
                "insertion_lateral_curve_m": [-0.0012, 0.0012],
                "insertion_progress_power": [0.92, 1.08],
            },
            "motion_styles": list(MOTION_STYLES),
            "style_variants": list(STYLE_VARIANTS),
            "side_signs": [-1.0, 1.0],
            "discrete_path_families_per_regime": (
                len(MOTION_STYLES) * len(STYLE_VARIANTS) * 2
            ),
            "plan": variation_plan,
            "accepted_by_regime": accepted_variations,
            "guarantee": (
                "Threading-style balanced 10 motion styles x 5 bend variants x 2 side signs, with independently sampled control points, speed/rotation profiles, approach geometry, and insertion curvature; every path rejoins the shared 115 mm visibility checkpoint"
                if not args.no_variation
                else "disabled by command line"
            ),
        },
        "regimes": {
            "full_visible": {
                "grasp_range_local_x_m": list(FULL_VISIBLE_GRASP_RANGE),
                "grasp_yaw_deg": GRASP_YAW_DEG["full_visible"],
                "fork_face_roll_deg": FORK_FACE_ROLL_DEG["full_visible"],
                "grasp_local_y_m": GRASP_LOCAL_Y_M["full_visible"],
                "status": "validated_shared_alignment_aperture_visible_9_of_9",
            },
            "partial_hidden": {
                "grasp_range_local_x_m": list(PARTIAL_HIDDEN_GRASP_RANGE),
                "grasp_yaw_deg": GRASP_YAW_DEG["partial_hidden"],
                "fork_face_roll_deg": FORK_FACE_ROLL_DEG["partial_hidden"],
                "grasp_local_y_m": GRASP_LOCAL_Y_M["partial_hidden"],
                "status": "validated_shared_alignment_aperture_hidden_0_of_9",
            },
        },
        "manifest": str(manifest_path),
    }
    (args.directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("output/ycb_fork_joint_dataset"))
    parser.add_argument("--video-dir", type=Path, default=Path("output/ycb_fork_joint_videos"))
    parser.add_argument("--regime", choices=(*REGIMES, "both"), default="both")
    parser.add_argument("--num-demos-per-regime", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--grasp-offset-override",
        type=float,
        help="Debug one fixed fork-local handle coordinate instead of sampling the regime range.",
    )
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Disable fork, rack, grasp-point, and motion-style randomization.",
    )
    args = parser.parse_args(argv)
    if args.num_demos_per_regime <= 0 or args.max_attempts <= 0:
        parser.error("demo and attempt counts must be positive")
    args.directory = args.directory.resolve()
    args.video_dir = args.video_dir.resolve()
    return args


def main():
    summary = collect(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
