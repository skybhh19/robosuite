"""Collect scripted demonstrations for the Threading task.

Example:
    $ python robosuite/scripts/collect_threading_scripted.py --num-demos 1 --render
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.models.objects.composite.needle import NEEDLE_SHAFT_HALF_LENGTH
from robosuite.wrappers import DataCollectionWrapper


POSITION_SCALE = np.array([0.05, 0.05, 0.05])
ORIENTATION_SCALE = np.array([0.5, 0.5, 0.5])
MOTION_STYLES = [
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
]
STYLE_VARIANTS = ["plain", "early_bend", "late_bend", "wide_bend", "soft_bend"]


def unit(vec, fallback=None):
    """Return a unit vector, or fallback if the norm is too small."""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return np.array(vec, dtype=float) / norm


def rotation_between(source, target):
    """Return a rotation matrix that maps source direction to target direction."""
    source = unit(source)
    target = unit(target)
    cross = np.cross(source, target)
    dot = np.clip(np.dot(source, target), -1.0, 1.0)
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = unit(np.cross(source, np.array([0.0, 0.0, 1.0])), fallback=[1.0, 0.0, 0.0])
        return T.rotation_matrix(np.pi, axis)[:3, :3]
    return T.rotation_matrix(np.arccos(dot), cross)[:3, :3]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quadratic_bezier(start, control, end, t):
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end


def cubic_bezier(start, control1, control2, end, t):
    t = np.clip(t, 0.0, 1.0)
    return (
        (1.0 - t) ** 3 * start
        + 3.0 * (1.0 - t) ** 2 * t * control1
        + 3.0 * (1.0 - t) * t**2 * control2
        + t**3 * end
    )


def get_eef_pose(env):
    """Return the right-arm end-effector site pose."""
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    pos = np.array(env.sim.data.site_xpos[site_id])
    quat = T.mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3))
    return pos, quat


def geom_pose(env, name):
    """Return geom position and rotation matrix."""
    geom_id = env.sim.model.geom_name2id(name)
    pos = np.array(env.sim.data.geom_xpos[geom_id])
    mat = np.array(env.sim.data.geom_xmat[geom_id]).reshape(3, 3)
    return pos, mat


def active_needle_shaft_half_length(env):
    """Return the shaft half-length configured by the active environment."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    needle = getattr(base_env, "needle", None)
    return float(getattr(needle, "shaft_half_length", NEEDLE_SHAFT_HALF_LENGTH))


def needle_state(env):
    """Return needle center, handle center, and local axes in world coordinates."""
    needle_center, needle_mat = geom_pose(env, "needle_obj_needle")
    handle_center, _ = geom_pose(env, "needle_obj_handle")
    yaxis = unit(needle_mat[:, 1])
    shaft_half_length = active_needle_shaft_half_length(env)
    return {
        "needle_center": needle_center,
        "handle_center": handle_center,
        "xaxis": unit(needle_mat[:, 0]),
        "yaxis": yaxis,
        "zaxis": unit(needle_mat[:, 2], fallback=[0.0, 0.0, 1.0]),
        "shaft_half_length": shaft_half_length,
        "tip": needle_center - shaft_half_length * yaxis,
        "handle_side": needle_center + shaft_half_length * yaxis,
    }


def ring_state(env):
    """Return ring center and opening normal."""
    ring_pos = np.zeros(3)
    ring_mat = None
    for i in range(env.tripod.num_ring_geoms):
        pos, mat = geom_pose(env, f"tripod_obj_ring_{i}")
        ring_pos += pos
        if ring_mat is None:
            ring_mat = mat
    ring_pos /= env.tripod.num_ring_geoms
    normal = unit(ring_mat[:, 0], fallback=[1.0, 0.0, 0.0])
    needle_center, _ = geom_pose(env, "needle_obj_needle")
    if np.dot(normal, ring_pos - needle_center) < 0:
        normal = -normal
    normal[2] = 0.0
    normal = unit(normal, fallback=[1.0, 0.0, 0.0])
    return {"center": ring_pos, "normal": normal}


def shaft_ring_distance(needle, ring):
    """Return the closest distance from the ring center to the needle shaft segment."""
    rel = ring["center"] - needle["needle_center"]
    half_length = needle.get("shaft_half_length", NEEDLE_SHAFT_HALF_LENGTH)
    t = np.clip(np.dot(rel, needle["yaxis"]), -half_length, half_length)
    closest = needle["needle_center"] + t * needle["yaxis"]
    return float(np.linalg.norm(closest - ring["center"]))


def insertion_progress(needle, ring):
    """Signed progress of the needle tip through the ring plane."""
    return float(np.dot(needle["tip"] - ring["center"], ring["normal"]))


def tripod_position(env):
    return np.array(env.sim.data.body_xpos[env.obj_body_id["tripod"]])


def target_action(env, target_pos, target_quat, gripper, prev_action, noise_state, noise_std, rng):
    """Convert a target pose into a smooth normalized OSC_POSE action."""
    eef_pos, eef_quat = get_eef_pose(env)
    pos_cmd = np.clip((target_pos - eef_pos) / POSITION_SCALE, -0.7, 0.7)

    quat_err = T.quat_distance(target_quat.copy(), eef_quat.copy())
    ori_err = T.quat2axisangle(quat_err)
    ori_cmd = np.clip(ori_err / ORIENTATION_SCALE, -0.45, 0.45)

    action = np.zeros(env.action_spec[0].shape)
    action[:3] = pos_cmd
    action[3:6] = ori_cmd
    action[-1] = gripper

    if noise_std > 0:
        noise_alpha = 0.96
        noise_state[:] = noise_alpha * noise_state + np.sqrt(1.0 - noise_alpha**2) * rng.normal(size=noise_state.shape)
        action[:-1] += noise_std * noise_state

    if prev_action is not None:
        action[:-1] = 0.6 * prev_action[:-1] + 0.4 * action[:-1]
        max_delta = 0.14
        action[:-1] = np.clip(action[:-1], prev_action[:-1] - max_delta, prev_action[:-1] + max_delta)
        action[-1] = np.clip(action[-1], prev_action[-1] - 0.12, prev_action[-1] + 0.12)

    low, high = env.action_spec
    return np.clip(action, low, high)


def hold_pose_steps(env, target_pos, target_quat, gripper, steps, policy_state, render=False, max_fr=None):
    """Track one pose target for a fixed number of control steps."""
    success = False
    for _ in range(steps):
        start = time.time()
        action = target_action(
            env=env,
            target_pos=target_pos,
            target_quat=target_quat,
            gripper=gripper,
            prev_action=policy_state["prev_action"],
            noise_state=policy_state["noise_state"],
            noise_std=policy_state["noise_std"],
            rng=policy_state["rng"],
        )
        policy_state["prev_action"] = action
        if policy_state.get("last_action_for_metrics") is not None:
            delta = action[:6] - policy_state["last_action_for_metrics"][:6]
            policy_state["action_delta_norms"].append(float(np.linalg.norm(delta)))
            if policy_state.get("last_delta_for_metrics") is not None:
                jerk = delta - policy_state["last_delta_for_metrics"]
                policy_state["action_jerk_norms"].append(float(np.linalg.norm(jerk)))
            policy_state["last_delta_for_metrics"] = delta
        policy_state["last_action_for_metrics"] = action.copy()
        env.step(action)
        success = success or env._check_success()
        if render:
            env.render()
        if max_fr is not None:
            elapsed = time.time() - start
            if elapsed < 1.0 / max_fr:
                time.sleep(1.0 / max_fr - elapsed)
    return success


def finalize_episode(env, success, cleanup_failed=True, stats=None):
    """Flush the wrapper and optionally remove failed attempts."""
    ep_dir = env.ep_directory
    if env.has_interaction:
        env._flush()
        env.has_interaction = False
    if ep_dir and os.path.exists(ep_dir) and stats is not None:
        with open(os.path.join(ep_dir, "policy_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    if not success and cleanup_failed and ep_dir and os.path.exists(ep_dir):
        shutil.rmtree(ep_dir)
    return ep_dir


class ThreadingScriptedPolicy:
    """Closed-loop scripted policy for grasping, lifting, aligning, and threading the needle."""

    def __init__(self, rng, action_noise_std=0.01):
        self.rng = rng
        self.action_noise_std = action_noise_std
        self.stats = []

    def _new_policy_state(self, env, noise_std=None):
        return {
            "prev_action": None,
            "target_pos": None,
            "target_quat": None,
            "noise_state": np.zeros(env.action_spec[0].shape[0] - 1),
            "noise_std": self.action_noise_std if noise_std is None else noise_std,
            "rng": self.rng,
            "last_action_for_metrics": None,
            "last_delta_for_metrics": None,
            "action_delta_norms": [],
            "action_jerk_norms": [],
        }

    def _record_metrics(self, env, stats):
        needle = needle_state(env)
        ring = ring_state(env)
        stats["env_success"] = bool(stats["env_success"] or env._check_success())
        stats["min_ring_distance"] = min(stats["min_ring_distance"], shaft_ring_distance(needle, ring))
        stats["max_insert_progress"] = max(stats["max_insert_progress"], insertion_progress(needle, ring))
        stats["final_insert_progress"] = insertion_progress(needle, ring)
        stats["tripod_displacement"] = float(np.linalg.norm(tripod_position(env) - stats["initial_tripod_pos"]))

    def _advance_target(self, env, desired_pos, desired_quat, policy_state, subgoal):
        if policy_state["target_pos"] is None:
            current_pos, current_quat = get_eef_pose(env)
            policy_state["target_pos"] = current_pos
            policy_state["target_quat"] = current_quat

        max_pos_step = 0.0042
        max_angle_step = 0.035
        if subgoal == "insert_through":
            max_pos_step = 0.0038
            max_angle_step = 0.028

        pos_delta = desired_pos - policy_state["target_pos"]
        pos_dist = np.linalg.norm(pos_delta)
        if pos_dist > max_pos_step:
            policy_state["target_pos"] = policy_state["target_pos"] + max_pos_step * pos_delta / pos_dist
        else:
            policy_state["target_pos"] = np.array(desired_pos)

        quat_delta = T.quat_distance(desired_quat.copy(), policy_state["target_quat"].copy())
        angle = np.linalg.norm(T.quat2axisangle(quat_delta))
        if angle > max_angle_step:
            policy_state["target_quat"] = T.quat_slerp(policy_state["target_quat"], desired_quat, max_angle_step / angle)
        else:
            policy_state["target_quat"] = np.array(desired_quat)
        return policy_state["target_pos"], policy_state["target_quat"]

    def _pose_error(self, env, desired_pos, desired_quat):
        eef_pos, eef_quat = get_eef_pose(env)
        pos_err = float(np.linalg.norm(desired_pos - eef_pos))
        quat_err = T.quat_distance(desired_quat.copy(), eef_quat.copy())
        ori_err = float(np.linalg.norm(T.quat2axisangle(quat_err)))
        return pos_err, ori_err

    def _track_target(
        self,
        env,
        target_fn,
        gripper,
        steps,
        policy_state,
        stats,
        subgoal,
        render=False,
        max_fr=None,
        stop_on_reach=True,
        min_steps=12,
    ):
        start_t = int(env.t)
        for _ in range(steps):
            desired_pos, desired_quat = target_fn()
            target_pos, target_quat = self._advance_target(env, desired_pos, desired_quat, policy_state, subgoal)
            hold_pose_steps(env, target_pos, target_quat, gripper, 1, policy_state, render, max_fr)
            self._record_metrics(env, stats)
            elapsed = int(env.t) - start_t
            if stop_on_reach and elapsed >= min_steps:
                pos_err, ori_err = self._pose_error(env, desired_pos, desired_quat)
                if pos_err < 0.012 and ori_err < 0.12:
                    break
        stats["subgoal_durations"][subgoal] = stats["subgoal_durations"].get(subgoal, 0) + int(env.t) - start_t

    def _fixed_target(self, target_pos, target_quat):
        return lambda: (np.array(target_pos), np.array(target_quat))

    def _alignment_target(
        self,
        offset,
        lateral_offset=0.0,
        vertical_offset=0.0,
        twist=0.0,
        tilt=0.0,
        rotation_fraction=1.0,
    ):
        def target(env):
            current_needle = needle_state(env)
            current_ring = ring_state(env)
            side = unit(np.cross([0.0, 0.0, 1.0], current_ring["normal"]), fallback=[0.0, 1.0, 0.0])
            align_rot = rotation_between(-current_needle["yaxis"], current_ring["normal"])
            align_quat = T.mat2quat(align_rot)
            partial_align_quat = T.quat_slerp(T.mat2quat(np.eye(3)), align_quat, np.clip(rotation_fraction, 0.0, 1.0))
            partial_align_rot = T.quat2mat(partial_align_quat)
            wobble_rot = (
                T.rotation_matrix(twist, current_ring["normal"])[:3, :3].dot(
                    T.rotation_matrix(tilt, side)[:3, :3]
                )
            )
            current_eef_pos, current_eef_quat = get_eef_pose(env)
            target_mat = wobble_rot.dot(partial_align_rot).dot(T.quat2mat(current_eef_quat))
            target_quat = T.mat2quat(target_mat)
            desired_tip = (
                current_ring["center"]
                + offset * current_ring["normal"]
                + lateral_offset * side
                + np.array([0.0, 0.0, vertical_offset])
            )
            target_pos = current_eef_pos + (desired_tip - current_needle["tip"])
            return target_pos, target_quat

        return target

    def _curved_alignment_target(self, offset, progress, curve):
        path_t = smoothstep(progress)
        rot_t = smoothstep((progress - curve["rotation_delay"]) / curve["rotation_span"])
        lateral = quadratic_bezier(curve["lateral_start"], curve["lateral_control"], curve["lateral_end"], path_t)
        vertical = quadratic_bezier(curve["vertical_start"], curve["vertical_control"], curve["vertical_end"], path_t)
        twist = quadratic_bezier(curve["twist_start"], curve["twist_control"], curve["twist_end"], rot_t)
        tilt = quadratic_bezier(curve["tilt_start"], curve["tilt_control"], curve["tilt_end"], rot_t)
        return self._alignment_target(offset, lateral, vertical, twist, tilt, rot_t)

    def _policy_success(self, env, stats):
        needle = needle_state(env)
        ring = ring_state(env)
        stats["final_ring_distance"] = shaft_ring_distance(needle, ring)
        stats["final_insert_progress"] = insertion_progress(needle, ring)
        checks = {
            "env_success": bool(stats["env_success"] or env._check_success()),
            "ring_crossed": stats["min_ring_distance"] < 0.018,
            "inserted_past_ring": stats["max_insert_progress"] > 0.026,
            "final_still_inserted": stats["final_insert_progress"] > 0.014,
            "tripod_stable": stats["tripod_displacement"] < 0.035,
            "hold_complete": bool(stats["hold_complete"]),
            "gripper_closed": bool(stats["gripper_closed"]),
        }
        stats["policy_checks"] = checks
        failed = [name for name, passed in checks.items() if not passed]
        stats["failure_reason"] = "none" if not failed else ",".join(failed)
        return not failed

    def rollout(self, env, render=False, max_fr=None, motion_style=None):
        env.reset()
        _, eef_quat = get_eef_pose(env)
        base_eef_mat = T.quat2mat(eef_quat)
        needle = needle_state(env)
        ring = ring_state(env)

        grasp_angle = 0.0
        grasp_tilt_x = 0.0
        grasp_tilt_y = 0.0
        grasp_offset_along = self.rng.uniform(-0.009, 0.009)
        grasp_offset_lateral = self.rng.uniform(-0.0015, 0.0015)
        grasp_offset_vertical = self.rng.uniform(-0.0008, 0.0018)
        grasp_offset = (
            grasp_offset_along * needle["yaxis"]
            + grasp_offset_lateral * needle["xaxis"]
            + grasp_offset_vertical * needle["zaxis"]
        )
        grasp_pos = needle["handle_center"] + grasp_offset
        grasp_mat = base_eef_mat
        grasp_quat = T.mat2quat(grasp_mat)

        pregrasp_height = self.rng.uniform(0.09, 0.13)
        descend_height = self.rng.uniform(0.007, 0.017)
        close_height = max(0.005, descend_height - self.rng.uniform(0.002, 0.006))
        toward_ring = ring["center"] - grasp_pos
        toward_ring[2] = 0.0
        toward_ring = unit(toward_ring, fallback=ring["normal"])
        lift_side = unit(np.cross([0.0, 0.0, 1.0], toward_ring), fallback=needle["xaxis"])
        if motion_style is None:
            motion_style = self.rng.choice(MOTION_STYLES)
        style_variant = self.rng.choice(STYLE_VARIANTS)
        side_sign = self.rng.choice([-1.0, 1.0])
        if motion_style == "direct_low":
            lift_toward = self.rng.uniform(0.075, 0.13) * toward_ring
            lift_lateral = self.rng.uniform(-0.018, 0.018) * lift_side
            lift_height = self.rng.uniform(0.115, 0.165)
            control_toward = self.rng.uniform(0.45, 0.8)
            control_side = self.rng.uniform(-0.02, 0.02)
            control_z_frac = self.rng.uniform(0.35, 0.65)
            lift_progress_power = self.rng.uniform(0.75, 1.05)
        elif motion_style == "high_arc":
            lift_toward = self.rng.uniform(0.025, 0.07) * toward_ring
            lift_lateral = side_sign * self.rng.uniform(0.015, 0.055) * lift_side
            lift_height = self.rng.uniform(0.215, 0.29)
            control_toward = self.rng.uniform(0.05, 0.4)
            control_side = side_sign * self.rng.uniform(0.03, 0.075)
            control_z_frac = self.rng.uniform(0.85, 1.2)
            lift_progress_power = self.rng.uniform(1.0, 1.45)
        elif motion_style == "side_sweep":
            lift_toward = self.rng.uniform(0.04, 0.085) * toward_ring
            lift_lateral = side_sign * self.rng.uniform(0.055, 0.095) * lift_side
            lift_height = self.rng.uniform(0.155, 0.23)
            control_toward = self.rng.uniform(0.15, 0.55)
            control_side = -side_sign * self.rng.uniform(0.035, 0.075)
            control_z_frac = self.rng.uniform(0.35, 0.75)
            lift_progress_power = self.rng.uniform(0.85, 1.25)
        elif motion_style == "early_approach":
            lift_toward = self.rng.uniform(0.105, 0.16) * toward_ring
            lift_lateral = self.rng.uniform(-0.025, 0.025) * lift_side
            lift_height = self.rng.uniform(0.13, 0.2)
            control_toward = self.rng.uniform(0.75, 1.1)
            control_side = self.rng.uniform(-0.025, 0.025)
            control_z_frac = self.rng.uniform(0.25, 0.55)
            lift_progress_power = self.rng.uniform(0.65, 0.95)
        elif motion_style == "delayed_approach":
            lift_toward = self.rng.uniform(0.01, 0.045) * toward_ring
            lift_lateral = side_sign * self.rng.uniform(0.025, 0.07) * lift_side
            lift_height = self.rng.uniform(0.17, 0.255)
            control_toward = self.rng.uniform(-0.15, 0.25)
            control_side = side_sign * self.rng.uniform(0.02, 0.07)
            control_z_frac = self.rng.uniform(0.55, 0.95)
            lift_progress_power = self.rng.uniform(1.15, 1.65)
        elif motion_style == "low_s_curve":
            lift_toward = self.rng.uniform(0.08, 0.14) * toward_ring
            lift_lateral = side_sign * self.rng.uniform(0.025, 0.06) * lift_side
            lift_height = self.rng.uniform(0.12, 0.18)
            control_toward = self.rng.uniform(0.25, 0.55)
            control_side = -side_sign * self.rng.uniform(0.045, 0.085)
            control_z_frac = self.rng.uniform(0.3, 0.6)
            lift_progress_power = self.rng.uniform(0.7, 1.05)
        elif motion_style == "vertical_first":
            lift_toward = self.rng.uniform(0.045, 0.09) * toward_ring
            lift_lateral = self.rng.uniform(-0.018, 0.018) * lift_side
            lift_height = self.rng.uniform(0.2, 0.27)
            control_toward = self.rng.uniform(-0.05, 0.18)
            control_side = self.rng.uniform(-0.018, 0.018)
            control_z_frac = self.rng.uniform(0.8, 1.1)
            lift_progress_power = self.rng.uniform(1.05, 1.45)
        elif motion_style == "shallow_sweep":
            lift_toward = self.rng.uniform(0.095, 0.17) * toward_ring
            lift_lateral = side_sign * self.rng.uniform(0.04, 0.08) * lift_side
            lift_height = self.rng.uniform(0.105, 0.155)
            control_toward = self.rng.uniform(0.55, 0.95)
            control_side = side_sign * self.rng.uniform(0.02, 0.06)
            control_z_frac = self.rng.uniform(0.25, 0.5)
            lift_progress_power = self.rng.uniform(0.6, 0.9)
        elif motion_style == "over_then_back":
            lift_toward = self.rng.uniform(0.08, 0.135) * toward_ring
            lift_lateral = side_sign * self.rng.uniform(0.02, 0.055) * lift_side
            lift_height = self.rng.uniform(0.17, 0.245)
            control_toward = self.rng.uniform(1.1, 1.55)
            control_side = -side_sign * self.rng.uniform(0.02, 0.06)
            control_z_frac = self.rng.uniform(0.45, 0.85)
            lift_progress_power = self.rng.uniform(0.8, 1.2)
        else:
            lift_toward = self.rng.uniform(0.055, 0.105) * toward_ring
            lift_lateral = self.rng.uniform(-0.012, 0.012) * lift_side
            lift_height = self.rng.uniform(0.11, 0.145)
            control_toward = self.rng.uniform(0.65, 1.0)
            control_side = self.rng.uniform(-0.012, 0.012)
            control_z_frac = self.rng.uniform(0.35, 0.55)
            lift_progress_power = self.rng.uniform(0.7, 0.95)
        lift_jitter = np.array([self.rng.uniform(-0.012, 0.012), self.rng.uniform(-0.012, 0.012), 0.0])
        lift_xy = lift_toward + lift_lateral + lift_jitter
        lift_start_pos = grasp_pos + np.array([0.0, 0.0, close_height])
        variant_side = self.rng.choice([-1.0, 1.0])
        if style_variant == "plain":
            control2_toward = self.rng.uniform(0.65, 1.05)
            control2_side = self.rng.uniform(-0.018, 0.018)
            control2_z_frac = self.rng.uniform(0.65, 1.0)
        elif style_variant == "early_bend":
            control_toward *= self.rng.uniform(1.05, 1.35)
            control2_toward = self.rng.uniform(0.75, 1.15)
            control2_side = -0.45 * control_side + self.rng.uniform(-0.018, 0.018)
            control2_z_frac = self.rng.uniform(0.65, 0.95)
            lift_progress_power *= self.rng.uniform(0.75, 0.95)
        elif style_variant == "late_bend":
            control_toward *= self.rng.uniform(0.55, 0.85)
            control2_toward = self.rng.uniform(0.95, 1.35)
            control2_side = control_side + variant_side * self.rng.uniform(0.015, 0.045)
            control2_z_frac = self.rng.uniform(0.75, 1.1)
            lift_progress_power *= self.rng.uniform(1.05, 1.3)
        elif style_variant == "wide_bend":
            control_side += variant_side * self.rng.uniform(0.02, 0.055)
            control2_toward = self.rng.uniform(0.75, 1.25)
            control2_side = -control_side * self.rng.uniform(0.35, 0.8)
            control2_z_frac = self.rng.uniform(0.55, 1.05)
        else:
            control_side *= self.rng.uniform(0.35, 0.7)
            control2_toward = self.rng.uniform(0.75, 1.1)
            control2_side = self.rng.uniform(-0.012, 0.012)
            control2_z_frac = self.rng.uniform(0.65, 0.95)
            lift_progress_power = 0.85 * lift_progress_power + 0.15
        lift_control_xy = (
            control_toward * lift_toward
            + control_side * lift_side
            + self.rng.uniform(-0.01, 0.01) * toward_ring
        )
        lift_control_z = control_z_frac * lift_height
        lift_control_pos = grasp_pos + lift_control_xy + np.array([0.0, 0.0, lift_control_z])
        lift_control2_xy = (
            control2_toward * lift_xy
            + control2_side * lift_side
            + self.rng.uniform(-0.01, 0.012) * toward_ring
        )
        lift_control2_z = control2_z_frac * lift_height
        lift_control2_pos = grasp_pos + lift_control2_xy + np.array([0.0, 0.0, lift_control2_z])
        lift_pos = grasp_pos + lift_xy + np.array([0.0, 0.0, lift_height])
        lift_progress_power = float(np.clip(lift_progress_power, 0.55, 1.75))
        lift_yaw = 0.0
        lift_pitch = 0.0
        lift_roll = 0.0
        lift_mat = grasp_mat
        lift_quat = T.mat2quat(lift_mat)
        durations = {
            "aim_approach": int(self.rng.randint(48, 70)),
            "aim_descend": int(self.rng.randint(48, 70)),
            "close_gripper": int(self.rng.randint(24, 36)),
            "lift_arc": int(self.rng.randint(48, 98)),
            "align": int(self.rng.randint(72, 105)),
            "pre_insert": int(self.rng.randint(26, 42)),
            "insert_through": int(self.rng.randint(110, 155)),
            "hold_after_insert": int(self.rng.randint(22, 34)),
        }
        pre_insert_offset = float(self.rng.uniform(-0.052, -0.038))
        align_offset = float(self.rng.uniform(-0.082, -0.062))
        insert_start_offset = float(self.rng.uniform(-0.045, -0.035))
        insert_end_offset = float(self.rng.uniform(0.036, 0.052))
        align_curve = {
            "lateral_start": float(self.rng.uniform(-0.002, 0.002)),
            "lateral_control": float(self.rng.uniform(-0.012, 0.012)),
            "lateral_end": float(self.rng.uniform(-0.002, 0.002)),
            "vertical_start": float(self.rng.uniform(-0.001, 0.002)),
            "vertical_control": float(self.rng.uniform(-0.004, 0.008)),
            "vertical_end": float(self.rng.uniform(-0.001, 0.0015)),
            "twist_start": 0.0,
            "twist_control": float(self.rng.uniform(np.deg2rad(-5.0), np.deg2rad(5.0))),
            "twist_end": float(self.rng.uniform(np.deg2rad(-1.5), np.deg2rad(1.5))),
            "tilt_start": 0.0,
            "tilt_control": float(self.rng.uniform(np.deg2rad(-3.0), np.deg2rad(3.0))),
            "tilt_end": float(self.rng.uniform(np.deg2rad(-1.0), np.deg2rad(1.0))),
            "rotation_delay": float(self.rng.uniform(0.08, 0.28)),
            "rotation_span": float(self.rng.uniform(0.58, 0.85)),
        }

        episode_noise_std = float(self.action_noise_std * self.rng.uniform(0.8, 1.25))
        policy_state = self._new_policy_state(env, noise_std=episode_noise_std)
        stats = {
            "success": False,
            "env_success": False,
            "policy_success": False,
            "failure_reason": "not_evaluated",
            "subgoal_durations": {},
            "min_ring_distance": float("inf"),
            "max_insert_progress": -float("inf"),
            "final_insert_progress": -float("inf"),
            "tripod_displacement": 0.0,
            "initial_tripod_pos": tripod_position(env),
            "gripper_closed": False,
            "hold_complete": False,
            "grasp_angle_deg": float(grasp_angle),
            "grasp_tilt_x_deg": float(np.rad2deg(grasp_tilt_x)),
            "grasp_tilt_y_deg": float(np.rad2deg(grasp_tilt_y)),
            "grasp_orientation_mode": "fixed_perpendicular",
            "grasp_offset_along": float(grasp_offset_along),
            "grasp_offset_lateral": float(grasp_offset_lateral),
            "grasp_offset_vertical": float(grasp_offset_vertical),
            "grasp_pos": grasp_pos.tolist(),
            "pregrasp_height": float(pregrasp_height),
            "descend_height": float(descend_height),
            "close_height": float(close_height),
            "lift_toward": lift_toward.tolist(),
            "lift_lateral": lift_lateral.tolist(),
            "lift_jitter": lift_jitter.tolist(),
            "lift_height": float(lift_height),
            "motion_style": str(motion_style),
            "style_variant": str(style_variant),
            "lift_control_toward": float(control_toward),
            "lift_control_side": float(control_side),
            "lift_control_z_frac": float(control_z_frac),
            "lift_control2_toward": float(control2_toward),
            "lift_control2_side": float(control2_side),
            "lift_control2_z_frac": float(control2_z_frac),
            "lift_start_pos": lift_start_pos.tolist(),
            "lift_control_pos": lift_control_pos.tolist(),
            "lift_control2_pos": lift_control2_pos.tolist(),
            "lift_progress_power": float(lift_progress_power),
            "lift_pos": lift_pos.tolist(),
            "lift_yaw_deg": float(np.rad2deg(lift_yaw)),
            "lift_pitch_deg": float(np.rad2deg(lift_pitch)),
            "lift_roll_deg": float(np.rad2deg(lift_roll)),
            "align_curve": {
                key: float(np.rad2deg(value)) if "twist" in key or "tilt" in key else float(value)
                for key, value in align_curve.items()
            },
            "action_noise_std": float(self.action_noise_std),
            "episode_noise_std": episode_noise_std,
            "planned_durations": durations,
            "planned_offsets": {
                "align": align_offset,
                "pre_insert": pre_insert_offset,
                "insert_start": insert_start_offset,
                "insert_end": insert_end_offset,
            },
        }

        # aim: approach and descend onto the randomized handle grasp point.
        self._track_target(
            env,
            self._fixed_target(grasp_pos + np.array([0.0, 0.0, pregrasp_height]), grasp_quat),
            -1.0,
            durations["aim_approach"],
            policy_state,
            stats,
            "aim_approach",
            render,
            max_fr,
        )
        self._track_target(
            env,
            self._fixed_target(grasp_pos + np.array([0.0, 0.0, descend_height]), grasp_quat),
            -1.0,
            durations["aim_descend"],
            policy_state,
            stats,
            "aim_descend",
            render,
            max_fr,
        )

        # close_gripper: close and dwell before lifting so the handle settles in the gripper.
        self._track_target(
            env,
            self._fixed_target(grasp_pos + np.array([0.0, 0.0, close_height]), grasp_quat),
            1.0,
            durations["close_gripper"],
            policy_state,
            stats,
            "close_gripper",
            render,
            max_fr,
            stop_on_reach=False,
        )
        stats["gripper_closed"] = True

        # lift: follow one continuous randomized arc instead of a fixed up-then-sideways sequence.
        for i in range(durations["lift_arc"]):
            progress = i / max(1, durations["lift_arc"] - 1)
            lift_t = smoothstep(progress**lift_progress_power)
            target_pos = cubic_bezier(lift_start_pos, lift_control_pos, lift_control2_pos, lift_pos, lift_t)
            self._track_target(
                env,
                self._fixed_target(target_pos, lift_quat),
                1.0,
                1,
                policy_state,
                stats,
                "lift_arc",
                render,
                max_fr,
            )

        # align: move forward while gradually rotating toward the ring normal, with smooth low-frequency variation.
        align_end_offset = pre_insert_offset - 0.012
        for i, offset in enumerate(np.linspace(align_offset, align_end_offset, durations["align"])):
            progress = i / max(1, durations["align"] - 1)
            self._track_target(
                env,
                lambda offset=offset, progress=progress: self._curved_alignment_target(
                    offset,
                    progress,
                    align_curve,
                )(env),
                1.0,
                1,
                policy_state,
                stats,
                "align",
                render,
                max_fr,
            )
        self._track_target(
            env,
            lambda: self._curved_alignment_target(
                pre_insert_offset,
                1.0,
                align_curve,
            )(env),
            1.0,
            durations["pre_insert"],
            policy_state,
            stats,
            "pre_insert",
            render,
            max_fr,
        )

        # insert_through: keep inserting past sparse env success instead of terminating early.
        for offset in np.linspace(insert_start_offset, insert_end_offset, durations["insert_through"]):
            progress = (offset - insert_start_offset) / max(1e-6, insert_end_offset - insert_start_offset)
            insert_curve = {
                **align_curve,
                "lateral_start": align_curve["lateral_end"],
                "lateral_control": align_curve["lateral_end"] * (1.0 - progress),
                "lateral_end": 0.0,
                "vertical_start": align_curve["vertical_end"],
                "vertical_control": align_curve["vertical_end"] * (1.0 - progress),
                "vertical_end": 0.0,
                "twist_start": align_curve["twist_end"],
                "twist_control": align_curve["twist_end"] * (1.0 - progress),
                "twist_end": 0.0,
                "tilt_start": align_curve["tilt_end"],
                "tilt_control": align_curve["tilt_end"] * (1.0 - progress),
                "tilt_end": 0.0,
                "rotation_delay": 0.0,
                "rotation_span": 0.35,
            }
            self._track_target(
                env,
                lambda offset=offset, progress=progress, insert_curve=insert_curve: self._curved_alignment_target(
                    offset,
                    progress,
                    insert_curve,
                )(env),
                1.0,
                1,
                policy_state,
                stats,
                "insert_through",
                render,
                max_fr,
            )

        # hold_after_insert: keep the gripper closed after insertion for a stable terminal segment.
        hold_pos, hold_quat = get_eef_pose(env)
        self._track_target(
            env,
            self._fixed_target(hold_pos, hold_quat),
            1.0,
            durations["hold_after_insert"],
            policy_state,
            stats,
            "hold_after_insert",
            render,
            max_fr,
            stop_on_reach=False,
        )
        stats["hold_complete"] = True
        self._record_metrics(env, stats)

        stats["policy_success"] = bool(self._policy_success(env, stats))
        stats["success"] = stats["policy_success"]
        stats["steps"] = int(env.t)
        if policy_state["action_delta_norms"]:
            stats["smoothness"] = {
                "mean_action_delta": float(np.mean(policy_state["action_delta_norms"])),
                "max_action_delta": float(np.max(policy_state["action_delta_norms"])),
                "mean_action_jerk": float(np.mean(policy_state["action_jerk_norms"])) if policy_state["action_jerk_norms"] else 0.0,
                "max_action_jerk": float(np.max(policy_state["action_jerk_norms"])) if policy_state["action_jerk_norms"] else 0.0,
            }
        else:
            stats["smoothness"] = {
                "mean_action_delta": 0.0,
                "max_action_delta": 0.0,
                "mean_action_jerk": 0.0,
                "max_action_jerk": 0.0,
            }
        stats["initial_tripod_pos"] = stats["initial_tripod_pos"].tolist()
        self.stats.append(stats)
        return bool(stats["policy_success"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Threading")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--directory", type=str, default=str(REPO_ROOT / "threading_scripted_demos"))
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-fr", type=int, default=None)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument(
        "--balanced-motion-styles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cycle through motion styles so large collections do not over-sample one style.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Count every smooth completed rollout toward --num-demos and keep failed attempts instead of requiring success.",
    )
    return parser.parse_args()


def make_motion_style_plan(rng, num_demos):
    plan = []
    while len(plan) < num_demos:
        cycle = list(MOTION_STYLES)
        rng.shuffle(cycle)
        plan.extend(cycle)
    return plan[:num_demos]


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    controller_config = suite.load_composite_controller_config(robot=args.robots[0])

    env = suite.make(
        args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        horizon=args.horizon,
        seed=args.seed,
    )
    env = DataCollectionWrapper(env, args.directory, collect_freq=1, flush_freq=args.horizon + 1)
    policy = ThreadingScriptedPolicy(rng=rng, action_noise_std=args.action_noise_std)
    motion_style_plan = make_motion_style_plan(rng, args.num_demos) if args.balanced_motion_styles else None

    successes = 0
    attempts = 0
    kept = 0
    while kept < args.num_demos and attempts < args.max_attempts:
        attempts += 1
        target_motion_style = motion_style_plan[kept] if motion_style_plan is not None else None
        success = policy.rollout(env, render=args.render, max_fr=args.max_fr, motion_style=target_motion_style)
        keep_episode = success or args.allow_failures or args.keep_failed
        ep_dir = finalize_episode(
            env,
            success=success,
            cleanup_failed=not keep_episode,
            stats=policy.stats[-1],
        )
        if success:
            successes += 1
        if keep_episode:
            kept += 1
        print(
            "attempt={} success={} kept={}/{} successes={} style={} variant={} steps={} ep_dir={}".format(
                attempts,
                success,
                kept,
                args.num_demos,
                successes,
                policy.stats[-1].get("motion_style"),
                policy.stats[-1].get("style_variant"),
                policy.stats[-1]["steps"],
                ep_dir,
            )
        )
        if not success:
            print(f"  failure_reason={policy.stats[-1]['failure_reason']}")

    env.close()
    if kept < args.num_demos:
        raise RuntimeError(f"Kept {kept}/{args.num_demos} demos after {attempts} attempts")


if __name__ == "__main__":
    main()
