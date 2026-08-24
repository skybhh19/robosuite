"""Collect scripted demonstrations for the Threading task.

Example:
    $ python robosuite/scripts/collect_threading_scripted_grasp_angle.py --num-demos 1 --render
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
from robosuite.utils.ik_utils import IKSolver
from robosuite.wrappers import DataCollectionWrapper


POSITION_SCALE = np.array([0.05, 0.05, 0.05])
ORIENTATION_SCALE = np.array([0.5, 0.5, 0.5])
PATH_COMMAND_POSITION_TOLERANCE = 1e-6
PATH_COMMAND_ORIENTATION_TOLERANCE_DEG = 1e-3
GRIPPER_CLOSE_COMMAND_THRESHOLD = 0.99
GRIPPER_CLOSE_MOTION_THRESHOLD = 1e-4
GRIPPER_CLOSE_STABLE_STEPS = 3
GRIPPER_CLOSE_MAX_STEPS = 32
# Reference steps define curve speed; independent caps bound OSC convergence time.
NOMINAL_STAGE_STEPS = {
    "aim_approach": 45,
    "aim_descend": 45,
    "close_gripper": 24,
    "grasp_probe_lift": 14,
    "lift_arc": 60,
    "align": 50,
    "pre_insert": 1,
    "insert_through": 65,
    "hold_after_insert": 10,
}
OSC_STAGE_MAX_STEPS = {
    "aim_continuous": 135,
    "close_gripper": GRIPPER_CLOSE_MAX_STEPS,
    "grasp_probe_lift": 22,
    "lift_arc": 150,
    "align": 90,
    "pre_insert": 20,
    "insert_through": 135,
    "hold_after_insert": 18,
}
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


def minimum_jerk(x):
    """Return a quintic blend with zero endpoint velocity and acceleration."""
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


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


def angle_deg(a, b):
    a = unit(a)
    b = unit(b)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def get_eef_pose(env):
    """Return the right-arm end-effector site pose."""
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    pos = np.array(env.sim.data.site_xpos[site_id])
    quat = T.mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3))
    return pos, quat


def get_eef_mat(env):
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    return np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3)


def geom_pose(env, name):
    """Return geom position and rotation matrix."""
    geom_id = env.sim.model.geom_name2id(name)
    pos = np.array(env.sim.data.geom_xpos[geom_id])
    mat = np.array(env.sim.data.geom_xmat[geom_id]).reshape(3, 3)
    return pos, mat


def needle_state(env):
    """Return needle center, handle center, and local axes in world coordinates."""
    needle_center, needle_mat = geom_pose(env, "needle_obj_needle")
    handle_center, _ = geom_pose(env, "needle_obj_handle")
    yaxis = unit(needle_mat[:, 1])
    return {
        "needle_center": needle_center,
        "handle_center": handle_center,
        "mat": needle_mat,
        "xaxis": unit(needle_mat[:, 0]),
        "yaxis": yaxis,
        "zaxis": unit(needle_mat[:, 2], fallback=[0.0, 0.0, 1.0]),
        "tip": needle_center - NEEDLE_SHAFT_HALF_LENGTH * yaxis,
        "handle_side": needle_center + NEEDLE_SHAFT_HALF_LENGTH * yaxis,
    }


def ring_state(env):
    """Return the ring center and an oriented orthonormal aperture frame."""
    ring_pos = np.zeros(3)
    ring_mat = None
    for i in range(env.tripod.num_ring_geoms):
        pos, mat = geom_pose(env, f"tripod_obj_ring_{i}")
        ring_pos += pos
        if ring_mat is None:
            ring_mat = mat
    ring_pos /= env.tripod.num_ring_geoms
    normal = unit(ring_mat[:, 0], fallback=[1.0, 0.0, 0.0])
    tangent = unit(ring_mat[:, 1], fallback=[1.0, 0.0, 0.0])
    vertical = unit(ring_mat[:, 2], fallback=[0.0, 0.0, 1.0])
    # Match D0's fixed insertion convention: approach from world +Y and pass
    # through the ring toward world -Y, never from the back side.
    env_name = env.unwrapped.__class__.__name__ if hasattr(env, "unwrapped") else env.__class__.__name__
    preferred_y = 1.0 if env_name == "Threading_D2" else -1.0
    if normal[1] * preferred_y < 0:
        normal = -normal
        tangent = -tangent
    # Match Threading._check_success(), which defines insertion against the
    # horizontal projection of the ring normal even if contact tilts the tripod.
    normal[2] = 0.0
    normal = unit(normal, fallback=[1.0, 0.0, 0.0])
    tangent = unit(tangent - np.dot(tangent, normal) * normal, fallback=[1.0, 0.0, 0.0])
    vertical = unit(np.cross(normal, tangent), fallback=vertical)
    tangent = unit(np.cross(vertical, normal), fallback=tangent)
    return {
        "center": ring_pos,
        "mat": ring_mat,
        "normal": normal,
        "tangent": tangent,
        "vertical": vertical,
    }


def ring_frame_offsets(point, ring):
    """Express a world-space point relative to the ring aperture frame."""
    delta = np.asarray(point, dtype=float) - np.asarray(ring["center"], dtype=float)
    return {
        "normal": float(np.dot(delta, ring["normal"])),
        "lateral": float(np.dot(delta, ring["tangent"])),
        "vertical": float(np.dot(delta, ring["vertical"])),
    }


def shaft_ring_distance(needle, ring):
    """Return the closest distance from the ring center to the needle shaft segment."""
    rel = ring["center"] - needle["needle_center"]
    t = np.clip(np.dot(rel, needle["yaxis"]), -NEEDLE_SHAFT_HALF_LENGTH, NEEDLE_SHAFT_HALF_LENGTH)
    closest = needle["needle_center"] + t * needle["yaxis"]
    return float(np.linalg.norm(closest - ring["center"]))


def insertion_progress(needle, ring):
    """Signed progress of the needle tip through the ring plane."""
    return float(np.dot(needle["tip"] - ring["center"], ring["normal"]))


def clean_ring_aperture_geometry(env):
    """Evaluate aperture crossing with the environment's canonical geometry test."""
    needle = needle_state(env)
    ring = ring_state(env)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    metrics = base_env._aperture_intersection_metrics(
        needle_pos=needle["needle_center"],
        needle_mat=needle["mat"],
        ring_pos=ring["center"],
        ring_mat=ring["mat"],
        ring_normal=ring["normal"],
    )
    if metrics["clean_aperture"]:
        reason = "clear"
    elif not metrics["finite_shaft_crosses_ring_plane"]:
        reason = "finite_shaft_misses_plane"
    else:
        reason = "outside_safe_aperture"
    return {
        "clear": bool(metrics["clean_aperture"]),
        "reason": reason,
        "margin_m": float(metrics["clean_aperture_margin"]),
        "tangent_offset_m": float(metrics["aperture_tangent_offset"]),
        "vertical_offset_m": float(metrics["aperture_vertical_offset"]),
    }


def tripod_position(env):
    return np.array(env.sim.data.body_xpos[env.obj_body_id["tripod"]])


def tripod_orientation(env):
    return np.array(env.sim.data.body_xmat[env.obj_body_id["tripod"]]).reshape(3, 3)


def rotation_error_deg(target_mat, current_mat):
    relative = np.asarray(target_mat).T.dot(np.asarray(current_mat))
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def local_axis_name(axis):
    axis = np.array(axis)
    idx = int(np.argmax(np.abs(axis)))
    sign = "+" if axis[idx] >= 0 else "-"
    return f"{sign}{'xyz'[idx]}"


def calibrate_gripper_axes(base_eef_mat, needle):
    """Infer EEF local axes for object-to-gripper approach and jaw closing."""
    world_z = np.array([0.0, 0.0, 1.0])
    candidates = []
    for idx in range(3):
        for sign in (-1.0, 1.0):
            local = np.zeros(3)
            local[idx] = sign
            candidates.append(local)

    approach_local = max(candidates, key=lambda local: float(np.dot(base_eef_mat.dot(local), world_z)))
    jaw_candidates = [local for local in candidates if abs(float(np.dot(local, approach_local))) < 0.5]
    jaw_local = max(jaw_candidates, key=lambda local: abs(float(np.dot(base_eef_mat.dot(local), needle["xaxis"]))))

    return {
        "approach_local": approach_local,
        "jaw_local": jaw_local,
        "approach_axis_name": local_axis_name(approach_local),
        "jaw_axis_name": local_axis_name(jaw_local),
        "base_world_x": base_eef_mat[:, 0].tolist(),
        "base_world_y": base_eef_mat[:, 1].tolist(),
        "base_world_z": base_eef_mat[:, 2].tolist(),
    }


def construct_grasp_mat(base_eef_mat, needle, grasp_angle_deg, axes):
    """Construct EEF rotation with a real target grasp angle and stable jaw axis."""
    needle_axis, needle_up, _ = needle_angle_basis(needle)
    desired_approach = (
        -np.cos(np.deg2rad(grasp_angle_deg)) * needle_axis
        + np.sin(np.deg2rad(grasp_angle_deg)) * needle_up
    )
    desired_approach = unit(desired_approach)

    desired_jaw = needle["xaxis"] - np.dot(needle["xaxis"], desired_approach) * desired_approach
    desired_jaw = unit(desired_jaw, fallback=np.cross(desired_approach, needle["yaxis"]))
    base_jaw_world = base_eef_mat.dot(axes["jaw_local"])
    if np.dot(desired_jaw, base_jaw_world) < 0:
        desired_jaw = -desired_jaw

    desired_third = unit(np.cross(desired_approach, desired_jaw))
    desired_jaw = unit(np.cross(desired_third, desired_approach))

    local_approach = unit(axes["approach_local"])
    local_jaw = unit(axes["jaw_local"] - np.dot(axes["jaw_local"], local_approach) * local_approach)
    local_third = unit(np.cross(local_approach, local_jaw))
    local_jaw = unit(np.cross(local_third, local_approach))

    local_basis = np.column_stack([local_approach, local_jaw, local_third])
    desired_basis = np.column_stack([desired_approach, desired_jaw, desired_third])
    grasp_mat = desired_basis.dot(local_basis.T)
    return grasp_mat, desired_approach, desired_jaw


def eef_approach_axis(eef_mat, axes):
    return unit(eef_mat.dot(axes["approach_local"]))


def measured_grasp_angle(eef_mat, needle, axes):
    return angle_deg(eef_approach_axis(eef_mat, axes), needle["yaxis"])


def eef_jaw_axis(eef_mat, axes):
    return unit(eef_mat.dot(axes["jaw_local"]))


def needle_angle_basis(needle):
    """Return needle shaft, projected-up, and side axes for grasp angles."""
    needle_axis = unit(needle["yaxis"])
    world_up = np.array([0.0, 0.0, 1.0])
    projected_up = world_up - np.dot(world_up, needle_axis) * needle_axis
    needle_up = unit(projected_up, fallback=needle["zaxis"])
    side = unit(np.cross(needle_axis, needle_up), fallback=needle["xaxis"])
    needle_up = unit(np.cross(side, needle_axis), fallback=needle_up)
    return needle_axis, needle_up, side


def visual_angle_axis(needle, visual_angle_deg):
    """Approach axis for the user-facing side-view grasp angle.

    90 degrees is straight down from above. Values below / above 90 move the
    pregrasp point to opposite sides of the handle in the needle-up plane.
    """
    needle_axis, up_axis, _ = needle_angle_basis(needle)
    side_component = -np.cos(np.deg2rad(visual_angle_deg)) * needle_axis
    up_component = np.sin(np.deg2rad(visual_angle_deg)) * up_axis
    return unit(side_component + up_component, fallback=up_axis)


def visual_grasp_angle_from_axis(axis, needle):
    needle_axis, up_axis, side_axis = needle_angle_basis(needle)
    projected = axis - np.dot(axis, side_axis) * side_axis
    projected = unit(projected, fallback=up_axis)
    signed_cos = -float(np.dot(projected, needle_axis))
    signed_sin = float(np.dot(projected, up_axis))
    return float(np.rad2deg(np.arctan2(signed_sin, signed_cos)))


def needle_to_eef_transform(env):
    needle = needle_state(env)
    eef_pos, _ = get_eef_pose(env)
    eef_mat = get_eef_mat(env)
    rel_pos = needle["mat"].T.dot(eef_pos - needle["needle_center"])
    rel_mat = needle["mat"].T.dot(eef_mat)
    return {"pos": rel_pos, "mat": rel_mat}


def eef_from_needle_pose(needle_center, needle_mat, needle_to_eef):
    target_pos = needle_center + needle_mat.dot(needle_to_eef["pos"])
    target_mat = needle_mat.dot(needle_to_eef["mat"])
    return target_pos, T.mat2quat(target_mat)


def needle_to_eef_error(reference, measured):
    """Measure translation and rotation drift of a grasp transform."""
    return {
        "translation_m": float(np.linalg.norm(np.asarray(reference["pos"]) - np.asarray(measured["pos"]))),
        "rotation_deg": rotation_error_deg(reference["mat"], measured["mat"]),
    }


def gripper_joint_positions(env):
    robot = env.robots[0]
    return np.asarray(robot.get_gripper_joint_positions(robot.arms[0]), dtype=float).copy()


def needle_grasp_contact(env):
    robot = env.robots[0]
    return bool(env._check_grasp(robot.gripper[robot.arms[0]], env.needle))


def shortest_orientation_error(target_quat, current_quat):
    """Return the equivalent relative rotation whose angle is at most pi."""
    quat_err = T.quat_distance(np.asarray(target_quat).copy(), np.asarray(current_quat).copy())
    if quat_err[3] < 0.0:
        quat_err = -quat_err
    return T.quat2axisangle(quat_err)


def target_action(env, target_pos, target_quat, gripper, prev_action, noise_state, noise_std, rng):
    """Convert a target pose into a smooth normalized OSC_POSE action."""
    eef_pos, eef_quat = get_eef_pose(env)
    pos_cmd = np.clip((target_pos - eef_pos) / POSITION_SCALE, -0.7, 0.7)

    ori_err = shortest_orientation_error(target_quat, eef_quat)
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


class JointPositionPoseAdapter:
    """Map world-frame EEF pose targets to absolute Panda joint targets."""

    def __init__(self, env, damping=0.12, integration_dt=0.05, max_dq=1.2):
        robot = env.robots[0]
        arm_name = robot.arms[0]
        arm_controller = robot.composite_controller.part_controllers[arm_name]
        if arm_controller.name != "JOINT_POSITION" or arm_controller.input_type != "absolute":
            raise ValueError("JointPositionPoseAdapter requires an absolute JOINT_POSITION arm controller")

        self.arm_dim = len(arm_controller.joint_names)
        self.qpos_indexes = np.asarray(arm_controller.qpos_index, dtype=int)
        self.max_target_step = 0.04
        robot_config = {
            "joint_names": list(arm_controller.joint_names),
            "end_effector_sites": [arm_controller.ref_name],
            # Avoid an undamped nullspace pseudo-inverse near singularities.
            "nullspace_gains": np.zeros(self.arm_dim),
        }
        self.ik = IKSolver(
            model=env.sim.model._model,
            data=env.sim.data._data,
            robot_config=robot_config,
            damping=damping,
            integration_dt=integration_dt,
            max_dq=max_dq,
            input_action_repr="absolute",
            input_rotation_repr="axis_angle",
            input_ref_frame="world",
        )
        self.ik.q0 = self.current_qpos(env)

    def current_qpos(self, env):
        return np.asarray(env.sim.data.qpos[self.qpos_indexes], dtype=float).copy()

    def jacobian_condition(self):
        jacobian = self.ik._compute_jacobian(self.ik.full_model, self.ik.full_model_data)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if singular_values[-1] < 1e-8:
            return float("inf")
        return float(singular_values[0] / singular_values[-1])

    def action(self, env, target_pos, target_quat, gripper, policy_state):
        target_axis_angle = T.quat2axisangle(np.asarray(target_quat, dtype=float))
        q_des = self.ik.solve(np.concatenate([target_pos, target_axis_angle]))

        noise_std = policy_state["noise_std"]
        if noise_std > 0:
            noise_alpha = 0.96
            noise_state = policy_state["noise_state"]
            noise_state[:] = noise_alpha * noise_state + np.sqrt(1.0 - noise_alpha**2) * policy_state["rng"].normal(
                size=noise_state.shape
            )
            q_des = q_des + noise_std * noise_state

        previous = policy_state["prev_action"]
        previous_q = self.current_qpos(env) if previous is None else previous[: self.arm_dim]
        q_des = 0.45 * previous_q + 0.55 * q_des
        q_des = np.clip(q_des, previous_q - self.max_target_step, previous_q + self.max_target_step)
        joint_ranges = env.sim.model.jnt_range[self.ik.dof_ids]
        q_des = np.clip(q_des, joint_ranges[:, 0], joint_ranges[:, 1])

        action = np.empty(self.arm_dim + 1, dtype=float)
        action[: self.arm_dim] = q_des
        action[-1] = gripper
        policy_state["jacobian_conditions"].append(self.jacobian_condition())
        policy_state["joint_target_step_norms"].append(float(np.linalg.norm(q_des - previous_q)))
        return action


def hold_pose_steps(env, target_pos, target_quat, gripper, steps, policy_state, render=False, max_fr=None):
    """Track one pose target for a fixed number of control steps."""
    success = False
    for _ in range(steps):
        start = time.time()
        action_adapter = policy_state.get("action_adapter")
        if action_adapter is None:
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
        else:
            action = action_adapter.action(env, target_pos, target_quat, gripper, policy_state)
        policy_state["prev_action"] = action
        if policy_state.get("last_action_for_metrics") is not None:
            delta = action[:-1] - policy_state["last_action_for_metrics"][:-1]
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


def json_safe(value):
    """Convert NumPy values and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def finalize_episode(env, success, cleanup_failed=True, stats=None):
    """Flush the wrapper and optionally remove failed attempts."""
    ep_dir = env.ep_directory
    if stats is not None and hasattr(env, "successful"):
        env.successful = bool(stats.get("collection_success", success))
    if env.has_interaction:
        env._flush()
        env.has_interaction = False
    if ep_dir and os.path.exists(ep_dir) and stats is not None:
        with open(os.path.join(ep_dir, "policy_stats.json"), "w") as f:
            json.dump(json_safe(stats), f, indent=2, allow_nan=False)
    if not success and cleanup_failed and ep_dir and os.path.exists(ep_dir):
        shutil.rmtree(ep_dir)
    return ep_dir


class ThreadingScriptedPolicy:
    """Closed-loop scripted policy for grasping, lifting, aligning, and threading the needle."""

    def __init__(
        self,
        rng,
        action_noise_std=0.01,
        grasp_angle_range=(80.0, 120.0),
        control_mode="osc_pose",
        collision_aware_threading=False,
    ):
        self.rng = rng
        self.action_noise_std = action_noise_std
        self.grasp_angle_range = grasp_angle_range
        self.control_mode = control_mode
        self.collision_aware_threading = collision_aware_threading
        self.stats = []

    def _new_policy_state(self, env, noise_std=None):
        action_adapter = JointPositionPoseAdapter(env) if self.control_mode == "joint_position" else None
        noise_dim = action_adapter.arm_dim if action_adapter is not None else env.action_spec[0].shape[0] - 1
        return {
            "prev_action": None,
            "target_pos": None,
            "target_quat": None,
            "noise_state": np.zeros(noise_dim),
            "noise_std": self.action_noise_std if noise_std is None else noise_std,
            "rng": self.rng,
            "last_action_for_metrics": None,
            "last_delta_for_metrics": None,
            "action_delta_norms": [],
            "action_jerk_norms": [],
            "action_adapter": action_adapter,
            "jacobian_conditions": [],
            "joint_target_step_norms": [],
        }

    def _record_metrics(self, env, stats):
        needle = needle_state(env)
        ring = ring_state(env)
        stats["env_success"] = bool(stats["env_success"] or env._check_success())
        stats["min_ring_distance"] = min(stats["min_ring_distance"], shaft_ring_distance(needle, ring))
        stats["max_insert_progress"] = max(stats["max_insert_progress"], insertion_progress(needle, ring))
        stats["final_insert_progress"] = insertion_progress(needle, ring)
        tripod_displacement = float(np.linalg.norm(tripod_position(env) - stats["initial_tripod_pos"]))
        tripod_rotation = rotation_error_deg(stats["initial_tripod_mat"], tripod_orientation(env))
        stats["final_tripod_displacement"] = tripod_displacement
        stats["final_tripod_rotation_deg"] = tripod_rotation
        stats["tripod_displacement"] = max(stats["tripod_displacement"], tripod_displacement)
        stats["tripod_rotation_deg"] = max(stats["tripod_rotation_deg"], tripod_rotation)
        clean_geometry = clean_ring_aperture_geometry(env)
        stats["clean_aperture_history"].append(bool(clean_geometry["clear"]))
        stats["final_clean_aperture_geometry"] = clean_geometry

    def _advance_target(self, env, desired_pos, desired_quat, policy_state, subgoal):
        if policy_state["target_pos"] is None:
            current_pos, current_quat = get_eef_pose(env)
            policy_state["target_pos"] = current_pos
            policy_state["target_quat"] = current_quat

        max_pos_step = 0.0052
        max_angle_step = 0.044
        if subgoal == "insert_through":
            max_pos_step = 0.0045
            max_angle_step = 0.034

        pos_delta = desired_pos - policy_state["target_pos"]
        pos_dist = np.linalg.norm(pos_delta)
        if pos_dist > max_pos_step:
            policy_state["target_pos"] = policy_state["target_pos"] + max_pos_step * pos_delta / pos_dist
        else:
            policy_state["target_pos"] = np.array(desired_pos)

        angle = np.linalg.norm(shortest_orientation_error(desired_quat, policy_state["target_quat"]))
        if angle > max_angle_step:
            policy_state["target_quat"] = T.quat_slerp(policy_state["target_quat"], desired_quat, max_angle_step / angle)
        else:
            policy_state["target_quat"] = np.array(desired_quat)
        return policy_state["target_pos"], policy_state["target_quat"]

    def _pose_error(self, env, desired_pos, desired_quat):
        eef_pos, eef_quat = get_eef_pose(env)
        pos_err = float(np.linalg.norm(desired_pos - eef_pos))
        ori_err = float(np.linalg.norm(shortest_orientation_error(desired_quat, eef_quat)))
        return pos_err, ori_err

    def _pose_completion(
        self,
        env,
        desired_pos,
        desired_quat,
        eef_step_motion,
        position_tolerance,
        orientation_tolerance_deg,
        motion_tolerance,
    ):
        position_error, orientation_error = self._pose_error(env, desired_pos, desired_quat)
        orientation_error_deg = float(np.rad2deg(orientation_error))
        return {
            "passed": bool(
                position_error < position_tolerance
                and orientation_error_deg < orientation_tolerance_deg
                and eef_step_motion < motion_tolerance
            ),
            "position_error_m": float(position_error),
            "orientation_error_deg": orientation_error_deg,
            "eef_step_motion_m": float(eef_step_motion),
            "position_tolerance_m": float(position_tolerance),
            "orientation_tolerance_deg": float(orientation_tolerance_deg),
            "motion_tolerance_m": float(motion_tolerance),
        }

    def _run_bounded_trajectory_stage(
        self,
        env,
        target_at_progress,
        gripper,
        nominal_steps,
        policy_state,
        stats,
        subgoal,
        completion_fn,
        render=False,
        max_fr=None,
        required_stable_steps=2,
        start_at_zero=False,
        completion_start_step=None,
        max_steps=None,
    ):
        """Follow a nominal trajectory, returning once its endpoint gate is stable."""
        nominal_steps = max(1, int(nominal_steps))
        completion_start_step = 1 if completion_start_step is None else max(1, int(completion_start_step))
        if max_steps is None:
            max_steps = nominal_steps + required_stable_steps - 1
        max_steps = max(nominal_steps, int(max_steps))
        stable_steps = 0
        final_details = None
        previous_eef_pos, _ = get_eef_pose(env)
        path_index = 0
        path_waypoints_completed = 0
        path_command_complete = False
        command_position_error = float("inf")
        command_orientation_error_deg = float("inf")

        for step in range(max_steps):
            if start_at_zero and nominal_steps > 1:
                progress = path_index / (nominal_steps - 1)
            else:
                progress = (path_index + 1) / nominal_steps
            desired_pos, desired_quat = target_at_progress(float(progress))
            self._track_target(
                env,
                self._fixed_target(desired_pos, desired_quat),
                gripper,
                1,
                policy_state,
                stats,
                subgoal,
                render,
                max_fr,
                stop_on_reach=False,
            )
            eef_pos, _ = get_eef_pose(env)
            eef_step_motion = float(np.linalg.norm(eef_pos - previous_eef_pos))
            previous_eef_pos = eef_pos

            command_position_error = float(np.linalg.norm(policy_state["target_pos"] - desired_pos))
            command_orientation_error_deg = float(
                np.rad2deg(
                    np.linalg.norm(
                        shortest_orientation_error(desired_quat, policy_state["target_quat"])
                    )
                )
            )
            waypoint_reached = (
                command_position_error <= PATH_COMMAND_POSITION_TOLERANCE
                and command_orientation_error_deg <= PATH_COMMAND_ORIENTATION_TOLERANCE_DEG
            )
            if waypoint_reached:
                path_waypoints_completed = max(path_waypoints_completed, path_index + 1)
                if path_index < nominal_steps - 1:
                    path_index += 1
                else:
                    path_command_complete = True

            if step + 1 < completion_start_step:
                continue
            final_pos, final_quat = target_at_progress(1.0)
            final_details = completion_fn(env, final_pos, final_quat, eef_step_motion)
            final_details.update(
                {
                    "path_command_complete": bool(path_command_complete),
                    "path_waypoints_completed": int(path_waypoints_completed),
                    "path_waypoints_total": int(nominal_steps),
                    "command_position_error_m": command_position_error,
                    "command_orientation_error_deg": command_orientation_error_deg,
                }
            )
            final_details["passed"] = bool(final_details["passed"] and path_command_complete)
            stable_steps = stable_steps + 1 if final_details["passed"] else 0
            if stable_steps >= required_stable_steps:
                break

        complete = stable_steps >= required_stable_steps
        actual_steps = int(stats["subgoal_durations"].get(subgoal, 0))
        stats["stage_outcomes"][subgoal] = {
            "complete": bool(complete),
            "timed_out": bool(not complete and actual_steps >= max_steps),
            "nominal_steps": int(nominal_steps),
            "max_steps": int(max_steps),
            "actual_steps": actual_steps,
            "stable_steps": int(stable_steps),
            "required_stable_steps": int(required_stable_steps),
            "path_command_complete": bool(path_command_complete),
            "path_waypoints_completed": int(path_waypoints_completed),
            "path_waypoints_total": int(nominal_steps),
            "final": final_details,
        }
        return complete

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

    def _settle_at_grasp(
        self,
        env,
        target_pos,
        target_quat,
        policy_state,
        stats,
        render=False,
        max_fr=None,
        max_steps=6,
        required_stable_steps=2,
    ):
        """Require a genuinely stable grasp pose before closing the fingers."""
        if self.control_mode == "joint_position":
            max_steps = max(max_steps, 10)
        settle_noise_std = policy_state["noise_std"]
        if self.control_mode == "joint_position":
            policy_state["noise_std"] = 0.0
        stable_steps = 0
        previous_pos, _ = get_eef_pose(env)
        best_pos_err = float("inf")
        best_ori_err = float("inf")
        max_stable_speed = 0.003
        try:
            for _ in range(max_steps):
                self._track_target(
                    env,
                    self._fixed_target(target_pos, target_quat),
                    -1.0,
                    1,
                    policy_state,
                    stats,
                    "grasp_settle",
                    render,
                    max_fr,
                    stop_on_reach=False,
                )
                eef_pos, _ = get_eef_pose(env)
                step_motion = float(np.linalg.norm(eef_pos - previous_pos))
                previous_pos = eef_pos
                pos_err, ori_err = self._pose_error(env, target_pos, target_quat)
                best_pos_err = min(best_pos_err, pos_err)
                best_ori_err = min(best_ori_err, ori_err)
                position_tolerance = 0.005 if self.control_mode == "joint_position" else 0.002
                if pos_err < position_tolerance and ori_err < np.deg2rad(2.0) and step_motion < max_stable_speed:
                    stable_steps += 1
                    if stable_steps >= required_stable_steps:
                        break
                else:
                    stable_steps = 0
        finally:
            policy_state["noise_std"] = settle_noise_std

        settled = stable_steps >= required_stable_steps
        stats["grasp_settle"] = {
            "passed": bool(settled),
            "stable_steps": int(stable_steps),
            "required_stable_steps": int(required_stable_steps),
            "best_position_error": float(best_pos_err),
            "best_orientation_error_deg": float(np.rad2deg(best_ori_err)),
        }
        return settled

    def _close_gripper_until_stable_contact(
        self,
        env,
        target_pos,
        target_quat,
        policy_state,
        stats,
        render=False,
        max_fr=None,
    ):
        """Close under OSC until the command, fingers, and grasp are stable."""
        previous_qpos = gripper_joint_positions(env)
        stable_steps = 0
        steps = 0
        finger_motion = float("inf")
        contact = False
        command = float(policy_state["prev_action"][-1]) if policy_state["prev_action"] is not None else -1.0

        for _ in range(GRIPPER_CLOSE_MAX_STEPS):
            self._track_target(
                env,
                self._fixed_target(target_pos, target_quat),
                1.0,
                1,
                policy_state,
                stats,
                "close_gripper",
                render,
                max_fr,
                stop_on_reach=False,
            )
            steps += 1
            current_qpos = gripper_joint_positions(env)
            finger_motion = float(np.linalg.norm(current_qpos - previous_qpos))
            previous_qpos = current_qpos
            command = float(policy_state["prev_action"][-1])
            contact = needle_grasp_contact(env)
            if (
                command >= GRIPPER_CLOSE_COMMAND_THRESHOLD
                and finger_motion <= GRIPPER_CLOSE_MOTION_THRESHOLD
                and contact
            ):
                stable_steps += 1
                if stable_steps >= GRIPPER_CLOSE_STABLE_STEPS:
                    break
            else:
                stable_steps = 0

        complete = stable_steps >= GRIPPER_CLOSE_STABLE_STEPS
        stats["gripper_close"] = {
            "complete": bool(complete),
            "timed_out": bool(not complete and steps >= GRIPPER_CLOSE_MAX_STEPS),
            "steps": int(steps),
            "max_steps": int(GRIPPER_CLOSE_MAX_STEPS),
            "stable_steps": int(stable_steps),
            "required_stable_steps": int(GRIPPER_CLOSE_STABLE_STEPS),
            "command": float(command),
            "command_fully_closed": bool(command >= GRIPPER_CLOSE_COMMAND_THRESHOLD),
            "finger_motion_m": float(finger_motion),
            "finger_motion_threshold_m": float(GRIPPER_CLOSE_MOTION_THRESHOLD),
            "contact": bool(contact),
        }
        stats.setdefault("stage_outcomes", {})["close_gripper"] = dict(stats["gripper_close"])
        return complete

    def _alignment_target(
        self,
        offset,
        lateral_offset=0.0,
        vertical_offset=0.0,
        twist=0.0,
        tilt=0.0,
        rotation_fraction=1.0,
        needle_to_eef=None,
        reference_needle_mat=None,
    ):
        if needle_to_eef is None or reference_needle_mat is None:
            raise ValueError("Alignment targets require a measured grasp transform and fixed reference needle frame")

        def target(env):
            current_ring = ring_state(env)
            side = current_ring["tangent"]
            vertical = current_ring["vertical"]
            reference_mat = np.asarray(reference_needle_mat, dtype=float)
            align_rot = rotation_between(-reference_mat[:, 1], current_ring["normal"])
            align_quat = T.mat2quat(align_rot)
            partial_align_quat = T.quat_slerp(T.mat2quat(np.eye(3)), align_quat, np.clip(rotation_fraction, 0.0, 1.0))
            partial_align_rot = T.quat2mat(partial_align_quat)
            wobble_rot = (
                T.rotation_matrix(twist, current_ring["normal"])[:3, :3].dot(
                    T.rotation_matrix(tilt, side)[:3, :3]
                )
            )
            target_needle_mat = wobble_rot.dot(partial_align_rot).dot(reference_mat)
            desired_tip = (
                current_ring["center"]
                + offset * current_ring["normal"]
                + lateral_offset * side
                + vertical_offset * vertical
            )
            target_needle_center = desired_tip + NEEDLE_SHAFT_HALF_LENGTH * target_needle_mat[:, 1]
            target_pos, target_quat = eef_from_needle_pose(target_needle_center, target_needle_mat, needle_to_eef)
            return target_pos, target_quat

        return target

    def _curved_alignment_target(
        self,
        offset,
        progress,
        curve,
        needle_to_eef=None,
        reference_needle_mat=None,
        alignment_fraction=None,
    ):
        path_t = smoothstep(progress)
        rot_t = smoothstep((progress - curve["rotation_delay"]) / curve["rotation_span"])
        lateral = quadratic_bezier(curve["lateral_start"], curve["lateral_control"], curve["lateral_end"], path_t)
        vertical = quadratic_bezier(curve["vertical_start"], curve["vertical_control"], curve["vertical_end"], path_t)
        twist = quadratic_bezier(curve["twist_start"], curve["twist_control"], curve["twist_end"], rot_t)
        tilt = quadratic_bezier(curve["tilt_start"], curve["tilt_control"], curve["tilt_end"], rot_t)
        return self._alignment_target(
            offset,
            lateral,
            vertical,
            twist,
            tilt,
            rot_t if alignment_fraction is None else alignment_fraction,
            needle_to_eef,
            reference_needle_mat,
        )

    def _needle_target_errors(self, env, offset, lateral_offset=0.0, vertical_offset=0.0):
        """Measure needle-tip tracking error in the moving ring frame."""
        needle = needle_state(env)
        ring = ring_state(env)
        side = ring["tangent"]
        desired_tip = (
            ring["center"]
            + offset * ring["normal"]
            + lateral_offset * side
            + vertical_offset * ring["vertical"]
        )
        error = needle["tip"] - desired_tip
        alignment_cosine = float(np.clip(np.dot(-needle["yaxis"], ring["normal"]), -1.0, 1.0))
        return {
            "normal_error_m": abs(float(np.dot(error, ring["normal"]))),
            "tangent_error_m": abs(float(np.dot(error, side))),
            "vertical_error_m": abs(float(np.dot(error, ring["vertical"]))),
            "orientation_error_deg": float(np.rad2deg(np.arccos(alignment_cosine))),
        }

    def _tripod_displacement(self, env, stats):
        current = float(np.linalg.norm(tripod_position(env) - stats["initial_tripod_pos"]))
        return max(float(stats["tripod_displacement"]), current)

    def _tripod_motion(self, env, stats, max_displacement=0.035, max_rotation_deg=5.0):
        displacement = self._tripod_displacement(env, stats)
        current_rotation = rotation_error_deg(stats["initial_tripod_mat"], tripod_orientation(env))
        rotation_deg = max(float(stats["tripod_rotation_deg"]), current_rotation)
        return {
            "stable": bool(displacement < max_displacement and rotation_deg < max_rotation_deg),
            "displacement_m": float(displacement),
            "rotation_deg": float(rotation_deg),
            "max_displacement_m": float(max_displacement),
            "max_rotation_deg": float(max_rotation_deg),
        }

    def _policy_success(self, env, stats):
        needle = needle_state(env)
        ring = ring_state(env)
        stats["final_ring_distance"] = shaft_ring_distance(needle, ring)
        stats["final_insert_progress"] = insertion_progress(needle, ring)
        env_success_now = bool(env._check_success())
        stats["env_success_debug"] = dict(getattr(env, "_threading_success_debug", {}))
        clean_geometry = clean_ring_aperture_geometry(env)
        history = stats.get("clean_aperture_history", [])
        hold_count = max(1, int(np.ceil(len(history) * 0.10)))
        hold_clear_fraction = float(np.mean(history[-hold_count:])) if history else 0.0
        stats["final_clean_aperture_geometry"] = clean_geometry
        stats["final_hold_clear_fraction"] = hold_clear_fraction
        checks = {
            "env_success": bool(stats["env_success"] or env_success_now),
            "ring_crossed": bool(clean_geometry["clear"]),
            "inserted_past_ring": stats["max_insert_progress"] > 0.026,
            "final_still_inserted": stats["final_insert_progress"] > 0.014,
            "tripod_stable": stats["tripod_displacement"] < 0.035 and stats["tripod_rotation_deg"] < 5.0,
            "hold_complete": bool(stats["hold_complete"]),
            "gripper_close_complete": bool(stats.get("gripper_close_complete", False)),
            "gripper_closed": bool(stats["gripper_closed"]),
            "grasp_contact": bool(stats.get("grasp_contact_after_lift", False)),
            "grasp_rigid": bool(stats.get("grasp_rigid_after_lift", False)),
            "grasp_axis_perpendicular": stats.get("actual_grasp_jaw_needle_axis_abs_dot", 1.0) < 0.15,
            "insert_direction_valid": bool(stats.get("insert_direction_valid", False)),
        }
        if stats.get("control_mode") == "osc_pose":
            checks["stage_sequence_complete"] = bool(
                stats.get("approach_complete", False)
                and stats.get("gripper_close_complete", False)
                and stats.get("probe_complete", False)
                and stats.get("lift_complete", False)
                and stats.get("align_complete", False)
                and stats.get("pre_insert_complete", False)
                and stats.get("insert_complete", False)
                and stats.get("hold_complete", False)
            )
        stats["policy_checks"] = checks
        failed = [name for name, passed in checks.items() if not passed]
        stats["failure_reason"] = "none" if not failed else ",".join(failed)
        return not failed

    def _finish_rollout(self, env, stats, policy_state, gripper_axes):
        """Finalize diagnostics for both completed and stage-aborted rollouts."""
        self._record_metrics(env, stats)
        insert_needle = needle_state(env)
        stats["actual_insert_angle_deg"] = float(
            visual_grasp_angle_from_axis(eef_approach_axis(get_eef_mat(env), gripper_axes), insert_needle)
        )
        stats["policy_success"] = bool(self._policy_success(env, stats))
        stats["success"] = stats["policy_success"]
        stats["steps"] = int(env.t)
        if policy_state["action_delta_norms"]:
            stats["smoothness"] = {
                "mean_action_delta": float(np.mean(policy_state["action_delta_norms"])),
                "max_action_delta": float(np.max(policy_state["action_delta_norms"])),
                "mean_action_jerk": float(np.mean(policy_state["action_jerk_norms"]))
                if policy_state["action_jerk_norms"]
                else 0.0,
                "max_action_jerk": float(np.max(policy_state["action_jerk_norms"]))
                if policy_state["action_jerk_norms"]
                else 0.0,
            }
        else:
            stats["smoothness"] = {
                "mean_action_delta": 0.0,
                "max_action_delta": 0.0,
                "mean_action_jerk": 0.0,
                "max_action_jerk": 0.0,
            }
        finite_conditions = [value for value in policy_state["jacobian_conditions"] if np.isfinite(value)]
        stats["joint_control"] = {
            "enabled": self.control_mode == "joint_position",
            "max_jacobian_condition": float(max(finite_conditions)) if finite_conditions else None,
            "mean_jacobian_condition": float(np.mean(finite_conditions)) if finite_conditions else None,
            "singular_condition_count": int(
                sum(not np.isfinite(value) for value in policy_state["jacobian_conditions"])
            ),
            "max_joint_target_step": float(max(policy_state["joint_target_step_norms"]))
            if policy_state["joint_target_step_norms"]
            else None,
            "mean_joint_target_step": float(np.mean(policy_state["joint_target_step_norms"]))
            if policy_state["joint_target_step_norms"]
            else None,
        }
        stats["initial_tripod_pos"] = stats["initial_tripod_pos"].tolist()
        stats["initial_tripod_mat"] = stats["initial_tripod_mat"].tolist()
        self.stats.append(stats)
        return bool(stats["policy_success"])

    def rollout(
        self,
        env,
        render=False,
        max_fr=None,
        motion_style=None,
        target_grasp_angle=None,
        full_quality_mode=False,
        post_reset_callback=None,
    ):
        env.reset()
        if post_reset_callback is not None:
            post_reset_callback(env)
        _, eef_quat = get_eef_pose(env)
        base_eef_mat = T.quat2mat(eef_quat)
        needle = needle_state(env)
        ring = ring_state(env)
        gripper_axes = calibrate_gripper_axes(base_eef_mat, needle)

        grasp_angle = float(target_grasp_angle) if target_grasp_angle is not None else self.rng.uniform(*self.grasp_angle_range)
        grasp_tilt_x = 0.0
        grasp_tilt_y = 0.0
        # Sample the lower handle region while retaining a margin from its edge.
        grasp_offset_along = self.rng.uniform(-0.0040, -0.0015)
        grasp_offset_lateral = self.rng.uniform(-0.0003, 0.0003)
        grasp_offset_vertical = self.rng.uniform(-0.0003, 0.0003)
        grasp_offset = (
            grasp_offset_along * needle["yaxis"]
            + grasp_offset_lateral * needle["xaxis"]
            + grasp_offset_vertical * needle["zaxis"]
        )
        grasp_pos = needle["handle_center"] + grasp_offset
        grasp_mat, grasp_to_gripper_axis, grasp_jaw_axis = construct_grasp_mat(
            base_eef_mat,
            needle,
            grasp_angle,
            gripper_axes,
        )
        grasp_quat = T.mat2quat(grasp_mat)
        measured_grasp_angle_value = visual_grasp_angle_from_axis(grasp_to_gripper_axis, needle)
        if full_quality_mode:
            pregrasp_height = self.rng.uniform(0.088, 0.115)
            descend_height = self.rng.uniform(0.026, 0.036)
        else:
            pregrasp_height = self.rng.uniform(0.105, 0.135)
            descend_height = self.rng.uniform(0.032, 0.044)
        # The Panda grip_site already represents the center between the fingers.
        # Adding a world-Z offset here systematically moves contact toward the
        # upper handle edge, especially for oblique grasps.
        close_height = 0.0
        close_pos = grasp_pos.copy()
        descend_pos = close_pos + grasp_to_gripper_axis * descend_height
        pregrasp_pos = close_pos + grasp_to_gripper_axis * pregrasp_height
        planned_needle_to_eef = {
            "pos": needle["mat"].T.dot(close_pos - needle["needle_center"]),
            "mat": needle["mat"].T.dot(grasp_mat),
        }
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
        if self.collision_aware_threading:
            # The randomized D0.5 tripod can put the entry side behind a support
            # relative to the robot. Clear the full tripod before moving around it.
            lift_height = max(lift_height, 0.20)
        if full_quality_mode:
            close_lift_rise = self.rng.uniform(0.006, 0.010)
            close_lift_forward = self.rng.uniform(0.002, 0.006) * toward_ring
        else:
            close_lift_rise = self.rng.uniform(0.003, 0.006)
            close_lift_forward = self.rng.uniform(0.0, 0.002) * toward_ring
        close_lift_end_pos = close_pos + close_lift_forward + np.array([0.0, 0.0, close_lift_rise])
        lift_jitter = np.array([self.rng.uniform(-0.012, 0.012), self.rng.uniform(-0.012, 0.012), 0.0])
        lift_xy = lift_toward + lift_lateral + lift_jitter
        lift_start_pos = close_lift_end_pos
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
        lift_control_pos = lift_start_pos + lift_control_xy + np.array([0.0, 0.0, lift_control_z])
        lift_control2_xy = (
            control2_toward * lift_xy
            + control2_side * lift_side
            + self.rng.uniform(-0.01, 0.012) * toward_ring
        )
        lift_control2_z = control2_z_frac * lift_height
        lift_control2_pos = lift_start_pos + lift_control2_xy + np.array([0.0, 0.0, lift_control2_z])
        lift_pos = lift_start_pos + lift_xy + np.array([0.0, 0.0, lift_height])
        lift_progress_power = float(np.clip(lift_progress_power, 0.55, 1.75))
        lift_yaw = 0.0
        lift_pitch = 0.0
        lift_roll = 0.0
        lift_mat = grasp_mat
        lift_quat = T.mat2quat(lift_mat)
        lift_align_rot = rotation_between(-needle["yaxis"], ring["normal"])
        lift_align_quat = T.mat2quat(lift_align_rot)
        lift_prealign_fraction = float(self.rng.uniform(0.28, 0.55))
        durations = dict(NOMINAL_STAGE_STEPS)
        durations["close_gripper_max"] = GRIPPER_CLOSE_MAX_STEPS
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
            "control_mode": self.control_mode,
            "action_representation": "absolute_joint_position" if self.control_mode == "joint_position" else "delta_eef_pose",
            "env_success": False,
            "policy_success": False,
            "failure_reason": "not_evaluated",
            "subgoal_durations": {},
            "stage_outcomes": {},
            "aborted_stage": None,
            "min_ring_distance": float("inf"),
            "max_insert_progress": -float("inf"),
            "final_insert_progress": -float("inf"),
            "clean_aperture_history": [],
            "tripod_displacement": 0.0,
            "final_tripod_displacement": 0.0,
            "tripod_rotation_deg": 0.0,
            "final_tripod_rotation_deg": 0.0,
            "initial_tripod_pos": tripod_position(env),
            "initial_tripod_mat": tripod_orientation(env),
            "gripper_closed": False,
            "gripper_command_closed": False,
            "grasp_contact_at_close": False,
            "grasp_contact_after_probe": False,
            "grasp_contact_after_lift": False,
            "grasp_rigid_after_probe": False,
            "grasp_rigid_after_lift": False,
            "hold_complete": False,
            "grasp_settled": False,
            "approach_complete": False,
            "probe_complete": False,
            "lift_complete": False,
            "align_complete": False,
            "pre_insert_complete": False,
            "insert_complete": False,
            "grasp_angle_deg": float(grasp_angle),
            "target_grasp_angle_deg": float(grasp_angle),
            "planned_grasp_angle_deg": float(measured_grasp_angle_value),
            "actual_close_angle_deg": None,
            "actual_lift_angle_deg": None,
            "actual_insert_angle_deg": None,
            "lift_angle_error_deg": None,
            "target_grasp_approach_angle_deg": float(grasp_angle),
            "grasp_approach_angle_deg": float(measured_grasp_angle_value),
            "gripper_approach_axis_name": gripper_axes["approach_axis_name"],
            "gripper_jaw_axis_name": gripper_axes["jaw_axis_name"],
            "planned_grasp_approach_axis": grasp_to_gripper_axis.tolist(),
            "planned_grasp_jaw_axis": grasp_jaw_axis.tolist(),
            "grasp_jaw_needle_axis_abs_dot": float(abs(np.dot(grasp_jaw_axis, needle["yaxis"]))),
            "planned_needle_to_eef": {
                "pos": planned_needle_to_eef["pos"].tolist(),
                "mat": planned_needle_to_eef["mat"].tolist(),
            },
            "planned_insert_normal": ring["normal"].tolist(),
            "insert_direction_preferred_y": -1.0,
            "insert_direction_valid": bool(ring["normal"][1] < 0.0),
            "grasp_tilt_x_deg": float(np.rad2deg(grasp_tilt_x)),
            "grasp_tilt_y_deg": float(np.rad2deg(grasp_tilt_y)),
            "grasp_offset_along": float(grasp_offset_along),
            "grasp_offset_lateral": float(grasp_offset_lateral),
            "grasp_offset_vertical": float(grasp_offset_vertical),
            "grasp_pos": grasp_pos.tolist(),
            "pregrasp_height": float(pregrasp_height),
            "descend_height": float(descend_height),
            "close_height": float(close_height),
            "pregrasp_pos": pregrasp_pos.tolist(),
            "descend_pos": descend_pos.tolist(),
            "close_pos": close_pos.tolist(),
            "close_lift_rise": float(close_lift_rise),
            "close_lift_forward": close_lift_forward.tolist(),
            "close_lift_end_pos": close_lift_end_pos.tolist(),
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
            "lift_prealign_fraction": float(lift_prealign_fraction),
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
            "full_quality_mode": bool(full_quality_mode),
            "collision_aware_threading": bool(self.collision_aware_threading),
        }

        # Follow one continuous approach curve instead of stopping at a
        # pregrasp waypoint and starting a second descend phase from rest.
        approach_start_pos, approach_start_quat = get_eef_pose(env)
        approach_steps = durations["aim_approach"] + durations["aim_descend"]
        high_angle_contact_mode = grasp_angle >= 108.0
        close_target_quat = grasp_quat

        def approach_target(progress):
            # Do not ease to zero velocity at the former pregrasp boundary or
            # before the short final verification window.
            path_t = progress
            target_pos = quadratic_bezier(approach_start_pos, pregrasp_pos, close_pos, path_t)
            if high_angle_contact_mode:
                orientation_progress = smoothstep((progress - 0.82) / 0.18)
            else:
                orientation_progress = smoothstep(min(1.0, progress / 0.7))
            target_quat = T.quat_slerp(approach_start_quat, grasp_quat, orientation_progress)
            return target_pos, target_quat

        if self.control_mode == "osc_pose":
            stats["approach_complete"] = self._run_bounded_trajectory_stage(
                env,
                approach_target,
                -1.0,
                approach_steps,
                policy_state,
                stats,
                "aim_continuous",
                lambda stage_env, target_pos, target_quat, motion: self._pose_completion(
                    stage_env,
                    target_pos,
                    target_quat,
                    motion,
                    position_tolerance=0.002,
                    orientation_tolerance_deg=2.0,
                    motion_tolerance=0.003,
                ),
                render,
                max_fr,
                max_steps=OSC_STAGE_MAX_STEPS["aim_continuous"],
            )
            approach_outcome = stats["stage_outcomes"]["aim_continuous"]
            stats["pregrasp_pose_settled"] = stats["approach_complete"]
            stats["grasp_settled"] = stats["approach_complete"]
            stats["grasp_settle"] = {
                "passed": bool(stats["approach_complete"]),
                "stable_steps": int(approach_outcome["stable_steps"]),
                "required_stable_steps": int(approach_outcome["required_stable_steps"]),
                "best_position_error": None
                if approach_outcome["final"] is None
                else float(approach_outcome["final"]["position_error_m"]),
                "best_orientation_error_deg": None
                if approach_outcome["final"] is None
                else float(approach_outcome["final"]["orientation_error_deg"]),
            }
            if not stats["approach_complete"]:
                stats["aborted_stage"] = "aim_continuous"
                return self._finish_rollout(env, stats, policy_state, gripper_axes)
        else:
            for i in range(approach_steps):
                target_pos, target_quat = approach_target((i + 1) / approach_steps)
                self._track_target(
                    env,
                    self._fixed_target(target_pos, target_quat),
                    -1.0,
                    1,
                    policy_state,
                    stats,
                    "aim_continuous",
                    render,
                    max_fr,
                    stop_on_reach=False,
                )

        # Closing is allowed only after the actual EEF pose has remained within
        # a tight tolerance for several consecutive control steps.
        if high_angle_contact_mode:
            _, close_target_quat = get_eef_pose(env)
        stats["high_angle_contact_mode"] = bool(high_angle_contact_mode)
        stats["close_target_angle_deg"] = float(
            visual_grasp_angle_from_axis(
                eef_approach_axis(T.quat2mat(close_target_quat), gripper_axes),
                needle_state(env),
            )
        )
        if self.control_mode != "osc_pose":
            stats["pregrasp_pose_settled"] = self._settle_at_grasp(
                env,
                close_pos,
                close_target_quat,
                policy_state,
                stats,
                render,
                max_fr,
            )
            stats["grasp_settled"] = stats["pregrasp_pose_settled"]

        # Require the rate-limited command, physical finger motion, and grasp
        # contact to remain complete together before starting the probe lift.
        if self.control_mode == "osc_pose":
            stats["gripper_close_complete"] = self._close_gripper_until_stable_contact(
                env,
                close_pos,
                close_target_quat,
                policy_state,
                stats,
                render,
                max_fr,
            )
            stats["gripper_command_closed"] = stats["gripper_close"]["command_fully_closed"]
        else:
            for _ in range(durations["close_gripper"]):
                self._track_target(
                    env,
                    self._fixed_target(close_pos, close_target_quat),
                    1.0,
                    1,
                    policy_state,
                    stats,
                    "close_gripper",
                    render,
                    max_fr,
                    stop_on_reach=False,
                )
            stats["gripper_close_complete"] = True
            stats["gripper_command_closed"] = True
        stats["gripper_qpos_at_close"] = gripper_joint_positions(env).tolist()
        stats["grasp_contact_at_close"] = needle_grasp_contact(env)
        if self.control_mode == "osc_pose" and not stats["gripper_close_complete"]:
            stats["aborted_stage"] = "close_gripper"
            return self._finish_rollout(env, stats, policy_state, gripper_axes)
        close_grasp_transform = needle_to_eef_transform(env)
        close_needle_center = needle_state(env)["needle_center"].copy()

        def probe_target(progress):
            target_pos = close_pos + smoothstep(progress) * (close_lift_end_pos - close_pos)
            return target_pos, close_target_quat

        def probe_completion(stage_env, target_pos, target_quat, motion):
            pose = self._pose_completion(
                stage_env,
                target_pos,
                target_quat,
                motion,
                position_tolerance=0.003,
                orientation_tolerance_deg=3.0,
                motion_tolerance=0.003,
            )
            measured_transform = needle_to_eef_transform(stage_env)
            transform_error = needle_to_eef_error(close_grasp_transform, measured_transform)
            needle_displacement = float(
                np.linalg.norm(needle_state(stage_env)["needle_center"] - close_needle_center)
            )
            contact = needle_grasp_contact(stage_env)
            tripod_motion = self._tripod_motion(stage_env, stats)
            pose.update(
                {
                    "contact": bool(contact),
                    "needle_displacement_m": needle_displacement,
                    "transform_translation_error_m": float(transform_error["translation_m"]),
                    "transform_rotation_error_deg": float(transform_error["rotation_deg"]),
                    "tripod_motion": tripod_motion,
                    "tripod_stable": bool(tripod_motion["stable"]),
                }
            )
            pose["passed"] = bool(
                pose["passed"]
                and contact
                and tripod_motion["stable"]
                and needle_displacement > 0.003
                and transform_error["translation_m"] < 0.006
                and transform_error["rotation_deg"] < 8.0
            )
            return pose

        if self.control_mode == "osc_pose":
            stats["probe_complete"] = self._run_bounded_trajectory_stage(
                env,
                probe_target,
                1.0,
                durations["grasp_probe_lift"],
                policy_state,
                stats,
                "grasp_probe_lift",
                probe_completion,
                render,
                max_fr,
                max_steps=OSC_STAGE_MAX_STEPS["grasp_probe_lift"],
            )
        else:
            for i in range(durations["grasp_probe_lift"]):
                target_pos, target_quat = probe_target((i + 1) / durations["grasp_probe_lift"])
                self._track_target(
                    env,
                    self._fixed_target(target_pos, target_quat),
                    1.0,
                    1,
                    policy_state,
                    stats,
                    "grasp_probe_lift",
                    render,
                    max_fr,
                    stop_on_reach=False,
                )
            stats["probe_complete"] = True
        probe_grasp_transform = needle_to_eef_transform(env)
        probe_grasp_error = needle_to_eef_error(close_grasp_transform, probe_grasp_transform)
        stats["grasp_contact_after_probe"] = needle_grasp_contact(env)
        stats["gripper_qpos_after_probe"] = gripper_joint_positions(env).tolist()
        stats["grasp_probe_needle_displacement_m"] = float(
            np.linalg.norm(needle_state(env)["needle_center"] - close_needle_center)
        )
        stats["grasp_transform_error_after_probe"] = probe_grasp_error
        stats["grasp_rigid_after_probe"] = bool(
            probe_grasp_error["translation_m"] < 0.006
            and probe_grasp_error["rotation_deg"] < 8.0
            and stats["grasp_probe_needle_displacement_m"] > 0.003
        )
        stats["gripper_closed"] = bool(stats["grasp_contact_after_probe"])
        close_needle = needle_state(env)
        stats["actual_close_angle_deg"] = float(
            visual_grasp_angle_from_axis(eef_approach_axis(get_eef_mat(env), gripper_axes), close_needle)
        )
        if self.control_mode == "osc_pose" and not stats["probe_complete"]:
            stats["aborted_stage"] = "grasp_probe_lift"
            return self._finish_rollout(env, stats, policy_state, gripper_axes)

        # lift: follow one continuous randomized arc and pre-rotate partway toward
        # the ring while moving, leaving less rotation for the align phase.
        def lift_target(progress):
            lift_t = smoothstep(progress**lift_progress_power)
            target_pos = cubic_bezier(lift_start_pos, lift_control_pos, lift_control2_pos, lift_pos, lift_t)
            rot_fraction = lift_prealign_fraction * smoothstep(progress)
            partial_lift_quat = T.quat_slerp(T.mat2quat(np.eye(3)), lift_align_quat, rot_fraction)
            partial_lift_rot = T.quat2mat(partial_lift_quat)
            target_quat = T.mat2quat(partial_lift_rot.dot(T.quat2mat(close_target_quat)))
            return target_pos, target_quat

        def lift_completion(stage_env, target_pos, target_quat, motion):
            pose = self._pose_completion(
                stage_env,
                target_pos,
                target_quat,
                motion,
                position_tolerance=0.006,
                orientation_tolerance_deg=4.0,
                motion_tolerance=0.005,
            )
            transform_error = needle_to_eef_error(close_grasp_transform, needle_to_eef_transform(stage_env))
            contact = needle_grasp_contact(stage_env)
            tripod_motion = self._tripod_motion(stage_env, stats)
            pose.update(
                {
                    "contact": bool(contact),
                    "transform_translation_error_m": float(transform_error["translation_m"]),
                    "transform_rotation_error_deg": float(transform_error["rotation_deg"]),
                    "tripod_motion": tripod_motion,
                    "tripod_stable": bool(tripod_motion["stable"]),
                }
            )
            pose["passed"] = bool(
                pose["passed"]
                and contact
                and tripod_motion["stable"]
                and transform_error["translation_m"] < 0.008
                and transform_error["rotation_deg"] < 12.0
            )
            return pose

        if self.control_mode == "osc_pose":
            stats["lift_complete"] = self._run_bounded_trajectory_stage(
                env,
                lift_target,
                1.0,
                durations["lift_arc"],
                policy_state,
                stats,
                "lift_arc",
                lift_completion,
                render,
                max_fr,
                required_stable_steps=2,
                start_at_zero=True,
                max_steps=OSC_STAGE_MAX_STEPS["lift_arc"],
            )
        else:
            for i in range(durations["lift_arc"]):
                progress = i / max(1, durations["lift_arc"] - 1)
                target_pos, target_quat = lift_target(progress)
                self._track_target(
                    env,
                    self._fixed_target(target_pos, target_quat),
                    1.0,
                    1,
                    policy_state,
                    stats,
                    "lift_arc",
                    render,
                    max_fr,
                )
            stats["lift_complete"] = True
        lift_needle = needle_state(env)
        actual_lift_angle = visual_grasp_angle_from_axis(eef_approach_axis(get_eef_mat(env), gripper_axes), lift_needle)
        stats["actual_lift_angle_deg"] = float(actual_lift_angle)
        stats["lift_angle_error_deg"] = float(actual_lift_angle - grasp_angle)

        measured_needle_to_eef = needle_to_eef_transform(env)
        lift_grasp_error = needle_to_eef_error(close_grasp_transform, measured_needle_to_eef)
        stats["measured_needle_to_eef_after_lift"] = {
            "pos": measured_needle_to_eef["pos"].tolist(),
            "mat": measured_needle_to_eef["mat"].tolist(),
        }
        stats["planned_to_measured_grasp_error"] = needle_to_eef_error(
            planned_needle_to_eef,
            measured_needle_to_eef,
        )
        stats["grasp_transform_error_after_lift"] = lift_grasp_error
        stats["grasp_contact_after_lift"] = needle_grasp_contact(env)
        stats["gripper_qpos_after_lift"] = gripper_joint_positions(env).tolist()
        stats["grasp_rigid_after_lift"] = bool(
            stats["grasp_rigid_after_probe"]
            and lift_grasp_error["translation_m"] < 0.008
            and lift_grasp_error["rotation_deg"] < 12.0
        )
        stats["actual_grasp_jaw_needle_axis_abs_dot"] = float(
            abs(np.dot(eef_jaw_axis(get_eef_mat(env), gripper_axes), lift_needle["yaxis"]))
        )
        if self.control_mode == "osc_pose" and not stats["lift_complete"]:
            stats["aborted_stage"] = "lift_arc"
            return self._finish_rollout(env, stats, policy_state, gripper_axes)
        align_reference_needle_mat = lift_needle["mat"].copy()
        alignment_start = ring_frame_offsets(lift_needle["tip"], ring_state(env))
        stats["alignment_profile"] = "measured_minimum_jerk"
        stats["alignment_start_offsets"] = dict(alignment_start)
        if self.collision_aware_threading:
            # First move above the entry side, then descend while still well
            # outside the ring. This avoids sweeping through a rotated support.
            self._track_target(
                env,
                lambda: self._alignment_target(
                    -0.075,
                    vertical_offset=0.080,
                    rotation_fraction=1.0,
                    needle_to_eef=measured_needle_to_eef,
                    reference_needle_mat=align_reference_needle_mat,
                )(env),
                1.0,
                100,
                policy_state,
                stats,
                "safe_stage_above",
                render,
                max_fr,
                min_steps=24,
            )
            if self._tripod_motion(env, stats, max_displacement=0.012)["stable"]:
                self._track_target(
                    env,
                    lambda: self._alignment_target(
                        -0.075,
                        vertical_offset=0.025,
                        rotation_fraction=1.0,
                        needle_to_eef=measured_needle_to_eef,
                        reference_needle_mat=align_reference_needle_mat,
                    )(env),
                    1.0,
                    80,
                    policy_state,
                    stats,
                    "safe_stage_entry",
                    render,
                    max_fr,
                    min_steps=20,
                )

        # Start from the measured post-lift tip pose and blend monotonically to
        # the pre-insertion pose. This avoids a one-off target reversal when the
        # lift hands control to alignment.
        align_end_offset = pre_insert_offset - 0.012
        def align_target(progress):
            blend = minimum_jerk(progress)
            offset = (1.0 - blend) * alignment_start["normal"] + blend * align_end_offset
            lateral = (1.0 - blend) * alignment_start["lateral"] + blend * align_curve["lateral_end"]
            vertical = (1.0 - blend) * alignment_start["vertical"] + blend * align_curve["vertical_end"]
            twist = blend * align_curve["twist_end"]
            tilt = blend * align_curve["tilt_end"]
            target_fn = self._alignment_target(
                offset,
                lateral_offset=lateral,
                vertical_offset=vertical,
                twist=twist,
                tilt=tilt,
                rotation_fraction=blend,
                needle_to_eef=measured_needle_to_eef,
                reference_needle_mat=align_reference_needle_mat,
            )
            return target_fn(env)

        def align_completion(stage_env, target_pos, target_quat, motion):
            pose = self._pose_completion(
                stage_env,
                target_pos,
                target_quat,
                motion,
                position_tolerance=0.006,
                orientation_tolerance_deg=4.0,
                motion_tolerance=0.005,
            )
            tracking = self._needle_target_errors(
                stage_env,
                align_end_offset,
                align_curve["lateral_end"],
                align_curve["vertical_end"],
            )
            contact = needle_grasp_contact(stage_env)
            tripod_motion = self._tripod_motion(stage_env, stats)
            pose.update(
                {
                    "needle_tracking": tracking,
                    "contact": bool(contact),
                    "tripod_motion": tripod_motion,
                    "tripod_stable": bool(tripod_motion["stable"]),
                }
            )
            pose["passed"] = bool(
                pose["passed"]
                and contact
                and tripod_motion["stable"]
                and tracking["normal_error_m"] < 0.008
                and tracking["tangent_error_m"] < 0.004
                and tracking["vertical_error_m"] < 0.004
                and tracking["orientation_error_deg"] < 5.0
            )
            return pose

        if self.control_mode == "osc_pose":
            stats["align_complete"] = self._run_bounded_trajectory_stage(
                env,
                align_target,
                1.0,
                durations["align"],
                policy_state,
                stats,
                "align",
                align_completion,
                render,
                max_fr,
                required_stable_steps=2,
                start_at_zero=True,
                max_steps=OSC_STAGE_MAX_STEPS["align"],
            )
            if not stats["align_complete"]:
                stats["aborted_stage"] = "align"
                return self._finish_rollout(env, stats, policy_state, gripper_axes)
        else:
            for i in range(durations["align"]):
                progress = i / max(1, durations["align"] - 1)
                target_pos, target_quat = align_target(progress)
                self._track_target(
                    env,
                    self._fixed_target(target_pos, target_quat),
                    1.0,
                    1,
                    policy_state,
                    stats,
                    "align",
                    render,
                    max_fr,
                )
            stats["align_complete"] = True
        if self.collision_aware_threading:
            stable_steps = 0
            gate_errors = None
            gate_steps = 0
            for _ in range(70):
                gate_steps += 1
                self._track_target(
                    env,
                    lambda: self._curved_alignment_target(
                        pre_insert_offset,
                        1.0,
                        align_curve,
                        measured_needle_to_eef,
                        align_reference_needle_mat,
                    )(env),
                    1.0,
                    1,
                    policy_state,
                    stats,
                    "pre_insert_gate",
                    render,
                    max_fr,
                    stop_on_reach=False,
                )
                gate_errors = self._needle_target_errors(env, pre_insert_offset)
                gate_tripod_motion = self._tripod_motion(env, stats, max_displacement=0.012)
                within_gate = (
                    gate_errors["normal_error_m"] < 0.008
                    and gate_errors["tangent_error_m"] < 0.004
                    and gate_errors["vertical_error_m"] < 0.004
                    and gate_errors["orientation_error_deg"] < 5.0
                    and gate_tripod_motion["stable"]
                )
                stable_steps = stable_steps + 1 if within_gate else 0
                if stable_steps >= 4 or not gate_tripod_motion["stable"]:
                    break
            stats["pre_insert_gate"] = {
                "passed": bool(stable_steps >= 4),
                "stable_steps": int(stable_steps),
                "steps": int(gate_steps),
                "errors": gate_errors,
                "tripod_motion": gate_tripod_motion,
            }
            stats["pre_insert_complete"] = bool(stable_steps >= 4)
        else:
            def pre_insert_target(progress):
                return self._curved_alignment_target(
                    pre_insert_offset,
                    1.0,
                    align_curve,
                    measured_needle_to_eef,
                    align_reference_needle_mat,
                )(env)

            def pre_insert_completion(stage_env, target_pos, target_quat, motion):
                pose = self._pose_completion(
                    stage_env,
                    target_pos,
                    target_quat,
                    motion,
                    position_tolerance=0.005,
                    orientation_tolerance_deg=3.5,
                    motion_tolerance=0.003,
                )
                tracking = self._needle_target_errors(stage_env, pre_insert_offset)
                contact = needle_grasp_contact(stage_env)
                tripod_motion = self._tripod_motion(stage_env, stats)
                pose.update(
                    {
                        "needle_tracking": tracking,
                        "contact": bool(contact),
                        "tripod_motion": tripod_motion,
                        "tripod_stable": bool(tripod_motion["stable"]),
                    }
                )
                pose["passed"] = bool(
                    pose["passed"]
                    and contact
                    and tripod_motion["stable"]
                    and tracking["normal_error_m"] < 0.006
                    and tracking["tangent_error_m"] < 0.003
                    and tracking["vertical_error_m"] < 0.003
                    and tracking["orientation_error_deg"] < 4.0
                )
                return pose

            if self.control_mode == "osc_pose":
                stats["pre_insert_complete"] = self._run_bounded_trajectory_stage(
                    env,
                    pre_insert_target,
                    1.0,
                    durations["pre_insert"],
                    policy_state,
                    stats,
                    "pre_insert",
                    pre_insert_completion,
                    render,
                    max_fr,
                    required_stable_steps=3,
                    completion_start_step=1,
                    max_steps=OSC_STAGE_MAX_STEPS["pre_insert"],
                )
            else:
                self._track_target(
                    env,
                    lambda: pre_insert_target(1.0),
                    1.0,
                    durations["pre_insert"],
                    policy_state,
                    stats,
                    "pre_insert",
                    render,
                    max_fr,
                )
                stats["pre_insert_complete"] = True

        if self.control_mode == "osc_pose" and not stats["pre_insert_complete"]:
            stats["aborted_stage"] = "pre_insert"
            return self._finish_rollout(env, stats, policy_state, gripper_axes)

        # insert_through: keep inserting past sparse env success instead of terminating early.
        insert_offsets = np.linspace(insert_start_offset, insert_end_offset, durations["insert_through"])
        gate_passed = (
            not self.collision_aware_threading
            or bool(stats.get("pre_insert_gate", {}).get("passed", False))
        )

        def insert_target(progress):
            offset = insert_start_offset + progress * (insert_end_offset - insert_start_offset)
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
            return self._curved_alignment_target(
                offset,
                progress,
                insert_curve,
                measured_needle_to_eef,
                align_reference_needle_mat,
                alignment_fraction=1.0,
            )(env)

        if self.control_mode == "osc_pose" and not self.collision_aware_threading:
            def insert_completion(stage_env, target_pos, target_quat, motion):
                pose = self._pose_completion(
                    stage_env,
                    target_pos,
                    target_quat,
                    motion,
                    position_tolerance=0.008,
                    orientation_tolerance_deg=5.0,
                    motion_tolerance=0.005,
                )
                tracking = self._needle_target_errors(stage_env, insert_end_offset)
                geometry = clean_ring_aperture_geometry(stage_env)
                current_progress = insertion_progress(needle_state(stage_env), ring_state(stage_env))
                contact = needle_grasp_contact(stage_env)
                tripod_motion = self._tripod_motion(stage_env, stats)
                pose.update(
                    {
                        "needle_tracking": tracking,
                        "clean_aperture": bool(geometry["clear"]),
                        "insertion_progress_m": float(current_progress),
                        "contact": bool(contact),
                        "tripod_motion": tripod_motion,
                        "tripod_stable": bool(tripod_motion["stable"]),
                    }
                )
                pose["passed"] = bool(
                    pose["passed"]
                    and geometry["clear"]
                    and stats["max_insert_progress"] > 0.026
                    and current_progress > 0.014
                    and contact
                    and tripod_motion["stable"]
                    and tracking["normal_error_m"] < 0.010
                    and tracking["tangent_error_m"] < 0.003
                    and tracking["vertical_error_m"] < 0.005
                    and tracking["orientation_error_deg"] < 4.0
                )
                return pose

            stats["insert_complete"] = self._run_bounded_trajectory_stage(
                env,
                insert_target,
                1.0,
                durations["insert_through"],
                policy_state,
                stats,
                "insert_through",
                insert_completion,
                render,
                max_fr,
                required_stable_steps=2,
                start_at_zero=True,
                max_steps=OSC_STAGE_MAX_STEPS["insert_through"],
            )
            insert_steps = stats["stage_outcomes"]["insert_through"]["actual_steps"]
            insert_index = len(insert_offsets) if stats["insert_complete"] else len(insert_offsets) - 1
            stats["closed_loop_insert"] = {
                "enabled": False,
                "completed": bool(stats["insert_complete"]),
                "steps": int(insert_steps),
                "offsets_completed": int(insert_index),
                "offsets_total": int(len(insert_offsets)),
                "tripod_abort": False,
            }
        else:
            insert_index = 0
            insert_steps = 0
            max_insert_steps = len(insert_offsets) * (5 if self.collision_aware_threading else 1)
            while gate_passed and insert_index < len(insert_offsets) and insert_steps < max_insert_steps:
                offset = insert_offsets[insert_index]
                progress = insert_index / max(1, len(insert_offsets) - 1)
                target_pos, target_quat = insert_target(progress)
                self._track_target(
                    env,
                    self._fixed_target(target_pos, target_quat),
                    1.0,
                    1,
                    policy_state,
                    stats,
                    "insert_through",
                    render,
                    max_fr,
                )
                insert_steps += 1
                if self.collision_aware_threading:
                    tracking = self._needle_target_errors(env, offset)
                    stats["closed_loop_insert_last_tracking"] = tracking
                    near_aperture = offset >= -0.015
                    well_tracked = (
                        tracking["normal_error_m"] < (0.007 if near_aperture else 0.010)
                        and tracking["tangent_error_m"] < (0.002 if near_aperture else 0.004)
                        and tracking["vertical_error_m"] < (0.002 if near_aperture else 0.004)
                        and tracking["orientation_error_deg"] < (2.5 if near_aperture else 6.0)
                    )
                    if not self._tripod_motion(env, stats, max_displacement=0.012)["stable"]:
                        break
                    if well_tracked:
                        insert_index += 1
                else:
                    insert_index += 1
            stats["insert_complete"] = bool(insert_index >= len(insert_offsets))
            stats["closed_loop_insert"] = {
                "enabled": bool(self.collision_aware_threading),
                "completed": bool(stats["insert_complete"]),
                "steps": int(insert_steps),
                "offsets_completed": int(insert_index),
                "offsets_total": int(len(insert_offsets)),
                "tripod_abort": bool(
                    self.collision_aware_threading
                    and not self._tripod_motion(env, stats, max_displacement=0.012)["stable"]
                ),
            }

        if self.control_mode == "osc_pose" and not stats["insert_complete"]:
            stats["aborted_stage"] = "insert_through"
            return self._finish_rollout(env, stats, policy_state, gripper_axes)

        # hold_after_insert: keep moving very slightly through the ring instead of
        # freezing into a zero-action terminal pause.
        if insert_index >= len(insert_offsets):
            hold_pos, hold_quat = get_eef_pose(env)
            hold_ring = ring_state(env)
            hold_drift = self.rng.uniform(0.0015, 0.004) * hold_ring["normal"]

            def hold_target(progress):
                target_pos = hold_pos + smoothstep(progress) * hold_drift
                return target_pos, hold_quat

            def hold_completion(stage_env, target_pos, target_quat, motion):
                pose = self._pose_completion(
                    stage_env,
                    target_pos,
                    target_quat,
                    motion,
                    # The commanded drift is only 1.5-4 mm, so this must be
                    # tighter than the shortest drift to prevent an immediate pass.
                    position_tolerance=0.0015,
                    orientation_tolerance_deg=4.0,
                    motion_tolerance=0.003,
                )
                geometry = clean_ring_aperture_geometry(stage_env)
                current_progress = insertion_progress(needle_state(stage_env), ring_state(stage_env))
                contact = needle_grasp_contact(stage_env)
                tripod_motion = self._tripod_motion(stage_env, stats)
                pose.update(
                    {
                        "clean_aperture": bool(geometry["clear"]),
                        "insertion_progress_m": float(current_progress),
                        "contact": bool(contact),
                        "tripod_motion": tripod_motion,
                        "tripod_stable": bool(tripod_motion["stable"]),
                    }
                )
                pose["passed"] = bool(
                    pose["passed"]
                    and geometry["clear"]
                    and current_progress > 0.014
                    and contact
                    and tripod_motion["stable"]
                )
                return pose

            if self.control_mode == "osc_pose":
                stats["hold_complete"] = self._run_bounded_trajectory_stage(
                    env,
                    hold_target,
                    1.0,
                    durations["hold_after_insert"],
                    policy_state,
                    stats,
                    "hold_after_insert",
                    hold_completion,
                    render,
                    max_fr,
                    start_at_zero=True,
                    max_steps=OSC_STAGE_MAX_STEPS["hold_after_insert"],
                )
            else:
                for i in range(durations["hold_after_insert"]):
                    progress = i / max(1, durations["hold_after_insert"] - 1)
                    target_pos, target_quat = hold_target(progress)
                    self._track_target(
                        env,
                        self._fixed_target(target_pos, target_quat),
                        1.0,
                        1,
                        policy_state,
                        stats,
                        "hold_after_insert",
                        render,
                        max_fr,
                        stop_on_reach=False,
                    )
                stats["hold_complete"] = True
        if self.control_mode == "osc_pose" and not stats["hold_complete"]:
            stats["aborted_stage"] = "hold_after_insert"
        return self._finish_rollout(env, stats, policy_state, gripper_axes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Threading")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--directory", type=str, default=str(REPO_ROOT / "threading_scripted_demos_grasp_angle"))
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--control-mode",
        choices=("osc_pose", "joint_position"),
        default="osc_pose",
        help="Use the existing delta OSC pose controller or damped IK feeding absolute joint-position targets.",
    )
    parser.add_argument("--grasp-angle-min", type=float, default=80.0)
    parser.add_argument("--grasp-angle-max", type=float, default=120.0)
    parser.add_argument(
        "--grasp-angle-list",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-kept-demo target grasp angles for debugging actual-vs-target calibration.",
    )
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
    parser.add_argument(
        "--collect-insert-angle-split",
        action="store_true",
        help="Keep only successful demos until both insert-angle buckets are full.",
    )
    parser.add_argument(
        "--collect-insert-angle-range",
        action="store_true",
        help="Keep only successful demos whose actual insert angle is inside [--insert-angle-min, --insert-angle-max].",
    )
    parser.add_argument(
        "--full-quality-mode",
        action="store_true",
        help="For full-only insert-angle-range collection, reduce extreme approach / pause artifacts without changing success filters.",
    )
    parser.add_argument(
        "--collision-aware-threading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use staged D0 alignment, a pre-insertion gate, and closed-loop insertion tracking.",
    )
    parser.add_argument("--insert-angle-threshold", type=float, default=95.0)
    parser.add_argument("--insert-angle-per-bin", type=int, default=25)
    parser.add_argument("--insert-angle-min", type=float, default=105.0)
    parser.add_argument("--insert-angle-max", type=float, default=130.0)
    parser.add_argument(
        "--balanced-insert-angle-range",
        action="store_true",
        help="For --collect-insert-angle-range, fill evenly spaced actual insert-angle bins instead of accepting the first in-range demos.",
    )
    parser.add_argument(
        "--range-use-balanced-motion-styles",
        action="store_true",
        help=(
            "For --collect-insert-angle-range, use the default balanced motion-style "
            "schedule instead of the range-specific high-angle style sampler."
        ),
    )
    parser.add_argument("--insert-angle-num-bins", type=int, default=5)
    parser.add_argument(
        "--insert-angle-edges",
        nargs="+",
        type=float,
        default=None,
        help="Custom balanced actual-angle bin edges, e.g. 90 97 100. Overrides min/max/num-bins.",
    )
    parser.add_argument(
        "--split-use-geometric-success",
        action="store_true",
        help="For insert-angle split collection, keep demos that pass geometric insertion checks even if env sparse success is false.",
    )
    parser.add_argument(
        "--smooth-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discard successful trajectories with action spikes or tripod shake.",
    )
    parser.add_argument("--max-action-delta", type=float, default=0.18)
    parser.add_argument("--max-action-jerk", type=float, default=0.095)
    parser.add_argument("--max-mean-action-delta", type=float, default=0.04)
    parser.add_argument("--max-tripod-displacement", type=float, default=0.02)
    parser.add_argument("--max-tripod-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--record-joint-training-fields",
        action="store_true",
        help=(
            "Save current robot0_joint_pos, absolute joint target, raw joint delta, "
            "and normalized joint-delta action labels in each action info."
        ),
    )
    parser.add_argument(
        "--joint-delta-scale",
        type=float,
        default=0.05,
        help="Joint delta in radians corresponding to normalized delta magnitude 1.",
    )
    return parser.parse_args()


def make_motion_style_plan(rng, num_demos):
    plan = []
    while len(plan) < num_demos:
        cycle = list(MOTION_STYLES)
        rng.shuffle(cycle)
        plan.extend(cycle)
    return plan[:num_demos]


def choose_split_target_angle(rng, split_counts, per_bin):
    low_deficit = max(0, per_bin - split_counts["lt"])
    high_deficit = max(0, per_bin - split_counts["ge"])
    if low_deficit <= 0 and high_deficit <= 0:
        return None
    if split_counts["lt"] + 5 < split_counts["ge"] and low_deficit > 0:
        desired_high = rng.rand() < 0.12
    elif split_counts["ge"] + 5 < split_counts["lt"] and high_deficit > 0:
        desired_high = rng.rand() < 0.88
    elif low_deficit > 0 and high_deficit > 0:
        desired_high = rng.rand() < high_deficit / float(low_deficit + high_deficit)
    else:
        desired_high = high_deficit > 0

    # These are target approach angles, not acceptance bins. They are intentionally
    # spread out so the accepted insert angles do not collapse to one value.
    if desired_high:
        return float(rng.choice([98.0, 104.0, 110.0, 118.0, 126.0, 134.0]))
    candidates = np.array([84.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0], dtype=float)
    weights = np.array([0.04, 0.10, 0.18, 0.17, 0.14, 0.15, 0.15, 0.07], dtype=float)
    return float(rng.choice(candidates, p=weights / weights.sum()))


def choose_large_insert_target_angle(rng):
    # These are target grasp approach angles, not the measured insert angles.
    # Higher targets bias the wrist camera / gripper pose toward larger measured
    # insert angles, but the final acceptance gate still uses actual_insert_angle_deg.
    candidates = np.array([104.0, 110.0, 118.0, 126.0, 134.0, 142.0], dtype=float)
    weights = np.array([0.08, 0.16, 0.25, 0.25, 0.18, 0.08], dtype=float)
    return float(rng.choice(candidates, p=weights / weights.sum()))


def choose_full_quality_target_angle(rng):
    # Softer full-only target distribution: still biased toward actual insert
    # angle >= 95, but avoids making every accepted trajectory an extreme slanted grasp.
    candidates = np.array([96.0, 98.0, 100.0, 104.0, 110.0, 118.0], dtype=float)
    weights = np.array([0.12, 0.20, 0.22, 0.20, 0.16, 0.10], dtype=float)
    return float(rng.choice(candidates, p=weights / weights.sum()))


def choose_large_insert_motion_style(rng):
    styles = np.array(
        [
            "high_arc",
            "vertical_first",
            "side_sweep",
            "shallow_sweep",
            "early_approach",
            "low_s_curve",
            "direct_low",
            "short_direct",
            "delayed_approach",
            "over_then_back",
        ],
        dtype=object,
    )
    weights = np.array([0.18, 0.17, 0.16, 0.13, 0.11, 0.10, 0.06, 0.04, 0.03, 0.02])
    return str(rng.choice(styles, p=weights / weights.sum()))


def choose_full_quality_motion_style(rng):
    # Keep the same style vocabulary as the main policy, but avoid over-sampling
    # the high-arc / vertical-first modes that made full demos look samey.
    styles = np.array(
        [
            "direct_low",
            "short_direct",
            "low_s_curve",
            "shallow_sweep",
            "early_approach",
            "delayed_approach",
            "side_sweep",
            "over_then_back",
            "vertical_first",
            "high_arc",
        ],
        dtype=object,
    )
    weights = np.array([0.15, 0.15, 0.14, 0.13, 0.12, 0.11, 0.09, 0.06, 0.03, 0.02])
    return str(rng.choice(styles, p=weights / weights.sum()))


def choose_split_motion_style(rng, split_counts, per_bin):
    if split_counts["lt"] + 5 < split_counts["ge"] and split_counts["lt"] < per_bin:
        styles = np.array(
            [
                "short_direct",
                "over_then_back",
                "delayed_approach",
                "direct_low",
                "low_s_curve",
                "vertical_first",
                "shallow_sweep",
                "side_sweep",
                "high_arc",
                "early_approach",
            ],
            dtype=object,
        )
        weights = np.array([0.22, 0.18, 0.14, 0.13, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02])
        return str(rng.choice(styles, p=weights / weights.sum()))
    if split_counts["ge"] + 5 < split_counts["lt"] and split_counts["ge"] < per_bin:
        return choose_large_insert_motion_style(rng)
    return str(rng.choice(MOTION_STYLES))


def insert_angle_bucket(angle, threshold):
    if angle is None or not np.isfinite(angle):
        return None
    if angle < 0.0 or angle > 180.0:
        return None
    return "lt" if angle < threshold else "ge"


def insert_angle_in_range(angle, min_angle, max_angle):
    if angle is None or not np.isfinite(angle):
        return False
    return min_angle <= float(angle) <= max_angle


def make_range_angle_edges(min_angle, max_angle, num_bins):
    if num_bins <= 0:
        raise ValueError("--insert-angle-num-bins must be positive")
    if max_angle <= min_angle:
        raise ValueError("--insert-angle-max must be greater than --insert-angle-min")
    return np.linspace(float(min_angle), float(max_angle), int(num_bins) + 1)


def range_angle_bucket(angle, edges):
    if angle is None or not np.isfinite(angle):
        return None
    angle = float(angle)
    if angle < edges[0] or angle > edges[-1]:
        return None
    for idx, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if low <= angle < high or (idx == len(edges) - 2 and angle <= high):
            return idx
    return None


def range_angle_counts_label(edges, counts):
    return {
        f"{edges[idx]:.1f}_{edges[idx + 1]:.1f}": int(counts.get(idx, 0))
        for idx in range(len(edges) - 1)
    }


def choose_balanced_range_bin(rng, counts, per_bin, num_bins):
    deficits = np.array([max(0, per_bin - counts.get(idx, 0)) for idx in range(num_bins)], dtype=float)
    if deficits.sum() <= 0:
        return None
    return int(rng.choice(np.arange(num_bins), p=deficits / deficits.sum()))


def choose_balanced_range_target_angle(rng, edges, target_bin):
    low = float(edges[target_bin])
    high = float(edges[target_bin + 1])
    center = 0.5 * (low + high)
    if high <= 105.0:
        # Around the near-vertical regime, the measured insert angle is
        # consistently about 1--2 degrees above the commanded grasp angle.
        # Sample across the accepted interval with that calibration offset.
        margin = min(0.35, 0.1 * (high - low))
        return float(rng.uniform(low + margin - 1.5, high - margin - 1.5))
    if target_bin == len(edges) - 2 and low >= 110.0:
        # Empirically, aiming the gripper too steeply for the highest accepted
        # insert-angle bin often flips the actual insert angle back to ~70-85 deg
        # or overshoots beyond the accepted range. A moderate 115-119 deg target
        # produces the cleanest 116-123 deg full-quality candidates.
        return float(rng.uniform(max(low - 1.5, 114.5), min(high - 3.5, 119.5)))
    # Target grasp angle is only a bias for the measured insert angle, so use a
    # slightly higher target for high bins and keep the lower bins less extreme.
    bias = np.interp(center, [edges[0], edges[-1]], [0.5, 7.0])
    jitter = rng.uniform(-0.35, 0.35) * (high - low)
    return float(np.clip(center + bias + jitter, 92.0, 142.0))


def insert_angle_subbin(angle, threshold):
    if angle < threshold:
        edges = [0.0, 70.0, 80.0, 90.0, threshold]
        prefix = "lt"
    else:
        edges = [threshold, 105.0, 115.0, 130.0, 180.0]
        prefix = "ge"
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= angle < high or (high == edges[-1] and angle <= high):
            return f"{prefix}_{low:.0f}_{high:.0f}"
    return f"{prefix}_other"


def prune_and_count_existing_split(directory, threshold, per_bin):
    counts = {"lt": 0, "ge": 0}
    subbins = {}
    directory = Path(os.path.abspath(os.path.expanduser(directory)))
    if not directory.exists():
        return counts, subbins, 0

    for ep_dir in sorted(directory.glob("ep_*")):
        stats_path = ep_dir / "policy_stats.json"
        keep = False
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            bucket = insert_angle_bucket(stats.get("actual_insert_angle_deg"), threshold)
            collection_success = bool(stats.get("success") or stats.get("collection_success"))
            if collection_success and bucket is not None and counts[bucket] < per_bin:
                keep = True
                counts[bucket] += 1
                subbin = insert_angle_subbin(stats["actual_insert_angle_deg"], threshold)
                subbins[subbin] = subbins.get(subbin, 0) + 1
        if not keep:
            shutil.rmtree(ep_dir, ignore_errors=True)
    return counts, subbins, counts["lt"] + counts["ge"]


def prune_and_count_existing_range(directory, edges, per_bin):
    counts = {idx: 0 for idx in range(len(edges) - 1)}
    directory = Path(os.path.abspath(os.path.expanduser(directory)))
    if not directory.exists():
        return counts, 0

    for ep_dir in sorted(directory.glob("ep_*")):
        stats_path = ep_dir / "policy_stats.json"
        keep = False
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            bucket = range_angle_bucket(stats.get("actual_insert_angle_deg"), edges)
            collection_success = bool(stats.get("collection_success") or stats.get("success"))
            if collection_success and bucket is not None and counts[bucket] < per_bin:
                keep = True
                counts[bucket] += 1
        if not keep:
            shutil.rmtree(ep_dir, ignore_errors=True)
    return counts, sum(counts.values())


def geometric_collection_success(stats):
    checks = stats.get("policy_checks", {})
    required = [
        "ring_crossed",
        "inserted_past_ring",
        "final_still_inserted",
        "tripod_stable",
        "hold_complete",
        "gripper_closed",
        "grasp_contact",
        "grasp_rigid",
    ]
    return all(bool(checks.get(name)) for name in required)


def smooth_collection_success(stats, args):
    smoothness = stats.get("smoothness", {})
    failures = []
    if smoothness.get("max_action_delta", float("inf")) > args.max_action_delta:
        failures.append("max_action_delta")
    if smoothness.get("max_action_jerk", float("inf")) > args.max_action_jerk:
        failures.append("max_action_jerk")
    if smoothness.get("mean_action_delta", float("inf")) > args.max_mean_action_delta:
        failures.append("mean_action_delta")
    if stats.get("tripod_displacement", float("inf")) > args.max_tripod_displacement:
        failures.append("tripod_displacement")
    if stats.get("tripod_rotation_deg", float("inf")) > args.max_tripod_rotation_deg:
        failures.append("tripod_rotation")
    stats["smooth_filter"] = {
        "enabled": bool(args.smooth_filter),
        "passed": not failures,
        "failure_reasons": failures,
        "thresholds": {
            "max_action_delta": float(args.max_action_delta),
            "max_action_jerk": float(args.max_action_jerk),
            "max_mean_action_delta": float(args.max_mean_action_delta),
            "max_tripod_displacement": float(args.max_tripod_displacement),
            "max_tripod_rotation_deg": float(args.max_tripod_rotation_deg),
        },
    }
    return not failures


def make_controller_config(robot, control_mode):
    config = suite.load_composite_controller_config(robot=robot)
    if control_mode == "osc_pose":
        return config

    arm_names = [name for name, part in config["body_parts"].items() if part.get("type", "").startswith("OSC")]
    if len(arm_names) != 1:
        raise ValueError(f"Expected one Panda OSC arm to replace, found {arm_names}")
    arm_name = arm_names[0]
    gripper_config = config["body_parts"][arm_name].get("gripper", {"type": "GRIP"})
    config["body_parts"][arm_name] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 100,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper_config,
    }
    return config


def main():
    args = parse_args()
    if args.collect_insert_angle_split and args.collect_insert_angle_range:
        raise ValueError("--collect-insert-angle-split and --collect-insert-angle-range are mutually exclusive")
    if args.allow_failures and (args.collect_insert_angle_split or args.collect_insert_angle_range):
        raise ValueError(
            "--allow-failures cannot fill successful angle-filtered collections; "
            "use --keep-failed to retain rejected attempts without counting them"
        )
    rng = np.random.RandomState(args.seed)
    if args.control_mode == "joint_position" and args.robots != ["Panda"]:
        raise ValueError("The joint-position experiment currently supports only one Panda robot")
    if args.record_joint_training_fields and args.control_mode != "joint_position":
        raise ValueError("--record-joint-training-fields requires --control-mode joint_position")
    controller_config = make_controller_config(args.robots[0], args.control_mode)

    split_counts = {"lt": 0, "ge": 0}
    split_subbins = {}
    range_edges = None
    range_counts = {}
    range_per_bin = None
    kept = 0
    if args.collect_insert_angle_split:
        split_counts, split_subbins, kept = prune_and_count_existing_split(
            args.directory,
            args.insert_angle_threshold,
            args.insert_angle_per_bin,
        )
        print(f"resume_insert_angle_split_counts={split_counts} kept={kept}")
    if args.collect_insert_angle_range and args.balanced_insert_angle_range:
        if args.insert_angle_edges is not None:
            range_edges = np.asarray(args.insert_angle_edges, dtype=float)
            if len(range_edges) < 2 or not np.all(np.diff(range_edges) > 0):
                raise ValueError("--insert-angle-edges must contain at least two strictly increasing values")
            args.insert_angle_min = float(range_edges[0])
            args.insert_angle_max = float(range_edges[-1])
            args.insert_angle_num_bins = len(range_edges) - 1
        else:
            range_edges = make_range_angle_edges(args.insert_angle_min, args.insert_angle_max, args.insert_angle_num_bins)
        if args.num_demos % args.insert_angle_num_bins != 0:
            raise ValueError("--num-demos must be divisible by --insert-angle-num-bins for balanced range collection")
        range_per_bin = args.num_demos // args.insert_angle_num_bins
        range_counts, kept = prune_and_count_existing_range(args.directory, range_edges, range_per_bin)
        print(
            "balanced_insert_angle_range_edges={} per_bin={} resume_counts={} kept={}".format(
                [round(float(v), 2) for v in range_edges],
                range_per_bin,
                range_angle_counts_label(range_edges, range_counts),
                kept,
            )
        )

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
    env = DataCollectionWrapper(
        env,
        args.directory,
        collect_freq=1,
        flush_freq=args.horizon + 1,
        record_joint_position_fields=args.record_joint_training_fields,
        joint_delta_scale=args.joint_delta_scale,
    )
    policy = ThreadingScriptedPolicy(
        rng=rng,
        action_noise_std=args.action_noise_std,
        grasp_angle_range=(args.grasp_angle_min, args.grasp_angle_max),
        control_mode=args.control_mode,
        collision_aware_threading=args.collision_aware_threading,
    )
    target_total = 2 * args.insert_angle_per_bin if args.collect_insert_angle_split else args.num_demos
    motion_style_plan = make_motion_style_plan(rng, target_total) if args.balanced_motion_styles else None

    successes = 0
    attempts = 0
    target_keep_count = target_total
    while kept < target_keep_count and attempts < args.max_attempts:
        attempts += 1
        if args.collect_insert_angle_split:
            target_motion_style = choose_split_motion_style(rng, split_counts, args.insert_angle_per_bin)
        elif args.collect_insert_angle_range:
            if args.range_use_balanced_motion_styles:
                target_motion_style = motion_style_plan[kept] if motion_style_plan is not None else None
            else:
                target_motion_style = (
                    choose_full_quality_motion_style(rng)
                    if args.full_quality_mode
                    else choose_large_insert_motion_style(rng)
                )
        else:
            target_motion_style = motion_style_plan[kept] if motion_style_plan is not None else None
        if args.collect_insert_angle_split:
            target_grasp_angle = choose_split_target_angle(rng, split_counts, args.insert_angle_per_bin)
        elif args.collect_insert_angle_range:
            if args.balanced_insert_angle_range:
                target_range_bin = choose_balanced_range_bin(
                    rng,
                    range_counts,
                    range_per_bin,
                    args.insert_angle_num_bins,
                )
                target_grasp_angle = choose_balanced_range_target_angle(rng, range_edges, target_range_bin)
            else:
                target_range_bin = None
                target_grasp_angle = (
                    choose_full_quality_target_angle(rng)
                    if args.full_quality_mode
                    else choose_large_insert_target_angle(rng)
                )
        else:
            target_range_bin = None
            target_grasp_angle = args.grasp_angle_list[kept % len(args.grasp_angle_list)] if args.grasp_angle_list else None
        success = policy.rollout(
            env,
            render=args.render,
            max_fr=args.max_fr,
            motion_style=target_motion_style,
            target_grasp_angle=target_grasp_angle,
            full_quality_mode=bool(args.full_quality_mode and args.collect_insert_angle_range),
        )
        stats = policy.stats[-1]
        insert_angle = stats.get("actual_insert_angle_deg")
        accepted_bucket = None
        if args.collect_insert_angle_split:
            collection_success = bool(success)
            if args.split_use_geometric_success:
                collection_success = bool(collection_success or geometric_collection_success(stats))
            if args.smooth_filter:
                collection_success = bool(collection_success and smooth_collection_success(stats, args))
            else:
                stats["smooth_filter"] = {"enabled": False, "passed": True, "failure_reasons": []}
            stats["collection_success"] = bool(collection_success)
            stats["collection_success_source"] = (
                "policy_success" if success and collection_success else (
                    "geometric_success" if collection_success else "failed"
                )
            )
            accepted_bucket = insert_angle_bucket(insert_angle, args.insert_angle_threshold) if collection_success else None
            count_episode = (
                collection_success
                and accepted_bucket is not None
                and split_counts[accepted_bucket] < args.insert_angle_per_bin
            )
            keep_episode = bool(count_episode or args.keep_failed)
        elif args.collect_insert_angle_range:
            collection_success = bool(success)
            if args.split_use_geometric_success:
                collection_success = bool(collection_success or geometric_collection_success(stats))
            smooth_success = smooth_collection_success(stats, args) if args.smooth_filter else True
            if not args.smooth_filter:
                stats["smooth_filter"] = {"enabled": False, "passed": True, "failure_reasons": []}
            angle_in_range = insert_angle_in_range(insert_angle, args.insert_angle_min, args.insert_angle_max)
            if args.balanced_insert_angle_range:
                accepted_bucket = range_angle_bucket(insert_angle, range_edges) if angle_in_range else None
                bucket_has_capacity = accepted_bucket is not None and range_counts[accepted_bucket] < range_per_bin
            else:
                accepted_bucket = None
                bucket_has_capacity = True
            stats["collection_success"] = bool(collection_success and smooth_success and angle_in_range)
            stats["collection_success_source"] = (
                "policy_success_insert_angle_range"
                if success and stats["collection_success"]
                else (
                    "geometric_success_insert_angle_range"
                    if collection_success and stats["collection_success"]
                    else ("angle_range_smooth_candidate" if stats["collection_success"] else "failed")
                )
            )
            stats["insert_angle_range"] = {
                "min": float(args.insert_angle_min),
                "max": float(args.insert_angle_max),
                "passed": bool(angle_in_range),
            }
            if args.balanced_insert_angle_range:
                stats["insert_angle_range"]["balanced_bins"] = range_angle_counts_label(range_edges, range_counts)
                stats["insert_angle_range"]["accepted_bin"] = None if accepted_bucket is None else int(accepted_bucket)
                stats["insert_angle_range"]["target_bin"] = None if target_range_bin is None else int(target_range_bin)
            count_episode = bool(stats["collection_success"] and bucket_has_capacity)
            keep_episode = bool(count_episode or args.keep_failed)
        else:
            smooth_success = smooth_collection_success(stats, args) if args.smooth_filter else True
            if not args.smooth_filter:
                stats["smooth_filter"] = {"enabled": False, "passed": True, "failure_reasons": []}
            stats["collection_success"] = bool(smooth_success and (success or args.allow_failures))
            stats["collection_success_source"] = (
                "policy_success_smooth"
                if success and smooth_success
                else ("allow_failures_smooth" if args.allow_failures and smooth_success else "failed")
            )
            count_episode = bool(stats["collection_success"])
            keep_episode = bool(count_episode or args.keep_failed)
        stats["episode_kept"] = bool(keep_episode)
        stats["counted_toward_target"] = bool(count_episode)
        ep_dir = finalize_episode(
            env,
            success=keep_episode,
            cleanup_failed=not keep_episode,
            stats=stats,
        )
        if success:
            successes += 1
        if count_episode:
            kept += 1
            if args.collect_insert_angle_split:
                split_counts[accepted_bucket] += 1
                subbin = insert_angle_subbin(insert_angle, args.insert_angle_threshold)
                split_subbins[subbin] = split_subbins.get(subbin, 0) + 1
            elif args.collect_insert_angle_range and args.balanced_insert_angle_range:
                range_counts[accepted_bucket] += 1
        print(
            "attempt={} success={} kept={}/{} successes={} style={} variant={} target_angle={} insert_angle={} bucket={} split={} steps={} ep_dir={}".format(
                attempts,
                success,
                kept,
                target_keep_count,
                successes,
                stats.get("motion_style"),
                stats.get("style_variant"),
                None if target_grasp_angle is None else round(float(target_grasp_angle), 1),
                None if insert_angle is None else round(float(insert_angle), 1),
                accepted_bucket,
                dict(split_counts) if args.collect_insert_angle_split else (
                    range_angle_counts_label(range_edges, range_counts)
                    if args.collect_insert_angle_range and args.balanced_insert_angle_range
                    else None
                ),
                stats["steps"],
                ep_dir,
            )
        )
        if not success:
            print(f"  failure_reason={stats['failure_reason']}")

    env.close()
    if kept < target_keep_count:
        raise RuntimeError(f"Kept {kept}/{target_keep_count} demos after {attempts} attempts")
    if args.collect_insert_angle_split:
        print(f"insert_angle_split_counts={split_counts}")
        print(f"insert_angle_subbins={split_subbins}")
    if args.collect_insert_angle_range:
        print(f"insert_angle_range=({args.insert_angle_min}, {args.insert_angle_max}) kept={kept}")
        if args.balanced_insert_angle_range:
            print(f"insert_angle_range_bins={range_angle_counts_label(range_edges, range_counts)}")


if __name__ == "__main__":
    main()
