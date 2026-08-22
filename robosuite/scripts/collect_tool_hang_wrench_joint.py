"""Collect clean or smoothly varied JOINT_POSITION ToolHang phase-2 demonstrations.

The assembled stand and frame are constructed at reset and rigidly fixed with
MuJoCo mocap welds. The robot then generates the wrench-only motion from
geometric stage waypoints using absolute Panda joint targets. PH data is used
only for aggregate duration and smoothness quality bounds, never for replay.
No simulator state, wrench qpos, or wrench qvel is written after execution
starts.

Examples:
    python robosuite/scripts/collect_tool_hang_wrench_joint.py --num-rollouts 20
    python robosuite/scripts/collect_tool_hang_wrench_joint.py --num-rollouts 1 \
        --video-dir output/tool_hang_clean --video-count 1
    python robosuite/scripts/collect_tool_hang_wrench_joint.py --num-rollouts 1 \
        --stop-after-stage insert --video-dir output/tool_hang_insert
"""

import argparse
from copy import deepcopy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.tool_hang_wrench_only import (
    RESET_ROBOT_QPOS,
    RESET_TOOL_QPOS,
    THREADING_STYLE_START_CLIP,
    THREADING_STYLE_START_STD,
    THREADING_STYLE_TASK_HOME_QPOS,
    ToolHangWrenchOnly,
)
from robosuite.utils.sim_utils import check_contact
from robosuite.wrappers import DataCollectionWrapper


macros.IMAGE_CONVENTION = "opencv"


def numpy_json_default(value):
    """Serialize numpy scalars emitted by geometric quality checks."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

CLEAN_STAGES = (
    "reset_fixture",
    "pregrasp",
    "descend",
    "close",
    "lift_verify",
    "transfer_rotate",
    "preinsert",
    "insert",
    "release_retreat",
)
DEFAULT_PH_DATASET = REPO_ROOT / "data" / "robomimic" / "tool_hang_ph" / "demo_v15.hdf5"
DEFAULT_PH_STATS = REPO_ROOT / "output" / "tool_hang_ph_analysis" / "tool_hang_ph_phase2_stats.json"
PH_REFERENCE_DEMO = "demo_74"
PH_PHASE2_START = 272
# The PH state is the measured response of an OSC controller. Six-frame arm
# lookahead compensates the absolute joint controller's tracking lag. It does
# not skip simulation steps or alter the gripper timing.
PH_JOINT_LOOKAHEAD = 6
PH_STAGE_ENDS = {
    "pregrasp": 314,
    "descend": 341,
    "close": 350,
    "lift_verify": 359,
    "transfer_rotate": 394,
    "preinsert": 420,
    "insert": 440,
    "release_retreat": 469,
}

# Calibrated after extending the silver handle to 20.5 cm. Across randomized
# reset poses, -10 mm remained visible while +20 mm was still transitional;
# +30 mm was selected as the conservative hidden-side boundary. Every retained
# episode is additionally checked using the actual preinsert wrist-camera ray.
VISIBILITY_CRITICAL_GRASP_X = 0.010
# Full visibility only requires grasping at or before the calibrated +10 mm
# wrist-occlusion boundary. The former -55..-10 mm support unnecessarily
# excluded the stable center and coupled the full label to near-ring contact:
# two-success qualification fell to 4.5% at -55..-46 mm. Use the continuous
# center region and let the measured wrist ray remain the final label gate.
FULL_VISIBLE_GRASP_RANGE = (0.0, VISIBILITY_CRITICAL_GRASP_X)
PARTIAL_HIDDEN_GRASP_RANGE = (0.030, 0.052)
BLACK_GRIP_EDGE_MARGIN = 0.012

# A clean demonstration must mechanically seat the ring while the wrench is
# still grasped. The native success check cannot be used before release because
# it intentionally rejects robot-tool contact, so these gates mirror its
# geometric conditions and add a tighter line-distance requirement.
PRE_RELEASE_LINE_DISTANCE_MAX = 0.004
PRE_RELEASE_INSERTION_MIN = 0.04
PRE_RELEASE_SEATED_STEPS = 10
INSERT_RELEASE_TRANSITION_STEPS = 20
INSERT_RELEASE_PREOPEN_STEPS = 12
POST_RELEASE_SUCCESS_STEPS = 20
CLEAN_GRASP_IK_ERROR_MAX = 0.160
CLEAN_TRANSFER_IK_ERROR_MAX = 0.420

# Clean small-end grasp waypoints solved from geometric EEF targets. The lift
# intentionally reverses the vertical descend by returning to PREGRASP_QPOS.
SCRIPT_PREGRASP_QPOS = np.array(
    [-0.72500307, 0.98443758, 0.46784276, -1.60987941, -0.69623419, 2.28977837, -0.48632227]
)
SCRIPT_CLOSE_QPOS = np.array(
    [-0.56902807, 1.09924060, 0.26488460, -1.55109106, -0.57003156, 2.49081986, -0.53208794]
)
def unit(vector, fallback=(1.0, 0.0, 0.0)):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return np.asarray(fallback, dtype=float) if norm < 1e-9 else vector / norm


def get_eef_pose(env):
    site_id = env.robots[0].eef_site_id["right"]
    position = np.asarray(env.sim.data.site_xpos[site_id], dtype=float).copy()
    matrix = np.asarray(env.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()
    return position, matrix


def body_pose(env, body_id):
    position = np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()
    matrix = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3).copy()
    return position, matrix


def make_controller_config(robot="Panda"):
    """Match the absolute JOINT_POSITION controller used by Threading."""
    config = suite.load_composite_controller_config(robot=robot)
    arm_names = [name for name, part in config["body_parts"].items() if part.get("type", "").startswith("OSC")]
    if len(arm_names) != 1:
        raise ValueError(f"Expected one OSC arm to replace, found {arm_names}")
    arm_name = arm_names[0]
    gripper = config["body_parts"][arm_name].get("gripper", {"type": "GRIP"})
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
        "gripper": gripper,
    }
    return config


def assemble_frame(env):
    """Construct the native assembled frame pose before enabling its weld."""
    base_position = np.asarray(env.sim.data.geom_xpos[env.obj_geom_id["stand_base"]], dtype=float)
    stand_qpos = np.asarray(env.sim.data.get_joint_qpos(env.stand.joints[0]), dtype=float)
    local_leg_x = (env.frame_args["frame_length"] - env.frame_args["frame_thickness"]) / 2.0
    tip = env.frame_args["tip_size"]
    local_tip_z = -env.frame_args["frame_height"] / 2.0 - 2.0 * tip[0] - tip[3]
    frame_position = np.array(
        [
            stand_qpos[0] + env.stand_args["mount_location"][0] - local_leg_x,
            stand_qpos[1] + env.stand_args["mount_location"][1],
            base_position[2] - local_tip_z + 0.002,
        ]
    )
    env.sim.data.set_joint_qpos(env.frame.joints[0], np.r_[frame_position, [1.0, 0.0, 0.0, 0.0]])
    env.sim.data.set_joint_qvel(env.frame.joints[0], np.zeros(6))
    env.sim.forward()
    return bool(env._check_frame_assembled())


class _LegacyCollectorToolHangWrenchOnly:
    """Deprecated in-script copy retained temporarily for old pickle imports.

    New collection and evaluation use the registered ToolHangWrenchOnly class
    imported from robosuite.environments.manipulation above.
    """

    STAND_ANCHOR = "tool_hang_stage2_stand_anchor"
    FRAME_ANCHOR = "tool_hang_stage2_frame_anchor"
    STAND_WELD = "tool_hang_stage2_stand_weld"
    FRAME_WELD = "tool_hang_stage2_frame_weld"
    # Extend the high-friction black grip across the full 16 cm metal handle.
    # handle and both holes retain the native ToolHang geometry.
    EXTENDED_GRIP_HALF_LENGTH = 0.080

    def _load_model(self):
        self.tool_grip_half_length = self.EXTENDED_GRIP_HALF_LENGTH
        # Preserve approximately the original black-grip mass after extending
        # its length by 12/8; this avoids changing release dynamics merely for
        # a larger stable graspable surface.
        self.tool_grip_density = 2000.0 * (0.040 / self.EXTENDED_GRIP_HALF_LENGTH)
        super()._load_model()

    def configure_reset_variation(self, variation=None):
        """Set episode-level reset offsets to apply on the next reset only."""
        self._pending_reset_variation = variation or {}
        # Keep an explicit sentinel because an empty dict means "fixed clean
        # reset", whereas no configured variation means a normal environment
        # reset that should randomize itself (as Threading does).
        self._has_pending_reset_variation = True

    def _sample_threading_style_robot_reset(self):
        """Sample the robot start inside the environment using its seeded RNG."""
        delta = np.clip(
            self.rng.normal(0.0, THREADING_STYLE_START_STD),
            -THREADING_STYLE_START_CLIP,
            THREADING_STYLE_START_CLIP,
        )
        qpos = THREADING_STYLE_TASK_HOME_QPOS + delta
        return {
            "robot_start_mode": "threading_continuous",
            "robot_start_center": THREADING_STYLE_TASK_HOME_QPOS.tolist(),
            "robot_start_index": None,
            "robot_qpos": qpos.tolist(),
            "robot_joint_delta_rad": (qpos - RESET_ROBOT_QPOS).tolist(),
        }

    def _sample_default_reset_variation(self):
        """Sample a complete phase-2 reset for bare ``env.reset()`` calls."""
        variation = self._sample_threading_style_robot_reset()
        variation.update(
            {
                "tool_translation_m": [
                    float(self.rng.uniform(0.020, 0.060) - RESET_TOOL_QPOS[0]),
                    float(self.rng.uniform(-0.220, -0.180) - RESET_TOOL_QPOS[1]),
                    0.0,
                ],
                "tool_yaw_rad": float(
                    self.rng.uniform(np.deg2rad(-120.0), np.deg2rad(-100.0))
                    - np.deg2rad(-100.74883)
                ),
                "fixture_translation_m": [
                    float(self.rng.uniform(-0.001, 0.001)),
                    float(self.rng.uniform(-0.001, 0.001)),
                    0.0,
                ],
                "fixture_yaw_rad": float(
                    self.rng.uniform(np.deg2rad(-0.10), np.deg2rad(0.10))
                ),
            }
        )
        return variation

    def edit_model_xml(self, xml_str):
        root = ET.fromstring(super().edit_model_xml(xml_str))
        worldbody = root.find("worldbody")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        if contact.find("./exclude[@name='tool_hang_stage2_fixture_internal']") is None:
            ET.SubElement(
                contact,
                "exclude",
                name="tool_hang_stage2_fixture_internal",
                body1=self.stand.root_body,
                body2=self.frame.root_body,
            )
        for name in (self.STAND_ANCHOR, self.FRAME_ANCHOR):
            if worldbody.find(f"./body[@name='{name}']") is None:
                ET.SubElement(worldbody, "body", name=name, mocap="true", pos="0 0 0", quat="1 0 0 0")
        # The weld fixes pose; gravity compensation removes the small steady
        # constraint deflection that would otherwise look like reset jitter.
        for body_name in (self.stand.root_body, self.frame.root_body):
            body = root.find(f".//body[@name='{body_name}']")
            if body is None:
                raise RuntimeError(f"Could not find fixture body {body_name} in ToolHang XML")
            body.set("gravcomp", "1")
        for name, body, anchor in (
            (self.STAND_WELD, self.stand.root_body, self.STAND_ANCHOR),
            (self.FRAME_WELD, self.frame.root_body, self.FRAME_ANCHOR),
        ):
            if equality.find(f"./weld[@name='{name}']") is None:
                ET.SubElement(
                    equality,
                    "weld",
                    name=name,
                    body1=body,
                    body2=anchor,
                    relpose="0 0 0 1 0 0 0",
                    active="false",
                    solref="-10000 -100",
                    solimp="0.9999 0.9999 0.0001",
                )
        return ET.tostring(root, encoding="unicode")

    def _equality_id(self, name):
        return mujoco.mj_name2id(self.sim.model._model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def _anchor_fixture(self):
        for key, anchor_name, weld_name in (
            ("stand", self.STAND_ANCHOR, self.STAND_WELD),
            ("frame", self.FRAME_ANCHOR, self.FRAME_WELD),
        ):
            body_id = self.obj_body_id[key]
            anchor_body_id = self.sim.model.body_name2id(anchor_name)
            mocap_id = self.sim.model.body_mocapid[anchor_body_id]
            anchor_position = np.asarray(self.sim.data.body_xpos[body_id]).copy()
            anchor_quat = np.asarray(self.sim.data.body_xquat[body_id]).copy()
            self.sim.data.mocap_pos[mocap_id] = anchor_position
            self.sim.data.mocap_quat[mocap_id] = anchor_quat
            # DataCollectionWrapper serializes and reloads the model XML at
            # episode start. Persist the mocap pose in the model as well as in
            # MjData so randomized fixture anchors survive that reload and the
            # recorded episode remains replayable.
            self.sim.model.body_pos[anchor_body_id] = anchor_position
            self.sim.model.body_quat[anchor_body_id] = anchor_quat
            self.sim.data.eq_active[self._equality_id(weld_name)] = 1
        self.sim.forward()
        self._fixture_reference = {
            key: np.r_[
                np.asarray(self.sim.data.body_xpos[self.obj_body_id[key]]).copy(),
                np.asarray(self.sim.data.body_xquat[self.obj_body_id[key]]).copy(),
            ]
            for key in ("stand", "frame")
        }

    def reset(self):
        if self.sim is not None:
            for name in (self.STAND_WELD, self.FRAME_WELD):
                self.sim.data.eq_active[self._equality_id(name)] = 0
        observation = super().reset()
        has_pending = bool(getattr(self, "_has_pending_reset_variation", False))
        variation = getattr(self, "_pending_reset_variation", {}) if has_pending else {}
        # A normal reset owns its randomization, matching Threading's reset
        # semantics. The collector may still configure fixed / PH resets. For
        # its threading_continuous mode it supplies object offsets and asks the
        # environment to sample only the robot start.
        if not self.deterministic_reset:
            if not has_pending:
                variation = self._sample_default_reset_variation()
            elif (
                variation.get("robot_start_mode") == "threading_continuous"
                and "robot_qpos" not in variation
            ):
                variation.update(self._sample_threading_style_robot_reset())
        robot_delta = np.asarray(variation.get("robot_joint_delta_rad", np.zeros(7)), dtype=float)
        robot_qpos = np.asarray(
            variation.get("robot_qpos", RESET_ROBOT_QPOS + robot_delta), dtype=float
        )
        self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = robot_qpos
        self.sim.data.qvel[self.robots[0]._ref_joint_vel_indexes] = 0.0
        gripper = self.robots[0].gripper["right"]
        for joint, value in zip(gripper.joints, RESET_GRIPPER_QPOS):
            self.sim.data.set_joint_qpos(joint, value)
            self.sim.data.set_joint_qvel(joint, 0.0)
        fixture_translation = np.asarray(
            variation.get("fixture_translation_m", np.zeros(3)), dtype=float
        )
        fixture_yaw = float(variation.get("fixture_yaw_rad", 0.0))
        fixture_pivot = RESET_STAND_QPOS[:3]
        stand_qpos = transform_free_joint_qpos(
            RESET_STAND_QPOS, fixture_translation, fixture_yaw, fixture_pivot
        )
        frame_qpos = transform_free_joint_qpos(
            RESET_FRAME_QPOS, fixture_translation, fixture_yaw, fixture_pivot
        )
        tool_translation = np.asarray(variation.get("tool_translation_m", np.zeros(3)), dtype=float)
        tool_yaw = float(variation.get("tool_yaw_rad", 0.0))
        tool_qpos = transform_free_joint_qpos(
            RESET_TOOL_QPOS, tool_translation, tool_yaw, RESET_TOOL_QPOS[:3]
        )
        for obj, qpos in (
            (self.stand, stand_qpos),
            (self.frame, frame_qpos),
            (self.tool, tool_qpos),
        ):
            self.sim.data.set_joint_qpos(obj.joints[0], qpos)
            self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()
        if not self._check_frame_assembled():
            raise RuntimeError("Deterministic ToolHang fixture reset is not assembled")
        self._anchor_fixture()
        self._applied_reset_variation = variation
        self._pending_reset_variation = {}
        self._has_pending_reset_variation = False
        self.robots[0].composite_controller.update_state()
        self.robots[0].composite_controller.reset()
        return self._get_observations(force_update=True)

    def load_phase2_reference_state(self, flattened_state, fixture_state=None):
        """Load reset state and anchor the fixture at its settled PH pose."""
        for name in (self.STAND_WELD, self.FRAME_WELD):
            self.sim.data.eq_active[self._equality_id(name)] = 0
        self.sim.set_state_from_flattened(np.asarray(flattened_state, dtype=float))
        if fixture_state is not None:
            fixture_qpos = np.asarray(fixture_state, dtype=float)[1 : 1 + self.sim.model.nq]
            for obj in (self.stand, self.frame):
                joint_id = self.sim.model.joint_name2id(obj.joints[0])
                qpos_address = self.sim.model.jnt_qposadr[joint_id]
                self.sim.data.set_joint_qpos(
                    obj.joints[0], fixture_qpos[qpos_address : qpos_address + 7]
                )
        for obj in (self.stand, self.frame):
            self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()
        self._anchor_fixture()
        return self._get_observations(force_update=True)

    def fixture_drift(self):
        drift = {}
        for key in ("stand", "frame"):
            reference = self._fixture_reference[key]
            position = np.asarray(self.sim.data.body_xpos[self.obj_body_id[key]])
            quat_wxyz = np.asarray(self.sim.data.body_xquat[self.obj_body_id[key]])
            quat_xyzw = np.r_[quat_wxyz[1:], quat_wxyz[0]]
            ref_xyzw = np.r_[reference[4:], reference[3]]
            drift[key] = {
                "position_m": float(np.linalg.norm(position - reference[:3])),
                "orientation_rad": float(
                    np.linalg.norm(T.quat2axisangle(T.quat_distance(ref_xyzw.copy(), quat_xyzw.copy())))
                ),
            }
        return drift


def refresh_collector_initial_state(env):
    if isinstance(env, DataCollectionWrapper):
        env._current_task_instance_state = np.asarray(env.sim.get_state().flatten()).copy()


def tool_hang_debug(env):
    hook_start = np.asarray(env.sim.data.site_xpos[env.obj_site_id["frame_hang_site"]]).copy()
    hook_end = np.asarray(env.sim.data.site_xpos[env.obj_site_id["frame_intersection_site"]]).copy()
    hook_vector = hook_end - hook_start
    hook_length = np.linalg.norm(hook_vector)
    hook_direction = hook_vector / hook_length
    hole_center = np.asarray(env.sim.data.site_xpos[env.obj_site_id["tool_hole1_center"]]).copy()
    relative = hole_center - hook_start
    along = float(np.dot(relative, hook_direction))
    residual = relative - along * hook_direction
    g1 = np.asarray(env.sim.data.geom_xpos[env.obj_geom_id["tool_hole1_hc_0"]]) - hook_start
    opposite = env.tool_args["ngeoms"] // 2
    g2 = np.asarray(env.sim.data.geom_xpos[env.obj_geom_id[f"tool_hole1_hc_{opposite}"]]) - hook_start
    tool_matrix = np.asarray(env.sim.data.body_xmat[env.obj_body_id["tool"]]).reshape(3, 3)
    ring_normal = tool_matrix[:, 2]
    frame_contact_pairs = []
    for contact_index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_index]
        geom1 = env.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
        geom2 = env.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
        if "frame" in geom1.lower() or "frame" in geom2.lower():
            frame_contact_pairs.append([geom1, geom2])
    return {
        "native_tool_on_frame": bool(env._check_tool_on_frame()),
        "hole_frame_contact": bool(
            check_contact(
                env.sim,
                [f"tool_hole1_hc_{index}" for index in range(env.tool_args["ngeoms"])],
                "frame_horizontal_frame",
            )
        ),
        "line_distance_m": float(np.linalg.norm(residual)),
        "line_residual_world_m": residual.tolist(),
        "line_distance_limit_m": float(
            env.tool_args["inner_radius_1"] - env.frame_args["frame_thickness"] / 2.0
        ),
        "hole_straddles_hook": bool(np.dot(np.cross(g1, hook_direction), np.cross(g2, hook_direction)) < 0),
        "normalized_insertion": along / hook_length,
        "hole_center": hole_center.tolist(),
        "ring_normal_world": ring_normal.tolist(),
        "ring_normal_up_dot": float(np.dot(ring_normal, np.array([0.0, 0.0, 1.0]))),
        "ring_normal_hook_dot": float(np.dot(ring_normal, hook_direction)),
        "frame_contact_pairs": frame_contact_pairs,
    }


class VideoRecorder:
    def __init__(self, path, fps=20):
        self.path = None if path is None else Path(path)
        self.writer = None
        self.fps = fps
        self.enabled = True

    def set_recording_enabled(self, enabled):
        """Exclude validation-only simulator steps from review videos."""
        self.enabled = bool(enabled)

    def append(self, observation):
        if self.path is None or not self.enabled:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                self.path, fps=self.fps, codec="libx264", quality=8, macro_block_size=None
            )
        agent = observation["agentview_image"]
        wrist = observation.get("robot0_eye_in_hand_image")
        self.writer.append_data(agent if wrist is None else np.concatenate([agent, wrist], axis=1))

    def close(self):
        if self.writer is not None:
            self.writer.close()


class GeometricJointPolicy:
    """Generate a new ToolHang trajectory from geometric stage targets."""

    # Desired final tool orientation in [hook axis, lateral axis, world up].
    HANG_TOOL_IN_HOOK_BASIS = np.array(
        [[-0.9899, -0.1171, -0.0799], [0.1360, -0.9437, -0.3014], [-0.0183, -0.3066, 0.9517]]
    )

    # Deliberate free-space transfer families. Every style converges to the
    # same exact preinsert and insertion anchors; only the collision-free
    # approach to the hook changes.
    VARIATION_STYLES = (
        "direct_low",
        "high_arc",
        "left_sweep",
        "right_sweep",
        "vertical_first",
    )

    def __init__(
        self,
        stop_after_stage=None,
        seed=0,
        variation=False,
        grasp_profile="normal",
        robot_start_mode="ph_empirical_safe",
        grasp_offset_range=None,
        robot_start_indexes=None,
        motion_style=None,
        grasp_offset_local_x_override=None,
        hang_yaw_deg=0.0,
        grasp_yaw_deg=None,
    ):
        self.stop_after_stage = stop_after_stage
        self.seed = int(seed)
        self.variation = bool(variation)
        if grasp_profile not in (
            "normal",
            "rear",
            "mixed",
            "balanced_visibility",
            "full_visible",
            "partial_hidden",
        ):
            raise ValueError(f"Unknown grasp profile: {grasp_profile}")
        if robot_start_mode not in (
            "fixed",
            "local",
            "threading_continuous",
            "ph_empirical",
            "ph_empirical_safe",
        ):
            raise ValueError(f"Unknown robot start mode: {robot_start_mode}")
        self.grasp_profile = grasp_profile
        self.robot_start_mode = robot_start_mode
        self.grasp_offset_range = grasp_offset_range
        if motion_style is not None and motion_style not in self.VARIATION_STYLES:
            raise ValueError(f"Unknown ToolHang motion style: {motion_style}")
        self.motion_style = motion_style
        self.grasp_offset_local_x_override = (
            None
            if grasp_offset_local_x_override is None
            else float(grasp_offset_local_x_override)
        )
        self.hang_yaw_deg = float(hang_yaw_deg)
        self.grasp_yaw_deg = None if grasp_yaw_deg is None else float(grasp_yaw_deg)
        self.robot_start_indexes = (
            None if robot_start_indexes is None else np.asarray(robot_start_indexes, dtype=int)
        )
        self.rng = np.random.RandomState(self.seed)
        self.ik_rng = np.random.RandomState(0)
        self.ph_robot_starts = (
            self._load_ph_robot_starts()
            if robot_start_mode in ("ph_empirical", "ph_empirical_safe")
            else None
        )
        if stop_after_stage is not None and stop_after_stage not in CLEAN_STAGES:
            raise ValueError(f"Unknown stop stage: {stop_after_stage}")

    @staticmethod
    def _load_ph_robot_starts():
        if not DEFAULT_PH_DATASET.is_file() or not DEFAULT_PH_STATS.is_file():
            raise FileNotFoundError(
                "PH empirical robot starts require both "
                f"{DEFAULT_PH_DATASET} and {DEFAULT_PH_STATS}"
            )
        phase_stats = json.loads(DEFAULT_PH_STATS.read_text())
        starts = []
        with h5py.File(DEFAULT_PH_DATASET, "r") as dataset:
            for name, result in phase_stats["demos"].items():
                phase_start = int(result["phase2_start"])
                starts.append(np.asarray(dataset[f"data/{name}/states"][phase_start, 1:8], dtype=float))
        return np.asarray(starts)

    @staticmethod
    def _grasped(env):
        return bool(env._check_grasp(env.robots[0].gripper, env.tool))

    @staticmethod
    def _joint_path_clear(env, start_qpos, target_qpos, samples=48):
        """Kinematically reject phase-2 starts whose direct departure hits the scene."""
        robot = env.robots[0]
        controller = robot.composite_controller.part_controllers[robot.arms[0]]
        indexes = np.asarray(controller.qpos_index, dtype=int)
        model = env.sim.model._model
        work = mujoco.MjData(model)
        work.qpos[:] = env.sim.data.qpos
        robot_geoms = {
            geom_id
            for geom_id, name in enumerate(env.sim.model.geom_names)
            if name is not None and name.startswith("robot0_")
        }
        previous_eef = None
        site_id = robot.eef_site_id["right"]
        for index in range(samples + 1):
            progress = GeometricJointPolicy._smooth(index / samples)
            work.qpos[indexes] = start_qpos + progress * (target_qpos - start_qpos)
            mujoco.mj_forward(model, work)
            eef = np.asarray(work.site(site_id).xpos, dtype=float)
            if previous_eef is not None and np.linalg.norm(eef - previous_eef) > 0.025:
                return False
            previous_eef = eef.copy()
            for contact_index in range(work.ncon):
                contact = work.contact[contact_index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if (geom1 in robot_geoms) != (geom2 in robot_geoms):
                    return False
        return True

    @classmethod
    def _robot_start_path_clear(cls, env, start_qpos, robot_start_mode):
        return cls._joint_path_clear(env, start_qpos, SCRIPT_PREGRASP_QPOS)

    @staticmethod
    def _set_robot_qpos(env, qpos):
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        base_env.sim.data.qpos[base_env.robots[0]._ref_joint_pos_indexes] = np.asarray(qpos)
        base_env.sim.data.qvel[base_env.robots[0]._ref_joint_vel_indexes] = 0.0
        base_env.sim.forward()
        base_env.robots[0].composite_controller.update_state()
        base_env.robots[0].composite_controller.reset()

    @staticmethod
    def _wrist_alignment(env, target):
        camera_id = env.sim.model.camera_name2id("robot0_eye_in_hand")
        camera_position = np.asarray(env.sim.data.cam_xpos[camera_id], dtype=float)
        camera_matrix = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        return float(np.dot(unit(np.asarray(target) - camera_position), unit(-camera_matrix[:, 2])))

    @staticmethod
    def _wrist_line_of_sight(env, target):
        """Project a target into the wrist image and test geometric occlusion."""
        camera_id = env.sim.model.camera_name2id("robot0_eye_in_hand")
        camera_position = np.asarray(env.sim.data.cam_xpos[camera_id], dtype=float).copy()
        camera_matrix = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        vector = np.asarray(target, dtype=float) - camera_position
        target_distance = float(np.linalg.norm(vector))
        camera_vector = camera_matrix.T.dot(vector)
        forward_depth = float(-camera_vector[2])
        tan_half_fovy = float(np.tan(np.deg2rad(env.sim.model.cam_fovy[camera_id]) / 2.0))
        normalized_x = float(camera_vector[0] / (forward_depth * tan_half_fovy)) if forward_depth > 0 else float("inf")
        normalized_y = float(camera_vector[1] / (forward_depth * tan_half_fovy)) if forward_depth > 0 else float("inf")
        in_frame = bool(forward_depth > 0 and abs(normalized_x) <= 1.0 and abs(normalized_y) <= 1.0)
        pixel_x_512 = float((normalized_x + 1.0) * 0.5 * 511.0) if np.isfinite(normalized_x) else None
        pixel_y_512 = float((1.0 - normalized_y) * 0.5 * 511.0) if np.isfinite(normalized_y) else None
        geom_id = np.array([-1], dtype=np.int32)
        hit_distance = float(
            mujoco.mj_ray(
                env.sim.model._model,
                env.sim.data._data,
                camera_position,
                unit(vector),
                np.ones(6, dtype=np.uint8),
                1,
                -1,
                geom_id,
            )
        )
        occluded = bool(0.0 <= hit_distance < target_distance - 0.003)
        hit_name = env.sim.model.geom_id2name(int(geom_id[0])) if geom_id[0] >= 0 else None
        return {
            "target_distance_m": target_distance,
            "first_hit_distance_m": hit_distance,
            "first_hit_geom": hit_name,
            "in_frame": in_frame,
            "normalized_image_xy": [normalized_x, normalized_y],
            "pixel_xy_at_512": [pixel_x_512, pixel_y_512],
            "center_ray_occluded": occluded,
            "center_ray_visible": bool(in_frame and not occluded),
            "hidden_from_wrist": bool(not in_frame or occluded),
        }

    def _global_ik(
        self, env, target_position, target_matrix, reference_qpos=None, restarts=28, position_weight=40.0
    ):
        robot = env.robots[0]
        controller = robot.composite_controller.part_controllers[robot.arms[0]]
        indexes = np.asarray(controller.qpos_index, dtype=int)
        model = env.sim.model._model
        site_id = robot.eef_site_id["right"]
        lower, upper = model.jnt_range[indexes].T
        reference = (
            np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()
            if reference_qpos is None
            else np.asarray(reference_qpos, dtype=float).copy()
        )
        work = mujoco.MjData(model)
        target_quat = T.mat2quat(target_matrix)

        def residual(qpos):
            work.qpos[:] = env.sim.data.qpos
            work.qpos[indexes] = qpos
            mujoco.mj_forward(model, work)
            position_error = np.asarray(work.site(site_id).xpos) - target_position
            matrix = np.asarray(work.site(site_id).xmat).reshape(3, 3)
            rotation_error = T.quat2axisangle(
                T.quat_distance(target_quat.copy(), T.mat2quat(matrix).copy())
            )
            return np.r_[position_weight * position_error, rotation_error, 0.002 * (qpos - reference)]

        starts = [np.clip(reference, lower + 1e-8, upper - 1e-8)]
        for _ in range(restarts - 1):
            starts.append(np.clip(reference + self.ik_rng.normal(0.0, 0.30, 7), lower, upper))
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
            solutions.append(
                (
                    float(np.linalg.norm(residual(result.x)[:6])),
                    float(np.linalg.norm(result.x - reference)),
                    result.x.copy(),
                )
            )
        solutions.sort(key=lambda item: (item[0], item[1]))
        return solutions[0][2], solutions[0][0]

    @staticmethod
    def _joint_pose(env, qpos):
        robot = env.robots[0]
        controller = robot.composite_controller.part_controllers[robot.arms[0]]
        indexes = np.asarray(controller.qpos_index, dtype=int)
        work = mujoco.MjData(env.sim.model._model)
        work.qpos[:] = env.sim.data.qpos
        work.qpos[indexes] = qpos
        mujoco.mj_forward(env.sim.model._model, work)
        site_id = robot.eef_site_id["right"]
        return (
            np.asarray(work.site(site_id).xpos).copy(),
            np.asarray(work.site(site_id).xmat).reshape(3, 3).copy(),
        )

    @staticmethod
    def _smooth(value):
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _hermite_progress(value, start_slope=0.0, end_slope=0.0):
        """Cubic progress with caller-controlled endpoint velocities."""
        value2 = value * value
        value3 = value2 * value
        return (
            (-2.0 * value3 + 3.0 * value2)
            + start_slope * (value3 - 2.0 * value2 + value)
            + end_slope * (value3 - value2)
        )

    def rollout(
        self,
        env,
        recorder=None,
        motion_style="clean",
        reset_variation_override=None,
        allow_reset_resample=True,
    ):
        # Clean mode is exactly deterministic. Variation is sampled once per
        # episode (never per control step), matching the smooth low-frequency
        # diversity used by the Threading scripted policy.
        if not self.variation:
            self.rng = np.random.RandomState(self.seed)
        # IK restart guesses stay deterministic across episodes. Diversity is
        # expressed only through explicit waypoints and timing, not accidental
        # jumps between inverse-kinematics branches.
        self.ik_rng = np.random.RandomState(0)
        recorder = recorder or VideoRecorder(None)
        if self.variation:
            if self.robot_start_mode in ("ph_empirical", "ph_empirical_safe"):
                if self.robot_start_mode == "ph_empirical_safe":
                    eligible = np.flatnonzero(
                        (self.ph_robot_starts[:, 5] > 2.3) & (self.ph_robot_starts[:, 6] < 0.0)
                    )
                    if self.robot_start_indexes is not None:
                        eligible = np.intersect1d(eligible, self.robot_start_indexes)
                        if not len(eligible):
                            raise ValueError("robot-start-indexes contains no PH-safe starts")
                    robot_start_index = int(eligible[self.rng.randint(len(eligible))])
                    robot_qpos = self.ph_robot_starts[robot_start_index].copy()
                    robot_qpos += self.rng.uniform(-0.010, 0.010, 7)
                else:
                    robot_start_index = int(self.rng.randint(len(self.ph_robot_starts)))
                    robot_qpos = self.ph_robot_starts[robot_start_index].copy()
                robot_joint_delta = robot_qpos - RESET_ROBOT_QPOS
            elif self.robot_start_mode == "threading_continuous":
                # Threading resets from the native Panda init pose with
                # independent Gaussian joint noise. Use the same continuous
                # construction, widened enough to be visually meaningful,
                # and rely on the same collision-screened departure below.
                robot_start_index = None
                robot_joint_delta = np.clip(
                    self.rng.normal(0.0, THREADING_STYLE_START_STD),
                    -THREADING_STYLE_START_CLIP,
                    THREADING_STYLE_START_CLIP,
                )
                robot_qpos = THREADING_STYLE_TASK_HOME_QPOS + robot_joint_delta
            elif self.robot_start_mode == "local":
                robot_start_index = None
                robot_joint_delta = self.rng.uniform(-0.010, 0.010, 7)
                robot_qpos = RESET_ROBOT_QPOS + robot_joint_delta
            else:
                robot_start_index = None
                robot_joint_delta = np.zeros(7)
                robot_qpos = RESET_ROBOT_QPOS.copy()
            reset_variation = {
                "robot_start_mode": self.robot_start_mode,
                "robot_start_center": (
                    THREADING_STYLE_TASK_HOME_QPOS.tolist()
                    if self.robot_start_mode == "threading_continuous"
                    else None
                ),
                "robot_start_index": robot_start_index,
                "robot_qpos": robot_qpos.tolist(),
                "robot_joint_delta_rad": robot_joint_delta.tolist(),
                "tool_translation_m": [
                    float(self.rng.uniform(0.020, 0.060) - RESET_TOOL_QPOS[0]),
                    float(self.rng.uniform(-0.220, -0.180) - RESET_TOOL_QPOS[1]),
                    0.0,
                ],
                # RESET_TOOL_QPOS has yaw -100.74883 deg. Applying this
                # relative yaw reproduces the native [-120, -100] deg sampler.
                "tool_yaw_rad": float(
                    self.rng.uniform(np.deg2rad(-120.0), np.deg2rad(-100.0))
                    - np.deg2rad(-100.74883)
                ),
                "fixture_translation_m": [
                    float(self.rng.uniform(-0.001, 0.001)),
                    float(self.rng.uniform(-0.001, 0.001)),
                    0.0,
                ],
                "fixture_yaw_rad": float(self.rng.uniform(np.deg2rad(-0.10), np.deg2rad(0.10))),
            }
            if self.robot_start_mode == "threading_continuous":
                # The environment owns this sampling, just like Threading's
                # robot reset. Keeping these keys absent also makes the same
                # behavior available to training / evaluation code that only
                # calls env.reset().
                for key in ("robot_start_index", "robot_qpos", "robot_joint_delta_rad"):
                    reset_variation.pop(key)
        else:
            reset_variation = {}
        if reset_variation_override is not None:
            # Retry mode freezes the entire physical state. Use a fresh copy
            # because the environment annotates the dict with applied reset
            # metadata, and no retry may mutate the pool manifest.
            reset_variation = deepcopy(reset_variation_override)
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        base_env.configure_reset_variation(reset_variation)
        observation = env.reset()
        if isinstance(env, DataCollectionWrapper):
            # DataCollectionWrapper performs an internal deterministic XML
            # reload. Re-anchor at the restored randomized fixture pose, then
            # refresh both serialized XML and state so playback sees exactly
            # the same mocap anchors as collection.
            base_env = env.unwrapped
            base_env._anchor_fixture()
            env._current_task_instance_xml = base_env.sim.model.get_xml()
            env._current_task_instance_state = np.asarray(
                base_env.sim.get_state().flatten()
            ).copy()
            observation = base_env._get_observations(force_update=True)
        if self.variation and self.robot_start_mode in (
            "threading_continuous",
            "ph_empirical",
            "ph_empirical_safe",
        ):
            base_env = env.unwrapped if hasattr(env, "unwrapped") else env
            selected_index = reset_variation["robot_start_index"]
            selected_qpos = np.asarray(reset_variation["robot_qpos"], dtype=float)
            selected_clear = self._robot_start_path_clear(
                base_env, selected_qpos, self.robot_start_mode
            )
            tries = 1
            while not selected_clear and allow_reset_resample and tries < 40:
                if self.robot_start_mode == "ph_empirical_safe":
                    eligible = np.flatnonzero(
                        (self.ph_robot_starts[:, 5] > 2.3) & (self.ph_robot_starts[:, 6] < 0.0)
                    )
                    if self.robot_start_indexes is not None:
                        eligible = np.intersect1d(eligible, self.robot_start_indexes)
                    selected_index = int(eligible[self.rng.randint(len(eligible))])
                    selected_qpos = self.ph_robot_starts[selected_index].copy()
                    selected_qpos += self.rng.uniform(-0.010, 0.010, 7)
                elif self.robot_start_mode == "ph_empirical":
                    selected_index = int(self.rng.randint(len(self.ph_robot_starts)))
                    selected_qpos = self.ph_robot_starts[selected_index].copy()
                else:
                    selected_qpos = THREADING_STYLE_TASK_HOME_QPOS + np.clip(
                        self.rng.normal(0.0, THREADING_STYLE_START_STD),
                        -THREADING_STYLE_START_CLIP,
                        THREADING_STYLE_START_CLIP,
                    )
                    selected_index = None
                selected_clear = self._robot_start_path_clear(
                    base_env, selected_qpos, self.robot_start_mode
                )
                tries += 1
            if not selected_clear:
                raise RuntimeError("Frozen phase-2 robot start is not collision-free")
            self._set_robot_qpos(base_env, selected_qpos)
            reset_variation["robot_start_index"] = selected_index
            reset_variation["robot_qpos"] = selected_qpos.tolist()
            reset_variation["robot_joint_delta_rad"] = (
                selected_qpos - RESET_ROBOT_QPOS
            ).tolist()
            reset_variation["robot_start_rejection_tries"] = tries
            observation = base_env._get_observations(force_update=True)
        reset_settle_steps = 0
        if self.variation and self.robot_start_mode == "threading_continuous":
            # Some broad reset configurations have a one-step torque transient
            # when the absolute joint controller first becomes active. Settle
            # that controller against the frozen reset target before defining
            # frame zero of the demonstration. This advances physics only: it
            # never writes wrench qpos / qvel and is deterministic per state.
            base_env = env.unwrapped if hasattr(env, "unwrapped") else env
            if not getattr(base_env, "_phase2_reset_settle_applied", False):
                settle_qpos = np.asarray(
                    base_env.sim.data.qpos[base_env.robots[0]._ref_joint_pos_indexes],
                    dtype=float,
                ).copy()
                for _ in range(10):
                    observation, _, _, _ = base_env.step(np.r_[settle_qpos, -1.0])
                    reset_settle_steps += 1
            # Refresh the weld reference after reset-only settling so the
            # fixture drift gate measures execution drift from frame zero.
            base_env._anchor_fixture()
            if isinstance(env, DataCollectionWrapper):
                env._current_task_instance_xml = base_env.sim.model.get_xml()
            base_env.robots[0].composite_controller.update_state()
        refresh_collector_initial_state(env)
        recorder.append(observation)
        robot = env.robots[0]
        controller = robot.composite_controller.part_controllers[robot.arms[0]]
        indexes = np.asarray(controller.qpos_index, dtype=int)
        initial_tool_position, initial_tool_matrix = body_pose(env, env.obj_body_id["tool"])
        hole_site = env.obj_site_id["tool_hole1_center"]
        initial_hole = np.asarray(env.sim.data.site_xpos[hole_site], dtype=float).copy()
        if self.variation:
            style = self.motion_style or str(self.rng.choice(self.VARIATION_STYLES))
            # Keep the contact and final insertion anchors exact. ToolHang's
            # hook clearance is only millimetres, so diversity belongs in the
            # free-space transfer, as it does in Threading's motion styles.
            requested_grasp_profile = self.grasp_profile
            episode_grasp_profile = requested_grasp_profile
            grasp_visibility_regime = None
            if episode_grasp_profile == "mixed":
                episode_grasp_profile = "normal" if self.rng.rand() < 0.5 else "rear"
            if requested_grasp_profile in (
                "balanced_visibility",
                "full_visible",
                "partial_hidden",
            ):
                usable_low = -ToolHangWrenchOnly.EXTENDED_GRIP_HALF_LENGTH + BLACK_GRIP_EDGE_MARGIN
                usable_high = ToolHangWrenchOnly.EXTENDED_GRIP_HALF_LENGTH - BLACK_GRIP_EDGE_MARGIN
                if FULL_VISIBLE_GRASP_RANGE[0] < usable_low or PARTIAL_HIDDEN_GRASP_RANGE[1] > usable_high:
                    raise RuntimeError("Balanced grasp range violates black-grip edge margin")
                sample_full = (
                    requested_grasp_profile == "full_visible"
                    or (
                        requested_grasp_profile == "balanced_visibility"
                        and self.rng.rand() < 0.5
                    )
                )
                if sample_full:
                    grasp_visibility_regime = "full_visible"
                    # Visibility categories must not select different IK
                    # policies. Use the same robust close-pose reference as
                    # partial_hidden; only grasp x and the measured camera-ray
                    # label differ between balanced regimes.
                    episode_grasp_profile = "rear"
                    default_grasp_range = FULL_VISIBLE_GRASP_RANGE
                else:
                    grasp_visibility_regime = "partial_hidden"
                    episode_grasp_profile = "rear"
                    default_grasp_range = PARTIAL_HIDDEN_GRASP_RANGE
            elif episode_grasp_profile == "normal":
                default_grasp_range = (-0.012, 0.012)
            else:
                default_grasp_range = (0.035, 0.055)
            grasp_range = self.grasp_offset_range or default_grasp_range
            grasp_offset_local_x = (
                float(self.rng.uniform(*grasp_range))
                if self.grasp_offset_local_x_override is None
                else self.grasp_offset_local_x_override
            )
            if not grasp_range[0] <= grasp_offset_local_x <= grasp_range[1]:
                raise RuntimeError(
                    f"Fixed grasp x={grasp_offset_local_x:.4f} is outside "
                    f"the {requested_grasp_profile} range {grasp_range}"
                )
            grasp_offset = initial_tool_matrix[:, 0] * grasp_offset_local_x
            motion_scale = float(self.rng.uniform(0.90, 1.10))
            transfer_along = transfer_side = transfer_up = 0.0
            insert_along = insert_side = insert_up = 0.0
            frame_offsets = np.zeros(4, dtype=int)
            retreat_along = retreat_side = retreat_up = 0.0
            retreat_frames = 15
        else:
            style = "clean"
            episode_grasp_profile = "normal"
            requested_grasp_profile = "normal"
            grasp_visibility_regime = None
            grasp_offset_local_x = 0.0
            grasp_offset = np.zeros(3)
            transfer_along = transfer_side = transfer_up = 0.0
            insert_along = insert_side = insert_up = 0.0
            frame_offsets = np.zeros(4, dtype=int)
            retreat_along = retreat_side = retreat_up = 0.0
            retreat_frames = 15
        variation_params = {
            "enabled": self.variation,
            "style": style,
            "grasp_profile": episode_grasp_profile,
            "requested_grasp_profile": requested_grasp_profile,
            "grasp_visibility_regime": grasp_visibility_regime,
            "visibility_critical_grasp_x_m": VISIBILITY_CRITICAL_GRASP_X,
            "visibility_regime_sample_probability": 0.5,
            "black_grip_edge_margin_m": BLACK_GRIP_EDGE_MARGIN,
            "black_grip_local_x_bounds_m": [
                -ToolHangWrenchOnly.EXTENDED_GRIP_HALF_LENGTH,
                ToolHangWrenchOnly.EXTENDED_GRIP_HALF_LENGTH,
            ],
            "grasp_offset_local_x_m": grasp_offset_local_x,
            "reset": reset_variation,
            "grasp_offset_world_m": grasp_offset.tolist(),
            "motion_style": style,
            "motion_scale": motion_scale if self.variation else 1.0,
            "transfer_offset_hook_basis_m": [transfer_along, transfer_side, transfer_up],
            "insert_offset_hook_basis_m": [insert_along, insert_side, insert_up],
            "frame_offsets": frame_offsets.tolist(),
            "retreat_offset_hook_basis_m": [retreat_along, retreat_side, retreat_up],
            "retreat_frames": retreat_frames,
        }

        pregrasp_qpos = SCRIPT_PREGRASP_QPOS.copy()
        close_qpos = SCRIPT_CLOSE_QPOS.copy()
        desired_close_position, _ = self._joint_pose(env, close_qpos)
        if self.variation:
            nominal_tool_matrix = T.quat2mat(
                np.r_[RESET_TOOL_QPOS[4:7], RESET_TOOL_QPOS[3]]
            )
            tool_rotation_delta = initial_tool_matrix.dot(nominal_tool_matrix.T)
            nominal_pregrasp_position, nominal_pregrasp_matrix = self._joint_pose(
                env, SCRIPT_PREGRASP_QPOS
            )
            nominal_close_position, nominal_close_matrix = self._joint_pose(env, SCRIPT_CLOSE_QPOS)
            pregrasp_position = initial_tool_position + tool_rotation_delta.dot(
                nominal_pregrasp_position - RESET_TOOL_QPOS[:3]
            )
            close_position = initial_tool_position + tool_rotation_delta.dot(
                nominal_close_position - RESET_TOOL_QPOS[:3]
            )
            pregrasp_matrix = tool_rotation_delta.dot(nominal_pregrasp_matrix)
            close_matrix = tool_rotation_delta.dot(nominal_close_matrix)
            # A small oblique grasp changes the wrist-camera baseline relative
            # to the handle. This is independent of the final wrench yaw and
            # is the degree of freedom that can move the hook out from behind
            # the black grip in the wrist image.
            episode_grasp_yaw_deg = (
                0.0 if self.grasp_yaw_deg is None else self.grasp_yaw_deg
            )
            grasp_yaw_rotation = T.rotation_matrix(
                np.deg2rad(episode_grasp_yaw_deg), initial_tool_matrix[:, 2]
            )[:3, :3]
            pregrasp_matrix = grasp_yaw_rotation.dot(pregrasp_matrix)
            close_matrix = grasp_yaw_rotation.dot(close_matrix)
            desired_close_position = close_position + grasp_offset
            pregrasp_qpos, pregrasp_ik_error = self._global_ik(
                env,
                pregrasp_position + grasp_offset,
                pregrasp_matrix,
                reference_qpos=SCRIPT_PREGRASP_QPOS,
                restarts=18,
            )
            close_qpos, close_ik_error = self._global_ik(
                env,
                close_position + grasp_offset,
                close_matrix,
                reference_qpos=(pregrasp_qpos if episode_grasp_profile == "rear" else SCRIPT_CLOSE_QPOS),
                restarts=18,
            )
            grasp_ik_mode = "balanced"
            # Only genuinely position-limited grasp poses benefit from a
            # stronger Cartesian position weight. Applying it universally
            # changes good contact geometry and lowers success. The residual
            # gate selects the hard native-yaw tail before any motion occurs.
            if max(pregrasp_ik_error, close_ik_error) > CLEAN_GRASP_IK_ERROR_MAX:
                # Candidate solvers must see the same deterministic restart
                # sequence; otherwise merely evaluating the balanced branch
                # changes the position-priority solution.
                self.ik_rng = np.random.RandomState(0)
                pregrasp_qpos, pregrasp_ik_error = self._global_ik(
                    env,
                    pregrasp_position + grasp_offset,
                    pregrasp_matrix,
                    reference_qpos=SCRIPT_PREGRASP_QPOS,
                    restarts=28,
                    position_weight=100.0,
                )
                close_qpos, close_ik_error = self._global_ik(
                    env,
                    close_position + grasp_offset,
                    close_matrix,
                    reference_qpos=(pregrasp_qpos if episode_grasp_profile == "rear" else SCRIPT_CLOSE_QPOS),
                    restarts=28,
                    position_weight=100.0,
                )
                grasp_ik_mode = "position_priority"
            # Solver residuals are not comparable across the balanced
            # (position_weight=40) and position-priority (weight=100) modes.
            # Re-evaluate both final solutions with one canonical metric so
            # entering the fallback cannot by itself make an otherwise better
            # pose fail the clean-grasp gate.
            canonical_grasp_errors = []
            grasp_position_errors = []
            grasp_orientation_errors = []
            for solved_qpos, target_position, target_matrix in (
                (pregrasp_qpos, pregrasp_position + grasp_offset, pregrasp_matrix),
                (close_qpos, close_position + grasp_offset, close_matrix),
            ):
                solved_position, solved_matrix = self._joint_pose(env, solved_qpos)
                position_error = float(np.linalg.norm(solved_position - target_position))
                orientation_error = float(
                    np.linalg.norm(
                        T.quat2axisangle(
                            T.quat_distance(
                                T.mat2quat(target_matrix).copy(),
                                T.mat2quat(solved_matrix).copy(),
                            )
                        )
                    )
                )
                grasp_position_errors.append(position_error)
                grasp_orientation_errors.append(orientation_error)
                canonical_grasp_errors.append(
                    float(np.hypot(40.0 * position_error, orientation_error))
                )
            variation_params["grasp_ik_mode"] = grasp_ik_mode
            variation_params["grasp_ik_solver_error"] = float(
                max(pregrasp_ik_error, close_ik_error)
            )
            variation_params["grasp_ik_position_error_m"] = float(
                max(grasp_position_errors)
            )
            variation_params["grasp_ik_orientation_error_rad"] = float(
                max(grasp_orientation_errors)
            )
            variation_params["grasp_ik_error"] = float(max(canonical_grasp_errors))
            variation_params["grasp_yaw_deg"] = float(episode_grasp_yaw_deg)

        # Full-visible grasps occupy local x=0..VISIBILITY_CRITICAL_GRASP_X,
        # closer to hole1 than the partial range at x=0.030..0.052. This
        # coordinate convention changed when the black grip was extended, but
        # the old negative-x formula remained and silently returned zero for
        # every demo. Restore the intended continuous clearance adjustment:
        # the closer a full grasp is to the ring, the larger the severity.
        near_ring_full_severity = (
            float(
                np.clip(
                    (VISIBILITY_CRITICAL_GRASP_X - grasp_offset_local_x)
                    / VISIBILITY_CRITICAL_GRASP_X,
                    0.0,
                    1.0,
                )
            )
            if grasp_visibility_regime == "full_visible"
            else 0.0
        )
        variation_params["near_ring_full_severity"] = near_ring_full_severity
        stage_checks = []
        joint_deltas, joint_jerks, eef_steps = [], [], []
        actual_joint_positions = [np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()]
        previous_action = previous_delta = None
        steps = stage_start = 0
        failure_reason = "none"
        requested_stage_complete = False

        def step(action):
            nonlocal previous_action, previous_delta, steps
            if previous_action is not None:
                delta = action[:-1] - previous_action[:-1]
                joint_deltas.append(float(np.linalg.norm(delta)))
                if previous_delta is not None:
                    joint_jerks.append(float(np.linalg.norm(delta - previous_delta)))
                previous_delta = delta
            old_eef, _ = get_eef_pose(env)
            obs, _, _, _ = env.step(action)
            new_eef, _ = get_eef_pose(env)
            eef_steps.append(float(np.linalg.norm(new_eef - old_eef)))
            actual_joint_positions.append(np.asarray(env.sim.data.qpos[indexes], dtype=float).copy())
            previous_action = action.copy()
            steps += 1
            recorder.append(obs)

        def move(
            target,
            gripper,
            frames,
            cartesian_parameterization=False,
            start_slope=0.0,
            end_slope=0.0,
        ):
            start = (
                np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()
                if previous_action is None
                else previous_action[:-1].copy()
            )
            if cartesian_parameterization:
                # Reparameterize the same straight joint-space line by its FK
                # arc length. This slows only Cartesian-sensitive regions and
                # adds no waypoint, detour, or IK branch change.
                grid = np.linspace(0.0, 1.0, 401)
                grid_qpos = start[None, :] + grid[:, None] * (target - start)[None, :]
                grid_eef = np.asarray(
                    [self._joint_pose(env, qpos)[0] for qpos in grid_qpos]
                )
                cartesian_distance = np.linalg.norm(np.diff(grid_eef, axis=0), axis=1)
                joint_distance = np.linalg.norm(np.diff(grid_qpos, axis=0), axis=1)
                cumulative = np.r_[0.0, np.cumsum(cartesian_distance + 0.02 * joint_distance)]
                normalized_time = np.arange(1, frames + 1, dtype=float) / frames
                progress_values = np.interp(
                    np.asarray(
                        [
                            self._hermite_progress(value, start_slope, end_slope)
                            for value in normalized_time
                        ]
                    )
                    * cumulative[-1],
                    cumulative,
                    grid,
                )
            else:
                progress_values = [
                    self._hermite_progress(
                        (frame + 1) / frames,
                        start_slope,
                        end_slope,
                    )
                    for frame in range(frames)
                ]
            for progress in progress_values:
                desired = start + progress * (target - start)
                if previous_action is not None:
                    prior = previous_action[:-1]
                    delta_limit = 0.015 if cartesian_parameterization else 0.030
                    desired = np.clip(desired, prior - delta_limit, prior + delta_limit)
                step(np.r_[desired, gripper])

        def move_through(targets, gripper, frames, start_slope=0.0, end_slope=0.0):
            """Traverse joint waypoints as one continuous, pause-free curve."""
            start = (
                np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()
                if previous_action is None
                else previous_action[:-1].copy()
            )
            points = np.vstack([start] + [np.asarray(target, dtype=float) for target in targets])
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            keep = np.r_[True, distances > 1e-7]
            points = points[keep]
            if len(points) == 1:
                hold(points[0], gripper, frames)
                return
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            knots = np.r_[0.0, np.cumsum(distances)]
            knots /= knots[-1]
            curve = PchipInterpolator(knots, points, axis=0)
            for frame in range(frames):
                progress = self._hermite_progress(
                    (frame + 1) / frames,
                    start_slope,
                    end_slope,
                )
                desired = np.asarray(curve(progress), dtype=float)
                if previous_action is not None:
                    prior = previous_action[:-1]
                    desired = np.clip(desired, prior - 0.030, prior + 0.030)
                step(np.r_[desired, gripper])

        def hold(target, gripper, frames):
            command = target if previous_action is None else previous_action[:-1].copy()
            for _ in range(frames):
                step(np.r_[command, gripper])

        validation_only_steps = {}

        def set_demo_recording(enabled):
            if isinstance(env, DataCollectionWrapper):
                env.set_recording_enabled(enabled)
            recorder.set_recording_enabled(enabled)

        def validation_hold(target, gripper, frames, reason):
            """Execute a physical hold without teaching a stationary action loop."""
            before = steps
            set_demo_recording(False)
            try:
                hold(target, gripper, frames)
            finally:
                set_demo_recording(True)
            validation_only_steps[reason] = validation_only_steps.get(reason, 0) + (
                steps - before
            )

        def finish_stage(name, passed, metrics=None):
            nonlocal stage_start, failure_reason, requested_stage_complete
            entry = {
                "name": name,
                "start_step": int(stage_start),
                "end_step": int(steps),
                "passed": bool(passed),
                "grasped": self._grasped(env),
                "stage_max_eef_step_m": float(np.max(eef_steps[stage_start:]))
                if len(eef_steps) > stage_start
                else 0.0,
                "stage_max_joint_jerk": float(
                    np.max(joint_jerks[max(0, stage_start - 2) :])
                )
                if len(joint_jerks) > max(0, stage_start - 2)
                else 0.0,
            }
            if metrics:
                entry.update(metrics)
            stage_checks.append(entry)
            stage_start = steps
            if not passed:
                failure_reason = name
                return False
            if self.stop_after_stage == name:
                requested_stage_complete = True
                return False
            return True

        drift = env.fixture_drift()
        running = finish_stage(
            "reset_fixture",
            env._check_frame_assembled()
            and max(value["position_m"] for value in drift.values()) < 1e-4
            and max(value["orientation_rad"] for value in drift.values()) < 1e-3,
            {"fixture_drift": drift},
        )

        if running:
            # Phase 2 begins at a safe observation pose near the wrench, so a
            # single direct joint segment reaches the overhead pregrasp.
            current_robot_qpos = np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()
            if self.robot_start_mode == "threading_continuous":
                # Decouple broad random resets from high-sensitivity regions:
                # first settle onto the authored Threading-style task home,
                # then use the validated direct task-home -> pregrasp line.
                # Start recording immediately after reset. The former six
                # hidden controller-sync steps left controller history that is
                # absent from MuJoCo state and caused replay parity drift.
                move(THREADING_STYLE_TASK_HOME_QPOS, -1.0, 20)
                pregrasp_frames = 50
                move(pregrasp_qpos, -1.0, pregrasp_frames)
            else:
                pregrasp_frames = max(
                    34,
                    min(54, int(np.ceil(np.max(np.abs(pregrasp_qpos - current_robot_qpos)) / 0.030)) + 4),
                )
                move(pregrasp_qpos, -1.0, pregrasp_frames + int(frame_offsets[0]))
            eef_position, _ = get_eef_pose(env)
            running = finish_stage(
                "pregrasp",
                not self._grasped(env) and eef_position[2] > initial_tool_position[2] + 0.05,
                {"eef_position": eef_position.tolist()},
            )
        if running:
            move(
                close_qpos,
                -1.0,
                24 + int(frame_offsets[1]),
                end_slope=0.15,
            )
            eef_position, _ = get_eef_pose(env)
            running = finish_stage(
                "descend",
                not self._grasped(env) and np.linalg.norm(eef_position - desired_close_position) < 0.015,
                {
                    "eef_position": eef_position.tolist(),
                    "grasp_target_position": desired_close_position.tolist(),
                    "grasp_target_error_m": float(
                        np.linalg.norm(eef_position - desired_close_position)
                    ),
                },
            )
        if running:
            close_grasp_run = 0
            # Close slowly, then begin the lift immediately after physical
            # grasp. Holding the exact same arm target for all close frames
            # creates a closed-gripper fixed point for feed-forward BC: the
            # rollout can remain at the grasp pose instead of transitioning to
            # lift. The small grasp-confirm lift below preserves stable finger
            # closure while making close -> lift continuous and monotonic.
            # Descend ends at gripper command -1.0. Begin closure immediately
            # instead of duplicating that exact 8-D action at the stage
            # boundary.
            close_commands = np.r_[np.linspace(-0.9, 1.0, 10), np.ones(8)]
            close_lift_target = close_qpos + 0.12 * (pregrasp_qpos - close_qpos)
            # Do not wait for `_grasped`: that predicate only becomes true
            # after several fully-closed frames, recreating the exact arm
            # target run we are trying to remove. Start a sub-centimetre lift
            # while the fingers are closing. The first four commands provide
            # contact acquisition; every fully-closed command is therefore
            # paired with upward arm motion.
            close_lift_start_frame = 4
            close_lift_frames = len(close_commands) - close_lift_start_frame
            for close_frame, close_command in enumerate(close_commands):
                close_lift_progress = np.clip(
                    (close_frame + 1 - close_lift_start_frame) / close_lift_frames,
                    0.0,
                    1.0,
                )
                progress = self._hermite_progress(close_lift_progress, 0.0, 0.5)
                close_arm_target = close_qpos + progress * (
                    close_lift_target - close_qpos
                )
                step(np.r_[close_arm_target, close_command])
                close_grasp_run = close_grasp_run + 1 if self._grasped(env) else 0
            running = finish_stage("close", self._grasped(env))
        if running:
            pre_lift_z = body_pose(env, env.obj_body_id["tool"])[0][2]
            move(
                pregrasp_qpos,
                1.0,
                18 + int(frame_offsets[2]),
                start_slope=0.09,
                end_slope=0.15,
            )
            lift = float(body_pose(env, env.obj_body_id["tool"])[0][2] - pre_lift_z)
            running = finish_stage(
                "lift_verify",
                self._grasped(env) and 0.05 < lift < 0.085,
                {"tool_lift_m": lift},
            )

        if running:
            tool_position, tool_matrix = body_pose(env, env.obj_body_id["tool"])
            eef_position, eef_matrix = get_eef_pose(env)
            local_hole = tool_matrix.T.dot(
                np.asarray(env.sim.data.site_xpos[hole_site]) - tool_position
            )
            tool_to_eef_position = tool_matrix.T.dot(eef_position - tool_position)
            tool_to_eef_matrix = tool_matrix.T.dot(eef_matrix)
            hook_start = np.asarray(env.sim.data.site_xpos[env.obj_site_id["frame_hang_site"]]).copy()
            hook_end = np.asarray(
                env.sim.data.site_xpos[env.obj_site_id["frame_intersection_site"]]
            ).copy()
            hook_vector = hook_end - hook_start
            hook_length = float(np.linalg.norm(hook_vector))
            hook_direction = hook_vector / hook_length
            world_up = np.array([0.0, 0.0, 1.0])
            side = unit(np.cross(world_up, hook_direction), fallback=(1.0, 0.0, 0.0))
            hook_basis = np.column_stack([hook_direction, side, world_up])
            # Use one reachable, downward-canted orientation for every demo.
            # A fully horizontal ring rests on top of the horizontal hook;
            # this modest cant gives vertical descent a component in the ring
            # plane without requiring the unreachable fully vertical pose.
            hang_cant_deg = -25.0
            hang_cant = np.deg2rad(hang_cant_deg)
            # Do not point the wrench handle directly down the hook axis. In
            # that configuration the wrist camera looks through the black
            # handle and the hook is geometrically occluded even for a
            # full-visible grasp. A single modest in-plane yaw keeps the ring
            # plane unchanged while exposing the hook beside the handle.
            hang_yaw_deg = self.hang_yaw_deg
            hang_yaw = np.deg2rad(hang_yaw_deg)
            handle_horizontal = unit(
                -np.cos(hang_yaw) * hook_direction + np.sin(hang_yaw) * side
            )
            hanging_tool_matrix = np.column_stack(
                [
                    np.cos(hang_cant) * handle_horizontal
                    + np.sin(hang_cant) * world_up,
                    unit(np.cross(world_up, handle_horizontal)),
                    np.cos(hang_cant) * world_up
                    - np.sin(hang_cant) * handle_horizontal,
                ]
            )
            # The final ring center sits just below the hook axis so the upper
            # inner rim bears the load while the gripper is still closed. The
            # horizontal target stays on the hook axis throughout the final
            # descent; with the ring plane leveled there is no need for a
            # collision-dependent lateral preload.
            seat_up_bias = -0.0030
            seat_side_bias = 0.0
            seat_along_fraction = 0.14
            variation_params["hang_cant_deg"] = float(hang_cant_deg)
            variation_params["hang_yaw_deg"] = float(hang_yaw_deg)
            variation_params["seat_along_fraction"] = float(seat_along_fraction)

            def eef_for_hole(hole_position):
                desired_tool_position = hole_position - hanging_tool_matrix.dot(local_hole)
                return (
                    desired_tool_position + hanging_tool_matrix.dot(tool_to_eef_position),
                    hanging_tool_matrix.dot(tool_to_eef_matrix),
                )

            high_hole = (
                hook_start
                + (seat_along_fraction * hook_length + transfer_along)
                * hook_direction
                + (seat_side_bias + transfer_side) * side
                + (0.0600 + transfer_up) * world_up
            )
            high_eef_position, high_eef_matrix = eef_for_hole(high_hole)
            lifted_hole = np.asarray(env.sim.data.site_xpos[hole_site]).copy()
            if not self.variation:
                control_holes = []
            elif style == "direct_low":
                # Preserve the low, nearly straight family but solve it as two
                # sequential Cartesian segments. The former single global IK
                # jump was the only style shared by both failed states in the
                # balanced pilot and frequently ended in a poor insertion
                # joint basin.
                control_holes = [
                    0.5 * (lifted_hole + high_hole) + 0.005 * world_up
                ]
            elif style == "high_arc":
                control_holes = [
                    high_hole
                    - (0.035 * motion_scale) * hook_direction
                    + (0.035 * motion_scale) * world_up
                ]
            elif style == "left_sweep":
                control_holes = [
                    high_hole
                    - (0.035 * motion_scale) * hook_direction
                    + (0.025 * motion_scale) * side
                    + (0.025 * motion_scale) * world_up
                ]
            elif style == "right_sweep":
                control_holes = [
                    high_hole
                    - (0.035 * motion_scale) * hook_direction
                    - (0.030 * motion_scale) * side
                    + (0.020 * motion_scale) * world_up
                ]
            elif style == "vertical_first":
                # Rise with the current wrist orientation before rotating the
                # wrench toward the hook. The old waypoint used the final
                # hanging orientation at the lifted XY position, forcing a
                # large in-place rotation; that made this style account for
                # nearly the entire full-vs-partial state-success gap.
                vertical_rise = 0.025
                vertical_hole = lifted_hole + (vertical_rise * motion_scale) * world_up
                blend_hole = (
                    0.45 * lifted_hole
                    + 0.55 * high_hole
                    + (0.012 * motion_scale) * world_up
                )
                control_holes = [vertical_hole, blend_hole]
                control_pose_overrides = [
                    (
                        eef_position + (vertical_rise * motion_scale) * world_up,
                        eef_matrix,
                    ),
                    None,
                ]
                variation_params["vertical_first_rise_m"] = float(vertical_rise)
            else:
                raise RuntimeError(f"Unhandled ToolHang motion style: {style}")

            if style != "vertical_first":
                control_pose_overrides = [None] * len(control_holes)

            variation_params["transfer_control_offsets_hook_basis_m"] = [
                hook_basis.T.dot(control - high_hole).tolist()
                for control in control_holes
            ]
            transfer_waypoint_joints = []
            transfer_ik_errors = []
            reference_qpos = np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()
            for control_hole, pose_override in zip(
                control_holes + [high_hole], control_pose_overrides + [None]
            ):
                if pose_override is None:
                    control_eef_position, control_eef_matrix = eef_for_hole(control_hole)
                else:
                    control_eef_position, control_eef_matrix = pose_override
                control_joints, control_error = self._global_ik(
                    env,
                    control_eef_position,
                    control_eef_matrix,
                    reference_qpos=reference_qpos,
                    restarts=55,
                )
                transfer_waypoint_joints.append(control_joints)
                transfer_ik_errors.append(float(control_error))
                reference_qpos = control_joints
            transfer_joints = transfer_waypoint_joints[-1]
            transfer_ik_error = max(transfer_ik_errors)
            transfer_path = [lifted_hole] + control_holes + [high_hole]
            transfer_path_length = float(
                sum(
                    np.linalg.norm(end - start)
                    for start, end in zip(transfer_path[:-1], transfer_path[1:])
                )
            )
            # The path is already rate-limited and parameterized as one
            # continuous curve. Sixty-plus frames duplicated slow free-space
            # motion and pushed otherwise clean Threading-reset trajectories
            # beyond the PH phase-2 frame envelope after adding replay-complete
            # grasp and seating dynamics.
            transfer_frames = (54 if style == "vertical_first" else 46) + int(
                frame_offsets[3]
            )
            variation_params["transfer_path_length_m"] = transfer_path_length
            variation_params["transfer_frames"] = transfer_frames
            if control_holes:
                move_through(
                    transfer_waypoint_joints,
                    1.0,
                    transfer_frames,
                    start_slope=0.25,
                    end_slope=0.25,
                )
            else:
                move(
                    transfer_joints,
                    1.0,
                    transfer_frames,
                    start_slope=0.25,
                    end_slope=0.25,
                )
            transfer_debug = tool_hang_debug(env)
            running = finish_stage(
                "transfer_rotate",
                self._grasped(env) and transfer_debug["line_distance_m"] < 0.100,
                {
                    "tool_debug": transfer_debug,
                    "motion_style": style,
                    "control_holes": [control.tolist() for control in control_holes],
                    "global_ik_error": transfer_ik_error,
                    "global_ik_errors": transfer_ik_errors,
                    "global_ik_final_error": float(transfer_ik_errors[-1]),
                },
            )

        if running:
            correction_count = 2
            for correction_index in range(correction_count):
                actual_hole = np.asarray(env.sim.data.site_xpos[hole_site]).copy()
                current_eef_position, current_eef_matrix = get_eef_pose(env)
                target_joints, _ = self._global_ik(
                    env,
                    current_eef_position + high_hole - actual_hole,
                    current_eef_matrix,
                    restarts=18,
                    position_weight=100.0,
                )
                move(
                    target_joints,
                    1.0,
                    6,
                    cartesian_parameterization=True,
                    start_slope=0.25,
                    end_slope=0.20 if correction_index == correction_count - 1 else 0.25,
                )
            preinsert_debug = tool_hang_debug(env)
            hole = np.asarray(env.sim.data.site_xpos[hole_site])
            hook_alignment = self._wrist_alignment(env, hook_start)
            hole_alignment = self._wrist_alignment(env, hole)
            hole_line_of_sight = self._wrist_line_of_sight(env, hole)
            hook_line_of_sight = self._wrist_line_of_sight(env, hook_start)
            hook_probe_visibility = {
                f"fraction_{fraction:.2f}": self._wrist_line_of_sight(
                    env, hook_start + fraction * hook_vector
                )
                for fraction in (0.0, 0.25, 0.50, 0.75, 1.0)
            }
            if grasp_visibility_regime is not None:
                # Visibility is a label / acceptance property, not a physical
                # policy stage. Applying it here changes which actions execute
                # for full versus partial trajectories and biases their state
                # success rates. collection_acceptance verifies the measured
                # camera ray after the common physical trajectory completes.
                visibility_gate = True
            else:
                visibility_gate = bool(
                    episode_grasp_profile == "rear"
                    or (hole_alignment > 0.85 and hook_alignment > 0.75)
                )
            running = finish_stage(
                "preinsert",
                self._grasped(env)
                and preinsert_debug["line_distance_m"] < 0.070
                and visibility_gate,
                {
                    "tool_debug": preinsert_debug,
                    "wrist_hole_alignment": hole_alignment,
                    "wrist_hook_alignment": hook_alignment,
                    "visibility_gate": visibility_gate,
                    "wrist_visibility": {
                        "tool_hole_center": hole_line_of_sight,
                        "hook_start": hook_line_of_sight,
                        "hook_axis_probes": hook_probe_visibility,
                    },
                },
            )

        if running:
            # All free-space styles converge here. Approach the hook through a
            # common three-part template, then keep the arm completely still
            # while verifying that the ring remains seated around the hook.
            # This replaces the former contact-only gate that released with an
            # 8--11 mm line error and relied on gravity to finish insertion.
            # At preinsert the ring is already aligned in hook-axis and lateral
            # coordinates. All retained trajectories now lower it vertically;
            # no diagonal side sweep is allowed in the precision phase.
            canonical_holes = [
                hook_start
                    + (seat_along_fraction * hook_length + insert_along) * hook_direction
                    + (seat_side_bias + insert_side) * side
                    + (0.0350 + insert_up) * world_up,
                hook_start
                    + (seat_along_fraction * hook_length + insert_along) * hook_direction
                    + (seat_side_bias + insert_side) * side
                    + (0.0180 + insert_up) * world_up,
                hook_start
                    + (seat_along_fraction * hook_length + insert_along) * hook_direction
                    + insert_side * side
                    + (0.0080 + insert_up) * world_up,
                hook_start
                    + (seat_along_fraction * hook_length + insert_along) * hook_direction
                    + insert_side * side
                    + (seat_up_bias + insert_up) * world_up,
            ]
            insertion_progress = []
            insertion_debug = []
            insertion_ik_errors = []
            target_joints = np.asarray(env.sim.data.qpos[indexes], dtype=float).copy()
            for phase_index, (target_hole, phase_frames) in enumerate(
                zip(canonical_holes, (8, 12, 10, 18))
            ):
                # Re-solve once halfway through the seating phase so contact
                # tracking error is corrected without introducing a new path
                # family or a release-dependent settling motion.
                # Re-solve the two free-space descent targets incrementally.
                # A single global IK solve can leave 7--12 mm of vertical
                # tracking error for near-ring full grasps even though there
                # is no contact. Multiple short Cartesian closes converge
                # before the final axial insertion begins.
                # Re-solve every free-space descent anchor after the previous
                # segment has physically tracked. One solve per anchor left
                # 8--20 mm of residual error on randomized grasps; once the
                # ring touched the hook, the bounded contact correction could
                # no longer recover. These extra short segments follow the
                # same vertical path and add no new motion mode.
                correction_frames = (
                    (8, 6)
                    if phase_index == 0
                    else (8, 6)
                    if phase_index == 1
                    else (6, 6, 6)
                )
                for correction_index, frames in enumerate(correction_frames):
                    actual_hole = np.asarray(env.sim.data.site_xpos[hole_site]).copy()
                    current_eef_position, current_eef_matrix = get_eef_pose(env)
                    target_joints, ik_error = self._global_ik(
                        env,
                        current_eef_position + target_hole - actual_hole,
                        current_eef_matrix,
                        reference_qpos=np.asarray(
                            env.sim.data.qpos[indexes], dtype=float
                        ).copy(),
                        restarts=28,
                        position_weight=100.0,
                    )
                    insertion_ik_errors.append(float(ik_error))
                    move(
                        target_joints,
                        1.0,
                        frames,
                        cartesian_parameterization=True,
                        start_slope=0.15 if phase_index == 0 and correction_index == 0 else 0.10,
                        end_slope=0.10,
                    )
                    debug = tool_hang_debug(env)
                    insertion_progress.append(debug["normalized_insertion"])
                    insertion_debug.append(debug)

            # Roll the ring edge over the hook with a small, controlled
            # closed-gripper correction. This used to live inside the release
            # branch for rear grasps, which hid the actual seating event from
            # the insert gate. It is now part of the canonical insertion for
            # every trajectory and is verified before any opening command.
            current_eef_position, current_eef_matrix = get_eef_pose(env)
            seat_tilt_joints, seat_tilt_ik_error = self._global_ik(
                env,
                current_eef_position
                + 0.0013 * hook_direction
                - 0.0005 * world_up,
                current_eef_matrix,
                reference_qpos=np.asarray(
                    env.sim.data.qpos[indexes], dtype=float
                ).copy(),
                restarts=24,
                position_weight=100.0,
            )
            move(
                seat_tilt_joints,
                1.0,
                8,
                cartesian_parameterization=True,
                start_slope=0.10,
                end_slope=0.10,
            )
            target_joints = seat_tilt_joints

            # Contact can leave the ring a few millimetres off the hook axis.
            # Finish with up to six bounded, slow line-error corrections while the
            # gripper remains closed. The command acts only through the robot;
            # the wrench qpos / qvel are never modified.
            for _ in range(6):
                correction_debug = tool_hang_debug(env)
                residual = np.asarray(
                    correction_debug["line_residual_world_m"], dtype=float
                )
                # Aim near the center of the clean gate. The former -3.5 mm
                # target sat only 0.5 mm inside the <=4 mm acceptance limit,
                # and its >=3 mm early-stop condition actively moved already
                # well-aligned rings away from the hook axis.
                desired_residual = -0.0015 * world_up
                along_deficit = max(
                    0.0,
                    (0.065 - correction_debug["normalized_insertion"])
                    * hook_length,
                )
                cartesian_correction = (
                    -0.65 * (residual - desired_residual)
                    + along_deficit * hook_direction
                )
                correction_norm = float(np.linalg.norm(cartesian_correction))
                correction_limit = 0.004
                if correction_norm > correction_limit:
                    cartesian_correction *= correction_limit / correction_norm
                current_eef_position, current_eef_matrix = get_eef_pose(env)
                target_joints, correction_ik_error = self._global_ik(
                    env,
                    current_eef_position + cartesian_correction,
                    current_eef_matrix,
                    reference_qpos=np.asarray(
                        env.sim.data.qpos[indexes], dtype=float
                    ).copy(),
                    restarts=24,
                    position_weight=100.0,
                )
                insertion_ik_errors.append(float(correction_ik_error))
                move(
                    target_joints,
                    1.0,
                    6,
                    cartesian_parameterization=True,
                    start_slope=0.10,
                    end_slope=0.10,
                )
                debug = tool_hang_debug(env)
                insertion_progress.append(debug["normalized_insertion"])
                insertion_debug.append(debug)
                if (
                    debug["hole_frame_contact"]
                    and debug["hole_straddles_hook"]
                    and debug["line_distance_m"] <= PRE_RELEASE_LINE_DISTANCE_MAX
                    and debug["normalized_insertion"] > PRE_RELEASE_INSERTION_MIN
                ):
                    break

            # Join insertion and release with one authored motion. The former
            # implementation held a fixed arm target for up to 32 frames and
            # then opened in place for another eight. Those repeated actions
            # taught BC to stop immediately before release. Here the arm keeps
            # advancing a small distance along the same seating direction
            # while the gripper opens monotonically. The first part verifies
            # stable seating; the remaining samples continue the exact same
            # curve through release.
            transition_start_joints = previous_action[:-1].copy()
            transition_start_eef, transition_eef_matrix = get_eef_pose(env)
            transition_target_joints, transition_ik_error = self._global_ik(
                env,
                transition_start_eef
                + 0.0020 * hook_direction
                - 0.0010 * world_up,
                transition_eef_matrix,
                reference_qpos=transition_start_joints,
                restarts=24,
                position_weight=100.0,
            )
            transition_progress = np.asarray(
                [
                    self._hermite_progress(
                        (frame + 1) / INSERT_RELEASE_TRANSITION_STEPS,
                        0.20,
                        0.20,
                    )
                    for frame in range(INSERT_RELEASE_TRANSITION_STEPS)
                ],
                dtype=float,
            )
            transition_gripper = np.r_[
                np.linspace(1.0, 0.75, INSERT_RELEASE_PREOPEN_STEPS),
                np.linspace(
                    0.75,
                    -1.0,
                    INSERT_RELEASE_TRANSITION_STEPS
                    - INSERT_RELEASE_PREOPEN_STEPS
                    + 1,
                )[1:],
            ]
            transition_joint_targets = transition_start_joints[None, :] + (
                transition_progress[:, None]
                * (transition_target_joints - transition_start_joints)[None, :]
            )
            transition_arm_deltas = np.linalg.norm(
                np.diff(
                    np.vstack([transition_start_joints, transition_joint_targets]),
                    axis=0,
                ),
                axis=1,
            )
            insert_release_continuous = bool(
                np.all(np.diff(transition_progress) > 0.0)
                and np.all(np.diff(transition_gripper) <= 1e-12)
                and np.min(transition_arm_deltas) > 1e-7
            )
            pre_seated_debug = tool_hang_debug(env)
            seated_run = 0
            seated_debug = []
            for target_joints, gripper_command in zip(
                transition_joint_targets[:INSERT_RELEASE_PREOPEN_STEPS],
                transition_gripper[:INSERT_RELEASE_PREOPEN_STEPS],
            ):
                step(np.r_[target_joints, gripper_command])
                debug = tool_hang_debug(env)
                seated_debug.append(debug)
                insertion_progress.append(debug["normalized_insertion"])
                insertion_debug.append(debug)
                seated = bool(
                    self._grasped(env)
                    and debug["hole_frame_contact"]
                    and debug["hole_straddles_hook"]
                    and debug["line_distance_m"] <= PRE_RELEASE_LINE_DISTANCE_MAX
                    and PRE_RELEASE_INSERTION_MIN
                    < debug["normalized_insertion"]
                    < 1.0
                )
                seated_run = seated_run + 1 if seated else 0

            insert_debug = tool_hang_debug(env)
            seating_progress = insertion_progress[-INSERT_RELEASE_PREOPEN_STEPS:]
            insertion_monotonic = bool(
                len(seating_progress) < 2
                or np.all(np.diff(seating_progress) >= -0.010)
            )
            running = finish_stage(
                "insert",
                seated_run >= PRE_RELEASE_SEATED_STEPS
                and insertion_monotonic
                and insert_release_continuous,
                {
                    "tool_debug": insert_debug,
                    "pre_seated_tool_debug": pre_seated_debug,
                    "insertion_progress": insertion_progress,
                    "insertion_debug": insertion_debug,
                    "insertion_ik_errors": insertion_ik_errors,
                    "seat_tilt_ik_error": float(seat_tilt_ik_error),
                    "insert_release_transition_ik_error": float(transition_ik_error),
                    "insertion_monotonic": insertion_monotonic,
                    "insert_release_continuous": insert_release_continuous,
                    "insert_release_min_arm_target_delta": float(
                        np.min(transition_arm_deltas)
                    ),
                    "insert_release_max_arm_target_delta": float(
                        np.max(transition_arm_deltas)
                    ),
                    "pre_release_seated_required": PRE_RELEASE_SEATED_STEPS,
                    "pre_release_seated_run": seated_run,
                    "pre_release_seated_debug": seated_debug,
                    "wrist_visibility": {
                        "tool_hole_center": self._wrist_line_of_sight(
                            env, np.asarray(env.sim.data.site_xpos[hole_site]).copy()
                        ),
                        "hook_start": self._wrist_line_of_sight(env, hook_start),
                    },
                },
            )

        if running:
            # Continue the already-started insertion-to-release curve. There
            # is no duplicated endpoint or stationary action at the stage
            # boundary. Then retreat along a fixed safe direction and require
            # twenty consecutive native-success frames.
            release_debug = []
            for target_joints, gripper_command in zip(
                transition_joint_targets[INSERT_RELEASE_PREOPEN_STEPS:],
                transition_gripper[INSERT_RELEASE_PREOPEN_STEPS:],
            ):
                step(np.r_[target_joints, gripper_command])
                release_debug.append(tool_hang_debug(env))
            # Seed retreat IK from the final 7-D command, not the complete
            # (steps, 7) transition matrix. Passing the matrix silently
            # broadcast through the solver and could select a distant joint
            # branch, producing a large target jump after an otherwise clean
            # release.
            target_joints = transition_target_joints[-1].copy()
            current_eef_position, current_eef_matrix = get_eef_pose(env)
            released_hole = np.asarray(env.sim.data.site_xpos[hole_site]).copy()
            retreat_away_from_ring = unit(
                current_eef_position - released_hole,
                fallback=side,
            )
            retreat_joints, retreat_ik_error = self._global_ik(
                env,
                current_eef_position
                + 0.050 * retreat_away_from_ring
                + 0.015 * world_up,
                current_eef_matrix,
                reference_qpos=target_joints,
                restarts=24,
            )
            move(
                retreat_joints,
                -1.0,
                12,
                cartesian_parameterization=True,
            )
            persistent_run = 0
            persistent_history = []
            persistent_start = steps
            set_demo_recording(False)
            # `move` rate-limits each joint and can legitimately stop short
            # of the raw IK target. Re-commanding `retreat_joints` here would
            # create one unrecorded but physically real jump before the
            # persistence audit, and it also contaminated the smoothness
            # metrics. Hold the final command actually issued by `move`.
            persistent_target = previous_action[:-1].copy()
            # Terminal audit only: never resume this episode's recording.
            # Resuming used to overwrite the last recorded post-state with a
            # state one second later while retaining the old action.
            for _ in range(POST_RELEASE_SUCCESS_STEPS):
                step(np.r_[persistent_target, -1.0])
                native_now = bool(env._check_success())
                persistent_history.append(native_now)
                persistent_run = persistent_run + 1 if native_now else 0
            validation_only_steps["post_release_persistence"] = (
                steps - persistent_start
            )
            persistent_success = persistent_run >= POST_RELEASE_SUCCESS_STEPS
            finish_stage(
                "release_retreat",
                persistent_success and not self._grasped(env),
                {
                    "pre_release_debug": insert_debug,
                    "release_debug": release_debug,
                    "retreat_ik_error": float(retreat_ik_error),
                    "persistent_success_required": POST_RELEASE_SUCCESS_STEPS,
                    "persistent_success_run": persistent_run,
                    "persistent_success_history": persistent_history,
                    "final_debug": tool_hang_debug(env),
                },
            )

        full_success = bool(
            failure_reason == "none"
            and not requested_stage_complete
            and len(stage_checks) == len(CLEAN_STAGES)
            and all(stage["passed"] for stage in stage_checks)
            and env._check_success()
        )
        stats = {
            "success": full_success,
            "collection_success": full_success,
            "requested_stage_complete": requested_stage_complete,
            "stop_after_stage": self.stop_after_stage,
            "failure_reason": failure_reason,
            "frame_assembled": bool(env._check_frame_assembled()),
            "tool_on_frame": bool(env._check_tool_on_frame()),
            "motion_style": "geometric_joint_script",
            "variation": variation_params,
            "trajectory_source": "generated_from_geometric_waypoints",
            "ph_frame_replay": False,
            "steps": int(steps),
            "training_recorded_steps": int(
                steps - sum(validation_only_steps.values())
            ),
            "validation_only_steps": validation_only_steps,
            "initial_tool_position": initial_tool_position.tolist(),
            "initial_hole_position": initial_hole.tolist(),
            "wrench_pose_assist_count": 0,
            "reset_controller_settle_steps": reset_settle_steps,
            "stage_checks": stage_checks,
            "final_debug": tool_hang_debug(env),
            "fixture_drift": env.fixture_drift(),
            "smoothness": {
                "eef_path_length_m": float(np.sum(eef_steps)) if eef_steps else 0.0,
                "mean_joint_target_delta": float(np.mean(joint_deltas)) if joint_deltas else 0.0,
                "max_joint_target_delta": float(np.max(joint_deltas)) if joint_deltas else 0.0,
                "mean_joint_target_jerk": float(np.mean(joint_jerks)) if joint_jerks else 0.0,
                "max_joint_target_jerk": float(np.max(joint_jerks)) if joint_jerks else 0.0,
                "mean_actual_joint_step": float(
                    np.mean(np.linalg.norm(np.diff(np.asarray(actual_joint_positions), axis=0), axis=1))
                )
                if len(actual_joint_positions) >= 2
                else 0.0,
                "max_actual_joint_step": float(
                    np.max(np.linalg.norm(np.diff(np.asarray(actual_joint_positions), axis=0), axis=1))
                )
                if len(actual_joint_positions) >= 2
                else 0.0,
                "max_actual_joint_step_index": int(
                    np.argmax(np.linalg.norm(np.diff(np.asarray(actual_joint_positions), axis=0), axis=1))
                )
                if len(actual_joint_positions) >= 2
                else None,
                "mean_actual_joint_second_difference": float(
                    np.mean(
                        np.linalg.norm(
                            np.diff(np.diff(np.asarray(actual_joint_positions), axis=0), axis=0), axis=1
                        )
                    )
                )
                if len(actual_joint_positions) >= 3
                else 0.0,
                "max_actual_joint_second_difference": float(
                    np.max(
                        np.linalg.norm(
                            np.diff(np.diff(np.asarray(actual_joint_positions), axis=0), axis=0), axis=1
                        )
                    )
                )
                if len(actual_joint_positions) >= 3
                else 0.0,
                "max_actual_joint_second_difference_index": int(
                    np.argmax(
                        np.linalg.norm(
                            np.diff(np.diff(np.asarray(actual_joint_positions), axis=0), axis=0), axis=1
                        )
                    )
                )
                if len(actual_joint_positions) >= 3
                else None,
                "mean_eef_step_m": float(np.mean(eef_steps)) if eef_steps else 0.0,
                "max_eef_step_m": float(np.max(eef_steps)) if eef_steps else 0.0,
                "max_eef_step_index": int(np.argmax(eef_steps)) if eef_steps else None,
            },
        }
        recorder.close()
        return full_success, stats


class PHJointReferenceBaseline:
    """Legacy replay baseline retained only for explicit code comparison.

    The collection CLI never instantiates this class; new demonstrations use
    :class:`GeometricJointPolicy` exclusively.
    """

    def __init__(self, dataset_path, stop_after_stage=None):
        self.dataset_path = Path(dataset_path)
        self.stop_after_stage = stop_after_stage
        if stop_after_stage is not None and stop_after_stage not in CLEAN_STAGES:
            raise ValueError(f"Unknown stop stage: {stop_after_stage}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Official ToolHang PH dataset is required: {self.dataset_path}")
        with h5py.File(self.dataset_path, "r") as dataset:
            group = dataset[f"data/{PH_REFERENCE_DEMO}"]
            self.reference_states = np.asarray(group["states"], dtype=float)
            self.reference_actions = np.asarray(group["actions"], dtype=float)

    @staticmethod
    def _grasped(env):
        return bool(env._check_grasp(env.robots[0].gripper, env.tool))

    @staticmethod
    def _wrist_alignment(env, target):
        camera_id = env.sim.model.camera_name2id("robot0_eye_in_hand")
        camera_position = np.asarray(env.sim.data.cam_xpos[camera_id], dtype=float)
        camera_matrix = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        return float(np.dot(unit(np.asarray(target) - camera_position), unit(-camera_matrix[:, 2])))

    def rollout(self, env, recorder=None, motion_style="clean"):
        recorder = recorder or VideoRecorder(None)
        env.reset()
        observation = env.load_phase2_reference_state(
            self.reference_states[PH_PHASE2_START],
            fixture_state=self.reference_states[PH_STAGE_ENDS["close"]],
        )
        refresh_collector_initial_state(env)
        recorder.append(observation)

        robot = env.robots[0]
        controller = robot.composite_controller.part_controllers[robot.arms[0]]
        qpos_indexes = np.asarray(controller.qpos_index, dtype=int)
        qpos_slice = slice(1, 1 + int(env.sim.model.nq))
        initial_tool_position, initial_tool_matrix = body_pose(env, env.obj_body_id["tool"])
        initial_hole = np.asarray(env.sim.data.site_xpos[env.obj_site_id["tool_hole1_center"]]).copy()
        stage_checks = []
        stage_start = 0
        joint_deltas, joint_jerks, eef_steps = [], [], []
        previous_action = previous_delta = None
        failure_reason = "none"
        requested_stage_complete = False
        steps = 0

        def finish_stage(name, passed, metrics=None):
            nonlocal stage_start, failure_reason, requested_stage_complete
            entry = {
                "name": name,
                "start_step": int(stage_start),
                "end_step": int(steps),
                "passed": bool(passed),
                "grasped": self._grasped(env),
            }
            if metrics:
                entry.update(metrics)
            stage_checks.append(entry)
            stage_start = steps
            if not passed:
                failure_reason = name
                return False
            if self.stop_after_stage == name:
                requested_stage_complete = True
                return False
            return True

        drift = env.fixture_drift()
        running = finish_stage(
            "reset_fixture",
            bool(
                env._check_frame_assembled()
                and max(value["position_m"] for value in drift.values()) < 0.0015
                and max(value["orientation_rad"] for value in drift.values()) < 0.03
            ),
            {"fixture_drift": drift},
        )

        stage_by_frame = {frame: name for name, frame in PH_STAGE_ENDS.items()}
        for reference_index in range(PH_PHASE2_START, PH_STAGE_ENDS["release_retreat"]):
            if not running:
                break
            target_index = min(reference_index + PH_JOINT_LOOKAHEAD, PH_STAGE_ENDS["release_retreat"])
            target_qpos = self.reference_states[target_index, qpos_slice][qpos_indexes]
            gripper_command = self.reference_actions[reference_index, -1]
            # JOINT_POSITION reaches the contact segment a few frames later
            # than PH's OSC controller. Keep the real grasp closed through the
            # insert gate, then use the unchanged PH release / retreat motion.
            if 432 <= reference_index < PH_STAGE_ENDS["insert"]:
                gripper_command = 1.0
            action = np.r_[target_qpos, gripper_command]
            if previous_action is not None:
                delta = action[:-1] - previous_action[:-1]
                joint_deltas.append(float(np.linalg.norm(delta)))
                if previous_delta is not None:
                    joint_jerks.append(float(np.linalg.norm(delta - previous_delta)))
                previous_delta = delta
            old_eef, _ = get_eef_pose(env)
            observation, _, _, _ = env.step(action)
            new_eef, _ = get_eef_pose(env)
            eef_steps.append(float(np.linalg.norm(new_eef - old_eef)))
            previous_action = action.copy()
            steps += 1
            recorder.append(observation)

            stage_name = stage_by_frame.get(reference_index + 1)
            if stage_name is None or stage_name == "release_retreat":
                continue
            tool_position, _ = body_pose(env, env.obj_body_id["tool"])
            eef_position, _ = get_eef_pose(env)
            eef_tool_distance = float(np.linalg.norm(eef_position - tool_position))
            lift = float(tool_position[2] - initial_tool_position[2])
            debug = tool_hang_debug(env)
            metrics = {
                "eef_tool_distance_m": eef_tool_distance,
                "tool_lift_m": lift,
                "tool_debug": debug,
            }
            if stage_name == "pregrasp":
                passed = not self._grasped(env) and eef_tool_distance < 0.080
            elif stage_name == "descend":
                passed = not self._grasped(env) and eef_tool_distance < 0.035
            elif stage_name == "close":
                passed = self._grasped(env)
            elif stage_name == "lift_verify":
                passed = self._grasped(env) and 0.05 < lift < 0.085
            elif stage_name == "transfer_rotate":
                passed = self._grasped(env) and lift > 0.08
            elif stage_name == "preinsert":
                hole = np.asarray(env.sim.data.site_xpos[env.obj_site_id["tool_hole1_center"]])
                hook = np.asarray(env.sim.data.site_xpos[env.obj_site_id["frame_hang_site"]])
                hole_alignment = self._wrist_alignment(env, hole)
                hook_alignment = self._wrist_alignment(env, hook)
                metrics.update(
                    wrist_hole_alignment=hole_alignment,
                    wrist_hook_alignment=hook_alignment,
                )
                passed = bool(
                    self._grasped(env)
                    and debug["line_distance_m"] < 0.025
                    and hole_alignment > 0.90
                    and hook_alignment > 0.75
                )
            else:
                passed = bool(
                    self._grasped(env)
                    and debug["hole_frame_contact"]
                    and debug["line_distance_m"] < 0.010
                    and debug["normalized_insertion"] > 0.04
                )
            running = finish_stage(stage_name, passed, metrics)

        if running:
            final_qpos = self.reference_states[PH_STAGE_ENDS["release_retreat"], qpos_slice][qpos_indexes]
            persistent_success = True
            for _ in range(15):
                old_eef, _ = get_eef_pose(env)
                observation, _, _, _ = env.step(np.r_[final_qpos, -1.0])
                new_eef, _ = get_eef_pose(env)
                eef_steps.append(float(np.linalg.norm(new_eef - old_eef)))
                steps += 1
                recorder.append(observation)
                persistent_success &= bool(env._check_success())
            finish_stage(
                "release_retreat",
                persistent_success and not self._grasped(env),
                {"persistent_success_15": persistent_success, "final_debug": tool_hang_debug(env)},
            )

        full_success = bool(
            failure_reason == "none"
            and not requested_stage_complete
            and len(stage_checks) == len(CLEAN_STAGES)
            and all(stage["passed"] for stage in stage_checks)
            and env._check_success()
        )
        stats = {
            "success": full_success,
            "collection_success": full_success,
            "requested_stage_complete": requested_stage_complete,
            "stop_after_stage": self.stop_after_stage,
            "failure_reason": failure_reason,
            "frame_assembled": bool(env._check_frame_assembled()),
            "tool_on_frame": bool(env._check_tool_on_frame()),
            "motion_style": "ph_joint_reference_clean",
            "reference_demo": PH_REFERENCE_DEMO,
            "reference_phase2_start": PH_PHASE2_START,
            "joint_lookahead_frames": PH_JOINT_LOOKAHEAD,
            "steps": int(steps),
            "initial_tool_position": initial_tool_position.tolist(),
            "initial_tool_yaw_deg": float(
                np.rad2deg(np.arctan2(initial_tool_matrix[1, 0], initial_tool_matrix[0, 0]))
            ),
            "initial_hole_position": initial_hole.tolist(),
            "wrench_pose_assist_count": 0,
            "stage_checks": stage_checks,
            "final_debug": tool_hang_debug(env),
            "fixture_drift": env.fixture_drift(),
            "smoothness": {
                "eef_path_length_m": float(np.sum(eef_steps)) if eef_steps else 0.0,
                "mean_joint_target_delta": float(np.mean(joint_deltas)) if joint_deltas else 0.0,
                "max_joint_target_delta": float(np.max(joint_deltas)) if joint_deltas else 0.0,
                "mean_joint_target_jerk": float(np.mean(joint_jerks)) if joint_jerks else 0.0,
                "max_joint_target_jerk": float(np.max(joint_jerks)) if joint_jerks else 0.0,
                "mean_eef_step_m": float(np.mean(eef_steps)) if eef_steps else 0.0,
                "max_eef_step_m": float(np.max(eef_steps)) if eef_steps else 0.0,
            },
        }
        recorder.close()
        return full_success, stats


def finalize_episode(env, success, stats, keep_failed):
    if not isinstance(env, DataCollectionWrapper):
        return
    episode_directory = env.ep_directory
    if env.has_interaction:
        if env.collect_freq == 1:
            recorded_actions = len(env.action_infos)
            recorded_states = len(env.states)
            expected_actions = int(stats.get("training_recorded_steps", recorded_actions))
            recorded_times = np.asarray(
                [np.asarray(state, dtype=float)[0] for state in env.states], dtype=float
            )
            state_dts = np.diff(recorded_times)
            expected_dt = float(env.control_timestep)
            integrity = {
                "recorded_actions": recorded_actions,
                "recorded_states": recorded_states,
                "states_equal_actions_plus_one": recorded_states
                == recorded_actions + 1,
                "recorded_actions_match_policy_stats": recorded_actions
                == expected_actions,
                "expected_state_dt_s": expected_dt,
                "max_state_dt_s": float(np.max(state_dts)) if len(state_dts) else 0.0,
                "min_state_dt_s": float(np.min(state_dts)) if len(state_dts) else 0.0,
                "contiguous_control_steps": bool(
                    len(state_dts) == 0
                    or np.allclose(state_dts, expected_dt, atol=1e-8, rtol=0.0)
                ),
            }
            stats["recording_integrity"] = integrity
            if not all(
                (
                    integrity["states_equal_actions_plus_one"],
                    integrity["recorded_actions_match_policy_stats"],
                    integrity["contiguous_control_steps"],
                )
            ):
                raise RuntimeError(f"Invalid masked demonstration recording: {integrity}")
        env._flush()
        env.has_interaction = False
    if episode_directory and os.path.isdir(episode_directory):
        with open(os.path.join(episode_directory, "policy_stats.json"), "w") as stream:
            json.dump(stats, stream, indent=2, default=numpy_json_default)
        if not success and not keep_failed:
            shutil.rmtree(episode_directory)


def collection_acceptance(native_success, stats, wrist_requirement="any", require_ph_quality=False):
    checks = {"native_success": bool(native_success)}
    preinsert = next(
        (stage for stage in stats.get("stage_checks", []) if stage.get("name") == "preinsert"),
        {},
    )
    hole_visibility = preinsert.get("wrist_visibility", {}).get("tool_hole_center", {})
    hook_visibility = preinsert.get("wrist_visibility", {}).get("hook_start", {})
    hole_xy = np.asarray(hole_visibility.get("normalized_image_xy", [np.inf, np.inf]), dtype=float)
    hook_xy = np.asarray(hook_visibility.get("normalized_image_xy", [np.inf, np.inf]), dtype=float)
    relative_pixel_separation = (
        float(np.linalg.norm(hole_xy - hook_xy))
        if np.all(np.isfinite(hole_xy)) and np.all(np.isfinite(hook_xy))
        else 0.0
    )
    both_with_margin = bool(
        np.all(np.isfinite(hole_xy))
        and np.all(np.isfinite(hook_xy))
        and np.max(np.abs(hole_xy)) <= 0.995
        and np.max(np.abs(hook_xy)) <= 0.995
    )
    if wrist_requirement == "visible":
        checks["wrist_visible"] = bool(hole_visibility.get("center_ray_visible"))
    elif wrist_requirement == "hidden":
        checks["wrist_hidden"] = bool(hole_visibility.get("hidden_from_wrist"))
    visibility_regime = stats.get("variation", {}).get("grasp_visibility_regime")
    if visibility_regime == "full_visible":
        checks["balanced_full_ring_visible"] = bool(
            hole_visibility.get("center_ray_visible")
        )
        checks["balanced_full_hook_visible"] = bool(
            hook_visibility.get("center_ray_visible")
        )
        checks["balanced_full_image_margin"] = both_with_margin
        checks["balanced_full_relative_separation"] = relative_pixel_separation >= 0.08
    elif visibility_regime == "partial_hidden":
        checks["balanced_partial_hidden"] = bool(hole_visibility.get("hidden_from_wrist"))
    checks["recording_contiguous"] = bool(
        stats.get("recording_integrity", {}).get("contiguous_control_steps", True)
    )
    if require_ph_quality:
        smooth = stats.get("smoothness", {})
        insert = next(
            (stage for stage in stats.get("stage_checks", []) if stage.get("name") == "insert"),
            {},
        )
        transfer = next(
            (
                stage
                for stage in stats.get("stage_checks", [])
                if stage.get("name") == "transfer_rotate"
            ),
            {},
        )
        insert_debug = insert.get("tool_debug", {})
        threading_style_start = (
            stats.get("variation", {}).get("reset", {}).get("robot_start_mode")
            == "threading_continuous"
        )
        frame_low, frame_high = (140, 325) if threading_style_start else (165, 240)
        quality_frames = stats.get("training_recorded_steps", stats.get("steps", 0))
        checks.update(
            {
                "clean_pre_release_contact": bool(insert_debug.get("hole_frame_contact")),
                "clean_pre_release_straddle": bool(insert_debug.get("hole_straddles_hook")),
                "clean_pre_release_line_distance": insert_debug.get(
                    "line_distance_m", float("inf")
                )
                <= PRE_RELEASE_LINE_DISTANCE_MAX,
                "clean_pre_release_insertion": PRE_RELEASE_INSERTION_MIN
                < insert_debug.get("normalized_insertion", float("-inf"))
                < 1.0,
                "clean_pre_release_moving_seated_run": insert.get(
                    "pre_release_seated_run", 0
                )
                >= PRE_RELEASE_SEATED_STEPS,
                "clean_insertion_monotonic": bool(insert.get("insertion_monotonic")),
                "clean_insert_release_continuous": bool(
                    insert.get("insert_release_continuous")
                ),
                "clean_grasp_ik": stats.get("variation", {}).get(
                    "grasp_ik_error", float("inf")
                )
                <= CLEAN_GRASP_IK_ERROR_MAX,
                "clean_transfer_ik": transfer.get(
                    "global_ik_final_error",
                    transfer.get("global_ik_error", float("inf")),
                )
                <= CLEAN_TRANSFER_IK_ERROR_MAX,
                f"ph_frames_{frame_low}_{frame_high}": frame_low
                <= quality_frames
                <= frame_high,
                "clean_max_joint_target_delta": smooth.get(
                    "max_joint_target_delta", float("inf")
                )
                <= 0.065,
                "clean_max_joint_target_jerk": smooth.get(
                    "max_joint_target_jerk", float("inf")
                )
                <= 0.060,
                "ph_max_actual_joint_step": smooth.get("max_actual_joint_step", float("inf"))
                <= 0.07899320978201528,
                "ph_max_actual_joint_second_difference": smooth.get(
                    "max_actual_joint_second_difference", float("inf")
                )
                <= 0.021638167481016324,
                "ph_max_eef_step": smooth.get("max_eef_step_m", float("inf"))
                <= 0.017764508323362925,
            }
        )
    return checks, all(checks.values())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--directory", type=Path, default=REPO_ROOT / "tool_hang_wrench_joint_demos")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--video-count", type=int, default=3)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-joint-training-fields", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-delta-scale", type=float, default=0.05)
    parser.add_argument("--stop-after-stage", choices=CLEAN_STAGES, default=None)
    parser.add_argument(
        "--variation",
        action="store_true",
        help="Sample one smooth Threading-style waypoint variation per episode.",
    )
    parser.add_argument(
        "--grasp-profile",
        choices=(
            "normal",
            "rear",
            "mixed",
            "balanced_visibility",
            "full_visible",
            "partial_hidden",
        ),
        default="normal",
        help="Sample continuous grasp offsets, including calibrated balanced or forced full / partial visibility regions.",
    )
    parser.add_argument(
        "--robot-start-mode",
        choices=("fixed", "local", "threading_continuous", "ph_empirical", "ph_empirical_safe"),
        default="threading_continuous",
        help="Robot reset distribution; threading_continuous uses Threading-matched sigma=0.02 Gaussian noise around a task-specific home, without PH states.",
    )
    parser.add_argument(
        "--grasp-offset-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=None,
        help="Override the continuous local-X grasp-offset range for calibration.",
    )
    parser.add_argument(
        "--motion-style",
        choices=GeometricJointPolicy.VARIATION_STYLES,
        default=None,
        help="Force one ToolHang hook-approach family; variation mode samples a family when omitted.",
    )
    parser.add_argument(
        "--grasp-offset-local-x",
        type=float,
        default=None,
        help="Freeze one local-X grasp coordinate across retries.",
    )
    parser.add_argument(
        "--hang-yaw-deg",
        type=float,
        default=0.0,
        help="In-plane wrench yaw at the hook; positive yaw exposes the hook beside the handle.",
    )
    parser.add_argument(
        "--grasp-yaw-deg",
        type=float,
        default=None,
        help="Override oblique gripper yaw; the default keeps the stable orthogonal grasp.",
    )
    parser.add_argument(
        "--robot-start-indexes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of official PH phase-2 start indexes for targeted robustness sampling.",
    )
    parser.add_argument(
        "--require-wrist",
        choices=("any", "visible", "hidden"),
        default="any",
        help="Keep only demos whose wrench-hole center has the requested preinsert wrist visibility.",
    )
    parser.add_argument(
        "--require-ph-quality",
        action="store_true",
        help="Keep only native successes within the PH timing and actual-motion P90 gates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_camera = args.video_dir is not None
    env = suite.make(
        "ToolHangWrenchOnly",
        robots=["Panda"],
        controller_configs=make_controller_config("Panda"),
        initialization_noise=None,
        ignore_done=True,
        use_camera_obs=use_camera,
        use_object_obs=True,
        has_renderer=args.render,
        has_offscreen_renderer=use_camera,
        camera_names=["agentview", "robot0_eye_in_hand"] if use_camera else None,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        horizon=args.horizon,
        hard_reset=False,
        seed=args.seed,
    )
    if args.collect:
        env = DataCollectionWrapper(
            env,
            str(args.directory),
            collect_freq=1,
            flush_freq=args.horizon + 1,
            record_joint_position_fields=args.record_joint_training_fields,
            joint_delta_scale=args.joint_delta_scale,
            reload_from_xml_on_episode_start=False,
        )

    policy = GeometricJointPolicy(
        stop_after_stage=args.stop_after_stage,
        seed=args.seed,
        variation=args.variation,
        grasp_profile=args.grasp_profile,
        robot_start_mode=args.robot_start_mode,
        grasp_offset_range=None if args.grasp_offset_range is None else tuple(args.grasp_offset_range),
        robot_start_indexes=args.robot_start_indexes,
        motion_style=args.motion_style,
        grasp_offset_local_x_override=args.grasp_offset_local_x,
        hang_yaw_deg=args.hang_yaw_deg,
        grasp_yaw_deg=args.grasp_yaw_deg,
    )
    results = []
    attempts = successes = 0
    max_attempts = args.max_attempts or args.num_rollouts
    started = time.time()
    try:
        while attempts < max_attempts and (attempts < args.num_rollouts if not args.collect else successes < args.num_rollouts):
            attempts += 1
            video_path = None
            if args.video_dir is not None and attempts <= args.video_count:
                video_path = args.video_dir / f"rollout_{attempts:03d}.mp4"
            native_success, stats = policy.rollout(env, VideoRecorder(video_path))
            acceptance_checks, success = collection_acceptance(
                native_success,
                stats,
                wrist_requirement=args.require_wrist,
                require_ph_quality=args.require_ph_quality,
            )
            stats["native_success"] = bool(native_success)
            stats["acceptance_checks"] = acceptance_checks
            stats["accepted"] = bool(success)
            if native_success and not success:
                stats["failure_reason"] = "acceptance:" + ",".join(
                    name for name, passed in acceptance_checks.items() if not passed
                )
            stats["success"] = bool(success)
            successes += int(success)
            stats["attempt"] = attempts
            results.append(stats)
            finalize_episode(env, success, stats, args.keep_failed)
            print(
                f"attempt={attempts} success={success} rate={successes / attempts:.1%} "
                f"max_dq={stats['smoothness']['max_joint_target_delta']:.4f} "
                f"max_eef_step={stats['smoothness']['max_eef_step_m']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "seed": args.seed,
        "variation": args.variation,
        "attempts": attempts,
        "successes": successes,
        "success_rate": successes / attempts if attempts else 0.0,
        "elapsed_seconds": time.time() - started,
        "motion_style_counts": {
            style: sum(item.get("variation", {}).get("style") == style for item in results)
            for style in sorted({item.get("variation", {}).get("style", "unknown") for item in results})
        },
        "smoothness": {
            key: max(item["smoothness"][key] for item in results) if results else 0.0
            for key in ("max_joint_target_delta", "max_joint_target_jerk", "max_eef_step_m")
        },
        "rollouts": results,
    }
    output_directory = args.directory if args.collect else (args.summary_dir or args.video_dir or REPO_ROOT / "output")
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "tool_hang_wrench_joint_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=numpy_json_default,
        )
        + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rollouts"}, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
