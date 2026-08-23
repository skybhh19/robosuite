"""ToolHang phase-2 environment with an assembled fixture and reset randomization."""

import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.tool_hang import ToolHang


# Authored phase-2 reference configuration. These values define reset state,
# never a replayed trajectory or an execution-time object assist.
RESET_ROBOT_QPOS = np.array(
    [0.08073644, 0.90338240, -0.11503691, -1.48327730, -0.09632931, 1.53187796, 0.35768220]
)
# Start fully open. The stock Panda value is half-open; making reset itself
# open avoids a visually different frame before the policy's open command.
RESET_GRIPPER_QPOS = np.array([0.040, -0.040])
RESET_STAND_QPOS = np.array(
    [-0.08000954, 0.00001946, 0.87989457, 1.0, -0.00002121, 0.00000039, 0.00004552]
)
RESET_FRAME_QPOS = np.array(
    [-0.07911001, 0.00217305, 0.94521802, 0.70699097, 0.00000380, 0.01625869, 0.70703566]
)
RESET_TOOL_QPOS = np.array(
    [0.05106344, -0.21424905, 0.81489473, 0.63776799, -0.00005003, -0.00004143, -0.77022853]
)

THREADING_DEPARTURE_QPOS = np.array([-0.25, 0.65, 0.10, -2.00, -0.25, 2.55, 0.05])
THREADING_STYLE_TASK_HOME_QPOS = THREADING_DEPARTURE_QPOS.copy()
# Match the actual accepted Threading reset spread: robosuite's default is a
# per-joint Gaussian with sigma 0.02 rad. A 3-sigma clip keeps the state pool
# bounded without creating a visibly different distribution.
THREADING_STYLE_START_STD = np.full(7, 0.020)
THREADING_STYLE_START_CLIP = np.full(7, 0.060)


def _yaw_matrix(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _transform_free_joint_qpos(qpos, translation, yaw, pivot):
    """Apply one rigid world transform to a free-joint pose (wxyz quaternion)."""
    qpos = np.asarray(qpos, dtype=float)
    rotation = _yaw_matrix(yaw)
    position = np.asarray(pivot) + rotation.dot(qpos[:3] - pivot) + translation
    nominal_xyzw = np.r_[qpos[4:7], qpos[3]]
    matrix = rotation.dot(T.quat2mat(nominal_xyzw))
    xyzw = T.mat2quat(matrix)
    return np.r_[position, xyzw[3], xyzw[:3]]


class ToolHangWrenchOnly(ToolHang):
    """ToolHang phase 2 with a completed fixture and self-randomizing reset.

    A normal ``reset()`` samples a Threading-style continuous robot start,
    native ToolHang wrench XY / yaw, and a tiny assembled-fixture transform.
    Sampling uses the environment RNG, so equal seeds produce equal reset
    sequences. ``configure_reset_variation`` remains available for exact
    replay, clean scripted tests, and externally specified evaluation splits.
    """

    STAND_ANCHOR = "tool_hang_stage2_stand_anchor"
    FRAME_ANCHOR = "tool_hang_stage2_frame_anchor"
    STAND_WELD = "tool_hang_stage2_stand_weld"
    FRAME_WELD = "tool_hang_stage2_frame_weld"
    # Move both rings 2 cm farther from the wrench center while retaining the
    # 16 cm central high-friction grip. This matches the intended real-world
    # wrist-camera occlusion geometry instead of covering the original metal
    # handle almost edge-to-edge with black material.
    EXTENDED_HANDLE_HALF_LENGTH = 0.1025  # 20.5 cm full silver handle
    EXTENDED_GRIP_HALF_LENGTH = 0.080
    # ToolHang's long opaque handle lies directly between the stock centered
    # Panda wrist camera and the hook during pre-insertion. Use a task-local
    # side-mounted camera baseline (and a modestly wider lens) so the full
    # regime can observe both the ring and hook without changing arm motion.
    WRIST_CAMERA_POS = np.array([0.05, 0.06, 0.0])
    # Keep Panda's native 75 degree lens. The lateral baseline reveals the
    # hook, while the narrower field preserves the intended full / partial
    # distinction as the ring moves farther from the camera.
    WRIST_CAMERA_FOVY_DEG = 75.0
    RESET_CONTROLLER_SETTLE_STEPS = 10

    def _load_model(self):
        self.tool_handle_half_length = self.EXTENDED_HANDLE_HALF_LENGTH
        self.tool_grip_half_length = self.EXTENDED_GRIP_HALF_LENGTH
        self.tool_grip_density = 2000.0 * (0.040 / self.EXTENDED_GRIP_HALF_LENGTH)
        self.tool_grip_friction = (2.0, 0.01, 0.0001)
        super()._load_model()

    def _setup_references(self):
        super()._setup_references()
        camera_id = self.sim.model.camera_name2id("robot0_eye_in_hand")
        self.sim.model.cam_pos[camera_id] = self.WRIST_CAMERA_POS
        self.sim.model.cam_fovy[camera_id] = self.WRIST_CAMERA_FOVY_DEG
        self.sim.forward()

    def configure_reset_variation(self, variation=None):
        """Apply one caller-provided variation on the next reset only."""
        self._pending_reset_variation = variation or {}
        self._has_pending_reset_variation = True

    def _sample_threading_style_robot_reset(self):
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

    def sample_reset_variation(self):
        """Return one seeded reset sample without applying it to the simulator."""
        return deepcopy(self._sample_default_reset_variation())

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
        if not self.deterministic_reset:
            if not has_pending:
                variation = self._sample_default_reset_variation()
            elif variation.get("robot_start_mode") == "threading_continuous" and "robot_qpos" not in variation:
                variation.update(self._sample_threading_style_robot_reset())

        robot_delta = np.asarray(variation.get("robot_joint_delta_rad", np.zeros(7)), dtype=float)
        robot_qpos = np.asarray(variation.get("robot_qpos", RESET_ROBOT_QPOS + robot_delta), dtype=float)
        self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = robot_qpos
        self.sim.data.qvel[self.robots[0]._ref_joint_vel_indexes] = 0.0
        gripper = self.robots[0].gripper["right"]
        for joint, value in zip(gripper.joints, RESET_GRIPPER_QPOS):
            self.sim.data.set_joint_qpos(joint, value)
            self.sim.data.set_joint_qvel(joint, 0.0)

        fixture_translation = np.asarray(variation.get("fixture_translation_m", np.zeros(3)), dtype=float)
        fixture_yaw = float(variation.get("fixture_yaw_rad", 0.0))
        fixture_pivot = RESET_STAND_QPOS[:3]
        stand_qpos = _transform_free_joint_qpos(
            RESET_STAND_QPOS, fixture_translation, fixture_yaw, fixture_pivot
        )
        frame_qpos = _transform_free_joint_qpos(
            RESET_FRAME_QPOS, fixture_translation, fixture_yaw, fixture_pivot
        )
        tool_translation = np.asarray(variation.get("tool_translation_m", np.zeros(3)), dtype=float)
        tool_yaw = float(variation.get("tool_yaw_rad", 0.0))
        tool_qpos = _transform_free_joint_qpos(
            RESET_TOOL_QPOS, tool_translation, tool_yaw, RESET_TOOL_QPOS[:3]
        )
        for obj, qpos in ((self.stand, stand_qpos), (self.frame, frame_qpos), (self.tool, tool_qpos)):
            self.sim.data.set_joint_qpos(obj.joints[0], qpos)
            self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()
        if not self._check_frame_assembled():
            raise RuntimeError("ToolHangWrenchOnly fixture reset is not assembled")
        self._anchor_fixture()
        self._applied_reset_variation = variation
        self._pending_reset_variation = {}
        self._has_pending_reset_variation = False
        self.robots[0].composite_controller.update_state()
        self.robots[0].composite_controller.reset()
        # Data collection historically discarded ten controller-settling
        # steps before frame zero, while a normal evaluation reset exposed
        # the freshly reset controller immediately. That made policy rollout
        # start from different controller / velocity state than every demo.
        # Make settling part of the registered environment reset so bare
        # training and evaluation resets reproduce the demonstrated state.
        settle_action = np.r_[robot_qpos, -1.0]
        for _ in range(self.RESET_CONTROLLER_SETTLE_STEPS):
            super().step(settle_action)
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        self.sim_state_initial = self.sim.get_state()
        self._anchor_fixture()
        self._phase2_reset_settle_applied = True
        return self._get_observations(force_update=True)

    def load_phase2_reference_state(self, flattened_state, fixture_state=None):
        """Load a reference state and anchor its assembled fixture pose."""
        for name in (self.STAND_WELD, self.FRAME_WELD):
            self.sim.data.eq_active[self._equality_id(name)] = 0
        self.sim.set_state_from_flattened(np.asarray(flattened_state, dtype=float))
        if fixture_state is not None:
            fixture_qpos = np.asarray(fixture_state, dtype=float)[1 : 1 + self.sim.model.nq]
            for obj in (self.stand, self.frame):
                joint_id = self.sim.model.joint_name2id(obj.joints[0])
                qpos_address = self.sim.model.jnt_qposadr[joint_id]
                self.sim.data.set_joint_qpos(obj.joints[0], fixture_qpos[qpos_address : qpos_address + 7])
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
