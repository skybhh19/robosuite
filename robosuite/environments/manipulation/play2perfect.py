"""Robot-independent robosuite ports of two Play2Perfect assembly tasks."""

from collections import OrderedDict
import xml.etree.ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    Play2PerfectBeamFixtureObject,
    Play2PerfectBeamPart0Object,
    Play2PerfectBeamPart2Object,
    Play2PerfectForkObject,
    Play2PerfectForkRackObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.utils.fork_appearance import APPEARANCE_PROFILES, apply_fork_appearance
from robosuite.utils.fork_legacy_geometry import EASY_GEOMETRY_PROFILES, LEGACY_PROFILES, PRECISION_PROFILE, PRECISION_WRIST_X_M, PRECISION_WRIST_TILT_DEG, PROFILE as LEGACY_SHAPE_PROFILE, apply_legacy_shape
from robosuite.utils.fork_legacy_geometry import redistribute_fork_mass
from robosuite.utils.observables import Observable, sensor


def _wxyz_to_xyzw(quat):
    quat = np.asarray(quat, dtype=float)
    return quat[[1, 2, 3, 0]]


class _Play2PerfectAssemblyBase(ManipulationEnv):
    """Common table setup, observations, and pose-error utilities."""

    task_objects = ()
    fixture_object = None

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        if getattr(self, 'fork_agentview_profile', 'default') == 'side_hidden':
            from robosuite.utils.fork_agentview_camera import side_agentview_camera
            position, quaternion, fovy = side_agentview_camera()
            arena.set_camera(camera_name='agentview', pos=position, quat=quaternion,
                             camera_attribs={'fovy': str(fovy)})
        arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array("0.75 0 1.55"),
            quat=string_to_array("0.653 0.271 0.271 0.653"),
        )
        # Front view of the Fabrica assembly. Its long part-0 beam lies on
        # world Y, so the generic +X view makes it look incorrectly lateral.
        arena.set_camera(
            camera_name="assemblyview",
            pos=string_to_array("0 -0.75 1.55"),
            quat=string_to_array("0.92362 0.38331 0 0"),
        )
        self._load_task_objects()
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[*self.task_objects, self.fixture_object],
        )

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = {
            obj.name: self.sim.model.body_name2id(obj.root_body)
            for obj in (*self.task_objects, self.fixture_object)
        }

    def _body_pose(self, obj):
        body_id = self.object_body_ids[obj.name]
        pos = np.array(self.sim.data.body_xpos[body_id])
        quat = T.mat2quat(np.array(self.sim.data.body_xmat[body_id]).reshape(3, 3))
        return pos, quat

    def _goal_pose(self, local_pos, local_quat_wxyz):
        fixture_pos, fixture_quat = self._body_pose(self.fixture_object)
        fixture_pose = T.pose2mat((fixture_pos, fixture_quat))
        local_pose = T.pose2mat((np.asarray(local_pos), _wxyz_to_xyzw(local_quat_wxyz)))
        return T.mat2pose(fixture_pose @ local_pose)

    def _pose_error(self, obj, local_pos, local_quat_wxyz):
        pos, quat = self._body_pose(obj)
        goal_pos, goal_quat = self._goal_pose(local_pos, local_quat_wxyz)
        pos_error = float(np.linalg.norm(pos - goal_pos))
        delta = T.quat_distance(goal_quat.copy(), quat.copy())
        rot_error = float(np.linalg.norm(T.quat2axisangle(delta)))
        return pos_error, rot_error

    @staticmethod
    def _pose_is_close(error, pos_tolerance, rot_tolerance):
        return error[0] < pos_tolerance and error[1] < rot_tolerance

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        def make_pos_sensor(obj):
            @sensor(modality=modality)
            def object_pos(obs_cache):
                return self._body_pose(obj)[0]

            object_pos.__name__ = f"{obj.name}_pos"
            return object_pos

        def make_quat_sensor(obj):
            @sensor(modality=modality)
            def object_quat(obs_cache):
                return self._body_pose(obj)[1]

            object_quat.__name__ = f"{obj.name}_quat"
            return object_quat

        for obj in (*self.task_objects, self.fixture_object):
            for obs_sensor in (make_pos_sensor(obj), make_quat_sensor(obj)):
                observables[obs_sensor.__name__] = Observable(
                    name=obs_sensor.__name__, sensor=obs_sensor, sampling_rate=self.control_freq
                )
        for name, obs_sensor in self._task_observable_sensors().items():
            observables[name] = Observable(name=name, sensor=obs_sensor, sampling_rate=self.control_freq)
        return observables

    def _task_observable_sensors(self):
        return OrderedDict()

    def _set_free_object_pose(self, obj, pos, quat_wxyz):
        qpos = np.concatenate((np.asarray(pos, dtype=float), np.asarray(quat_wxyz, dtype=float)))
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)

    def _sample_yaw_quat(self, center=0.0, half_width=np.pi):
        yaw = self.rng.uniform(center - half_width, center + half_width)
        return np.array((np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)))


class YCBForkInRack(_Play2PerfectAssemblyBase):
    """Insert a real-scale YCB fork into the narrow opening of a rack."""

    # The released reset data contains a pre-insert waypoint at 0.141277 m
    # and the actual seated pose at 0.063277 m above the rack frame.
    FORK_GOAL_POS = np.array((0.0, 0.0, 0.063277))
    FORK_GOAL_QUAT = np.array((np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0))
    POS_TOLERANCE = 0.018
    ROT_TOLERANCE = np.deg2rad(20.0)

    def __init__(
        self, *args, grasp_contact_profile="legacy", fork_geometry_profile="standard",
        fork_appearance_profile="default", fork_handle_extension_m=0.0,
        fork_grasp_front_extension_m=0.0,
        fork_wrist_camera_x_m=None, fork_wrist_camera_tilt_deg=None,
        fork_wrist_camera_z_m=0.0, fork_wrist_camera_y_m=0.0,
        fork_wrist_camera_pan_deg=0.0, fork_wrist_camera_roll_deg=0.0,
        fork_wrist_camera_fovy_deg=40.0,
        fork_agentview_profile="default",
        fork_continuous_grasp=False, fork_pad_contact_radius_m=0.0,
        fork_guide_slide_friction=0.2, fork_mass_redistribution_kg=0.0,
        fork_finger_torsional_friction_m=0.005, fork_pad_contact_dynamics='legacy',
        fork_gripper_command_rate_per_s=0., fork_gripper_position_gain=0., **kwargs
    ):
        if grasp_contact_profile not in ("legacy", "balanced", "moderate"):
            raise ValueError(f"Unknown fork grasp contact profile: {grasp_contact_profile}")
        if fork_geometry_profile not in (
            "standard", "thick_wide", "equal_grasp_wide", "uniform_grasp_wide", *EASY_GEOMETRY_PROFILES
        ):
            raise ValueError(f"Unknown fork geometry profile: {fork_geometry_profile}")
        self.grasp_contact_profile = grasp_contact_profile
        from robosuite.utils.fork_grasp_contacts import CONTACT_DYNAMICS
        if fork_pad_contact_dynamics not in CONTACT_DYNAMICS:
            raise ValueError('Unknown fork pad contact dynamics')
        if fork_pad_contact_dynamics != 'legacy' and grasp_contact_profile != 'moderate':
            raise ValueError('Grasp contact dynamics requires the moderate profile')
        self.fork_pad_contact_dynamics = fork_pad_contact_dynamics
        if (not np.isfinite(fork_gripper_command_rate_per_s)
                or fork_gripper_command_rate_per_s != 0 and not 2.8 <= fork_gripper_command_rate_per_s <= 12.):
            raise ValueError('Invalid time-scaled Panda command rate')
        self.fork_gripper_command_rate_per_s = float(fork_gripper_command_rate_per_s)
        from robosuite.utils.fork_gripper_timing import validate_gripper_position_gain
        self.fork_gripper_position_gain = validate_gripper_position_gain(fork_gripper_position_gain)
        if self.fork_gripper_position_gain and grasp_contact_profile != 'moderate':
            raise ValueError('Adjusted gripper gain requires the moderate contact profile')
        if (not np.isfinite(fork_finger_torsional_friction_m)
                or not .001 <= fork_finger_torsional_friction_m <= .015):
            raise ValueError('Finger torsional contact parameter must be within 1..15 mm')
        if grasp_contact_profile != 'moderate' and fork_finger_torsional_friction_m != .005:
            raise ValueError('Adjusted finger torsion requires the moderate contact profile')
        self.fork_finger_torsional_friction_m = float(fork_finger_torsional_friction_m)
        if fork_agentview_profile not in ('default', 'side_hidden'):
            raise ValueError(f'Unknown Fork agentview profile: {fork_agentview_profile}')
        self.fork_agentview_profile = fork_agentview_profile
        self.fork_geometry_profile = fork_geometry_profile
        if not np.isfinite(fork_handle_extension_m) or not 0 <= fork_handle_extension_m <= .08:
            raise ValueError("Fork handle extension must be between 0 and 80 mm")
        if fork_handle_extension_m and fork_geometry_profile != PRECISION_PROFILE:
            raise ValueError("Extended handle requires the opt-in precision profile")
        self.fork_handle_extension_m = float(fork_handle_extension_m)
        if not np.isfinite(fork_grasp_front_extension_m) or not 0 <= fork_grasp_front_extension_m <= .025:
            raise ValueError('Front grasp extension must be between 0 and 25 mm')
        if fork_grasp_front_extension_m and fork_geometry_profile != PRECISION_PROFILE:
            raise ValueError('Front grasp extension requires the opt-in precision profile')
        self.fork_grasp_front_extension_m = float(fork_grasp_front_extension_m)
        if fork_continuous_grasp and fork_geometry_profile != PRECISION_PROFILE:
            raise ValueError("Continuous grasp rail requires the precision profile")
        self.fork_continuous_grasp = bool(fork_continuous_grasp)
        if not np.isfinite(fork_guide_slide_friction) or not .1 <= fork_guide_slide_friction <= .6:
            raise ValueError('Passive guide sliding friction must be between 0.1 and 0.6')
        if fork_guide_slide_friction != .2 and fork_geometry_profile != PRECISION_PROFILE:
            raise ValueError('Adjusted passive guide friction requires the precision profile')
        self.fork_guide_slide_friction = float(fork_guide_slide_friction)
        if not np.isfinite(fork_mass_redistribution_kg) or not 0 <= fork_mass_redistribution_kg <= .015:
            raise ValueError('Redistributed fork mass must be between zero and 15 grams')
        if fork_mass_redistribution_kg and fork_geometry_profile != PRECISION_PROFILE:
            raise ValueError('Density redistribution requires the opt-in precision profile')
        self.fork_mass_redistribution_kg = float(fork_mass_redistribution_kg)
        # Keep zero-valued diagnostic metadata readable. The finite-pad
        # experiment did not improve robustness and is not a supported profile.
        if not np.isfinite(fork_pad_contact_radius_m) or fork_pad_contact_radius_m != 0:
            raise ValueError('Finite pad contact experiment is retired; use the moderate default contacts')
        self.FORK_GOAL_POS = type(self).FORK_GOAL_POS.copy()
        self.FORK_GOAL_POS[2] += self.fork_handle_extension_m
        self.fork_wrist_camera_x_m = fork_wrist_camera_x_m
        self.fork_wrist_camera_tilt_deg = fork_wrist_camera_tilt_deg
        if not np.isfinite(fork_wrist_camera_z_m) or not -.10 <= fork_wrist_camera_z_m <= .12:
            raise ValueError("Wrist camera Z offset is outside the calibrated bracket")
        self.fork_wrist_camera_z_m = float(fork_wrist_camera_z_m)
        camera_y_limit = .30 if fork_geometry_profile == PRECISION_PROFILE else .15
        if not np.isfinite(fork_wrist_camera_y_m) or abs(fork_wrist_camera_y_m) > camera_y_limit:
            raise ValueError(f'Wrist camera Y must be between {-1000 * camera_y_limit:g} '
                             f'and {1000 * camera_y_limit:g} mm')
        self.fork_wrist_camera_y_m = float(fork_wrist_camera_y_m)
        if not np.isfinite(fork_wrist_camera_pan_deg) or abs(fork_wrist_camera_pan_deg) > 90:
            raise ValueError('Wrist camera pan must be between -90 and 90 degrees')
        self.fork_wrist_camera_pan_deg = float(fork_wrist_camera_pan_deg)
        if not np.isfinite(fork_wrist_camera_roll_deg) or abs(fork_wrist_camera_roll_deg) > 45:
            raise ValueError('Wrist camera roll must be between -45 and 45 degrees')
        if fork_wrist_camera_roll_deg and fork_geometry_profile != PRECISION_PROFILE:
            raise ValueError('Optical-axis roll requires the opt-in precision profile')
        self.fork_wrist_camera_roll_deg = float(fork_wrist_camera_roll_deg)
        if not np.isfinite(fork_wrist_camera_fovy_deg) or not 25 <= fork_wrist_camera_fovy_deg <= 90:
            raise ValueError('Shared wrist camera FOV must be within [25,90] degrees')
        self.fork_wrist_camera_fovy_deg = float(fork_wrist_camera_fovy_deg)
        if fork_appearance_profile not in APPEARANCE_PROFILES:
            raise ValueError(f"Unknown fork appearance profile: {fork_appearance_profile}")
        if fork_appearance_profile != "default" and fork_geometry_profile not in ("easy_grasp", "easy_grasp_guided"):
            raise ValueError("legacy_style appearance requires easy_grasp or easy_grasp_guided geometry")
        self.fork_appearance_profile = fork_appearance_profile
        # Optional byte-identical input cache for cloud-offloaded assets.
        # Unset by default; does not alter XML or relative mesh resolution.
        from robosuite.utils.fork_xml_cache import enable_from_environment
        enable_from_environment()
        super().__init__(*args, **kwargs)

    def _setup_references(self):
        super()._setup_references()
        if self.fork_gripper_command_rate_per_s:
            from robosuite.utils.fork_gripper_timing import configure_gripper_timing
            configure_gripper_timing(self.robots[0].gripper['right'], self.control_freq,
                                     self.fork_gripper_command_rate_per_s)

    def _load_task_objects(self):
        self.fork = Play2PerfectForkObject(name="fork")
        self.rack = Play2PerfectForkRackObject(name="rack")
        # Shift the fixture into the Panda's side-grasp workspace. The
        # fork-to-rack goal remains identical in the rack-local frame.
        self.rack.set_pos((-0.08, 0.10, self.table_offset[2] + 0.0425))
        if self.fork_geometry_profile in EASY_GEOMETRY_PROFILES:
            # Compile the new geometry, not just runtime geom_size: MuJoCo's
            # static BVH/bounds must also represent the longer/thicker rail.
            self._configure_easy_grasp_xml()
            if self.fork_geometry_profile in ("easy_grasp_guided", *LEGACY_PROFILES):
                self._configure_rack_guide_xml()
            if self.fork_geometry_profile in LEGACY_PROFILES:
                apply_legacy_shape(self.fork.get_obj(), self.rack.get_obj(), self.fork.asset, self.rack.asset,
                                   precision=self.fork_geometry_profile == PRECISION_PROFILE,
                                   extension_m=self.fork_handle_extension_m,
                                   front_extension_m=self.fork_grasp_front_extension_m,
                                   guide_slide_friction=self.fork_guide_slide_friction)
                additions = []
                redistribute_fork_mass(self.fork.get_obj(), self.fork_mass_redistribution_kg)
                if self.fork_handle_extension_m:
                    length = self.fork_handle_extension_m
                    additions.append((length, 800. * .018 * .020 * length,
                                      np.array((-.088 - length / 2, 0., 0.))))
                if self.fork_grasp_front_extension_m:
                    # Approximate the added rounded shank at the same 800
                    # kg/m^3 synthetic-handle density, rather than keeping
                    # the old inertia for a visibly larger physical object.
                    length = self.fork_grasp_front_extension_m
                    area = .018 * .020 - (4 - np.pi) * .003**2
                    additions.append((length, 800. * area * length,
                                      np.array((.017 + length / 2, 0., 0.))))
                for length, added_mass, added_center in additions:
                    # Add a constant-density tail, instead of silently giving
                    # a longer object its previous centre of mass / inertia.
                    inertial = self.fork.get_obj().find('.//inertial')
                    old_mass = float(inertial.get('mass'))
                    old_center = np.fromstring(inertial.get('pos'), sep=' ')
                    xx, yy, zz, xy, xz, yz = np.fromstring(inertial.get('fullinertia'), sep=' ')
                    tensor = np.array(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))
                    mass = old_mass + added_mass
                    center = (old_mass * old_center + added_mass * added_center) / mass
                    added_tensor = np.diag(added_mass / 12 * np.array((.018**2 + .020**2,
                                            length**2 + .018**2, length**2 + .020**2)))
                    for component_mass, component_center in ((old_mass, old_center), (added_mass, added_center)):
                        delta = component_center - center
                        tensor += component_mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
                    tensor += added_tensor
                    inertial.set('mass', str(mass))
                    inertial.set('pos', ' '.join(map(str, center)))
                    inertial.set('fullinertia', ' '.join(map(str, (tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                                   tensor[0, 1], tensor[0, 2], tensor[1, 2]))))
                if self.fork_continuous_grasp:
                    # The red handle and silver neck formerly supplied two
                    # simultaneous contact manifolds near the neck grasp.
                    # One convex rail uses their actual union, preserving the
                    # visible surfaces/section rather than a larger hidden box.
                    vertices = []
                    for name in ('fork_legacy_handle_mesh', 'fork_legacy_neck_mesh'):
                        vertices.extend(np.fromstring(self.fork.asset.find(f"mesh[@name='{name}']").get('vertex'), sep=' '))
                    ET.SubElement(self.fork.asset, 'mesh', name='fork_continuous_grasp_mesh',
                                  vertex=' '.join(map(str, vertices)))
                    self.fork.get_obj().find(".//geom[@name='fork_handle_collision']").set('mesh', 'fork_continuous_grasp_mesh')
                    neck = self.fork.get_obj().find(".//geom[@name='fork_neck_collision']")
                    neck.set('contype', '0'); neck.set('conaffinity', '0')
            else:
                apply_fork_appearance(self.fork.get_obj(), self.rack.get_obj(), self.fork_appearance_profile)
        if self.grasp_contact_profile == "moderate":
            # condim is discrete: mj_saveLastXML does not reliably copy a
            # runtime edit back to MJCF. Bake it into both object and pads.
            for geom in self.fork.get_obj().findall(".//geom"):
                if "handle_collision" in geom.get("name", "") or "neck_collision" in geom.get("name", ""):
                    geom.set("condim", "4")
                    geom.set("friction", "1 0.005 0.0001")
            for geom in self.robots[0].robot_model.root.findall(".//geom"):
                if "finger" in geom.get("name", "") or "pad" in geom.get("name", ""):
                    geom.set("condim", "4")
                    geom.set("friction", "1.5 0.005 0.0001")
        self.task_objects = (self.fork,)
        self.fixture_object = self.rack
        from robosuite.utils.fork_grasp_contacts import add_grasp_contact_pairs
        add_grasp_contact_pairs(self.robots[0].robot_model.contact,
            self.fork_pad_contact_dynamics, self.fork_finger_torsional_friction_m)

    def _configure_easy_grasp_xml(self):
        fork = self.fork.get_obj()
        rack = self.rack.get_obj()
        handle = fork.find(".//geom[@name='fork_handle_collision']")
        handle.set("pos", "-0.0365 0 0")
        handle.set("size", "0.0625 0.012 0.012")
        handle.set("group", "1")
        handle.set("rgba", "0.78 0.12 0.12 1")
        neck = fork.find(".//geom[@name='fork_neck_collision']")
        neck.set("contype", "0")
        neck.set("conaffinity", "0")
        for name in ("fork_handle_visual", "fork_handle_visual_thick"):
            fork.find(f".//geom[@name='{name}']").set("rgba", "0.78 0.12 0.12 0")
        for name in ("rack_left", "rack_right"):
            rack.find(f".//geom[@name='{name}']").set("size", "0.012858 0.035 0.04")
        for name in ("rack_front", "rack_back"):
            rack.find(f".//geom[@name='{name}']").set("size", "0.014 0.008673 0.04")
        for name, pos in (
            ("rack_slot_rim_left", "-0.015 0 0.0435"),
            ("rack_slot_rim_right", "0.015 0 0.0435"),
            ("rack_slot_rim_front", "0 0.015 0.0435"),
            ("rack_slot_rim_back", "0 -0.015 0.0435"),
        ):
            rack.find(f".//geom[@name='{name}']").set("pos", pos)
            rack.find(f".//geom[@name='{name}']").set(
                "size", "0.001 0.016 0.00045" if "left" in name or "right" in name else "0.016 0.001 0.00045"
            )
        rack.find(".//geom[@name='rack_visual']").set("rgba", "0.65 0.65 0.65 0")
        for name in ("rack_base", "rack_left", "rack_right", "rack_front", "rack_back"):
            rack.find(f".//geom[@name='{name}']").set("group", "1")

    def _reset_internal(self):
        super()._reset_internal()
        # easy_grasp geometry is already baked into the compiled MJCF.
        if self.fork_geometry_profile in (
            "thick_wide", "equal_grasp_wide", "uniform_grasp_wide"
        ):
            model = self.sim.model
            model.geom_size[model.geom_name2id("fork_handle_collision")] = (
                0.036, 0.0100, 0.0077
            )
            model.geom_size[model.geom_name2id("fork_neck_collision")] = (
                (0.030, 0.0100, 0.0077)
                if self.fork_geometry_profile == "equal_grasp_wide"
                else (0.030, 0.0093, 0.0065)
            )
            if self.fork_geometry_profile == "equal_grasp_wide":
                # Leave a small non-grasped gap between the two equal contact
                # boxes. This prevents a finger body from simultaneously
                # loading both segments when OSC tracking lands a few mm off.
                model.geom_size[model.geom_name2id("fork_neck_collision"), 0] = 0.022
            if self.fork_geometry_profile == "uniform_grasp_wide":
                # One continuous rail covers both Full and Partial grasp
                # intervals, so label-dependent grasp points see identical
                # width, thickness, friction, and contact dimensionality.
                handle_id = model.geom_name2id("fork_handle_collision")
                model.geom_pos[handle_id, 0] = -0.0365
                model.geom_size[handle_id] = (0.0625, 0.0100, 0.0077)
                neck_id = model.geom_name2id("fork_neck_collision")
                model.geom_contype[neck_id] = 0
                model.geom_conaffinity[neck_id] = 0
            model.geom_rgba[model.geom_name2id("fork_handle_visual"), 3] = 0.0
            thick_handle_visual_id = model.geom_name2id("fork_handle_visual_thick")
            model.geom_rgba[thick_handle_visual_id, 3] = 1.0
            model.geom_group[thick_handle_visual_id] = 1
            if self.fork_geometry_profile in ("equal_grasp_wide", "uniform_grasp_wide"):
                model.geom_rgba[model.geom_name2id("fork_steel_visual"), 3] = 0.0
                thick_steel_visual_id = model.geom_name2id("fork_steel_visual_thick")
                model.geom_rgba[thick_steel_visual_id, 3] = 1.0
                model.geom_group[thick_steel_visual_id] = 1
            # Widen both aperture axes by 3 mm by thinning the four walls.
            # Keep the outer rack footprint fixed for finger clearance.
            for name in ("rack_left", "rack_right"):
                model.geom_size[model.geom_name2id(name), 0] = 0.0166415
            for name in ("rack_front", "rack_back"):
                model.geom_size[model.geom_name2id(name), 1] = 0.010827
            # Absolute coordinates avoid accumulation with hard_reset=False.
            for name, axis, value in (
                ("rack_slot_rim_left", 0, -0.0112),
                ("rack_slot_rim_right", 0, 0.0112),
                ("rack_slot_rim_front", 1, 0.0129),
                ("rack_slot_rim_back", 1, -0.0129),
            ):
                model.geom_pos[model.geom_name2id(name), axis] = value
            # The source rack mesh has a fixed aperture. Render the exact new
            # collision walls instead, so RGB and contact geometry agree.
            model.geom_rgba[model.geom_name2id("rack_visual"), 3] = 0.0
            for name in ("rack_base", "rack_left", "rack_right", "rack_front", "rack_back"):
                model.geom_group[model.geom_name2id(name)] = 1
        if self.grasp_contact_profile == "balanced":
            # One shared physical setup for both observation labels. The Full
            # grasp lands on the handle, whose torsional friction prevents
            # contact-induced rotation during insertion; Partial lands on the
            # neck. This is an object property, never a regime switch.
            for geom_id, name in enumerate(self.sim.model.geom_names):
                if not name:
                    continue
                if name.startswith("fork_") and "handle_collision" in name:
                    self.sim.model.geom_friction[geom_id] = (4.0, 0.3, 0.02)
                    self.sim.model.geom_condim[geom_id] = 6
                elif name.startswith("fork_") and "neck_collision" in name:
                    self.sim.model.geom_friction[geom_id] = (
                        (4.0, 0.3, 0.02)
                        if self.fork_geometry_profile == "equal_grasp_wide"
                        else (4.0, 0.08, 0.02)
                    )
                    self.sim.model.geom_condim[geom_id] = 6
                elif "finger" in name or "pad" in name:
                    self.sim.model.geom_friction[geom_id] = (8.0, 0.08, 0.02)
                    self.sim.model.geom_condim[geom_id] = 6
            for actuator_id, name in enumerate(self.sim.model.actuator_names):
                if name and ("gripper" in name or "finger" in name):
                    self.sim.model.actuator_forcerange[actuator_id] = (-100.0, 100.0)
                    self.sim.model.actuator_forcelimited[actuator_id] = 1
            wrist_camera_id = self.sim.model.camera_name2id("robot0_eye_in_hand")
            self.sim.model.cam_fovy[wrist_camera_id] = 40.0
        if self.grasp_contact_profile == "moderate":
            self.configure_moderate_grasp_contacts()
        if not self.deterministic_reset:
            pos = (
                self.rng.uniform(0.08, 0.16),
                self.rng.uniform(-0.14, -0.06),
                self.table_offset[2] + 0.010,
            )
            self._set_free_object_pose(self.fork, pos, self._sample_yaw_quat(0.0, np.pi / 4.0))

    def _configure_rack_guide_xml(self):
        """Shared passive keyed seat: original 28 mm mouth, 24.4 mm floor.

        Four convex guide inserts taper over 10 mm to a 25 mm long lower
        sleeve. They remove free axial twist after release without changing
        the fork, friction, top aperture, cameras, or goal pose. Compile the
        actual guide geometry, including its BVH; never edit runtime bounds.
        """
        for label, axis, sign in (("left", 0, -1), ("right", 0, 1),
                                  ("front", 1, 1), ("back", 1, -1)):
            vertices = []
            for z, depths in ((-0.0375, (0.014, 0.0122)),
                              (-0.0125, (0.014, 0.0122)),
                              (-0.0025, (0.014,))):
                for depth in depths:
                    for cross in (-0.014, 0.014):
                        vertex = [0.0, 0.0, z]
                        vertex[axis] = sign * depth
                        vertex[1 - axis] = cross
                        vertices.append(vertex)
            mesh_name = f"rack_guide_{label}_mesh"
            ET.SubElement(self.rack.asset, "mesh", name=mesh_name,
                          vertex=" ".join(str(v) for p in vertices for v in p))
            ET.SubElement(self.rack.get_obj(), "geom", name=f"rack_guide_{label}",
                          type="mesh", mesh=mesh_name, group="1", contype="1", conaffinity="1",
                          rgba="0.47 0.47 0.47 1", friction="0.5 0.001 0.00001",
                          priority="1", solref="0.006 1", solimp="0.99 0.99 0.001")

    def configure_moderate_grasp_contacts(self):
        """One shared, non-sticky contact setup, also restored by collectors."""
        for geom_id, name in enumerate(self.sim.model.geom_names):
            if not name:
                continue
            if name.startswith("fork_") and ("handle_collision" in name or "neck_collision" in name):
                self.sim.model.geom_friction[geom_id] = (1.0, 0.005, 0.0001)
                self.sim.model.geom_condim[geom_id] = 4
            elif "finger" in name or "pad" in name:
                # Shared soft-finger anti-twist parameter; default preserves
                # exact V11 contacts. Sliding/rolling, force, and condim stay
                # unchanged. No label enters this configuration.
                self.sim.model.geom_friction[geom_id] = (
                    1.5, self.fork_finger_torsional_friction_m, 0.0001)
                self.sim.model.geom_condim[geom_id] = 4
        for actuator_id, name in enumerate(self.sim.model.actuator_names):
            if name and ("gripper" in name or "finger" in name):
                self.sim.model.actuator_forcerange[actuator_id] = (-60.0, 60.0)
                self.sim.model.actuator_forcelimited[actuator_id] = 1
        from robosuite.utils.fork_gripper_timing import configure_gripper_position_gain
        configure_gripper_position_gain(self.sim.model, self.fork_gripper_position_gain)
        self.sim.model.cam_fovy[self.sim.model.camera_name2id("robot0_eye_in_hand")] = self.fork_wrist_camera_fovy_deg
        camera = self.sim.model.camera_name2id("robot0_eye_in_hand")
        # Identical mount for both labels. The precision mount shifts forward
        # and tilts back for a larger Full-visible / Partial-hidden margin.
        # Other profiles retain their original shared 6-degree mount.
        base = np.array((0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0))
        half_tilt_deg = PRECISION_WRIST_TILT_DEG / 2 if self.fork_geometry_profile == PRECISION_PROFILE else 3.0
        if self.fork_geometry_profile == PRECISION_PROFILE:
            camera_x = PRECISION_WRIST_X_M if self.fork_wrist_camera_x_m is None else self.fork_wrist_camera_x_m
            self.sim.model.cam_pos[camera] = (camera_x, self.fork_wrist_camera_y_m,
                                            self.fork_wrist_camera_z_m)
            if self.fork_wrist_camera_tilt_deg is not None:
                half_tilt_deg = self.fork_wrist_camera_tilt_deg / 2
        tilt = np.array((np.cos(np.deg2rad(half_tilt_deg)), np.sin(np.deg2rad(half_tilt_deg)), 0.0, 0.0))
        quat = T.quat_multiply(base[[1, 2, 3, 0]], tilt[[1, 2, 3, 0]])
        if self.fork_geometry_profile == PRECISION_PROFILE and self.fork_wrist_camera_pan_deg:
            half_pan = np.deg2rad(self.fork_wrist_camera_pan_deg / 2)
            quat = T.quat_multiply(quat, np.array((0., np.sin(half_pan), 0., np.cos(half_pan))))
        self.sim.model.cam_quat[camera] = quat[[3, 0, 1, 2]]
        if self.fork_geometry_profile == PRECISION_PROFILE and self.fork_wrist_camera_roll_deg:
            from robosuite.utils.fork_legacy_geometry import shared_wrist_quaternion
            self.sim.model.cam_quat[camera] = shared_wrist_quaternion(
                2 * half_tilt_deg, self.fork_wrist_camera_pan_deg, self.fork_wrist_camera_roll_deg)

    def _fork_error(self):
        return self._pose_error(self.fork, self.FORK_GOAL_POS, self.FORK_GOAL_QUAT)

    def _check_success(self):
        return self._pose_is_close(self._fork_error(), self.POS_TOLERANCE, self.ROT_TOLERANCE)

    def reward(self, action=None):
        pos_error, rot_error = self._fork_error()
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            reward = 0.7 * (1.0 - np.tanh(10.0 * pos_error)) + 0.3 * (
                1.0 - np.tanh(2.0 * rot_error)
            )
            reward = min(float(reward), 0.99)
        else:
            reward = 0.0
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _task_observable_sensors(self):
        modality = "object"

        @sensor(modality=modality)
        def fork_goal_pos(obs_cache):
            return self._goal_pose(self.FORK_GOAL_POS, self.FORK_GOAL_QUAT)[0]

        @sensor(modality=modality)
        def fork_goal_quat(obs_cache):
            return self._goal_pose(self.FORK_GOAL_POS, self.FORK_GOAL_QUAT)[1]

        @sensor(modality=modality)
        def fork_pose_error(obs_cache):
            return np.asarray(self._fork_error())

        @sensor(modality=modality)
        def fork_inserted(obs_cache):
            return np.array([float(self._check_success())])

        return OrderedDict(
            (s.__name__, s) for s in (fork_goal_pos, fork_goal_quat, fork_pose_error, fork_inserted)
        )


class MultiPartAssembly(_Play2PerfectAssemblyBase):
    """Two-step Fabrica beam assembly: insert part 2, then part 0."""

    PART2_GOAL_POS = np.array((-0.047625, 0.0, 0.113075179))
    PART2_GOAL_QUAT = np.array((np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0))
    PART0_GOAL_POS = np.array((-0.047625, 0.0, 0.213364889))
    PART0_GOAL_QUAT = np.array((0.5, 0.5, -0.5, -0.5))
    POS_TOLERANCE = 0.018
    ROT_TOLERANCE = np.deg2rad(20.0)

    def _load_task_objects(self):
        self.part0 = Play2PerfectBeamPart0Object(name="beam_part0")
        self.part2 = Play2PerfectBeamPart2Object(name="beam_part2")
        self.fixture = Play2PerfectBeamFixtureObject(name="beam_fixture")
        self.fixture.set_pos((0.0, 0.10, self.table_offset[2]))
        self.task_objects = (self.part2, self.part0)
        self.fixture_object = self.fixture

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            z = self.table_offset[2] + 0.020
            self._set_free_object_pose(
                self.part2,
                (self.rng.uniform(0.08, 0.14), self.rng.uniform(0.13, 0.19), z),
                self._sample_yaw_quat(np.pi / 2.0, np.pi / 6.0),
            )
            self._set_free_object_pose(
                self.part0,
                (self.rng.uniform(0.08, 0.14), self.rng.uniform(-0.15, -0.09), z),
                self._sample_yaw_quat(np.pi / 2.0, np.pi / 6.0),
            )

    def _part_errors(self):
        return (
            self._pose_error(self.part2, self.PART2_GOAL_POS, self.PART2_GOAL_QUAT),
            self._pose_error(self.part0, self.PART0_GOAL_POS, self.PART0_GOAL_QUAT),
        )

    def _stage_success(self):
        return tuple(
            self._pose_is_close(error, self.POS_TOLERANCE, self.ROT_TOLERANCE)
            for error in self._part_errors()
        )

    def _check_success(self):
        return all(self._stage_success())

    def reward(self, action=None):
        errors = self._part_errors()
        done = self._stage_success()
        if all(done):
            reward = 1.0
        elif self.reward_shaping:
            scores = [
                0.7 * (1.0 - np.tanh(10.0 * p)) + 0.3 * (1.0 - np.tanh(2.0 * r))
                for p, r in errors
            ]
            # Gate part-0 shaping on completion of part 2 so the dense reward
            # preserves the physical assembly order instead of encouraging two
            # independent insertions in parallel.
            reward = 0.45 * scores[0] + 0.05 * float(done[0])
            if done[0]:
                reward += 0.45 * scores[1] + 0.05 * float(done[1])
            reward = min(float(reward), 0.99)
        else:
            reward = 0.0
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _task_observable_sensors(self):
        modality = "object"
        sensors = OrderedDict()
        specs = (
            ("part2", self.part2, self.PART2_GOAL_POS, self.PART2_GOAL_QUAT),
            ("part0", self.part0, self.PART0_GOAL_POS, self.PART0_GOAL_QUAT),
        )
        for label, obj, local_pos, local_quat in specs:
            def goal_pos(obs_cache, lp=local_pos, lq=local_quat):
                return self._goal_pose(lp, lq)[0]

            def goal_quat(obs_cache, lp=local_pos, lq=local_quat):
                return self._goal_pose(lp, lq)[1]

            def pose_error(obs_cache, o=obj, lp=local_pos, lq=local_quat):
                return np.asarray(self._pose_error(o, lp, lq))

            goal_pos.__name__ = f"{label}_goal_pos"
            goal_quat.__name__ = f"{label}_goal_quat"
            pose_error.__name__ = f"{label}_pose_error"
            for obs_sensor in (goal_pos, goal_quat, pose_error):
                sensors[obs_sensor.__name__] = sensor(modality=modality)(obs_sensor)

        @sensor(modality=modality)
        def assembly_stage(obs_cache):
            part2_done, part0_done = self._stage_success()
            return np.array((float(part2_done), float(part0_done)))

        sensors[assembly_stage.__name__] = assembly_stage
        return sensors
