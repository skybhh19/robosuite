"""Robot-independent robosuite ports of two Play2Perfect assembly tasks."""

from collections import OrderedDict

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

    def _load_task_objects(self):
        self.fork = Play2PerfectForkObject(name="fork")
        self.rack = Play2PerfectForkRackObject(name="rack")
        # Shift the fixture into the Panda's side-grasp workspace. The
        # fork-to-rack goal remains identical in the rack-local frame.
        self.rack.set_pos((-0.08, 0.10, self.table_offset[2] + 0.0425))
        self.task_objects = (self.fork,)
        self.fixture_object = self.rack

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            pos = (
                self.rng.uniform(0.08, 0.16),
                self.rng.uniform(-0.14, -0.06),
                self.table_offset[2] + 0.010,
            )
            self._set_free_object_pose(self.fork, pos, self._sample_yaw_quat(0.0, np.pi / 4.0))

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
