"""Single-arm needle-threading task."""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import NeedleObject, RingTripodObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class Threading(ManipulationEnv):
    """Single-arm task where a robot inserts a needle through a small ring."""

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
        render_camera="frontview",
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
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
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

    def reward(self, action=None):
        """Sparse reward: 1 when the needle tip is inside the tripod ring."""
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        self._add_agentview_full_camera(mujoco_arena)

        self.needle = NeedleObject(name="needle_obj")
        self.tripod = RingTripodObject(name="tripod_obj")

        self._get_placement_initializer()
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.needle, self.tripod],
        )

    def _add_agentview_full_camera(self, arena):
        """Add MimicGen's wider tabletop camera."""
        arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array("0.753078462147161 2.062036796036723e-08 1.5194726087166726"),
            quat=string_to_array("0.6432409286499023 0.293668270111084 0.2936684489250183 0.6432408690452576"),
        )

    def _get_initial_placement_bounds(self):
        return {
            "needle": {
                "x": (-0.2, -0.05),
                "y": (0.15, 0.25),
                "z_rot": (-2.0 * np.pi / 3.0 + np.pi, -np.pi / 3.0 + np.pi),
                "reference": self.table_offset,
            },
            "tripod": {
                "x": (0.0, 0.0),
                "y": (-0.15, -0.15),
                "z_rot": (np.pi / 2.0, np.pi / 2.0),
                "reference": self.table_offset,
            },
        }

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="NeedleSampler",
                mujoco_objects=self.needle,
                x_range=bounds["needle"]["x"],
                y_range=bounds["needle"]["y"],
                rotation=bounds["needle"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["needle"]["reference"],
                z_offset=0.0,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="TripodSampler",
                mujoco_objects=self.tripod,
                x_range=bounds["tripod"]["x"],
                y_range=bounds["tripod"]["y"],
                rotation=bounds["tripod"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["tripod"]["reference"],
                z_offset=0.001,
                rng=self.rng,
            )
        )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {
            "needle": self.sim.model.body_name2id(self.needle.root_body),
            "tripod": self.sim.model.body_name2id(self.tripod.root_body),
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        self._threading_initial_tripod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["tripod"]])
        self._threading_max_insert_progress = -np.inf

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        @sensor(modality=modality)
        def world_pose_in_gripper(obs_cache):
            if f"{pf}eef_pos" not in obs_cache or f"{pf}eef_quat" not in obs_cache:
                return np.eye(4)
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"])))

        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        actives = [False]

        for obj_name in self.obj_body_id:
            obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
            sensors += obj_sensors
            names += obj_sensor_names
            actives += [True] * len(obj_sensors)

        for name, obs_sensor, active in zip(names, sensors, actives):
            observables[name] = Observable(
                name=name,
                sensor=obs_sensor,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            required = [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]
            if any(name not in obs_cache for name in required):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache.get(f"{obj_name}_to_{pf}eef_quat", np.zeros(4))

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]
        return sensors, names

    def _check_success(self):
        """Check whether the needle has cleanly crossed through the tripod ring."""
        needle_id = self.sim.model.geom_name2id("needle_obj_needle")
        needle_pos = np.array(self.sim.data.geom_xpos[needle_id])
        needle_mat = np.array(self.sim.data.geom_xmat[needle_id]).reshape(3, 3)
        needle_axis = self._unit_vector(needle_mat[:, 1])
        needle_tip = needle_pos - 0.06 * needle_axis

        ring_pos = np.zeros(3)
        ring_mat = None
        for i in range(self.tripod.num_ring_geoms):
            ring_id = self.sim.model.geom_name2id(f"tripod_obj_ring_{i}")
            ring_pos += np.array(self.sim.data.geom_xpos[ring_id])
            if ring_mat is None:
                ring_mat = np.array(self.sim.data.geom_xmat[ring_id]).reshape(3, 3)
        ring_pos /= self.tripod.num_ring_geoms

        ring_normal = self._unit_vector(ring_mat[:, 0], fallback=np.array([1.0, 0.0, 0.0]))
        if np.dot(ring_normal, ring_pos - needle_pos) < 0:
            ring_normal = -ring_normal
        ring_normal[2] = 0.0
        ring_normal = self._unit_vector(ring_normal, fallback=np.array([1.0, 0.0, 0.0]))

        rel = ring_pos - needle_pos
        t = np.clip(np.dot(rel, needle_axis), -0.06, 0.06)
        closest = needle_pos + t * needle_axis
        shaft_ring_distance = float(np.linalg.norm(closest - ring_pos))
        insert_progress = float(np.dot(needle_tip - ring_pos, ring_normal))
        self._threading_max_insert_progress = max(
            float(getattr(self, "_threading_max_insert_progress", -np.inf)),
            insert_progress,
        )

        current_tripod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["tripod"]])
        if float(self.sim.data.time) <= 1e-8:
            self._threading_initial_tripod_pos = current_tripod_pos.copy()
            self._threading_max_insert_progress = insert_progress
        initial_tripod_pos = getattr(self, "_threading_initial_tripod_pos", None)
        if initial_tripod_pos is None:
            initial_tripod_pos = current_tripod_pos.copy()
            self._threading_initial_tripod_pos = initial_tripod_pos
        tripod_displacement = float(np.linalg.norm(current_tripod_pos - initial_tripod_pos))

        return bool(
            shaft_ring_distance < 0.018
            and self._threading_max_insert_progress > 0.026
            and insert_progress > 0.014
            and tripod_displacement < 0.035
        )

    @staticmethod
    def _unit_vector(vec, fallback=None):
        vec = np.array(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.array(fallback if fallback is not None else np.zeros_like(vec), dtype=float)
        return vec / norm

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.needle)


class Threading_D0(Threading):
    """D0 shell: fixed tripod, needle in a modest region with limited top-down rotation."""


class Threading_D1(Threading_D0):
    """D1 shell: needle and tripod randomized in larger left/right table regions."""

    def _get_initial_placement_bounds(self):
        return {
            "needle": {
                "x": (-0.2, 0.05),
                "y": (0.15, 0.25),
                "z_rot": (-7.0 * np.pi / 6.0, np.pi / 6.0),
                "reference": self.table_offset,
            },
            "tripod": {
                "x": (-0.1, 0.15),
                "y": (-0.2, -0.1),
                "z_rot": (np.pi / 6.0, 5.0 * np.pi / 6.0),
                "reference": self.table_offset,
            },
        }


class Threading_D2(Threading_D1):
    """D2 shell: same difficulty as D1, but needle/tripod sides are reversed."""

    def _get_initial_placement_bounds(self):
        return {
            "needle": {
                "x": (-0.2, 0.05),
                "y": (-0.25, -0.15),
                "z_rot": (-7.0 * np.pi / 6.0, np.pi / 6.0),
                "reference": self.table_offset,
            },
            "tripod": {
                "x": (-0.1, 0.15),
                "y": (0.1, 0.2),
                "z_rot": (-5.0 * np.pi / 6.0, -np.pi / 6.0),
                "reference": self.table_offset,
            },
        }
