from collections import OrderedDict

import numpy as np
from copy import deepcopy

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler,SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, add_material

DEFAULT_CLEANUP_CONFIG = {
    'use_pnp_rew': True,
    'use_push_rew': True,
    'rew_type': 'sum',
    'num_pnp_objs': 1,
    'num_push_objs': 1,
    'shaped_push_rew': False,
    'push_scale_fac': 5.0,
}
class CleanUpPnP(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(0.6, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # Get config
        self.task_config = DEFAULT_CLEANUP_CONFIG.copy()

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
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
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        if self._check_success():
            return 1.0
        return 0.0


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        pnpmaterial = CustomMaterial(
            texture="Spam",
            tex_name="pnpobj_tex",
            mat_name="pnpobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        pushmaterial = CustomMaterial(
            texture="Jello",
            tex_name="pushobj_tex",
            mat_name="pushobj_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.pnp_objs = []
        num_pnp_objs = self.task_config['num_pnp_objs']
        for i in range(num_pnp_objs):
            if num_pnp_objs > 1:
                color = 0.25 + 0.75 * i / (num_pnp_objs - 1)
            else:
                color = 1.0
            pnp_size = np.array([0.04, 0.022, 0.033]) * 0.7
            obj = BoxObject(
                name="obj_pnp_{}".format(i),
                size_min=pnp_size,
                size_max=pnp_size,
                rgba=[color, 0, 0, 1],
                material=pnpmaterial,
            )
            self.pnp_objs.append(obj)

        self.push_objs = []
        num_push_objs = self.task_config['num_push_objs']
        for i in range(num_push_objs):
            if num_push_objs > 1:
                color = 0.25 + 0.75 * i / (num_push_objs - 1)
            else:
                color = 1.0
            push_size = np.array([0.0350, 0.0425, 0.025]) * 1.25
            obj = BoxObject(
                name="obj_push_{}".format(i),
                size_min=push_size,
                size_max=push_size,
                rgba=[0, color, 0, 1],
                material=pushmaterial,
            )
            self.push_objs.append(obj)

        self.grasp_objs = self.pnp_objs

        mujoco_objects = self.pnp_objs + self.push_objs

        # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(cubes)
        # else:

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=mujoco_objects,
            x_range=[0.0, 0.05],
            y_range=[-0.13, 0.13],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.table_body_id = self.sim.model.body_name2id("table")

        self.pnp_obj_body_ids = []
        for i in range(self.task_config['num_pnp_objs']):
            obj = self.pnp_objs[i]
            id = self.sim.model.body_name2id(obj.root_body)
            self.pnp_obj_body_ids.append(id)
        self.grasp_obj_body_ids = self.pnp_obj_body_ids

        self.push_obj_body_ids = []
        for i in range(self.task_config['num_push_objs']):
            obj = self.push_objs[i]
            id = self.sim.model.body_name2id(obj.root_body)
            self.push_obj_body_ids.append(id)

        self.bin_body_id = self.sim.model.body_name2id("bin")

    @property
    def obj_positions(self):
        pnp_obj_positions = [
            self.sim.data.body_xpos[self.pnp_obj_body_ids[i]].copy()
            for i in range(self.task_config['num_pnp_objs'])
        ]
        push_obj_positions = [
            self.sim.data.body_xpos[self.push_obj_body_ids[i]].copy()
            for i in range(self.task_config['num_push_objs'])
        ]
        return pnp_obj_positions + push_obj_positions

    @property
    def obj_quats(self):
        pnp_obj_quats = [
            convert_quat(
                np.array(self.sim.data.body_xquat[self.pnp_obj_body_ids[i]]), to="xyzw"
            )
            for i in range(self.task_config['num_pnp_objs'])
        ]
        push_obj_quats = [
            convert_quat(
                np.array(self.sim.data.body_xquat[self.push_obj_body_ids[i]]), to="xyzw"
            )
            for i in range(self.task_config['num_push_objs'])
        ]
        return pnp_obj_quats + push_obj_quats

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.obj_positions).flatten()

            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return np.array(self.obj_quats).flatten()

            @sensor(modality=modality)
            def object_centric(obs_cache):
                _objs_pos = np.array(self.obj_positions).copy().reshape(-1, 3)
                nobject = _objs_pos.shape[0]
                # print(obs_cache.keys())
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                gripper_to_objects_pos = np.stack([_obj_pos - gripper_site_pos for _obj_pos in _objs_pos])
                assert len(gripper_to_objects_pos) == nobject
                relative_objs_pos = []
                for i in range(nobject):
                    for j in range(i + 1, nobject):
                        relative_objs_pos.append(_objs_pos[j] - _objs_pos[i])
                assert len(relative_objs_pos) == (nobject - 1) * nobject / 2
                relative_objs_pos = np.array(relative_objs_pos)
                return np.concatenate([gripper_to_objects_pos.flatten(), relative_objs_pos.flatten()])

            # @sensor(modality=modality)
            # def obj_state(obs_cache):
            #     obj_pos = np.array(self.obj_positions).flatten()
            #     obj_quat = np.array(self.obj_quats).flatten()
            #     return np.concatenate([obj_pos, obj_quat])


            sensors = [obj_pos, obj_quat, object_centric]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def in_bin(self, obj_pos):
        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        return abs(obj_pos[0] - bin_pos[0]) < 0.10 \
               and abs(obj_pos[1] - bin_pos[1]) < 0.15 \
               and obj_pos[2] < self.table_offset[2] + 0.05

    def _check_success_pnp(self):
        for i in range(self.task_config['num_pnp_objs']):
            obj_pos = self.sim.data.body_xpos[self.pnp_obj_body_ids[i]]
            if not self.in_bin(obj_pos):
                return False
        return True

    def _check_success_push(self):
        for i in range(self.task_config['num_push_objs']):
            obj_pos = self.sim.data.body_xpos[self.push_obj_body_ids[i]]
            # target_pos_xy = self.table_offset[:2] + np.array([-0.15, 0.15])
            # d_push = np.linalg.norm(obj_pos[:2] - target_pos_xy)
            # if d_push > 0.09:
            #     return False
            if obj_pos[0] > -0.07:
                return False
        return True

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        if not self._check_success_pnp():
            return False
        # if not self._check_success_push():
        #     return False
        return True

    def _get_skill_info(self):
        pos_info = dict(
            grasp=[],
            push=[],
            reach=[],
        )

        bin_pos = self.sim.data.body_xpos[self.bin_body_id].copy()
        obj_positions = self.obj_positions
        num_pnp_objs = self.task_config['num_pnp_objs']

        pnp_objs = obj_positions[:num_pnp_objs]
        push_objs = obj_positions[num_pnp_objs:]

        drop_pos = bin_pos + [0, 0, 0.15]

        pos_info['grasp'] += pnp_objs
        pos_info['push'] += push_objs
        pos_info['reach'].append(drop_pos)

        info = {}
        for k in pos_info:
            info[k + '_pos'] = pos_info[k]

        return info

    def _get_env_info(self, action):
        env_info = {}
        env_info['success_pnp'] = self._check_success_pnp()
        env_info['success_push'] = self._check_success_push()
        env_info['success'] = self._check_success()

        return env_info

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube

        # if vis_settings["grippers"]:
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)

