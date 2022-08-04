import numpy as np
import robosuite.utils.transform_utils as T
from utils.env_utils import get_obs, get_eef_pos, get_eef_quat, get_axisangle_error
from utils.primitive_utils import unscale_action

class BaseSkill:
    def __init__(self,
                 env,
                 image_obs_in_info,
                 aff_type,
                 render,
                 global_xyz_bounds,
                 delta_xyz_scale,
                 yaw_bounds,
                 lift_height,
                 lift_thres,
                 reach_thres,
                 push_thres,
                 aff_thres,
                 yaw_thres,
                 aff_tanh_scaling,
                 binary_gripper,
                 env_idx,
                 **config
                 ):
        self._env = env
        self._env_idx = env_idx

        self._num_ac_calls = None
        self._params = None
        self._state = None
        self._normalize_params = None
        self.skill_obs_list = []
        self.skill_action_list = []
        self.skill_image_obs_list = []
        # self._aff_reward = None
        # self._aff_success = None

        self._config = dict(
            global_xyz_bounds=global_xyz_bounds,
            delta_xyz_scale=delta_xyz_scale,
            yaw_bounds=yaw_bounds,
            lift_height=lift_height,
            lift_thres=lift_thres,
            reach_thres=reach_thres,
            push_thres=push_thres,
            yaw_thres=yaw_thres,
            aff_thres=aff_thres,
            aff_type=aff_type,
            binary_gripper=binary_gripper,
            aff_tanh_scaling=aff_tanh_scaling,
            image_obs_in_info=image_obs_in_info,
            render=render,
            **config,
        )

        for k in ['global_xyz_bounds', 'delta_xyz_scale']:
            assert self._config[k] is not None
            self._config[k] = np.array(self._config[k])

        assert self._config['aff_type'] in [None, 'sparse', 'dense']

    def get_param_dim(self):
        raise NotImplementedError

    def get_param_spec(self):
        param_dim = self.get_param_dim()
        low = np.ones(param_dim) * -1.
        high = np.ones(param_dim) * 1.
        return low, high

    def _check_params_dim(self):
        if self._params is None:
            assert self.get_param_dim() == 0
        else:
            assert len(self._params) == self.get_param_dim()

    def _update_state(self):
        raise NotImplementedError

    def _get_aff_centers(self):
        raise NotImplementedError

    def _get_reach_pos(self):
        raise NotImplementedError

    def get_aff_reward_and_success(self, params, norm):
        self._reset(params, norm)
        assert self._num_ac_calls is None or self._num_ac_calls == 0

        if self._config['aff_type'] is None:
            return 1.0, True

        aff_centers = self._get_aff_centers()

        if aff_centers is None:
            return 1.0, True

        reach_pos = self._get_reach_pos()

        if not len(aff_centers):
            return 0.0, False

        thres = self._config['aff_thres']
        within_thres = (np.abs(aff_centers - reach_pos) <= thres)
        aff_success = np.any(np.all(within_thres, axis=1))  # close to one of the key points

        if self._config['aff_type'] == 'dense':
            if aff_success:
                aff_reward = 1.0
            else:
                dist = np.clip(np.abs(aff_centers - reach_pos) - thres, 0, None)
                min_dist = np.min(np.sum(dist, axis=1))
                aff_reward = 1.0 - np.tanh(self._config['aff_tanh_scaling'] * min_dist)
        else:
            aff_reward = float(aff_success)

        return aff_reward, aff_success

    def _reset(self, params, norm):
        self._params = np.array(params).copy()
        self._normalize_params = norm
        self._num_ac_calls = 0
        self._state = None

    def _get_pos_ac(self):
        raise NotImplementedError

    def _get_ori_ac(self):
        raise NotImplementedError

    def _get_gripper_ac(self):
        raise NotImplementedError

    def _get_binary_gripper_ac(self, gripper_action):
        if np.abs(gripper_action) < 0.10:
            gripper_action[:] = 0
        elif gripper_action < 0:
            gripper_action[:] = -1
        else:
            gripper_action[:] = 1
        return gripper_action

    def get_max_ac_calls(self):
        return self._config['max_ac_calls']

    # def get_aff_reward(self):
    #     return self._aff_reward
    #
    # def get_aff_success(self):
    #     return self._aff_success

    def _get_unnormalized_params(self, params, bounds):
        params = params.copy()
        params = np.clip(params, -1, 1)
        params = (params + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * params

    def _get_normalized_params(self, params, bounds):
        params = params.copy()
        low, high = bounds[0], bounds[1]
        params = (params - low) / (high - low)
        params = params * 2 - 1
        params = np.clip(params, -1, 1)
        return params

    def is_success(self):
        raise NotImplementedError

    def skill_done(self):
        return self.is_success() or (self._num_ac_calls >= self._config['max_ac_calls'])

    def _update_info(self, info):
        info['num_ac_calls'] = self._num_ac_calls
        info['skill_success'] = self.is_success()
        info['env_success'] = self._env.env_method('_check_success', indices=self._env_idx)[0]

    def _reached_goal_ori_y(self):
        if not self._config['use_ori_params']:
            return True
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_quat = get_eef_quat(obs)
        cur_y = T.mat2euler(T.quat2mat(cur_quat), axes='rxyz')[-1:]
        target_y = self._get_ori_ac()
        if target_y is None:
            return True
        target_y = target_y.copy()
        ee_yaw_diff = np.minimum(
            (cur_y - target_y) % (2 * np.pi),
            (target_y - cur_y) % (2 * np.pi)
        )
        return ee_yaw_diff[-1] <= self._config['yaw_thres']

    def _get_info(self):
        info = self._env.env_method('_get_skill_info', indices=self._env_idx)[0]
        return info

    def _get_action(self):
        self._update_state()
        self._num_ac_calls += 1

    def act(self, params, norm):
        self._reset(params, norm)
        image_obs = []
        reward_sum = 0
        obs_list = []
        action_list = []
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        obs_list.append(obs)
        while True:
            action = self._get_action()
            action_list.append(action)
            obs, reward, done, info = self._env.step(action)
            obs_list.append(obs)
            info['last_gripper_ac'] = action[-1:]
            if self._config['render']:
                self._env.render()
            reward_sum += reward
            if self._config['image_obs_in_info']:
                image_obs.append(obs['agentview_image'])

            if self.skill_done() or done:
                self._env_done = done
                break

        if self._config['image_obs_in_info']:
            info['image_obs'] = image_obs
        info['obs_list'] = obs_list
        info['action_list'] = action_list
        info['env_done'] = done
        self._update_info(info)
        return dict(obs=obs, info=info)

    def check_interesting_interaction(self):
        assert self.skill_done() or self._env_done
        return True

class AtomicSkill(BaseSkill):
    def __init__(self,
                 use_ori_params,
                 max_ac_calls,
                 **config
                 ):
        super().__init__(
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            **config
        )

    def get_param_dim(self):
        return 7
        # if self._config['use_ori_params']:
        #     return 5
        # else:
        #     return 4

    def _update_state(self):
        self._state = None

    # def _get_pos_ac(self):
    #     self._check_params_dim()
    #     pos = self._params[:3].copy()
    #     return pos
    #
    # def _get_ori_ac(self):
    #     self._check_params_dim()
    #     assert self._config['use_ori_params']
    #     ori_y = self._params[3:4].copy()
    #     return ori_y

    def _get_gripper_ac(self):
        self._check_params_dim()
        gripper_action = self._params[-1:].copy()
        if self._config['binary_gripper']:
            gripper_action = self._get_binary_gripper_ac(gripper_action)
        return gripper_action

    def get_aff_reward_and_success(self, params, norm):
        reward, success = 1.0, True
        return reward, success

    def is_success(self):
        return self._num_ac_calls >= self._config['max_ac_calls']

    def _get_action(self):
        super()._get_action()
        if self._config['max_ac_calls'] > 1:
            low, high = self.get_param_spec()
            return np.random.uniform(low, high)
        self._check_params_dim()
        # pos = self._get_pos_ac()
        gripper_action = self._get_gripper_ac()
        # if self._config['use_ori_params']:
        #     ori_rp = np.array([0, 0])
        #     ori_y = self._get_ori_ac()
        #     return np.concatenate([pos, ori_rp, ori_y, gripper_action])
        # else:
        #     return np.concatenate([pos, gripper_action])
        return np.concatenate([self._params[:-1], gripper_action])

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        return True

    def _test_start_state(self):
        return True

    def set_max_ac_calls(self, max_ac_calls):
        self._config['max_ac_calls'] = max_ac_calls

class GripperSkill(BaseSkill):
    def __init__(self,
                 max_ac_calls,
                 skill_type,
                 **config
                 ):
        super().__init__(
            max_ac_calls=max_ac_calls,
            _use_ori_params=False,
            **config
        )
        self._skill_type = skill_type

    def get_param_dim(self):
        self.pos_dim = None
        self.orn_dim = None
        self.gripper_dim = None
        return 0

    def _update_state(self):
        self._state = None

    def _get_pos_ac(self):
        return None

    def _get_ori_ac(self):
        return None

    def _get_reach_pos(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        eef_pos = get_eef_pos(obs)
        return eef_pos

    def _get_gripper_ac(self):
        if self._skill_type in ['close']:
            gripper_action = np.array([1, ])
        elif self._skill_type in ['open']:
            gripper_action = np.array([-1, ])
        else:
            raise ValueError

        return gripper_action

    def is_success(self):
        return self._num_ac_calls >= self._config['max_ac_calls']

    def _get_aff_centers(self):
        info = self._get_info()
        aff_centers = info.get('grasp_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            return np.concatenate([[0, 0, 0, 0, 0, 0], gripper_action])
        return np.concatenate([[0, 0, 0], gripper_action])

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        return True
        # for obj_id in range(len(self._env.grasp_objs)):
        #     obj = self._env.grasp_objs[obj_id]
        #     obj_pos = self._env.sim.data.body_xpos[self._env.pnp_obj_body_ids[obj_id]]
        #     obs = get_obs(self._env)
        #     eef_pos = get_eef_pos(obs)
        #     if np.linalg.norm(obj_pos - eef_pos) < 0.1 and not self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=obj):
        #         return True
        # return False

    def _test_start_state(self):
        return True

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED']

    def __init__(self,
                 max_ac_calls,
                 use_gripper_params,
                 use_ori_params,
                 **config):
        super().__init__(
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            **config
        )

    def get_param_dim(self):
        param_dim = 3
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            param_dim += 1
            self.orn_dim = (3, 4)
        if self._config['use_gripper_params']:
            if self.orn_dim is not None:
                self.gripper_dim = (4, 5)
            else:
                self.gripper_dim = (3, 4)
            param_dim += 1
        return param_dim

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self.initial_grasped = False
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                self.initial_grasped = True

    def _get_reach_pos(self):
        if self._normalize_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds']
            )
        else:
            pos = self._params[:3]
        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _update_state(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < lift_th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if reached_xyz and reached_ori_y:
            self._state = 'REACHED'
        else:
            if reached_xy and reached_ori_y:
                self._state = 'HOVERING'
            else:
                if reached_lift:
                    self._state = 'LIFTED'
                else:
                    self._state = 'INIT'
        assert self._state in ReachSkill.STATES

    def _get_pos_ac(self):
        self._check_params_dim()
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state in ['HOVERING', 'REACHED']:
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        if self._normalize_params:
            ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        else:
            ori_y = param_y
        return ori_y

    def _get_gripper_ac(self):
        self._check_params_dim()
        if self._config['use_gripper_params']:
            gripper_action = self._params[-1:].copy()
            if self._config['binary_gripper']:
                gripper_action = self._get_binary_gripper_ac(gripper_action)
            return gripper_action
        return np.array([0, ])

    def is_success(self):
        return self._state == 'REACHED'

    def _get_aff_centers(self):
        info = self._get_info()
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        pos_action = pos - cur_pos
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            action = np.concatenate([pos_action, ori_action, gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])

        action = unscale_action(self._env, action, self._env_idx)
        return action

    def _test_start_state(self):
        return True

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        end_grasped = False
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                end_grasped = True
        if (self.initial_grasped and (not end_grasped)) or ((not self.initial_grasped) and end_grasped):
            return False
        return True

class GraspSkill(BaseSkill):
    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'GRASPED']

    def __init__(self,
                 max_ac_calls,
                 max_reach_steps,
                 max_grasp_steps,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            max_reach_steps=max_reach_steps,
            max_grasp_steps=max_grasp_steps,
            **config
        )
        self._num_reach_steps = None
        self._num_grasp_steps = None

    def get_param_dim(self):
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            self.orn_dim = (3, 4)
            return 4
        else:
            return 3

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._num_reach_steps = 0
        self._num_grasp_steps = 0

    def _get_reach_pos(self):
        if self._normalize_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds']
            )
        else:
            pos = self._params[:3]
        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _update_state(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < lift_th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'GRASPED' or \
                (self._state == 'REACHED') or self._num_reach_steps >= self._config['max_reach_steps']:
            self._state = 'GRASPED'
            self._num_grasp_steps += 1
        elif self._state == 'REACHED' or (reached_xyz and reached_ori_y):
            self._state = 'REACHED'
            self._num_reach_steps += 1
        elif reached_xy and reached_ori_y:
            self._state = 'HOVERING'
            self._num_reach_steps += 1
        elif reached_lift:
            self._state = 'LIFTED'
            self._num_reach_steps += 1
        else:
            self._state = 'INIT'
            self._num_reach_steps += 1

        assert self._state in GraspSkill.STATES

    def _get_pos_ac(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'HOVERING':
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        elif self._state == 'GRASPED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        if self._normalize_params:
            ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        else:
            ori_y = param_y
        return ori_y

    def _get_gripper_ac(self):
        if self._state in ['GRASPED', 'REACHED']:
            gripper_action = np.array([1, ])
        else:
            gripper_action = np.array([-1, ])
        return gripper_action

    def is_success(self):
        return self._num_grasp_steps >= self._config['max_grasp_steps']

    def _get_aff_centers(self):
        info = self._get_info()
        aff_centers = info.get('grasp_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        pos_action = pos - cur_pos
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            action = np.concatenate([pos_action, ori_action, gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])

        action = unscale_action(self._env, action, self._env_idx)
        return action

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                return True
        return False

    def _test_start_state(self):
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                return False
        return True

class PlaceSkill(BaseSkill):
    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PLACED']

    def __init__(self,
                 max_ac_calls,
                 max_reach_steps,
                 max_place_steps,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            max_reach_steps=max_reach_steps,
            max_place_steps=max_place_steps,
            **config
        )
        self._num_reach_steps = None
        self._num_place_steps = None

    def get_param_dim(self):
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            self.orn_dim = (3, 4)
            return 4
        else:
            return 3

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._num_reach_steps = 0
        self._num_place_steps = 0
        self._initial_grasped_obj_body_id = None
        for obj_id in range(len(self._env.get_attr("grasp_objs", indices=self._env_idx)[0])):
            obj = self._env.get_attr("grasp_objs", indices=self._env_idx)[0][obj_id]
            if self._env.env_method("_check_grasp", dict(gripper=self._env.robots[0].gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                self._initial_grasped_obj_body_id = self._env.get_attr("pnp_obj_body_ids", indices=self._env_idx)[0][obj_id]
                break

    def _get_reach_pos(self):
        if self._normalize_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds']
            )
        else:
            pos = self._params[:3]
        return pos

    def _update_state(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < lift_th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'PLACED' or \
                (self._state == 'REACHED') or self._num_reach_steps >= self._config['max_reach_steps']:
            self._state = 'PLACED'
            self._num_place_steps += 1
        elif self._state == 'REACHED' or (reached_xyz and reached_ori_y):
            self._state = 'REACHED'
            self._num_reach_steps += 1
        elif reached_xy and reached_ori_y:
            self._state = 'HOVERING'
            self._num_reach_steps += 1
        elif reached_lift:
            self._state = 'LIFTED'
            self._num_reach_steps += 1
        else:
            self._state = 'INIT'
            self._num_reach_steps += 1

        assert self._state in PlaceSkill.STATES

    def _get_pos_ac(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING':
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        elif self._state == 'PLACED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        if self._normalize_params:
            ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        else:
            ori_y = param_y
        return ori_y

    def _get_gripper_ac(self):
        if self._state in ['PLACED', 'REACHED']:
            gripper_action = np.array([-1, ])
        else:
            gripper_action = np.array([1, ])
        return gripper_action

    def is_success(self):
        return self._num_place_steps >= self._config['max_place_steps']

    def _get_aff_centers(self):
        info = self._get_info()
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        pos_action = pos - cur_pos
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            action = np.concatenate([pos_action, ori_action, gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])

        action = unscale_action(self._env, action, self._env_idx)
        return action

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        end_obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        eef_pos = get_eef_pos(end_obs)
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                return False
        if self._initial_grasped_obj_body_id is None:
            return False
        if np.linalg.norm(eef_pos[:2] - self._env.sim.data.body_xpos[self._initial_grasped_obj_body_id][:2]) > 0.03:
            return False
        return True

    def _test_start_state(self):
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            print(self._env.get_attr("robots", indices=self._env_idx))
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                return True
        return False

class PushSkill(BaseSkill):
    """
    params: reach_pos (3) + reach ori (1) + push_delta_pos (3)
    """
    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PUSHED']

    def __init__(self,
                 max_ac_calls,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            **config
        )

    def get_param_dim(self):
        # no gripper dim in params
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            self.orn_dim = (3, 4)
            self.delta_dim = (4, 7)
            return 7
        else:
            self.delta_dim = (3, 6)
            return 6

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._initial_push_obj_pos = []
        for obj_id in range(len(self._env.get_attr("push_objs", indices=self._env_idx)[0])):
            self._initial_push_obj_pos.append(self._env.get_attr("sim", indices=self._env_idx)[0].data.body_xpos[self._env.get_attr("push_obj_body_ids", indices=self._env_idx)[0][obj_id]].copy())

    def _get_reach_pos(self):
        if self._normalize_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds'])
        else:
            pos = self._params[:3]

        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _get_push_pos(self):
        src_pos = self._get_reach_pos()
        pos = src_pos.copy()

        delta_pos = self._params[-3:].copy()
        if self._normalize_params:
            delta_pos = np.clip(delta_pos, -1, 1)
            delta_pos *= self._config['delta_xyz_scale']
        pos += delta_pos

        pos = np.clip(pos, self._config['global_xyz_bounds'][0], self._config['global_xyz_bounds'][1] - [0.04, 0, 0])
        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _update_state(self):
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        src_pos = self._get_reach_pos()
        target_pos = self._get_push_pos()

        reach_th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        push_th = self._config['push_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_src_xy = (np.linalg.norm(cur_pos[0:2] - src_pos[0:2]) < lift_th)
        reached_src_xyz = (np.linalg.norm(cur_pos - src_pos) < reach_th)
        reached_target_xyz = (np.linalg.norm(cur_pos - target_pos) < push_th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'REACHED' and reached_target_xyz:
            self._state = 'PUSHED'
        else:
            if self._state == 'REACHED' or (reached_src_xyz and reached_ori_y):
                self._state = 'REACHED'
            else:
                if reached_src_xy and reached_ori_y:
                    self._state = 'HOVERING'
                else:
                    if reached_lift:
                        self._state = 'LIFTED'
                    else:
                        self._state = 'INIT'
            assert self._state in PushSkill.STATES


    def _get_pos_ac(self):
        self._check_params_dim()
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        src_pos = self._get_reach_pos()
        target_pos = self._get_push_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = src_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'HOVERING':
            pos = src_pos.copy()
        elif self._state == 'REACHED':
            pos = target_pos.copy()
        elif self._state == 'PUSHED':
            pos = target_pos.copy()
        else:
            raise NotImplementedError
        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        if self._normalize_params:
            ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        else:
            ori_y = param_y
        return ori_y

    def _get_gripper_ac(self):
        self._check_params_dim()
        gripper_action = np.array([-1, ])
        return gripper_action

    def is_success(self):
        return self._state == 'PUSHED'

    def _get_aff_centers(self):
        info = self._get_info()
        aff_centers = info.get('push_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        if self._state == 'REACHED':
            pos_action = (pos - cur_pos) * 1.5
        else:
            pos_action = pos - cur_pos
        # print("action target pos", pos)
        # print(self._state)
        # print("cur pos", cur_pos)
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            action = np.concatenate([pos_action, ori_action, gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])
        action = unscale_action(self._env, action, self._env_idx)
        return action

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        for obj_id in range(len(self._env.get_attr("push_objs", indices=self._env_idx)[0])):
            obj_pos = self._env.get_attr("sim", indices=self._env_idx)[0].data.body_xpos[self._env.get_attr("push_obj_body_ids", indices=self._env_idx)[0][obj_id]].copy()
            initial_obj_pos = self._initial_push_obj_pos[obj_id]
            obs = (self._env.env_method('_get_observations', dict(force_update=True), indices=self._env_idx)[0])
            eef_pos = get_eef_pos(obs)
            if np.linalg.norm(obj_pos - eef_pos) < 0.1 and np.linalg.norm(obj_pos - initial_obj_pos) > 0.04:
                return True
        return False

    def _test_start_state(self):
        for obj in self._env.get_attr("grasp_objs", indices=self._env_idx)[0]:
            robot = self._env.get_attr("robots", indices=self._env_idx)[0][0]
            if self._env.env_method("_check_grasp", dict(gripper=robot.gripper, object_geoms=obj),
                                 indices=self._env_idx)[0]:
                return False
        return True

