import numpy as np
import robosuite.utils.transform_utils as T
from utils.env_utils import get_obs, get_eef_pos, get_eef_quat, get_axisangle_error
from utils.primitive_utils import unscale_action

class BaseSkill:
    def __init__(self,
                 skill_type,
                 env,
                 image_obs_in_info,
                 reach_thres,
                 aff_thres,
                 aff_type,
                 binary_gripper,
                 aff_tanh_scaling,
                 global_xyz_bounds=np.array([
                    [-0.30, -0.30, 0.80],
                    [0.15, 0.30, 0.90]
                 ]),
                 delta_xyz_bounds=np.array([0.15, 0.15, 0.05]),
                 yaw_bounds=np.array([
                     [-np.pi / 2],
                     [np.pi / 2]
                 ]),
                 lift_height=0.95,
                 yaw_thres=0.20,
                 **config
                 ):
        self._skill_type = skill_type
        self._env = env

        self._num_ac_calls = None
        self._params = None
        self._state = None
        # self._aff_reward = None
        # self._aff_success = None

        self._config = dict(
            global_xyz_bounds=global_xyz_bounds,
            delta_xyz_bounds=delta_xyz_bounds,
            yaw_bounds=yaw_bounds,
            lift_height=lift_height,
            reach_thres=reach_thres,
            yaw_thres=yaw_thres,
            aff_thres=aff_thres,
            aff_type=aff_type,
            binary_gripper=binary_gripper,
            aff_tanh_scaling=aff_tanh_scaling,
            image_obs_in_info=image_obs_in_info,
            **config,
        )

        for k in ['global_xyz_bounds', 'delta_xyz_scale']:
            assert self._config[k] is not None
            self._config[k] = np.array(self._config[k])

        assert self._config['aff_type'] in [None, 'sparse', 'dense']

    def get_param_dim(self):
        raise NotImplementedError

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

    def get_aff_reward_and_success(self):
        if self._config['aff_type'] is None:
            return 1.0, True

        aff_centers = self._get_aff_centers()
        reach_pos = self._get_reach_pos()

        if aff_centers is None:
            return 1.0, True

        if len(aff_centers):
            return 0.0, False

        thres = self._config['aff_thers']
        within_thres = (np.abs(aff_centers - reach_pos) <= thres)
        aff_success = np.any(np.all(within_thres, axis=1)) # close to one of the key points

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

    def _reset(self, params):
        self._params = np.array(params).copy()
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
        params = np.clip(params, -1, 1)
        params = (params + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * params

    def _get_env_skill_info(self):
        info = self._env._get_skill_info()
        robot = self._env.robots[0]
        info['cur_ee_pos'] = np.array(robot.sim.data.site_xpos[robot.eef_site_id])

        # double check
        obs = get_obs(self._env)
        assert np.linalg.norm(info['cur_ee_pos'] - get_eef_pos(obs)) < 1e-4

        return info

    def is_success(self):
        raise NotImplementedError

    def skill_done(self):
        return self.is_success() or (self._num_ac_calls >= self._config['max_ac_calls'])
    
    def _update_info(self, info):
        info['num_ac_calls'] = self._num_ac_calls
        info['skill_success'] = self.is_success()

    def _reached_goal_ori_y(self):
        if not self._config['use_ori_params']:
            return True
        obs = get_obs(self._env)
        cur_quat = get_eef_quat(obs)
        cur_y = T.mat2euler(T.quat2mat(cur_quat), axes='rxyz')
        target_y = self._get_ori_ac().copy()
        ee_yaw_diff = np.minimum(
            (cur_y - target_y) % (2 * np.pi),
            (target_y - cur_y) % (2 * np.pi)
        )
        return ee_yaw_diff[-1] <= self._config['yaw_thres']

    def _get_info(self):
        info = self._env._get_skill_info()
        robot = self._env.robots[0]
        info['cur_ee_pos'] = np.array(robot.sim.data.site_xpos[robot.eef_site_id])
        return info

    def _get_action(self):
        self._update_state()
        self._num_ac_calls += 1

    def forward(self, params):
        self._reset(params)
        image_obs = []
        reward_sum = 0
        while True:
            action = self._get_action()
            obs, reward, done, info = self._env.step(action)
            reward_sum += reward
            if self._config['image_obs_in_info']:
                image_obs.append(obs['agentview_image'])

            if self.skill_done():
                break

        if self._config['image_obs_in_info']:
            info['image_obs'] = image_obs
        self._update_info(info)
        return obs, reward, done, info


class AtomicSkill(BaseSkill):
    def __init__(self,
                 skill_type,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            skill_type,
            use_ori_params=use_ori_params,
            max_ac_calls=1,
            **config
        )

    def get_param_dim(self):
        if self._config['use_ori_params']:
            return 5
        else:
            return 4

    def _get_pos_ac(self):
        self._check_params_dim()
        pos = self._params[:3].copy()
        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        ori_y = self._params[3:4].copy()
        return ori_y

    def _get_gripper_ac(self):
        self._check_params_dim()
        gripper_action = self._params[-1:].copy()
        if self._config['binary_gripper']:
            gripper_action = self._get_binary_gripper_ac(gripper_action)
        return gripper_action

    def get_aff_reward_and_success(self):
        reward, success = 1.0, True
        return reward, success

    def is_success(self):
        return True

    def _get_action(self):
        super()._get_action()
        pos = self._get_pos_ac()
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            ori_rp = np.array([0, 0])
            ori_y = self._get_ori_ac()
            return np.concatenate([pos, ori_rp, ori_y, gripper_action])
        else:
            return np.concatenate([pos, gripper_action])

class GripperSkill(BaseSkill):
    def __init__(self,
                 skill_type,
                 max_ac_calls=4,
                 use_aff=True,
                 **config
                 ):
        super().__init__(
            skill_type,
            max_ac_calls=max_ac_calls,
            **config
        )
        self._use_aff = use_aff

    def get_param_dim(self):
        return 0

    def _update_state(self):
        self._state = None

    def get_pos_ac(self):
        return None

    def get_ori_ac(self):
        return None

    def get_gripper_ac(self):
        if self._skill_type in ['close']:
            gripper_action = np.array([1, ])
        elif self._skill_type in ['open']:
            gripper_action = np.array([-1, ])
        else:
            raise ValueError

        return gripper_action

    def is_success(self):
        return self._num_ac_calls == self._config['max_ac_calls']

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

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED']

    def __init__(self,
                 skill_type,
                 use_gripper_params=False,
                 use_ori_params=True,
                 max_ac_calls=15,
                 use_aff=True,
                 **config):
        super().__init__(
            skill_type,
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            **config
        )

    def get_param_dim(self):
        param_dim = 3
        if self._config['use_ori_params']:
            param_dim += 1
        if self._config['use_gripper_params']:
            param_dim += 1
        return param_dim

    def _get_reach_pos(self):
        pos = self._get_unnormalized_params(
            self._params[:3], self._config['global_xyz_bounds']
        )
        return pos

    def _update_state(self):
        obs = get_obs(self._env)
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th)
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
        obs = get_obs(self._env)
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state in ['HOVERING', 'REACHED']:
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        if self._state == 'INIT':
            return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
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
        obs = get_obs(self._env)
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

        action = unscale_action(self._env, action)
        return action

class GraspSkill(BaseSkill):
    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'GRASPED']

    def __init__(self,
                 skill_type,
                 use_ori_params=True,
                 max_ac_calls=20,
                 max_reach_steps=15,
                 max_grasp_steps=5,
                 **config
                 ):
        super().__init__(
            skill_type,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            max_reach_steps=max_reach_steps,
            max_grasp_steps=max_grasp_steps,
            **config
        )
        self._num_reach_steps = None
        self._num_grasp_steps = None

    def get_param_dim(self):
        if self._config['use_ori_params']:
            return 4
        else:
            return 3

    def _reset(self, params):
        super()._reset(params)
        self._num_reach_steps = 0
        self._num_grasp_steps = 0

    def _get_reach_pos(self):
        pos = self._get_unnormalized_params(
            self._params[:3], self._config['global_xyz_bounds']
        )
        return pos

    def _update_state(self):
        obs = get_obs(self._env)
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'GRASPED' or \
                (self._state == 'REACHED' and (self._num_reach_steps >= self._config['num_reach_steps'])):
            self._state = 'GRASPED'
            self._num_grasp_steps += 1
        elif self._state == 'REACHED' or (reached_xyz and reached_ori_y):
            self._state = 'REACHED'
            self._num_reach_steps += 1
        elif reached_xy and reached_ori_y:
            self._state = 'HOVERING'
        elif reached_lift:
            self._state = 'LIFTED'
        else:
            self._state = 'INIT'

        assert self._state in GraspSkill.STATES

    def get_pos_ac(self):
        obs = get_obs(self._env)
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
        elif self._state == 'GRASPED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        if self._state == 'INIT':
            return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        return ori_y

    def _get_gripper_ac(self):
        if self._state in ['GRASPED', 'REACHED']:
            gripper_action = np.array([1, ])
        else:
            gripper_action = np.array([-1, ])
        return gripper_action

    def is_success(self):
        return self._num_grasp_steps >= self._config['num_grasp_steps']

    def _get_aff_centers(self):
        info = self._get_info()
        aff_centers = info.get('grasp_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = get_obs(self._env)
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

        action = unscale_action(self._env, action)
        return action

class PushSkill(BaseSkill):
    """
    params: reach_pos (3) + reach ori (1) + push_delta_pos (3)
    """
    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PUSHED']

    def __init__(self,
                 skill_type,
                 max_ac_calls=20,
                 use_ori_params=True,
                 **config
                 ):
        super().__init__(
            skill_type,
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            **config
        )

    def get_param_dim(self):
        # no gripper dim in params
        if self._config['use_ori_params']:
            return 7
        else:
            return 6

    def _get_reach_pos(self):
        pos = self._get_unnormalized_params(
            self._params[:3], self._config['global_xyz_bounds'])
        pos = pos.copy()
        return pos

    def _get_push_pos(self):
        src_pos = self._get_reach_pos()
        pos = src_pos.copy()

        delta_pos = self._params[-3:].copy()
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config['delta_xyz_scale']
        pos += delta_pos

        return pos

    def _update_state(self):
        obs = get_obs(self._env)
        cur_pos = get_eef_pos(obs)
        src_pos = self._get_reach_pos()
        target_pos = self._get_push_pos()

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_src_xy = (np.linalg.norm(cur_pos[0:2] - src_pos[0:2]) < th)
        reached_src_xyz = (np.linalg.norm(cur_pos - src_pos) < th)
        reached_target_xyz = (np.linalg.norm(cur_pos - target_pos) < th)
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
        obs = get_obs(self._env)
        cur_pos = get_eef_pos(obs)
        src_pos = self._get_reach_pos()
        target_pos = self._get_push_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            pos = src_pos.copy()
            pos[2] = self._config['lift_height']
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
        if self._state == 'INIT':
            return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        return ori_y

    def _get_gripper_ac(self):
        self._check_params_dim()
        gripper_action = np.array([0, ])
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
        obs = get_obs(self._env)
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

        action = unscale_action(self._env, action)
        return action
