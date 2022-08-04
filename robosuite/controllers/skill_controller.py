import collections
import copy
import numpy as np
from collections import OrderedDict

from robosuite.controllers.skills import (
    AtomicSkill,
    ReachSkill,
    GraspSkill,
    PushSkill,
    GripperSkill,
    PlaceSkill,
)

PRIMITIVE_TO_ID = {
    'reach': 0,
    'place': 1,
    'grasp': 2,
    'push': 3,
    'atomic': 4
}

DELTA_XYZ_SCALE = np.array([0.2, 0.2, 0.02])

ID_TO_PRIMITIVE = [
    'reach',
    'place',
    'grasp',
    'push',
    'atomic'
   ]

NON_ATOMIC_PRIMITIVES = [
    'reach',
    'place',
    'grasp',
    'push'
]
GLOBAL_XYZ_BOUNDS = np.array([
                [-0.32, -0.26, 0.80],
                [0.20, 0.26, 1.0]
            ])
class SkillController:

    def __init__(self,
                 env,
                 controller_type,
                 image_obs_in_info=False,
                 aff_type='sparse',
                 render=False,
                 reach_use_gripper=False,
                 env_idx=None):

        self._env = env
        self._env_idx = env_idx
        if controller_type == 'OSC_POSE':
            _use_ori_params = True
        elif controller_type == 'OSC_POSITION':
            _use_ori_params = False
        else:
            raise NotImplementedError
        base_config = dict(
            env=self._env,
            aff_type=aff_type,
            image_obs_in_info=image_obs_in_info,
            render=render,
            use_ori_params=_use_ori_params,
            global_xyz_bounds=GLOBAL_XYZ_BOUNDS,
            delta_xyz_scale=DELTA_XYZ_SCALE,
            yaw_bounds=np.array([
                [-np.pi / 2],
                [np.pi / 2]
            ]),
            lift_height=0.95,
            lift_thres=0.02,
            reach_thres=0.01,
            push_thres=0.015,
            aff_thres=0.08,
            yaw_thres=0.20,
            grasp_thres=0.03,
            aff_tanh_scaling=10.0,
            binary_gripper=False,
            env_idx=env_idx,
        )

        self.atomic = AtomicSkill(
            max_ac_calls=1,
            **base_config
        )

        # self.gripper_release = GripperSkill(
        #     max_ac_calls=10,
        #     skill_type='open',
        #     **base_config
        # )

        self.place = PlaceSkill(
            max_ac_calls=80,
            max_reach_steps=70,
            max_place_steps=10,
            **base_config
        )

        self.reach = ReachSkill(
            max_ac_calls=70,
            use_gripper_params=reach_use_gripper,
            **base_config
        )

        self.grasp = GraspSkill(
            max_ac_calls=80,
            max_reach_steps=70,
            max_grasp_steps=10,
            **base_config
        )

        self.push = PushSkill(
            max_ac_calls=120,
            **base_config
        )

        self.name_to_skill = OrderedDict(
            atomic=self.atomic,
            # gripper_release=self.gripper_release,
            place=self.place,
            reach=self.reach,
            grasp=self.grasp,
            push=self.push
        )

    def get_skill(self, p_name):
        return self.name_to_skill[p_name]

    def test_start_state(self, p_name):
        return self.name_to_skill[p_name]._test_start_state()

    def reset_skill(self, p_name, output, norm):
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        skill_args = output[:param_dim]
        try:
            if norm or p_name == 'atomic':
                assert (skill_args <= 1.).all() and (skill_args >= -1.).all()
            else:
                norm_args = self.get_normalized_params(p_name=p_name, unnorm_params=skill_args)
                assert (norm_args <= 1.).all and (skill_args >= -1.).all()
        except:
            print("p", p_name)
            print("args", skill_args)
            raise ValueError
        skill._reset(skill_args, norm)


    def step_action(self, p_name):
        skill = self.name_to_skill[p_name]
        action = skill._get_action()
        skill.skill_action_list.append(action)
        return action

    def skill_done(self, p_name):
        skill = self.name_to_skill[p_name]
        return skill.skill_done()

    def skill_success(self, p_name):
        skill = self.name_to_skill[p_name]
        return skill.is_success()

    def skill_interest(self, p_name):
        skill = self.name_to_skill[p_name]
        return skill.check_interesting_interaction()


    def execute(self, p_name, output, norm, **kwargs):
        # len(args) = maximal argument length
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        skill_args = output[:param_dim]
        try:
            if norm or p_name == 'atomic':
                assert (skill_args <= 1.).all() and (skill_args >= -1.).all()
            else:
                norm_args = self.get_normalized_params(p_name=p_name, unnorm_params=skill_args)
                assert (norm_args <= 1.).all and (skill_args >= -1.).all()
        except:
            print("p", p_name)
            print("args", skill_args)
            raise ValueError
        ret = skill.act(skill_args, norm=norm)
        if ret is not None:
            ret['info']['interest_interaction'] = skill.check_interesting_interaction()
        return ret

    def get_normalized_params(self, p_name, unnorm_params):
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        skill_unnorm_params = unnorm_params[:param_dim]
        pad_len = len(unnorm_params) - param_dim
        norm_params = []
        if skill.pos_dim is not None:
            norm_params.append(skill._get_normalized_params(skill_unnorm_params[skill.pos_dim[0], skill.pos_dim[1]], skill._config['global_xyz_bounds']))
        if skill.orn_dim is not None:
            norm_params.append(skill._get_normalized_params(skill_unnorm_params[skill.orn_dim[0], skill.orn_dim[1]], skill._config['yaw_bounds']))
        if skill.gripper_dim is not None:
            norm_params.append(skill_unnorm_params[skill.gripper_dim[0], skill.gripper_dim[1]])
        if p_name == 'push':
            assert skill.delta_dim is not None
            unnorm_delta = skill_unnorm_params[skill.delta_dim[0], skill.delta_dim[1]]
            norm_delta = unnorm_delta / skill._config['delta_xyz_scale']
            norm_delta = np.clip(norm_delta, -1, 1)
            norm_params.append(norm_delta)
        return np.concatenate([np.concatenate(norm_params), np.zeros(pad_len)])

    def get_unnormalized_params(self, p_name, norm_params):
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        skill_norm_params = norm_params[:param_dim]
        pad_len = len(norm_params) - param_dim
        unnorm_params = []
        if skill.pos_dim is not None:
            unnorm_params.append(skill._get_unnormalized_params(skill_norm_params[skill.pos_dim[0], skill.pos_dim[1]], skill._config['global_xyz_bounds']))
        if skill.orn_dim is not None:
            unnorm_params.append(skill._get_unnormalized_params(skill_norm_params[skill.orn_dim[0], skill.orn_dim[1]], skill._config['yaw_bound']))
        if skill.gripper is not None:
            unnorm_params.append(skill_norm_params[skill.gripper_dim[0], skill.gripper_dim[1]])
        if p_name == 'push':
            assert skill.delta_dim is not None
            norm_delta = skill_norm_params[skill.delta_dim[0], skill.delta_dim[1]]
            norm_delta = np.clip(norm_delta, -1, 1)
            unnorm_delta = norm_delta * skill._config['delta_xyz_scale']
            unnorm_params.append(unnorm_delta)
        return np.concatenate([np.concatenate(unnorm_params), np.zeros(pad_len)])