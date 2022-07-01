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

ID_TO_PRIMITIVE = ['reach', 'place', 'grasp', 'push', 'atomic']

class SkillController:

    def __init__(self,
                 env,
                 controller_type,
                 image_obs_in_info=False,
                 aff_type='sparse',
                 render=False,
                 reach_use_gripper=True):

        self._env = env
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
            global_xyz_bounds=np.array([
                [-0.25, -0.25, 0.80],
                [0.08, 0.25, 0.90]
            ]),
            delta_xyz_scale=DELTA_XYZ_SCALE,
            yaw_bounds=np.array([
                [-np.pi / 2],
                [np.pi / 2]
            ]),
            lift_height=0.95,
            lift_thres=0.02,
            reach_thres=0.02,
            push_thres=0.015,
            aff_thres=0.08,
            yaw_thres=0.20,
            aff_tanh_scaling=10.0,
            binary_gripper=False,
        )

        self.atomic = AtomicSkill(
            max_ac_calls=1,
            **base_config
        )

        self.gripper_release = GripperSkill(
            max_ac_calls=10,
            skill_type='open',
            **base_config
        )

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
            gripper_release=self.gripper_release,
            place=self.place,
            reach=self.reach,
            grasp=self.grasp,
            push=self.push
        )

    def get_skill(self, p_name):
        return self.name_to_skill[p_name]

    def test_start_state(self, p_name):
        return self.name_to_skill[p_name]._test_start_state()

    def execute(self, p_name, output, norm, **kwargs):
        # len(args) = maximal argument length
        skill = self.name_to_skill[p_name]
        skill_dim = skill.get_param_dim()
        skill_args = output[:skill_dim]
        ret = skill.act(skill_args, norm=norm)
        if ret is not None:
            ret['info']['interest_interaction'] = skill.check_interesting_interaction()
        return ret

