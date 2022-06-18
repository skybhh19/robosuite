import collections
import copy
import numpy as np

from robosuite.controllers.skills import (
    AtomicSkill,
    ReachSkill,
    GraspSkill,
    PushSkill,
    GripperSkill,
)

class SkillController:

    def __init__(self,
                 env,
                 controller_type,
                 image_obs_in_info=False,
                 aff_type='sparse'):

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
            use_ori_params=_use_ori_params,
            global_xyz_bounds=np.array([
                [-0.30, -0.30, 0.80],
                [0.15, 0.30, 0.90]
            ]),
            delta_xyz_scale=np.array([0.15, 0.15, 0.05]),
            yaw_bounds=np.array([
                [-np.pi / 2],
                [np.pi / 2]
            ]),
            lift_height=0.95,
            reach_thres=0.02,
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
            max_ac_calls=4,
            skill_type='open',
            **base_config
        )

        self.reach = ReachSkill(
            max_ac_calls=15,
            use_gripper_params=True,
            **base_config
        )

        self.grasp = GraspSkill(
            max_ac_calls=20,
            max_reach_steps=15,
            max_grasp_steps=5,
            **base_config
        )

        self.push = PushSkill(
            max_ac_calls=20,
            **base_config
        )

        self.name_to_skill = dict(
            atomic=self.atomic,
            gripper_release=self.gripper_release,
            reach=self.reach,
            grasp=self.grasp,
            push=self.push
        )

    def execute(self, p_name, output, norm, **kwargs):
        # len(args) = maximal argument length
        skill = self.name_to_skill[p_name]
        skill_dim = skill.get_param_dim()
        skill_args = output[:skill_dim]
        ret = skill.act(skill_args, norm=norm)
        if ret is not None:
            ret['info']['interest_interaction'] = skill.check_interesting_interaction()
        return ret

