import collections
import copy
import numpy as np
from robosuite.controllers.skills import (
    AtomicSkill,
    ReachSkill,
    GraspSkill,
    PushSkill,
    GripperReleaseSkill,
)

def _setup_config(config):
    pass
class SkillController:

    SKILL_NAMES = [
        'atomic',
        'reach',
        'grasp',
        'push',
        'open'
    ]

    def __init__(self,
               env,
               config):
        self._env = env
        _setup_config(config)

    def execute(self, p, args, image_obs_in_info=False, **kwargs):
        image_obs = []
        reward_sum = 0
        while True:
            action_ll