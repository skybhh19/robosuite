import robosuite as suite

import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from robosuite.environments.base import REGISTERED_ENVS, MujocoEnv

# device = SpaceMouse(9583, 50734, pos_sensitivity=1.0, rot_sensitivity=0.1)
# device.start_control()

options = {}
options["robots"] = ["Panda"]

options["controller_configs"] = suite.load_controller_config(default_controller="OSC_POSITION")

options["env_name"] = "Kitchen"


env = suite.make(**options,
                 has_renderer=True,
                 render_camera="agentview",
                 has_offscreen_renderer=False,
                 ignore_done=True,
                 use_camera_obs=False,
                 horizon=1000,
                 control_freq=20,
                 use_object_obs=True,)
obs = env.reset()
# for i in range(10):
#     env.reset()
#     obs, _, _, _ = env.step([0, 0, 0, 0])
#     env.render()
#     input()
for _ in range(1000):
    env.render()
    obs, _, _, _ = env.step([0, 0, 0, 0])
    # obs = get_obs(env)
    # print(obs['robot0_eef_pos'])
    # obs, _, _, _ = env.step([0.1] * 4)
    # obs = get_obs(env)
    # for k in obs.keys():
    #     print(k, obs[k])
    # print(obs['obj_pos'][:3] - obs['robot0_eef_pos'])
    # print(obs['obj_pos'][3:] - obs['robot0_eef_pos'])
    # print(obs['obj_pos'][:3] - obs['obj_pos'][3:])
    # input()
    # env.render()

