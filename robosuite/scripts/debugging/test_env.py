import robosuite as suite
import matplotlib.pyplot as plt
import robosuite.utils.macros as macros
# macros.IMAGE_CONVENTION = "opencv"

from robosuite.environments.base import REGISTERED_ENVS, MujocoEnv

# device = SpaceMouse(9583, 50734, pos_sensitivity=1.0, rot_sensitivity=0.1)
# device.start_control()

options = {}
options["robots"] = ["Panda"]

options["controller_configs"] = suite.load_controller_config(default_controller="OSC_POSITION")

options["env_name"] = "CleanUpMediumSmallInit"


env = suite.make(**options,
                 has_renderer=False,
                 render_camera="frontview",
                 has_offscreen_renderer=True,
                 ignore_done=True,
                 use_camera_obs=True,
                 horizon=1000,
                 control_freq=20,
                 camera_widths=256,
                 camera_heights=256,
                 use_object_obs=True,)
# for _ in range(100):
# obs = env.reset()

    # env.render()
    # input()
# for i in range(1000):
#     env.reset()
#     print(i)
for _ in range(1000):
    # env.render()
    obs, _, _, _ = env.step([0, 0, 0, 1])
    # print(obs['robot0_gripper_qpos'])
    print(obs.keys())
    print(obs['agentview_image'])
    #
    plt.imshow(obs['agentview_image'][::-1])
    plt.show()
    input()

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

