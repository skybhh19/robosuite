"""Construct the published V17 environment and check its action/camera contract.

Run from the repository root with:
    python -m robosuite.scripts.check_fork_v17_environment
Add --render to check both native 256px camera observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import robosuite
from robosuite.utils.fork_legacy_geometry import shared_wrist_quaternion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    path = Path(robosuite.__file__).parent / 'configs/fork_v17/environment.json'
    metadata = json.loads(path.read_text())
    kwargs = dict(metadata['env_kwargs'])
    if not args.render:
        kwargs.update(has_offscreen_renderer=False, use_camera_obs=False)
    env = robosuite.make(metadata['env_name'], **kwargs)
    try:
        obs = env.reset()
        assert env.control_freq == 20 and env.action_dim == 8
        camera = env.sim.model.camera_name2id('robot0_eye_in_hand')
        expected_pos = [kwargs['fork_wrist_camera_' + axis + '_m'] for axis in 'xyz']
        expected_quat = shared_wrist_quaternion(kwargs['fork_wrist_camera_tilt_deg'],
            kwargs['fork_wrist_camera_pan_deg'], kwargs['fork_wrist_camera_roll_deg'])
        np.testing.assert_allclose(env.sim.model.cam_pos[camera], expected_pos, atol=1e-10)
        np.testing.assert_allclose(env.sim.model.cam_quat[camera], expected_quat, atol=1e-10)
        assert env.sim.model.cam_fovy[camera] == 50
        for _ in range(3):
            obs, _, _, _ = env.step(np.r_[obs['robot0_joint_pos'], -1.])
            if args.render:
                for name in ('agentview', 'robot0_eye_in_hand'):
                    pixels = obs[name + '_image']
                    assert pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8
        print(json.dumps(dict(env=metadata['env_name'], control_hz=env.control_freq,
                             action_dim=env.action_dim, camera_fovy=50, rendered=args.render)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
