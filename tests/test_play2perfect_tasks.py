import numpy as np
import pytest

import robosuite as suite
import robosuite.utils.transform_utils as T


def _make_env(name):
    return suite.make(
        name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        seed=7,
    )


@pytest.mark.parametrize(
    ("name", "expected_keys"),
    (
        ("YCBForkInRack", {"fork_pos", "rack_pos", "fork_goal_pos", "fork_inserted"}),
        (
            "MultiPartAssembly",
            {"beam_part0_pos", "beam_part2_pos", "part0_goal_pos", "part2_goal_pos", "assembly_stage"},
        ),
    ),
)
def test_play2perfect_task_reset_and_step(name, expected_keys):
    env = _make_env(name)
    try:
        obs = env.reset()
        assert expected_keys.issubset(obs)
        _, reward, done, _ = env.step(np.zeros(env.action_dim))
        assert reward == 0.0
        assert not done
        assert not env._check_success()
    finally:
        env.close()


@pytest.mark.parametrize("name", ("YCBForkInRack", "MultiPartAssembly"))
def test_play2perfect_goal_pose_is_successful_and_physically_stable(name):
    env = _make_env(name)
    try:
        env.reset()
        if name == "YCBForkInRack":
            specs = [(env.fork, env.FORK_GOAL_POS, env.FORK_GOAL_QUAT)]
        else:
            specs = [
                (env.part2, env.PART2_GOAL_POS, env.PART2_GOAL_QUAT),
                (env.part0, env.PART0_GOAL_POS, env.PART0_GOAL_QUAT),
            ]
        for obj, local_pos, local_quat in specs:
            goal_pos, goal_quat_xyzw = env._goal_pose(local_pos, local_quat)
            env._set_free_object_pose(obj, goal_pos, goal_quat_xyzw[[3, 0, 1, 2]])
        env.sim.forward()
        assert env._check_success()
        assert env.reward() == 1.0
        for _ in range(120):
            env.sim.step()
        assert env._check_success()
    finally:
        env.close()


def test_multi_part_assembly_matches_released_beam_frame_and_has_full_fixture_visual():
    env = _make_env("MultiPartAssembly")
    try:
        env.reset()
        # Released beam_3x canonical transforms: part 2's long local X axis
        # becomes world +Z, while part 0's becomes world -Y.
        part2_rot = T.quat2mat(env.PART2_GOAL_QUAT[[1, 2, 3, 0]])
        part0_rot = T.quat2mat(env.PART0_GOAL_QUAT[[1, 2, 3, 0]])
        np.testing.assert_allclose(part2_rot[:, 0], (0.0, 0.0, 1.0), atol=1e-7)
        np.testing.assert_allclose(part0_rot[:, 0], (0.0, -1.0, 0.0), atol=1e-7)

        # The released SDF mesh is only a local hole patch. The four visual
        # primitives are required to show the complete part-6 receiver.
        visual_geoms = {
            "beam_fixture_side_pos_y_visual",
            "beam_fixture_side_neg_y_visual",
            "beam_fixture_left_end_visual",
            "beam_fixture_right_bulk_visual",
        }
        assert visual_geoms.issubset(set(env.sim.model.geom_names))
        assert "assemblyview" in env.sim.model.camera_names
    finally:
        env.close()
