from unittest.mock import patch

import numpy as np

from robosuite.scripts.collect_tool_hang_balanced_state_retries import (
    generate_state_pool,
)
from robosuite.scripts.collect_tool_hang_wrench_joint import GeometricJointPolicy


class FakeResetEnv:
    def __init__(self, seed=7):
        self.unwrapped = self
        self.rng = np.random.RandomState(seed)
        self.pending = None

    def sample_reset_variation(self):
        return {"robot_qpos": np.zeros(7).tolist()}

    def configure_reset_variation(self, variation):
        self.pending = variation

    def reset(self):
        return {}

    def _check_frame_assembled(self):
        return True


def test_fixture_randomization_is_bounded_and_reproducible():
    kwargs = dict(
        count=40,
        assignment_seed=91,
        fixture_x_range_m=(-0.035, 0.035),
        fixture_y_range_m=(-0.025, 0.025),
        fixture_yaw_range_deg=(-20.0, 20.0),
    )
    with patch.object(GeometricJointPolicy, "_robot_start_path_clear", return_value=True):
        first, _ = generate_state_pool(FakeResetEnv(), **kwargs)
        second, _ = generate_state_pool(FakeResetEnv(), **kwargs)

    first_variations = [entry["reset_variation"] for entry in first]
    second_variations = [entry["reset_variation"] for entry in second]
    assert first_variations == second_variations
    assert len({tuple(v["fixture_translation_m"][:2]) for v in first_variations}) == 40
    for variation in first_variations:
        x, y, z = variation["fixture_translation_m"]
        assert -0.035 <= x <= 0.035
        assert -0.025 <= y <= 0.025
        assert z == 0.0
        assert np.deg2rad(-20.0) <= variation["fixture_yaw_rad"] <= np.deg2rad(20.0)


def test_default_fixture_remains_fixed_for_existing_callers():
    with patch.object(GeometricJointPolicy, "_robot_start_path_clear", return_value=True):
        entries, _ = generate_state_pool(FakeResetEnv(), 4, 13)
    for entry in entries:
        variation = entry["reset_variation"]
        assert variation["fixture_translation_m"] == [0.0, 0.0, 0.0]
        assert variation["fixture_yaw_rad"] == 0.0
