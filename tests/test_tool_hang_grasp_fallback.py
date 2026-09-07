"""Extract the actual selector block to check its gate and RNG isolation."""
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
import numpy as np


class GraspFallbackTests(unittest.TestCase):
    def run_selector(self, baseline_error, candidate_error, fallback=1.0):
        text = (Path(__file__).resolve().parents[1] / "robosuite/scripts/collect_tool_hang_wrench_joint.py").read_text()
        start = text.index("            initial_grasp_rng_state = self.ik_rng.get_state()")
        end = text.index('\n            pregrasp_qpos = selected_grasp["pregrasp_qpos"]', start)
        selector = textwrap.dedent(text[start:end])
        policy = SimpleNamespace(ik_rng=np.random.RandomState(27), grasp_ik_yaw_fallback_deg=fallback)
        calls = []
        states = {}
        def solve(yaw):
            calls.append(yaw)
            # Simulate solvers consuming different numbers of random starts.
            policy.ik_rng.normal(size=3 if yaw == 0 else 8)
            states[yaw] = policy.ik_rng.get_state()
            return {"metrics": {"grasp_ik_error": baseline_error if yaw == 0 else candidate_error},
                    "rng_state": states[yaw]}
        scope = {"self": policy, "solve_grasp_yaw": solve, "episode_grasp_yaw_deg": 0.0,
                 "CLEAN_GRASP_IK_ERROR_MAX": .18, "variation_params": {}}
        exec(selector, scope)
        selected = scope["selected_grasp"]
        expected = np.random.RandomState(27)
        chose_candidate = fallback is not None and baseline_error > .18 and candidate_error < baseline_error
        expected.normal(size=8 if chose_candidate else 3)
        np.testing.assert_array_equal(policy.ik_rng.normal(size=20), expected.normal(size=20))
        return calls, scope["variation_params"], selected

    def test_valid_grasp_not_touched(self):
        calls, audit, _ = self.run_selector(.10, .01)
        self.assertEqual(calls, [0.0])
        self.assertFalse(audit["grasp_ik_yaw_fallback"]["evaluated"])

    def test_existing_quality_threshold_not_narrowed(self):
        calls, _, _ = self.run_selector(.18, .01)
        self.assertEqual(calls, [0.0])

    def test_improving_fallback_selected(self):
        calls, audit, selected = self.run_selector(.23, .09)
        self.assertEqual(calls, [0.0, 1.0])
        self.assertTrue(audit["grasp_ik_yaw_fallback"]["selected"])
        self.assertEqual(selected["metrics"]["grasp_ik_error"], .09)

    def test_worse_fallback_restores_original_rng(self):
        calls, audit, selected = self.run_selector(.23, .30)
        self.assertEqual(calls, [0.0, 1.0])
        self.assertFalse(audit["grasp_ik_yaw_fallback"]["selected"])
        self.assertEqual(selected["metrics"]["grasp_ik_error"], .23)

    def test_default_disabled(self):
        calls, audit, _ = self.run_selector(.23, .09, None)
        self.assertEqual(calls, [0.0])
        self.assertEqual(audit, {})


if __name__ == "__main__":
    unittest.main()
