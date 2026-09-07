"""CLI wiring checks; rollout success is assessed by the paired evaluator."""
import sys
import unittest
from unittest.mock import patch

from robosuite.scripts import collect_tool_hang_wrench_joint as core
from robosuite.scripts import collect_tool_hang_two_candidate_pool as pool


class RobustPresetTests(unittest.TestCase):
    def parse(self, module, *flags):
        required = ["--output-dir", "/tmp/toolhang-preset-test"] if module is pool else []
        with patch.object(sys, "argv", ["test", *required, *flags]):
            return module.parse_args()

    def test_both_collectors_enable_complete_shared_policy(self):
        for module in (core, pool):
            args = self.parse(module, "--policy-preset", "robust_joint")
            self.assertEqual(args.controller_backend, "joint_position")
            for name, value in core.ROBUST_JOINT_OPTIONS.items():
                self.assertEqual(getattr(args, name), value)

    def test_legacy_defaults_are_preserved(self):
        for module in (core, pool):
            args = self.parse(module)
            self.assertEqual(args.controller_backend, "osc_pose")
            self.assertFalse(args.line_correction_memory)
            self.assertFalse(args.transfer_retime)

    def test_explicit_parameters_override_preset_defaults(self):
        args = self.parse(core, "--policy-preset", "robust_joint",
                          "--grasp-ik-yaw-fallback-deg", "2")
        self.assertEqual(args.grasp_ik_yaw_fallback_deg, 2)

    def test_insertion_retime_preset_adds_only_insertion_retime(self):
        args = self.parse(core, "--policy-preset", "robust_joint_insertion_retime")
        self.assertEqual(args.controller_backend, "joint_position")
        self.assertTrue(args.insertion_retime)
        self.assertEqual(args.transfer_ik_recovery_restarts, 120)
        for name, value in core.ROBUST_JOINT_OPTIONS.items():
            self.assertEqual(getattr(args, name), value)

    def test_incompatible_controller_or_memory_flags_rejected(self):
        for flags in (("--policy-preset", "robust_joint", "--controller-backend", "osc_pose"),
                      ("--line-correction-memory-unseated-only",),
                      ("--line-correction-memory",)):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.parse(core, *flags)


if __name__ == "__main__":
    unittest.main()
