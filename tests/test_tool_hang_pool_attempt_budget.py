"""Synthetic protocol tests; these are not evidence of rollout performance."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "robosuite/scripts/audit_tool_hang_multistyle_pool.py"
spec = importlib.util.spec_from_file_location("pool_audit", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit = module.audited_attempt_prefix


class AttemptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.history = [{"retry": i, "accepted": i % 2 == 0} for i in range(1, 5)]

    def test_all_attempts_including_failures_retained(self):
        self.assertEqual(audit(self.history, 4, True), self.history)
        self.assertEqual(sum(x["accepted"] for x in audit(self.history, 4, True)), 2)

    def test_order_does_not_change_denominator(self):
        self.assertEqual(len(audit(list(reversed(self.history)), 4, True)), 4)

    def test_missing_or_duplicate_ids_rejected(self):
        for history in (self.history[1:], self.history[:3] + [self.history[2]]):
            with self.assertRaises(ValueError):
                audit(history, 4, True)

    def test_extra_retries_require_explicit_legacy_mode(self):
        history = self.history + [{"retry": 5, "accepted": True}]
        with self.assertRaises(ValueError):
            audit(history, 4, True)
        self.assertEqual(audit(history, 4, False), self.history)

    def test_invalid_values_not_silently_coerced(self):
        for field, value in (("retry", 1.5), ("retry", True), ("retry", 0),
                             ("accepted", "False"), ("accepted", 1)):
            history = deepcopy(self.history)
            history[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit(history, 4, True)

    def test_invalid_budget_rejected(self):
        for budget in (0, -1, True, 4.0):
            with self.assertRaises(ValueError):
                audit(self.history, budget, True)


if __name__ == "__main__":
    unittest.main()
