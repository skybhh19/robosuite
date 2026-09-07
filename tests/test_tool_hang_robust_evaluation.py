"""Ensure evaluation cannot inflate rates by dropping failed/missing trials."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from robosuite.scripts import evaluate_tool_hang_robust_joint as evaluator


class EvaluationAccountingTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "manifest.json"
        evaluator.write(manifest, {"states": [{"state_id": 1}, {"state_id": 2}], "runtime": {}})
        for sid in (1, 2):
            results = {v: {r: {"stats": {"success": True, "accepted": True}}
                           for r in evaluator.REGIMES} for v in evaluator.VERSIONS}
            if sid == 2:
                results["robust_joint"]["full_visible"] = {"exception": {"type": "RuntimeError"}}
                results["legacy"]["partial_hidden"]["stats"]["accepted"] = False
            evaluator.write(root / "pairs" / f"pair_{sid}.json", {
                "source_entry": {"state_id": sid}, "manifest_sha256": evaluator.sha(manifest),
                "results": results})

    def test_exceptions_count_in_denominator_and_quality_has_own_rate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with redirect_stdout(io.StringIO()):
                evaluator.summarize(SimpleNamespace(output=root))
            import json
            summary = json.loads((root / "summary.json").read_text())
            full = summary["policies"]["robust_joint"]["full_visible"]
            self.assertEqual((full["attempts"], full["physical_successes"]), (2, 1))
            self.assertEqual(full["physical_failures"], {"exception": 1})
            partial = summary["policies"]["legacy"]["partial_hidden"]
            self.assertEqual((partial["physical_rate"], partial["strict_rate"]), (1, .5))
            self.assertFalse(summary["both_physical_point_estimates_at_least_90_percent"])

    def test_missing_result_prevents_reporting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "pairs" / "pair_2.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete evaluation"):
                evaluator.summarize(SimpleNamespace(output=root))

    def test_changed_manifest_prevents_mixing_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            manifest = root / "manifest.json"
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "mixed experiment provenance"):
                evaluator.summarize(SimpleNamespace(output=root))


if __name__ == "__main__":
    unittest.main()
