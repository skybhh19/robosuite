#!/usr/bin/env python3
"""Summarize ToolHang collection failures from rollout JSON files."""

import argparse
import collections
import json
from pathlib import Path


def summarize(root):
    output = {}
    for regime in ("full", "partial"):
        rows = [json.loads(path.read_text()) for path in sorted((root / "trials").glob(f"pair_*_{regime}.json"))]
        physical_failures = [row for row in rows if not row.get("physical_success")]
        quality_rejections = [row for row in rows if row.get("physical_success") and not row.get("accepted")]
        failure_reasons = collections.Counter()
        failed_stages = collections.Counter()
        quality_checks = collections.Counter()
        examples = collections.defaultdict(list)
        for row in physical_failures:
            stats = row.get("stats", {})
            reason = stats.get("failure_reason") or "missing_failure_reason"
            failure_reasons[reason] += 1
            stage_checks = stats.get("stage_checks", [])
            failed = [stage for stage in stage_checks if not stage.get("success", stage.get("passed", True))]
            stage_name = (failed[-1] if failed else stage_checks[-1]).get("name", "unknown") if stage_checks else "no_stage"
            failed_stages[stage_name] += 1
            if len(examples[reason]) < 3:
                debug = stats.get("final_debug", {})
                examples[reason].append({
                    "pair": row.get("pair"),
                    "steps": row.get("steps"),
                    "last_stage": stage_name,
                    "line_distance_m": debug.get("line_distance_m"),
                    "normalized_insertion": debug.get("normalized_insertion"),
                    "hole_straddles_hook": debug.get("hole_straddles_hook"),
                    "tool_on_frame": debug.get("tool_on_frame"),
                })
        for row in quality_rejections:
            checks = row.get("stats", {}).get("acceptance_checks", {})
            for name, passed in checks.items():
                if not passed:
                    quality_checks[name] += 1
        n = len(rows)
        physical = n - len(physical_failures)
        accepted = sum(bool(row.get("accepted")) for row in rows)
        output[regime] = {
            "rollouts": n,
            "physical_success": physical,
            "physical_success_rate": physical / n,
            "accepted": accepted,
            "accepted_rate": accepted / n,
            "physical_failures": len(physical_failures),
            "quality_only_rejections": len(quality_rejections),
            "failure_reasons": dict(failure_reasons.most_common()),
            "failed_stages": dict(failed_stages.most_common()),
            "quality_rejection_checks": dict(quality_checks.most_common()),
            "examples": dict(examples),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"v1_400_demo_source_pool": summarize(args.v1_root), "v2_300_demo_source_pool": summarize(args.v2_root)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
