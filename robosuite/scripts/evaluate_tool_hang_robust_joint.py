"""Fixed-budget paired evaluation of legacy and robust_joint, without retries.

Prepare freezes random resets before any policy runs. Run executes both policies
and both grasp regimes on every assigned reset. Summarize refuses incomplete
coverage and counts exceptions as failures. This does not collect a dataset.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robosuite.scripts.collect_tool_hang_wrench_joint import ROBUST_JOINT_OPTIONS, numpy_json_default
from robosuite.scripts.collect_tool_hang_balanced_state_retries import generate_state_pool, make_env
from robosuite.scripts.run_tool_hang_paired_insertion_diagnostics import trial
from robosuite.scripts.run_tool_hang_shared_improvements import compact
from robosuite.scripts.run_tool_hang_frozen_confirmation import source_tree_sha

REGIMES = ("full_visible", "partial_hidden")
VERSIONS = ("legacy", "robust_joint")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=numpy_json_default) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z*z/count
    center = (p + z*z/(2*count))/scale
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/scale
    return [center-half, center+half]


def prepare(args):
    if args.pairs <= 0 or args.pairs % 40:
        raise ValueError("pairs must be a positive multiple of 40")
    path = args.output / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    env = make_env(args.seed, 84, 84, True, "joint_position")
    try:
        entries, screened = generate_state_pool(env, args.pairs, args.seed + 1)
    finally:
        env.close()
    signatures = set()
    for entry in entries:
        entry["state_id"] += 900000
        reset = dict(entry["reset_variation"])
        reset.pop("pool_candidate_index", None)
        digest = hashlib.sha256(json.dumps(reset, sort_keys=True).encode()).hexdigest()
        if digest in signatures:
            raise ValueError("duplicate physical reset")
        signatures.add(digest)
    write(path, dict(purpose="fixed_budget_paired_evaluation", seed=args.seed,
                    pairs=args.pairs, screened_candidates=screened, states=entries,
                    no_policy_outcomes_used=True, attempts_per_policy_regime_state=1,
                    physical_rate_target=.90, controller_backend="joint_position",
                    fixture_randomization=False, full_grasp_range_m=[-.005,.005],
                    partial_grasp_range_m=[.045,.055],
                    policies={"legacy": {}, "robust_joint": ROBUST_JOINT_OPTIONS},
                    runtime={"python": sys.version, "mujoco": mujoco.__version__,
                             "numpy": np.__version__},
                    source_tree_sha256=source_tree_sha(ROOT)))
    print(json.dumps(dict(manifest=str(path), pairs=args.pairs, rollouts=4*args.pairs)), flush=True)


def run(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if source_tree_sha(ROOT) != manifest["source_tree_sha256"]:
        raise ValueError("source changed after policy freeze")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    for entry in manifest["states"][args.shard_index::args.shard_count]:
        path = args.output / "pairs" / f"pair_{entry['state_id']}.json"
        pending = path.with_suffix(".pending.json")
        if path.exists() or pending.exists():
            raise FileExistsError("refusing implicit retry: " + str(path))
        row = dict(source_entry=entry, manifest_sha256=sha(manifest_path), results={})
        initial = None
        for version in VERSIONS:
            row["results"][version] = {}
            for regime in REGIMES:
                row["active"] = [version, regime]
                write(pending, row)
                try:
                    result = compact(trial(deepcopy(entry), regime, False,
                                           manifest["policies"][version], manifest["seed"]))
                except Exception as exc:
                    result = dict(exception=dict(type=type(exc).__name__, message=str(exc),
                                                 traceback=traceback.format_exc()))
                if "exception" not in result:
                    if initial is None:
                        initial = result["initial_hashes"]
                    if initial != result["initial_hashes"]:
                        raise ValueError("paired physics mismatch")
                row["results"][version][regime] = result
                write(pending, row)
        row.pop("active")
        successful_runs = [row["results"][v][r] for v in VERSIONS for r in REGIMES
                           if "exception" not in row["results"][v][r]]
        for result in successful_runs:
            reference = successful_runs[0]
            if result["policy_seed"] != reference["policy_seed"]:
                raise ValueError("paired policy seed mismatch")
            for key in ("reset", "motion_scale", "frame_offsets"):
                if result["stats"]["variation"][key] != reference["stats"]["variation"][key]:
                    raise ValueError("paired randomization mismatch: " + key)
        for version in VERSIONS:
            f, p = [row["results"][version][r] for r in REGIMES]
            if "exception" not in f and "exception" not in p:
                difference = (p["stats"]["variation"]["grasp_offset_local_x_m"]
                              - f["stats"]["variation"]["grasp_offset_local_x_m"])
                if not math.isclose(difference, .05, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("paired grasp quantiles differ")
        row["paired_initial_hashes"] = initial
        write(path, row)
        pending.unlink()
        print(json.dumps(dict(state_id=entry["state_id"], outcomes={
            v: {r: row["results"][v][r].get("stats", {}).get("success", False)
                for r in REGIMES} for v in VERSIONS})), flush=True)


def outcomes(result):
    stats = result.get("stats", {})
    return bool(stats.get("success", False)), bool(stats.get("accepted", False))


def summarize(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(p.read_text()) for p in sorted((args.output / "pairs").glob("pair_*.json"))]
    expected = {e["state_id"] for e in manifest["states"]}
    if len(rows) != len(expected) or {x["source_entry"]["state_id"] for x in rows} != expected:
        raise ValueError(f"incomplete evaluation: {len(rows)}/{len(expected)} pairs")
    if any(x["manifest_sha256"] != sha(manifest_path) for x in rows):
        raise ValueError("mixed experiment provenance")
    n = len(rows)
    summary = dict(pairs=n, rollouts=4*n, manifest_sha256=sha(manifest_path),
                   runtime=manifest["runtime"], policies={}, paired_changes={})
    for version in VERSIONS:
        summary["policies"][version] = {}
        for regime in REGIMES:
            results = [row["results"][version][regime] for row in rows]
            physical = sum(outcomes(x)[0] for x in results)
            strict = sum(outcomes(x)[1] for x in results)
            failures = Counter()
            for x in results:
                if "exception" in x:
                    failures["exception"] += 1
                elif not outcomes(x)[0]:
                    stages = x["stats"].get("stage_checks", [])
                    first = next((s["name"] for s in stages if not s.get("passed", False)), "native_success")
                    failures[first] += 1
            summary["policies"][version][regime] = dict(
                attempts=n, physical_successes=physical, physical_rate=physical/n,
                physical_wilson95=wilson(physical,n), strict_successes=strict,
                strict_rate=strict/n, strict_wilson95=wilson(strict,n),
                quality_rejections_after_physical_success=physical-strict,
                physical_failures=dict(failures))
    for regime in REGIMES:
        summary["paired_changes"][regime] = {}
        for index, metric in enumerate(("physical", "strict")):
            pairs = [(outcomes(row["results"]["legacy"][regime])[index],
                      outcomes(row["results"]["robust_joint"][regime])[index]) for row in rows]
            summary["paired_changes"][regime][metric] = dict(
                gains=sum(not a and b for a,b in pairs), losses=sum(a and not b for a,b in pairs))
    summary["both_physical_point_estimates_at_least_90_percent"] = all(
        summary["policies"]["robust_joint"][r]["physical_rate"] >= .90 for r in REGIMES)
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=202609031)
    parser.add_argument("--shard-count", type=int, default=40)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
