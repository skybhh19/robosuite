"""Merge disjoint balanced Threading collection shards into one dataset."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robosuite.scripts.collect_threading_balanced_initial_states import (
    write_hdf5,
)
from robosuite.scripts.collect_threading_scripted_grasp_angle import (
    JOINT_ACTION_NOISE_SCALE_RAD,
    JOINT_TRAJECTORY_TIMING_PROFILE,
    make_controller_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--angles", nargs="+", type=float, required=True)
    parser.add_argument("--partial-angles", nargs="+", type=float, required=True)
    parser.add_argument("--full-angles", nargs="+", type=float, required=True)
    parser.add_argument("--rollouts-per-angle", type=int, required=True)
    parser.add_argument("--joint-margin-rad", type=float, default=0.05)
    parser.add_argument("--required-approach-path-profile", type=str, default=None)
    parser.add_argument("--grasp-offset-along-min-m", type=float, default=None)
    parser.add_argument("--grasp-offset-along-max-m", type=float, default=None)
    parser.add_argument("--require-clean-pregrasp-contact", action="store_true")
    parser.add_argument("--require-joint-approach-nonstreaming", action="store_true")
    parser.add_argument(
        "--joint-approach-waypoint-position-tolerance-m", type=float, default=None
    )
    parser.add_argument(
        "--joint-approach-waypoint-orientation-tolerance-deg", type=float, default=None
    )
    return parser.parse_args()


def load_json(path):
    with path.open() as stream:
        return json.load(stream)


def write_json(path, payload):
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def main():
    args = parse_args()
    output = args.output.resolve()
    raw_output = output / "raw"
    dataset_output = output / "dataset"
    raw_output.mkdir(parents=True, exist_ok=True)
    allowed_angles = {float(angle) for angle in args.angles}

    source_payloads = []
    all_attempts = []
    episode_records = []
    for source_arg in args.sources:
        source = source_arg.resolve()
        summary = load_json(source / "collection_summary.json")
        attempts = []
        with (source / "attempt_log.jsonl").open() as stream:
            attempts = [json.loads(line) for line in stream if line.strip()]
        selected_attempts = [
            item
            for item in attempts
            if float(item["target_grasp_angle_deg"]) in allowed_angles
        ]
        all_attempts.extend(selected_attempts)

        selected_episode_count = 0
        for episode_dir in sorted((source / "raw").glob("ep_*")):
            stats = load_json(episode_dir / "policy_stats.json")
            angle = float(stats["target_grasp_angle_deg"])
            if angle not in allowed_angles:
                continue
            if not bool(stats.get("collection_success")):
                raise ValueError(f"Unaccepted episode present in source: {episode_dir}")
            if not bool(stats.get("env_check_success_final")):
                raise ValueError(f"Episode lacks final env success: {episode_dir}")
            safety = stats.get("joint_safety", {})
            if not bool(safety.get("passed")) or float(safety["minimum_margin_rad"]) < args.joint_margin_rad:
                raise ValueError(f"Episode violates joint margin: {episode_dir}")
            if (
                args.required_approach_path_profile is not None
                and stats.get("approach_path_profile") != args.required_approach_path_profile
            ):
                raise ValueError(f"Episode has unexpected approach profile: {episode_dir}")
            grasp_offset = float(stats.get("grasp_offset_along", np.nan))
            if args.grasp_offset_along_min_m is not None and grasp_offset < args.grasp_offset_along_min_m:
                raise ValueError(f"Episode grasp offset is below requested range: {episode_dir}")
            if args.grasp_offset_along_max_m is not None and grasp_offset > args.grasp_offset_along_max_m:
                raise ValueError(f"Episode grasp offset is above requested range: {episode_dir}")
            pregrasp_contact = stats.get("pregrasp_contact_check", {})
            clean_pregrasp_contact = bool(
                pregrasp_contact.get("passed", False)
                and not pregrasp_contact.get("detected", True)
            )
            if args.require_clean_pregrasp_contact and not clean_pregrasp_contact:
                raise ValueError(f"Episode failed clean pregrasp-contact gate: {episode_dir}")
            approach_outcome = stats.get("stage_outcomes", {}).get("aim_continuous", {})
            stream_waypoints = bool(approach_outcome.get("stream_waypoints", True))
            waypoint_position_tolerance_m = float(
                approach_outcome.get("waypoint_position_tolerance_m", np.nan)
            )
            waypoint_orientation_tolerance_deg = float(
                approach_outcome.get("waypoint_orientation_tolerance_deg", np.nan)
            )
            if args.require_joint_approach_nonstreaming and stream_waypoints:
                raise ValueError(f"Episode used streaming joint approach: {episode_dir}")
            if (
                args.joint_approach_waypoint_position_tolerance_m is not None
                and not np.isclose(
                    waypoint_position_tolerance_m,
                    args.joint_approach_waypoint_position_tolerance_m,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(f"Episode has unexpected position waypoint gate: {episode_dir}")
            if (
                args.joint_approach_waypoint_orientation_tolerance_deg is not None
                and not np.isclose(
                    waypoint_orientation_tolerance_deg,
                    args.joint_approach_waypoint_orientation_tolerance_deg,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(f"Episode has unexpected orientation waypoint gate: {episode_dir}")
            destination = raw_output / episode_dir.name
            if episode_dir.resolve() != destination.resolve():
                if destination.exists():
                    raise FileExistsError(f"Episode-name collision: {destination}")
                shutil.copytree(episode_dir, destination)
            episode_records.append(
                {
                    "episode_dir": destination.name,
                    "source": str(source),
                    "target_grasp_angle_deg": angle,
                    "initial_state_sampling": stats["initial_state_sampling"],
                    "minimum_joint_margin_rad": float(safety["minimum_margin_rad"]),
                    "minimum_joint_margin_joint": int(safety["minimum_margin_joint"]),
                    "pregrasp_contact_passed": bool(pregrasp_contact.get("passed", False)),
                    "pregrasp_contact_detected": bool(pregrasp_contact.get("detected", True)),
                    "joint_approach_stream_waypoints": stream_waypoints,
                    "joint_approach_waypoint_position_tolerance_m": waypoint_position_tolerance_m,
                    "joint_approach_waypoint_orientation_tolerance_deg": waypoint_orientation_tolerance_deg,
                }
            )
            selected_episode_count += 1
        source_payloads.append(
            {
                "path": str(source),
                "seed": summary.get("seed"),
                "selected_episode_count": selected_episode_count,
                "selected_attempt_count": len(selected_attempts),
            }
        )

    counts = Counter(item["target_grasp_angle_deg"] for item in episode_records)
    expected = {float(angle): args.rollouts_per_angle for angle in args.angles}
    if dict(counts) != expected:
        raise ValueError(f"Merged per-angle counts {dict(counts)} do not match {expected}")

    all_attempts.sort(
        key=lambda item: (
            float(item["target_grasp_angle_deg"]),
            int(item["slot"]),
            int(item["replacement_index"]),
            int(item["retry_index"]),
        )
    )
    with (output / "attempt_log.jsonl").open("w") as stream:
        for item in all_attempts:
            stream.write(json.dumps(item) + "\n")

    by_angle = {}
    for angle in args.angles:
        angle = float(angle)
        attempts = [item for item in all_attempts if float(item["target_grasp_angle_deg"]) == angle]
        accepted = [item for item in attempts if bool(item["accepted_success"])]
        replaced = {
            (int(item["slot"]), int(item["replacement_index"]))
            for item in attempts
            if int(item["retry_index"]) == 10 and not bool(item["accepted_success"])
        }
        by_angle[f"{angle:g}"] = {
            "target_grasp_angle_deg": angle,
            "target_successes": args.rollouts_per_angle,
            "initial_states_sampled_up_front": args.rollouts_per_angle,
            "successful_trajectories": len(accepted),
            "rollout_attempts": len(attempts),
            "retry_attempt_success_rate": len(accepted) / len(attempts),
            "final_env_success_attempts": sum(bool(item["env_check_success_final"]) for item in attempts),
            "final_env_success_attempt_rate": sum(
                bool(item["env_check_success_final"]) for item in attempts
            )
            / len(attempts),
            "joint_margin_pass_attempts": sum(bool(item["joint_margin_passed"]) for item in attempts),
            "joint_margin_fail_attempts": sum(not bool(item["joint_margin_passed"]) for item in attempts),
            "pregrasp_contact_pass_attempts": sum(
                bool(item.get("clean_pregrasp_contact", False)) for item in attempts
            ),
            "pregrasp_contact_rejection_attempts": sum(
                not bool(item.get("clean_pregrasp_contact", False)) for item in attempts
            ),
            "rejected_attempt_reasons": dict(
                Counter(
                    str(item.get("failure_reason", "unknown"))
                    for item in attempts
                    if not bool(item["accepted_success"])
                )
            ),
            "initial_states_replaced": len(replaced),
            "initial_states_sampled": args.rollouts_per_angle + len(replaced),
            "retry_histogram": dict(Counter(str(item["retry_index"]) for item in accepted)),
            "minimum_accepted_joint_margin_rad": min(
                item["minimum_joint_margin_rad"]
                for item in episode_records
                if item["target_grasp_angle_deg"] == angle
            ),
        }

    controller = make_controller_config("Panda", "joint_position")
    env_info = {
        "env_name": "Threading_D08",
        "robots": ["Panda"],
        "controller_configs": controller,
        "control_mode": "joint_position",
        "action_representation": "absolute_joint_position",
        "trajectory_timing_profile": JOINT_TRAJECTORY_TIMING_PROFILE,
        "joint_action_noise_scale_rad": JOINT_ACTION_NOISE_SCALE_RAD,
    }
    hdf5_path, hdf5_count, labels_path = write_hdf5(
        raw_output,
        dataset_output,
        env_info,
        args.partial_angles,
        args.full_angles,
    )
    expected_total = len(args.angles) * args.rollouts_per_angle
    if hdf5_count != expected_total:
        raise ValueError(f"Merged HDF5 has {hdf5_count} demos; expected {expected_total}")

    summary = {
        "status": "complete",
        "environment": "Threading_D08",
        "controller": "JOINT_POSITION",
        "control_mode": "joint_position",
        "action_representation": "absolute_joint_position",
        "success_criterion": "final env._check_success() AND minimum actual Panda joint margin",
        "minimum_joint_margin_rad": args.joint_margin_rad,
        "approach_path_profile": args.required_approach_path_profile,
        "grasp_offset_along_range_m": [
            args.grasp_offset_along_min_m,
            args.grasp_offset_along_max_m,
        ],
        "require_clean_pregrasp_contact": args.require_clean_pregrasp_contact,
        "joint_approach_stream_waypoints": (
            False if args.require_joint_approach_nonstreaming else None
        ),
        "joint_approach_waypoint_position_tolerance_m": (
            args.joint_approach_waypoint_position_tolerance_m
        ),
        "joint_approach_waypoint_orientation_tolerance_deg": (
            args.joint_approach_waypoint_orientation_tolerance_deg
        ),
        "angles_deg": [float(angle) for angle in args.angles],
        "partial_observability_angles_deg": [float(angle) for angle in args.partial_angles],
        "full_observability_angles_deg": [float(angle) for angle in args.full_angles],
        "rollouts_per_angle": args.rollouts_per_angle,
        "max_retries_per_initial_state": 10,
        "failed_simulator_states_saved": False,
        "angles": by_angle,
        "totals": {
            "successful_trajectories": expected_total,
            "rollout_attempts": len(all_attempts),
            "retry_attempt_success_rate": expected_total / len(all_attempts),
            "initial_states_sampled": sum(item["initial_states_sampled"] for item in by_angle.values()),
            "initial_states_replaced": sum(item["initial_states_replaced"] for item in by_angle.values()),
            "pregrasp_contact_rejection_attempts": sum(
                item["pregrasp_contact_rejection_attempts"] for item in by_angle.values()
            ),
            "joint_margin_fail_attempts": sum(
                item["joint_margin_fail_attempts"] for item in by_angle.values()
            ),
        },
        "hdf5_path": str(hdf5_path),
        "hdf5_demonstrations": hdf5_count,
        "observability_labels_path": str(labels_path),
        "observability_label_rules": {
            "partial": [float(angle) for angle in args.partial_angles],
            "full": [float(angle) for angle in args.full_angles],
        },
        "collection_shards": source_payloads,
    }
    write_json(output / "collection_summary.json", summary)
    write_json(
        output / "collection_manifest.json",
        {
            "collection_shards": source_payloads,
            "episodes": sorted(
                episode_records,
                key=lambda item: (
                    item["target_grasp_angle_deg"],
                    item["initial_state_sampling"]["slot"],
                ),
            ),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
