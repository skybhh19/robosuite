"""Collect one strict-success ToolHang demo for each frozen reset state.

The physical reset-state pool is generated before any full / partial labels
are assigned. Each state is then retried at most N times without changing its
robot, wrench, or fixture pose. A slot that exhausts its retries receives a
new screened physical state while retaining its assigned regime, grasp bin,
and motion style. Only the first strict success is retained.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite
from robosuite.scripts.collect_tool_hang_wrench_joint import (
    FULL_VISIBLE_GRASP_RANGE,
    PARTIAL_HIDDEN_GRASP_RANGE,
    GeometricJointPolicy,
    VideoRecorder,
    collection_acceptance,
    finalize_episode,
    make_controller_config,
    controller_metadata,
    numpy_json_default,
)
from robosuite.wrappers import DataCollectionWrapper


def make_env(
    seed,
    camera_height,
    camera_width,
    headless=False,
    controller_backend="osc_pose",
):
    return suite.make(
        "ToolHangWrenchOnly",
        robots=["Panda"],
        controller_configs=make_controller_config("Panda", controller_backend),
        initialization_noise=None,
        ignore_done=True,
        use_camera_obs=not headless,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=not headless,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        horizon=700,
        hard_reset=False,
        seed=seed,
    )


def sample_screened_state(env, candidates):
    """Sample one assembled, collision-screened physical reset state."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    while True:
        candidates += 1
        variation = base_env.sample_reset_variation()
        variation["fixture_translation_m"] = [0.0, 0.0, 0.0]
        variation["fixture_yaw_rad"] = 0.0
        base_env.configure_reset_variation(deepcopy(variation))
        base_env.reset()
        qpos = np.asarray(variation["robot_qpos"], dtype=float)
        if not base_env._check_frame_assembled():
            continue
        if not GeometricJointPolicy._robot_start_path_clear(
            base_env, qpos, "threading_continuous"
        ):
            continue
        variation["pool_candidate_index"] = candidates
        return variation, candidates


def generate_state_pool(env, count, assignment_seed):
    """Generate and collision-screen all states before assigning regimes."""
    states = []
    candidates = 0
    while len(states) < count:
        variation, candidates = sample_screened_state(env, candidates)
        states.append(variation)

    # Regime assignment happens only after every physical state is frozen.
    assignment_rng = np.random.RandomState(assignment_seed)
    permutation = assignment_rng.permutation(count)
    regimes = [None] * count
    half = count // 2
    for index in permutation[:half]:
        regimes[int(index)] = "full_visible"
    for index in permutation[half:]:
        regimes[int(index)] = "partial_hidden"
    # Assign grasp coordinates and motion families only after the complete
    # physical state pool exists. Each motion family receives equal full and
    # partial counts, so neither factor changes the reset-state distribution.
    style_rng = np.random.RandomState(assignment_seed + 1)
    grasp_rng = np.random.RandomState(assignment_seed + 2)
    by_regime = {
        regime: [index for index, value in enumerate(regimes) if value == regime]
        for regime in ("full_visible", "partial_hidden")
    }
    styles = GeometricJointPolicy.VARIATION_STYLES
    motion_styles = [None] * count
    grasp_positions = [None] * count
    for regime, indexes in by_regime.items():
        indexes = np.asarray(indexes, dtype=int)
        style_rng.shuffle(indexes)
        for rank, index in enumerate(indexes):
            motion_styles[int(index)] = styles[rank % len(styles)]
        low, high = (
            FULL_VISIBLE_GRASP_RANGE
            if regime == "full_visible"
            else PARTIAL_HIDDEN_GRASP_RANGE
        )
        # Stratification guarantees broad accepted coverage rather than
        # relying on 50 independent draws that can cluster by chance.
        bins = np.linspace(low, high, len(indexes) + 1)
        samples = np.asarray(
            [grasp_rng.uniform(bins[i], bins[i + 1]) for i in range(len(indexes))]
        )
        grasp_rng.shuffle(samples)
        for index, grasp_x in zip(indexes, samples):
            grasp_positions[int(index)] = float(grasp_x)
    return [
        {
            "state_id": index + 1,
            "regime": regimes[index],
            "motion_style": motion_styles[index],
            "grasp_offset_local_x_m": grasp_positions[index],
            "grasp_offset_range_m": [
                max(
                    FULL_VISIBLE_GRASP_RANGE[0]
                    if regimes[index] == "full_visible"
                    else PARTIAL_HIDDEN_GRASP_RANGE[0],
                    grasp_positions[index] - 0.0025,
                ),
                min(
                    FULL_VISIBLE_GRASP_RANGE[1]
                    if regimes[index] == "full_visible"
                    else PARTIAL_HIDDEN_GRASP_RANGE[1],
                    grasp_positions[index] + 0.0025,
                ),
            ],
            "reset_variation": state,
        }
        for index, state in enumerate(states)
    ], candidates


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=numpy_json_default) + "\n")


def freeze_manifest_fixtures(state_pool):
    """Normalize old and new manifests to the fixed assembled fixture."""
    for entry in state_pool:
        variation = entry["reset_variation"]
        variation["fixture_translation_m"] = [0.0, 0.0, 0.0]
        variation["fixture_yaw_rad"] = 0.0
    return state_pool


def write_dataset_metadata(output_dir, raw_dir, results):
    """Write a flat label table and a short schema note for downstream users."""
    episode_by_state = {}
    for episode_dir in sorted(raw_dir.glob("ep_*")):
        stats_path = episode_dir / "policy_stats.json"
        if not stats_path.is_file():
            continue
        state_id = int(json.loads(stats_path.read_text())["state_id"])
        episode_by_state[state_id] = episode_dir.name

    fieldnames = [
        "state_id",
        "episode_dir",
        "grasp_regime",
        "grasp_offset_local_x_m",
        "grasp_offset_local_x_mm",
        "motion_style",
        "retry",
        "replacement_count",
        "total_slot_attempts",
        "steps",
        "native_success",
        "accepted",
        "tool_on_frame",
        "wrench_pose_assist_count",
        "video",
        "controller_backend",
        "action_space",
    ]
    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for stats in sorted(results, key=lambda item: int(item["state_id"])):
            state_id = int(stats["state_id"])
            grasp_x = float(stats["variation"]["grasp_offset_local_x_m"])
            writer.writerow(
                {
                    "state_id": state_id,
                    "episode_dir": episode_by_state[state_id],
                    "grasp_regime": stats["assigned_regime"],
                    "grasp_offset_local_x_m": f"{grasp_x:.9f}",
                    "grasp_offset_local_x_mm": f"{1000.0 * grasp_x:.6f}",
                    "motion_style": stats["variation"]["motion_style"],
                    "retry": int(stats["retry"]),
                    "replacement_count": int(stats.get("replacement_count", 0)),
                    "total_slot_attempts": int(stats["total_slot_attempts"]),
                    "steps": int(stats["steps"]),
                    "native_success": bool(stats["native_success"]),
                    "accepted": bool(stats["accepted"]),
                    "tool_on_frame": bool(stats["tool_on_frame"]),
                    "wrench_pose_assist_count": int(stats["wrench_pose_assist_count"]),
                    "video": stats["video"],
                    "controller_backend": stats.get("controller_backend", "joint_position"),
                    "action_space": stats.get(
                        "action_space", "absolute_joint_position_plus_gripper"
                    ),
                }
            )

    regime_counts = Counter(item["assigned_regime"] for item in results)
    controller_counts = Counter(
        item.get("controller_backend", "joint_position") for item in results
    )
    controller_configs = {
        backend: controller_metadata(backend) for backend in controller_counts
    }
    write_json(
        output_dir / "controller_metadata.json",
        {
            "controllers": controller_configs,
            "no_temporal_subsampling": True,
            "fixture_translation_m": [0.0, 0.0, 0.0],
            "fixture_yaw_rad": 0.0,
        },
    )
    readme = f"""# ToolHang phase-2 scripted dataset

This directory contains {len(results)} strict-success demonstrations:
{regime_counts.get('full_visible', 0)} `full_visible` grasps and
{regime_counts.get('partial_hidden', 0)} `partial_hidden` grasps. Controllers:
{dict(controller_counts)}. The initial physical states were sampled before
grasp regimes and motion styles were assigned. Fixture translation and yaw are
fixed at zero.

## Files

- `raw_demos/ep_*/state_*.npz`: simulator states and per-step action labels.
- `raw_demos/ep_*/model.xml`: exact MuJoCo model used for the episode.
- `raw_demos/ep_*/policy_stats.json`: stage gates, reset variation, trajectory
  variation, native success, and smoothness diagnostics.
- `labels.csv`: one flat row per retained demonstration, including the raw
  episode directory, grasp regime, continuous grasp coordinate, and motion style.
- `state_manifest.json`: the pre-generated reset-state pool and assignments.
- `tool_hang_wrench_joint_summary.json`: collection-level and rollout summaries.
- `controller_metadata.json`: exact action dimensions, frames, scaling, gains,
  and control frequency for each real rollout backend.

Inside each NPZ, `action_infos` stores the action actually executed at every
control step. Joint-position episodes additionally store absolute joint targets
and joint-position diagnostics. OSC episodes store the real normalized 6-D
world-frame delta pose action plus gripper; they are never relabeled from a
joint trajectory after collection.
"""
    (output_dir / "README.md").write_text(readme)


def load_progress(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tool_hang_registered_success100",
    )
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--max-replacements-per-slot", type=int, default=20)
    parser.add_argument(
        "--max-regime-success-rate-gap",
        type=float,
        default=0.10,
        help="Reject the completed collection if full / partial attempt success rates differ by more than this fraction.",
    )
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--assignment-seed", type=int, default=6101)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument(
        "--controller-backend",
        choices=("joint_position", "osc_pose"),
        default="osc_pose",
    )
    parser.add_argument("--high-hole-height-m", type=float, default=0.060)
    parser.add_argument("--seat-along-fraction", type=float, default=0.10)
    parser.add_argument("--hang-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics and strict gates without creating a video renderer.",
    )
    parser.add_argument(
        "--motion-style",
        choices=GeometricJointPolicy.VARIATION_STYLES,
        default="high_arc",
        help="Use the same validated transfer style for every full / partial state.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.states <= 0 or args.states % 2:
        raise ValueError("--states must be a positive even number")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")
    if args.max_replacements_per_slot < 0:
        raise ValueError("--max-replacements-per-slot must be non-negative")

    output_dir = args.output_dir.resolve()
    video_dir = output_dir / "videos"
    failed_video_dir = output_dir / "_attempt_videos"
    raw_dir = output_dir / "raw_demos"
    manifest_path = output_dir / "state_manifest.json"
    progress_path = output_dir / "collection_progress.json"
    summary_path = output_dir / "tool_hang_wrench_joint_summary.json"
    video_dir.mkdir(parents=True, exist_ok=True)
    failed_video_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.seed,
        args.camera_height,
        args.camera_width,
        args.headless,
        args.controller_backend,
    )
    if args.resume and manifest_path.is_file():
        manifest_payload = json.loads(manifest_path.read_text())
        state_pool = manifest_payload["states"]
        candidate_count = manifest_payload["candidate_count"]
    else:
        state_pool, candidate_count = generate_state_pool(
            env, args.states, args.assignment_seed
        )
        manifest_payload = {
            "seed": args.seed,
            "assignment_seed": args.assignment_seed,
            "state_count": args.states,
            "candidate_count": candidate_count,
            "assignment_after_generation": True,
            "states": state_pool,
        }
        write_json(manifest_path, manifest_payload)

    state_pool = freeze_manifest_fixtures(state_pool)
    manifest_payload["fixture_randomization"] = False
    manifest_payload["controller_backend"] = args.controller_backend
    manifest_payload["states"] = state_pool
    write_json(manifest_path, manifest_payload)

    if args.motion_style is not None:
        for entry in state_pool:
            entry["motion_style"] = args.motion_style
        manifest_payload["forced_motion_style"] = args.motion_style
        manifest_payload["states"] = state_pool
        write_json(manifest_path, manifest_payload)

    prior_progress = load_progress(progress_path) if args.resume else {}
    existing = prior_progress.get("rollouts", [])
    completed = {int(item["state_id"]): item for item in existing if item.get("accepted")}
    results = [completed[key] for key in sorted(completed)]
    replacement_events = list(prior_progress.get("replacement_events", []))
    env = DataCollectionWrapper(
        env,
        str(raw_dir),
        collect_freq=1,
        flush_freq=701,
        record_joint_position_fields=args.controller_backend == "joint_position",
        joint_delta_scale=0.05,
        reload_from_xml_on_episode_start=False,
    )

    started = time.time()
    failed_states = []
    try:
        for entry in state_pool:
            state_id = int(entry["state_id"])
            if state_id in completed:
                continue
            regime = entry["regime"]
            profile = "full_visible" if regime == "full_visible" else "partial_hidden"
            accepted_stats = None
            final_video = video_dir / f"rollout_{state_id:03d}.mp4"
            replacement_index = int(entry.get("replacement_count", 0))
            while accepted_stats is None:
                policy = GeometricJointPolicy(
                    seed=args.seed + 1000 + state_id + 100000 * replacement_index,
                    variation=True,
                    grasp_profile=profile,
                    robot_start_mode="threading_continuous",
                    motion_style=entry["motion_style"],
                    grasp_offset_range=tuple(entry["grasp_offset_range_m"]),
                    controller_backend=args.controller_backend,
                    high_hole_height_m=args.high_hole_height_m,
                    seat_along_fraction=args.seat_along_fraction,
                    hang_yaw_deg=args.hang_yaw_deg,
                )
                attempt_history = []
                for retry in range(1, args.max_retries + 1):
                    attempt_video = failed_video_dir / (
                        f"state_{state_id:03d}_replacement_{replacement_index:02d}"
                        f"_try_{retry:02d}.mp4"
                    )
                    native_success, stats = policy.rollout(
                        env,
                        VideoRecorder(None if args.headless else attempt_video),
                        reset_variation_override=entry["reset_variation"],
                        allow_reset_resample=False,
                    )
                    checks, accepted = collection_acceptance(
                        native_success,
                        stats,
                        wrist_requirement="any",
                        require_ph_quality=True,
                    )
                    stats["native_success"] = bool(native_success)
                    stats["acceptance_checks"] = checks
                    stats["accepted"] = bool(accepted)
                    stats["success"] = bool(accepted)
                    stats["state_id"] = state_id
                    stats["assigned_regime"] = regime
                    stats["retry"] = retry
                    stats["max_retries"] = args.max_retries
                    stats["replacement_index"] = replacement_index
                    if native_success and not accepted:
                        stats["failure_reason"] = "acceptance:" + ",".join(
                            name for name, passed in checks.items() if not passed
                        )
                    finalize_episode(env, accepted, stats, keep_failed=False)
                    attempt_history.append(
                        {
                            "retry": retry,
                            "native_success": bool(native_success),
                            "accepted": bool(accepted),
                            "failure_reason": stats.get("failure_reason"),
                            "acceptance_checks": checks,
                            "stage_checks": stats.get("stage_checks", []),
                            "final_debug": stats.get("final_debug", {}),
                            "smoothness": stats.get("smoothness", {}),
                            "visibility_diagnostics": stats.get(
                                "visibility_diagnostics", {}
                            ),
                        }
                    )
                    if accepted:
                        if not args.headless:
                            shutil.move(str(attempt_video), str(final_video))
                        stats["retry_history"] = attempt_history
                        stats["replacement_count"] = replacement_index
                        stats["total_slot_attempts"] = (
                            replacement_index * args.max_retries + retry
                        )
                        stats["video"] = None if args.headless else final_video.name
                        accepted_stats = stats
                        results.append(stats)
                        print(
                            f"state={state_id:03d}/{args.states} regime={regime} "
                            f"replacement={replacement_index} accepted retry={retry} "
                            f"total={len(results)}/{args.states}",
                            flush=True,
                        )
                        break
                    attempt_video.unlink(missing_ok=True)
                    last_stage = stats.get("stage_checks", [])[-1] if stats.get("stage_checks") else {}
                    last_debug = last_stage.get("tool_debug", stats.get("final_debug", {}))
                    print(
                        f"state={state_id:03d}/{args.states} regime={regime} "
                        f"replacement={replacement_index} retry={retry}/{args.max_retries} "
                        f"failed={stats.get('failure_reason')} "
                        f"line_mm={1000.0 * last_debug.get('line_distance_m', float('nan')):.1f} "
                        f"straddle={last_debug.get('hole_straddles_hook')} "
                        f"depth={last_debug.get('normalized_insertion', float('nan')):.3f}",
                        flush=True,
                    )
                if accepted_stats is not None:
                    break

                replacement_event = {
                    "state_id": state_id,
                    "regime": regime,
                    "motion_style": entry["motion_style"],
                    "replacement_index": replacement_index,
                    "reset_variation": deepcopy(entry["reset_variation"]),
                    "attempts": attempt_history,
                }
                replacement_events.append(replacement_event)
                entry.setdefault("replacement_history", []).append(replacement_event)
                if replacement_index >= args.max_replacements_per_slot:
                    failed_states.append(replacement_event)
                    break

                replacement_index += 1
                replacement_state, candidate_count = sample_screened_state(
                    env, candidate_count
                )
                replacement_state["replacement_for_state_id"] = state_id
                replacement_state["replacement_index"] = replacement_index
                entry["reset_variation"] = replacement_state
                entry["replacement_count"] = replacement_index
                manifest_payload["candidate_count"] = candidate_count
                manifest_payload["states"] = state_pool
                write_json(manifest_path, manifest_payload)
                print(
                    f"state={state_id:03d}/{args.states} regime={regime} "
                    f"sampling replacement={replacement_index}",
                    flush=True,
                )
            results.sort(key=lambda item: int(item["state_id"]))
            write_json(
                progress_path,
                {
                    "states": args.states,
                    "max_retries": args.max_retries,
                    "rollouts": results,
                    "failed_states": failed_states,
                    "replacement_events": replacement_events,
                },
            )
    finally:
        env.close()

    regime_attempts = {
        regime: sum(
            int(item.get("total_slot_attempts", item["retry"]))
            for item in results
            if item["assigned_regime"] == regime
        )
        for regime in ("full_visible", "partial_hidden")
    }
    regime_successes = {
        regime: sum(item["assigned_regime"] == regime for item in results)
        for regime in ("full_visible", "partial_hidden")
    }
    regime_attempt_success_rates = {
        regime: (
            regime_successes[regime] / regime_attempts[regime]
            if regime_attempts[regime]
            else 0.0
        )
        for regime in regime_attempts
    }
    regime_success_rate_gap = abs(
        regime_attempt_success_rates["full_visible"]
        - regime_attempt_success_rates["partial_hidden"]
    )
    regime_rate_balanced = bool(
        regime_success_rate_gap <= args.max_regime_success_rate_gap
    )
    summary = {
        "seed": args.seed,
        "assignment_seed": args.assignment_seed,
        "state_count": args.states,
        "candidate_count": candidate_count,
        "max_retries_per_state": args.max_retries,
        "max_replacements_per_slot": args.max_replacements_per_slot,
        "attempts": sum(int(item.get("total_slot_attempts", item["retry"])) for item in results)
        + sum(len(item["attempts"]) for item in failed_states),
        "successes": len(results),
        "success_rate": len(results) / args.states,
        "attempts_by_regime": regime_attempts,
        "successes_by_regime": regime_successes,
        "attempt_success_rates_by_regime": regime_attempt_success_rates,
        "regime_success_rate_gap": regime_success_rate_gap,
        "max_regime_success_rate_gap": args.max_regime_success_rate_gap,
        "regime_success_rates_balanced": regime_rate_balanced,
        "elapsed_seconds": time.time() - started,
        "failed_states": failed_states,
        "replacement_events": replacement_events,
        "replacement_count": len(replacement_events),
        "motion_style_counts": dict(
            Counter(item["variation"]["motion_style"] for item in results)
        ),
        "rollouts": sorted(results, key=lambda item: int(item["state_id"])),
    }
    write_json(summary_path, summary)
    if len(results) == args.states and regime_rate_balanced:
        write_dataset_metadata(output_dir, raw_dir, results)
    print(
        json.dumps(
            {
                "successes": len(results),
                "states": args.states,
                "failed_state_ids": [item["state_id"] for item in failed_states],
                "summary": str(summary_path),
            },
            indent=2,
        )
    )
    if len(results) != args.states:
        raise SystemExit(2)
    if not regime_rate_balanced:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
