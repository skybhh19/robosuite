"""Collect a balanced ToolHang dataset with two strict candidates per state."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robosuite.scripts.collect_tool_hang_balanced_state_retries import (
    freeze_manifest_fixtures,
    generate_state_pool,
    make_env,
    write_dataset_metadata,
    write_json,
)
from robosuite.scripts.collect_tool_hang_wrench_joint import (
    GeometricJointPolicy,
    VideoRecorder,
    collection_acceptance,
    controller_metadata,
    finalize_episode,
)
from robosuite.wrappers import DataCollectionWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=220)
    parser.add_argument("--final-states", type=int, default=200)
    parser.add_argument("--successes-per-state", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--max-regime-success-rate-gap",
        type=float,
        default=0.10,
        help="Reject final assembly if full / partial per-attempt success rates differ by more than this fraction.",
    )
    parser.add_argument(
        "--fixed-attempts-per-state",
        type=int,
        default=0,
        help=(
            "Run exactly this many attempts even after enough candidates are found. "
            "Extra successes contribute to the unbiased rate but are not retained."
        ),
    )
    parser.add_argument("--seed", type=int, default=22800)
    parser.add_argument("--assignment-seed", type=int, default=22801)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--controller-backend",
        choices=("joint_position", "osc_pose"),
        default="osc_pose",
    )
    parser.add_argument("--high-hole-height-m", type=float, default=0.060)
    parser.add_argument("--seat-along-fraction", type=float, default=0.10)
    parser.add_argument("--hang-yaw-deg", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--keep-rejected-candidates",
        action="store_true",
        help=(
            "Keep the non-winning strict candidates so replay parity can be "
            "checked before final best-of-two assembly."
        ),
    )
    parser.add_argument(
        "--regime", choices=("all", "full_visible", "partial_hidden"), default="all"
    )
    parser.add_argument("--grasp-bin-index", type=int, choices=range(5))
    parser.add_argument(
        "--motion-style",
        choices=GeometricJointPolicy.VARIATION_STYLES,
        help="Collect only manifest states assigned to this motion style.",
    )
    parser.add_argument(
        "--policy-motion-style",
        choices=GeometricJointPolicy.VARIATION_STYLES,
        default="high_arc",
        help="Override the policy style while preserving each frozen reset and grasp state.",
    )
    parser.add_argument(
        "--vertical-fallback-every",
        type=int,
        default=0,
        help=(
            "For vertical_first states, use high_arc on every Nth retry. "
            "Each style keeps its own deterministic retry seed sequence."
        ),
    )
    parser.add_argument(
        "--vertical-retry-style-cycle",
        help=(
            "Comma-separated retry path cycle for vertical_first states, for example "
            "vertical_first,high_arc,left_sweep,vertical_first,high_arc,direct_low. "
            "Each path keeps an independent deterministic retry seed sequence."
        ),
    )
    parser.add_argument("--limit-states", type=int)
    parser.add_argument(
        "--state-ids",
        help="Comma-separated frozen state ids to collect, for focused regression tests.",
    )
    return parser.parse_args()


def insert_stage(stats):
    return next(
        (stage for stage in stats.get("stage_checks", []) if stage.get("name") == "insert"),
        {},
    )


def transfer_stage(stats):
    return next(
        (
            stage
            for stage in stats.get("stage_checks", [])
            if stage.get("name") == "transfer_rotate"
        ),
        {},
    )


def quality_components(stats):
    """A deterministic within-state score; lower is better."""
    insert = insert_stage(stats)
    transfer = transfer_stage(stats)
    smooth = stats.get("smoothness", {})
    pre_hold = insert.get(
        "pre_seated_tool_debug",
        insert.get("pre_hold_tool_debug", insert.get("tool_debug", {})),
    )
    return {
        "pre_release_line_distance_m": float(
            pre_hold.get("line_distance_m", float("inf"))
        ),
        "max_actual_joint_second_difference": float(
            smooth.get("max_actual_joint_second_difference", float("inf"))
        ),
        "max_joint_target_jerk": float(
            smooth.get("max_joint_target_jerk", float("inf"))
        ),
        "transfer_final_ik_error": float(
            transfer.get("global_ik_final_error", float("inf"))
        ),
        "grasp_ik_error": float(
            stats.get("variation", {}).get("grasp_ik_error", float("inf"))
        ),
        "training_frame_deviation_from_305": abs(
            int(stats.get("training_recorded_steps", stats.get("steps", 0))) - 305
        ),
    }


def quality_key(stats):
    values = quality_components(stats)
    return (
        values["pre_release_line_distance_m"],
        values["max_actual_joint_second_difference"],
        values["max_joint_target_jerk"],
        values["transfer_final_ik_error"],
        values["grasp_ik_error"],
        values["training_frame_deviation_from_305"],
    )


def remove_candidate(output_dir, candidate):
    episode = candidate.get("episode_dir")
    if episode:
        shutil.rmtree(output_dir / "raw_demos" / episode, ignore_errors=True)
    video = candidate.get("candidate_video")
    if video:
        (output_dir / "_candidate_videos" / video).unlink(missing_ok=True)


def progress_payload(args, states, records, attempts):
    return {
        "states": args.states,
        "final_states": args.final_states,
        "successes_per_state": args.successes_per_state,
        "max_attempts": args.max_attempts,
        "fixed_attempts_per_state": args.fixed_attempts_per_state,
        "attempts": attempts,
        "state_records": records,
        "eligible_states": sum(
            record.get("status") == "eligible" for record in records.values()
        ),
        "failed_states": sum(
            record.get("status") == "failed" for record in records.values()
        ),
    }


def main():
    args = parse_args()
    if args.states <= 0 or args.states % 2:
        raise ValueError("--states must be a positive even number")
    if args.final_states <= 0 or args.final_states % 2 or args.final_states > args.states:
        raise ValueError("--final-states must be positive, even, and <= --states")
    if args.successes_per_state < 1:
        raise ValueError("--successes-per-state must be positive")
    if args.max_attempts < args.successes_per_state:
        raise ValueError("--max-attempts must allow the requested successes")
    if not 0 <= args.fixed_attempts_per_state <= args.max_attempts:
        raise ValueError("--fixed-attempts-per-state must be between 0 and --max-attempts")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    if args.vertical_fallback_every < 0:
        raise ValueError("--vertical-fallback-every must be nonnegative")
    vertical_retry_style_cycle = None
    if args.vertical_retry_style_cycle:
        vertical_retry_style_cycle = tuple(
            style.strip()
            for style in args.vertical_retry_style_cycle.split(",")
            if style.strip()
        )
        invalid_styles = set(vertical_retry_style_cycle) - set(
            GeometricJointPolicy.VARIATION_STYLES
        )
        if not vertical_retry_style_cycle or invalid_styles:
            raise ValueError(
                f"Invalid --vertical-retry-style-cycle: {sorted(invalid_styles)}"
            )
        if args.vertical_fallback_every:
            raise ValueError(
                "Use either --vertical-retry-style-cycle or "
                "--vertical-fallback-every, not both"
            )

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw_demos"
    candidate_video_dir = output_dir / "_candidate_videos"
    video_dir = output_dir / "videos"
    manifest_path = output_dir / "state_manifest.json"
    progress_path = output_dir / "collection_progress.json"
    summary_path = output_dir / "tool_hang_wrench_joint_summary.json"
    for directory in (raw_dir, candidate_video_dir, video_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.seed,
        args.camera_height,
        args.camera_width,
        args.headless,
        args.controller_backend,
    )
    if args.source_manifest is not None:
        manifest = json.loads(args.source_manifest.resolve().read_text())
        state_pool = manifest["states"]
        candidate_count = int(manifest["candidate_count"])
        write_json(manifest_path, manifest)
    elif args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        state_pool = manifest["states"]
        candidate_count = int(manifest["candidate_count"])
    else:
        if any(raw_dir.iterdir()) or any(candidate_video_dir.iterdir()) or any(video_dir.iterdir()):
            raise RuntimeError(f"Output directory is not empty: {output_dir}")
        state_pool, candidate_count = generate_state_pool(
            env, args.states, args.assignment_seed
        )
        manifest = {
            "seed": args.seed,
            "assignment_seed": args.assignment_seed,
            "state_count": args.states,
            "candidate_count": candidate_count,
            "assignment_after_generation": True,
            "states": state_pool,
        }
        write_json(manifest_path, manifest)

    state_pool = freeze_manifest_fixtures(state_pool)
    manifest["fixture_randomization"] = False
    manifest["controller_backend"] = args.controller_backend
    manifest["controller_config"] = controller_metadata(args.controller_backend)
    manifest["placement"] = {
        "high_hole_height_m": args.high_hole_height_m,
        "seat_along_fraction": args.seat_along_fraction,
        "hang_yaw_deg": args.hang_yaw_deg,
    }
    manifest["states"] = state_pool
    write_json(manifest_path, manifest)

    state_pool = [
        entry
        for entry in state_pool
        if (int(entry["state_id"]) - 1) % args.shard_count == args.shard_index
        and (args.regime == "all" or entry["regime"] == args.regime)
    ]
    if args.state_ids:
        requested_state_ids = {
            int(value.strip()) for value in args.state_ids.split(",") if value.strip()
        }
        state_pool = [
            entry
            for entry in state_pool
            if int(entry["state_id"]) in requested_state_ids
        ]
        found_state_ids = {int(entry["state_id"]) for entry in state_pool}
        if found_state_ids != requested_state_ids:
            raise ValueError(
                f"Requested state ids not available after filters: "
                f"{sorted(requested_state_ids - found_state_ids)}"
            )
    if args.grasp_bin_index is not None:
        low = -0.055 + 0.009 * args.grasp_bin_index
        high = low + 0.009
        state_pool = [
            entry
            for entry in state_pool
            if low
            <= 0.5 * sum(entry["grasp_offset_range_m"])
            <= high + 1e-12
        ]
    if args.motion_style is not None:
        state_pool = [
            entry for entry in state_pool if entry["motion_style"] == args.motion_style
        ]
    if args.limit_states is not None:
        state_pool = state_pool[: args.limit_states]

    if args.prepare_only:
        env.close()
        print(json.dumps({"prepared_states": len(manifest["states"]), "manifest": str(manifest_path)}, indent=2))
        return

    prior = json.loads(progress_path.read_text()) if args.resume and progress_path.is_file() else {}
    records = {str(key): value for key, value in prior.get("state_records", {}).items()}
    attempts_total = int(prior.get("attempts", 0))
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
    try:
        for entry in state_pool:
            state_id = int(entry["state_id"])
            key = str(state_id)
            record = records.get(
                key,
                {
                    "state_id": state_id,
                    "assigned_regime": entry["regime"],
                    "motion_style": entry["motion_style"],
                    "attempts": 0,
                    "candidates": [],
                    "attempt_history": [],
                    "status": "pending",
                },
            )
            if record.get("status") in ("eligible", "failed"):
                continue

            profile = (
                "full_visible" if entry["regime"] == "full_visible" else "partial_hidden"
            )
            while (
                (
                    len(record["candidates"]) < args.successes_per_state
                    or int(record["attempts"]) < args.fixed_attempts_per_state
                )
                and int(record["attempts"]) < args.max_attempts
            ):
                retry = int(record["attempts"]) + 1
                candidate_index = len(record["candidates"]) + 1
                assigned_policy_style = args.policy_motion_style or entry["motion_style"]
                use_vertical_cycle = (
                    vertical_retry_style_cycle is not None
                    and assigned_policy_style == "vertical_first"
                )
                use_vertical_fallback = (
                    args.vertical_fallback_every > 0
                    and assigned_policy_style == "vertical_first"
                    and retry % args.vertical_fallback_every == 0
                )
                if use_vertical_cycle:
                    attempt_motion_style = vertical_retry_style_cycle[
                        (retry - 1) % len(vertical_retry_style_cycle)
                    ]
                    style_retry = sum(
                        vertical_retry_style_cycle[index % len(vertical_retry_style_cycle)]
                        == attempt_motion_style
                        for index in range(retry)
                    )
                elif (
                    args.vertical_fallback_every > 0
                    and assigned_policy_style == "vertical_first"
                ):
                    attempt_motion_style = (
                        "high_arc" if use_vertical_fallback else assigned_policy_style
                    )
                    fallback_attempts = retry // args.vertical_fallback_every
                    style_retry = (
                        fallback_attempts
                        if use_vertical_fallback
                        else retry - fallback_attempts
                    )
                else:
                    attempt_motion_style = assigned_policy_style
                    style_retry = retry
                # Attempt-indexed seeds make an interrupted/resumed run exactly
                # reproduce the same remaining candidate sequence. When the
                # robust vertical fallback is enabled, each path advances its
                # own seed sequence so neither path loses its validated basin.
                policy = GeometricJointPolicy(
                    seed=args.seed + 1000 + state_id + 100000 * style_retry,
                    variation=True,
                    grasp_profile=profile,
                    robot_start_mode="threading_continuous",
                    motion_style=attempt_motion_style,
                    grasp_offset_range=tuple(entry["grasp_offset_range_m"]),
                    controller_backend=args.controller_backend,
                    high_hole_height_m=args.high_hole_height_m,
                    seat_along_fraction=args.seat_along_fraction,
                    hang_yaw_deg=args.hang_yaw_deg,
                )
                candidate_video = candidate_video_dir / (
                    f"state_{state_id:03d}_candidate_{candidate_index:02d}_try_{retry:02d}.mp4"
                )
                native_success, stats = policy.rollout(
                    env,
                    VideoRecorder(None if args.headless else candidate_video),
                    reset_variation_override=entry["reset_variation"],
                    allow_reset_resample=False,
                )
                checks, accepted = collection_acceptance(
                    native_success, stats, wrist_requirement="any", require_ph_quality=True
                )
                stats.update(
                    {
                        "native_success": bool(native_success),
                        "acceptance_checks": checks,
                        "accepted": bool(accepted),
                        "success": bool(accepted),
                        "state_id": state_id,
                        "assigned_regime": entry["regime"],
                        "retry": retry,
                        "assigned_motion_style": entry["motion_style"],
                        "attempt_motion_style": attempt_motion_style,
                        "style_retry": style_retry,
                        "max_retries": args.max_attempts,
                        "replacement_index": 0,
                        "replacement_count": 0,
                        "total_slot_attempts": retry,
                    }
                )
                if native_success and not accepted:
                    stats["failure_reason"] = "acceptance:" + ",".join(
                        name for name, passed in checks.items() if not passed
                    )
                retain_candidate = bool(
                    accepted and len(record["candidates"]) < args.successes_per_state
                )
                episode_dir = Path(env.ep_directory).name
                finalize_episode(env, retain_candidate, stats, keep_failed=False)
                attempts_total += 1
                record["attempts"] = retry
                record["attempt_history"].append(
                    {
                        "retry": retry,
                        "assigned_motion_style": entry["motion_style"],
                        "attempt_motion_style": attempt_motion_style,
                        "style_retry": style_retry,
                        "native_success": bool(native_success),
                        "accepted": bool(accepted),
                        "failure_reason": stats.get("failure_reason"),
                        "variation": stats.get("variation", {}),
                        "final_debug": stats.get("final_debug", {}),
                        "stage_checks": stats.get("stage_checks", []),
                        "smoothness": stats.get("smoothness", {}),
                        "visibility_diagnostics": stats.get(
                            "visibility_diagnostics", {}
                        ),
                    }
                )
                if retain_candidate:
                    candidate = stats
                    candidate["episode_dir"] = episode_dir
                    candidate["candidate_video"] = None if args.headless else candidate_video.name
                    candidate["quality_components"] = quality_components(candidate)
                    record["candidates"].append(candidate)
                    print(
                        f"state={state_id:03d}/{args.states} regime={entry['regime']} "
                        f"candidate={len(record['candidates'])}/{args.successes_per_state} "
                        f"attempt={retry}/{args.max_attempts}",
                        flush=True,
                    )
                else:
                    candidate_video.unlink(missing_ok=True)
                    print(
                        f"state={state_id:03d}/{args.states} regime={entry['regime']} "
                        f"attempt={retry}/{args.max_attempts} "
                        f"{'extra_success_not_retained' if accepted else 'failed=' + str(stats.get('failure_reason'))}",
                        flush=True,
                    )
                records[key] = record
                write_json(
                    progress_path,
                    progress_payload(args, state_pool, records, attempts_total),
                )

            if len(record["candidates"]) >= args.successes_per_state:
                ranked = sorted(record["candidates"], key=quality_key)
                winner = ranked[0]
                losers = ranked[1:]
                if not args.keep_rejected_candidates:
                    for loser in losers:
                        remove_candidate(output_dir, loser)
                final_video = video_dir / f"rollout_{state_id:03d}.mp4"
                if not args.headless:
                    shutil.move(
                        str(candidate_video_dir / winner["candidate_video"]),
                        str(final_video),
                    )
                winner["video"] = None if args.headless else final_video.name
                winner["candidate_count"] = len(ranked)
                winner["selected_candidate_retry"] = int(winner["retry"])
                winner["rejected_candidate_retries"] = [int(item["retry"]) for item in losers]
                winner["selection_rule"] = "lexicographic_within_frozen_state"
                record["winner"] = winner
                record["rejected_candidates"] = (
                    losers if args.keep_rejected_candidates else []
                )
                record["candidates"] = []
                record["status"] = "eligible"
            else:
                for candidate in record["candidates"]:
                    remove_candidate(output_dir, candidate)
                record["candidates"] = []
                record["status"] = "failed"
            records[key] = record
            write_json(
                progress_path,
                progress_payload(args, state_pool, records, attempts_total),
            )
    finally:
        env.close()

    if args.collect_only:
        selected = sorted(
            (
                record["winner"]
                for record in records.values()
                if record.get("status") == "eligible"
            ),
            key=lambda item: int(item["state_id"]),
        )
        write_json(
            summary_path,
            {
                "seed": args.seed,
                "assignment_seed": args.assignment_seed,
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "attempts": attempts_total,
                "eligible_state_count": len(selected),
                "failed_state_count": sum(
                    record.get("status") == "failed" for record in records.values()
                ),
                "rollouts": selected,
            },
        )
        print(json.dumps({"shard": args.shard_index, "eligible": len(selected)}, indent=2))
        return

    target_per_regime = args.final_states // 2
    selected = []
    overflow = []
    for regime in ("full_visible", "partial_hidden"):
        eligible = sorted(
            (
                record
                for record in records.values()
                if record.get("status") == "eligible"
                and record["assigned_regime"] == regime
            ),
            key=lambda item: int(item["state_id"]),
        )
        selected.extend(record["winner"] for record in eligible[:target_per_regime])
        overflow.extend(record for record in eligible[target_per_regime:])

    if len(selected) == args.final_states:
        for record in overflow:
            remove_candidate(
                output_dir,
                {
                    "episode_dir": record["winner"].get("episode_dir"),
                    "candidate_video": None,
                },
            )
            video_name = record["winner"].get("video")
            if video_name:
                (video_dir / video_name).unlink(missing_ok=True)
            record["status"] = "overflow_not_selected"

    selected.sort(key=lambda item: int(item["state_id"]))
    regime_attempts = {
        regime: sum(
            len(record.get("attempt_history", []))
            for record in records.values()
            if record.get("assigned_regime") == regime
        )
        for regime in ("full_visible", "partial_hidden")
    }
    regime_attempt_successes = {
        regime: sum(
            bool(attempt.get("accepted"))
            for record in records.values()
            if record.get("assigned_regime") == regime
            for attempt in record.get("attempt_history", [])
        )
        for regime in regime_attempts
    }
    regime_attempt_success_rates = {
        regime: (
            regime_attempt_successes[regime] / regime_attempts[regime]
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
        "sampled_states": args.states,
        "state_count": args.final_states,
        "successes_per_state_required": args.successes_per_state,
        "max_attempts_per_state": args.max_attempts,
        "attempts": attempts_total,
        "eligible_state_count": sum(
            record.get("status") in ("eligible", "overflow_not_selected")
            for record in records.values()
        ),
        "failed_state_count": sum(
            record.get("status") == "failed" for record in records.values()
        ),
        "successes": len(selected),
        "attempts_by_regime": regime_attempts,
        "attempt_successes_by_regime": regime_attempt_successes,
        "attempt_success_rates_by_regime": regime_attempt_success_rates,
        "regime_success_rate_gap": regime_success_rate_gap,
        "max_regime_success_rate_gap": args.max_regime_success_rate_gap,
        "regime_success_rates_balanced": regime_rate_balanced,
        "regime_counts": dict(Counter(item["assigned_regime"] for item in selected)),
        "motion_style_counts": dict(
            Counter(item["variation"]["motion_style"] for item in selected)
        ),
        "selection": (
            "best of two within each frozen state; first 100 eligible states in "
            "the pre-generated order for each regime"
        ),
        "elapsed_seconds": time.time() - started,
        "rollouts": selected,
    }
    write_json(summary_path, summary)
    if len(selected) == args.final_states and regime_rate_balanced:
        write_dataset_metadata(output_dir, raw_dir, selected)
    print(json.dumps({
        "selected": len(selected),
        "required": args.final_states,
        "eligible": summary["eligible_state_count"],
        "failed_states": summary["failed_state_count"],
        "summary": str(summary_path),
    }, indent=2))
    if len(selected) != args.final_states:
        raise SystemExit(2)
    if not regime_rate_balanced:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
