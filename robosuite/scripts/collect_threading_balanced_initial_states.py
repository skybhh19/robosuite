"""Collect Threading demos with bounded retries per sampled initial state.

Each requested grasp angle starts with an independently sampled pool of initial
states. A state is retried with policy randomness up to ``--max-retries`` times.
Only trajectories that pass both the policy's composite success check and the
final environment ``_check_success()`` call are retained. Exhausted states are
replaced until the requested number of successful trajectories is reached.
"""

import argparse
import csv
import datetime
import hashlib
import json
import sys
from collections import Counter, deque
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite
from robosuite.scripts.collect_threading_scripted_grasp_angle import (
    ThreadingScriptedPolicy,
    finalize_episode,
    json_safe,
    make_controller_config,
)
from robosuite.wrappers import DataCollectionWrapper


DEFAULT_OUTPUT = REPO_ROOT / "threading_d05_bc_88_96_15_per_angle_balanced_initial_states"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--environment", type=str, default="Threading_D05")
    parser.add_argument("--angles", nargs="+", type=float, default=list(range(88, 97)))
    parser.add_argument("--rollouts-per-angle", type=int, default=15)
    parser.add_argument("--initial-states-per-angle", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--collision-aware-threading",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def make_env(env_name, controller_config, seed, horizon, offscreen=False):
    return suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=controller_config,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        horizon=horizon,
        seed=seed,
    )


def state_identifier(state):
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()[:16]


def sample_initial_state(sampler_env, serial, slot, replacement_index):
    sampler_env.reset()
    state = np.asarray(sampler_env.sim.get_state().flatten(), dtype=float).copy()
    return {
        "serial": int(serial),
        "slot": int(slot),
        "replacement_index": int(replacement_index),
        "state_id": state_identifier(state),
        "state": state,
    }


def restore_initial_state(wrapped_env, initial_state):
    """Restore a sampled state after the policy-owned reset and sync logging."""
    state = np.asarray(initial_state["state"], dtype=float)
    wrapped_env.sim.set_state_from_flattened(state)
    wrapped_env.sim.forward()

    base_env = wrapped_env.unwrapped
    base_env._threading_initial_tripod_pos = None
    base_env._threading_max_insert_progress = -np.inf

    # DataCollectionWrapper starts the episode before this callback. Replace
    # its recorded initial state so replay begins from the sampled state.
    wrapped_env._current_task_instance_state = state.copy()
    wrapped_env.successful = False


def empty_angle_summary(angle, target_successes, initial_count):
    return {
        "target_grasp_angle_deg": float(angle),
        "target_successes": int(target_successes),
        "initial_states_sampled_up_front": int(initial_count),
        "successful_trajectories": 0,
        "rollout_attempts": 0,
        "retry_attempt_success_rate": 0.0,
        "initial_states_sampled": int(initial_count),
        "initial_states_replaced": 0,
        "successful_initial_states": 0,
        "retry_histogram": {},
        "failure_reasons": {},
        "aborted_stages": {},
    }


def update_rates(summary):
    total_attempts = sum(item["rollout_attempts"] for item in summary["angles"].values())
    total_successes = sum(item["successful_trajectories"] for item in summary["angles"].values())
    total_replacements = sum(item["initial_states_replaced"] for item in summary["angles"].values())
    total_states = sum(item["initial_states_sampled"] for item in summary["angles"].values())
    summary["totals"] = {
        "successful_trajectories": int(total_successes),
        "rollout_attempts": int(total_attempts),
        "retry_attempt_success_rate": float(total_successes / total_attempts) if total_attempts else 0.0,
        "initial_states_sampled": int(total_states),
        "initial_states_replaced": int(total_replacements),
    }
    for item in summary["angles"].values():
        attempts = item["rollout_attempts"]
        item["retry_attempt_success_rate"] = (
            float(item["successful_trajectories"] / attempts) if attempts else 0.0
        )


def write_summary(path, summary):
    update_rates(summary)
    with path.open("w") as output:
        json.dump(json_safe(summary), output, indent=2, allow_nan=False)


def append_attempt(path, record):
    with path.open("a") as output:
        output.write(json.dumps(json_safe(record), allow_nan=False) + "\n")


def load_raw_episode(ep_dir):
    states = []
    actions = []
    successful = False
    env_name = None
    for state_path in sorted(ep_dir.glob("state_*.npz")):
        data = np.load(state_path, allow_pickle=True)
        env_name = str(data["env"])
        states.extend(data["states"])
        actions.extend(info["actions"] for info in data["action_infos"])
        successful = bool(successful or data["successful"])
    if not successful or not states:
        return None
    if len(states) != len(actions) + 1:
        raise ValueError(
            f"Episode {ep_dir.name} has {len(states)} states for {len(actions)} actions"
        )
    with (ep_dir / "model.xml").open() as model_file:
        model_xml = model_file.read()
    with (ep_dir / "policy_stats.json").open() as stats_file:
        stats = json.load(stats_file)
    return {
        "states": np.asarray(states[:-1]),
        "actions": np.asarray(actions),
        "model_xml": model_xml,
        "stats": stats,
        "env_name": env_name,
    }


def observability_label(angle):
    angle = float(angle)
    if 86.0 <= angle <= 90.0:
        return "partial"
    if 93.0 <= angle <= 97.0:
        return "full"
    return "unlabeled"


def write_hdf5(raw_dir, dataset_dir, env_info):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "demo.hdf5"
    episodes = []
    for ep_dir in raw_dir.glob("ep_*"):
        episode = load_raw_episode(ep_dir)
        if episode is not None:
            episodes.append((ep_dir, episode))
    episodes.sort(
        key=lambda item: (
            item[1]["stats"]["target_grasp_angle_deg"],
            item[1]["stats"]["initial_state_sampling"]["slot"],
        )
    )

    label_rows = []
    with h5py.File(output_path, "w") as dataset:
        data_group = dataset.create_group("data")
        for index, (ep_dir, episode) in enumerate(episodes, start=1):
            demo_name = f"demo_{index}"
            demo = data_group.create_group(demo_name)
            demo.create_dataset("states", data=episode["states"])
            demo.create_dataset("actions", data=episode["actions"])
            demo.attrs["model_file"] = episode["model_xml"]
            angle = float(episode["stats"]["target_grasp_angle_deg"])
            sampling = episode["stats"]["initial_state_sampling"]
            label = observability_label(angle)
            demo.attrs["target_grasp_angle_deg"] = angle
            demo.attrs["observability"] = label
            demo.attrs["initial_state_id"] = sampling["state_id"]
            demo.attrs["retry_index"] = int(sampling["retry_index"])
            demo.attrs["raw_episode"] = ep_dir.name
            label_rows.append(
                {
                    "demo": demo_name,
                    "raw_episode": ep_dir.name,
                    "target_grasp_angle_deg": f"{angle:g}",
                    "observability": label,
                    "initial_state_id": sampling["state_id"],
                    "retry_index": int(sampling["retry_index"]),
                }
            )

        now = datetime.datetime.now()
        data_group.attrs["date"] = now.strftime("%m-%d-%Y")
        data_group.attrs["time"] = now.strftime("%H:%M:%S")
        data_group.attrs["repository_version"] = suite.__version__
        data_group.attrs["env"] = env_info["env_name"]
        data_group.attrs["env_info"] = json.dumps(env_info)
        data_group.attrs["num_demos"] = len(episodes)

    labels_path = dataset_dir / "observability_labels.csv"
    with labels_path.open("w", newline="") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=(
                "demo",
                "raw_episode",
                "target_grasp_angle_deg",
                "observability",
                "initial_state_id",
                "retry_index",
            ),
        )
        writer.writeheader()
        writer.writerows(label_rows)
    return output_path, len(episodes), labels_path


def main():
    args = parse_args()
    if args.rollouts_per_angle <= 0:
        raise ValueError("--rollouts-per-angle must be positive")
    if args.initial_states_per_angle <= 0:
        raise ValueError("--initial-states-per-angle must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")

    raw_dir = args.output / "raw"
    dataset_dir = args.output / "dataset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "collection_summary.json"
    attempt_log_path = args.output / "attempt_log.jsonl"

    controller_config = make_controller_config("Panda", "osc_pose")
    env_info = {
        "env_name": args.environment,
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    sampler_env = make_env(args.environment, controller_config, args.seed, args.horizon)
    collection_base_env = make_env(args.environment, controller_config, args.seed + 1, args.horizon)
    collection_env = DataCollectionWrapper(
        collection_base_env,
        str(raw_dir),
        collect_freq=1,
        flush_freq=args.horizon + 1,
    )
    policy = ThreadingScriptedPolicy(
        rng=np.random.RandomState(args.seed + 2),
        action_noise_std=args.action_noise_std,
        grasp_angle_range=(min(args.angles), max(args.angles)),
        control_mode="osc_pose",
        collision_aware_threading=args.collision_aware_threading,
    )

    summary = {
        "status": "in_progress",
        "environment": args.environment,
        "controller": "OSC_POSE",
        "success_criterion": "policy composite success AND final env._check_success()",
        "angles_deg": [float(angle) for angle in args.angles],
        "rollouts_per_angle": int(args.rollouts_per_angle),
        "initial_states_sampled_up_front_per_angle": int(args.initial_states_per_angle),
        "max_retries_per_initial_state": int(args.max_retries),
        "seed": int(args.seed),
        "action_noise_std": float(args.action_noise_std),
        "failed_simulator_states_saved": False,
        "angles": {
            f"{float(angle):g}": empty_angle_summary(
                angle,
                args.rollouts_per_angle,
                args.initial_states_per_angle,
            )
            for angle in args.angles
        },
        "totals": {},
    }
    write_summary(summary_path, summary)

    state_serial = 0
    try:
        for angle in args.angles:
            angle_key = f"{float(angle):g}"
            angle_summary = summary["angles"][angle_key]
            failure_reasons = Counter()
            aborted_stages = Counter()
            retry_histogram = Counter()

            pending = deque()
            next_slot = 0
            for slot in range(args.initial_states_per_angle):
                pending.append(sample_initial_state(sampler_env, state_serial, slot, 0))
                state_serial += 1
                next_slot += 1

            while angle_summary["successful_trajectories"] < args.rollouts_per_angle:
                if not pending:
                    pending.append(sample_initial_state(sampler_env, state_serial, next_slot, 0))
                    state_serial += 1
                    next_slot += 1
                    angle_summary["initial_states_sampled"] += 1

                initial_state = pending.popleft()
                state_succeeded = False
                for retry_index in range(1, args.max_retries + 1):
                    policy.rollout(
                        collection_env,
                        target_grasp_angle=float(angle),
                        post_reset_callback=lambda env, state=initial_state: restore_initial_state(
                            env, state
                        ),
                    )
                    stats = policy.stats[-1]
                    final_env_success = bool(collection_env._check_success())
                    composite_success = bool(stats.get("policy_success", False))
                    accepted_success = bool(composite_success and final_env_success)
                    final_debug = dict(
                        getattr(collection_env.unwrapped, "_threading_success_debug", {})
                    )

                    angle_summary["rollout_attempts"] += 1
                    stats["env_check_success_final"] = final_env_success
                    stats["env_success_debug_final"] = final_debug
                    stats["collection_success"] = accepted_success
                    stats["collection_success_source"] = (
                        "policy_composite_and_final_env_check_success"
                        if accepted_success
                        else "failed_composite_or_final_env_check"
                    )
                    stats["episode_kept"] = accepted_success
                    stats["counted_toward_target"] = accepted_success
                    stats["initial_state_sampling"] = {
                        "state_id": initial_state["state_id"],
                        "serial": initial_state["serial"],
                        "slot": initial_state["slot"],
                        "replacement_index": initial_state["replacement_index"],
                        "retry_index": retry_index,
                        "max_retries": args.max_retries,
                    }

                    ep_dir = finalize_episode(
                        collection_env,
                        success=accepted_success,
                        cleanup_failed=True,
                        stats=stats,
                    )
                    failure_reason = stats.get("failure_reason", "unknown")
                    aborted_stage = stats.get("aborted_stage")
                    if not accepted_success:
                        failure_reasons[failure_reason] += 1
                        if aborted_stage:
                            aborted_stages[aborted_stage] += 1

                    attempt_record = {
                        "target_grasp_angle_deg": float(angle),
                        "state_id": initial_state["state_id"],
                        "slot": initial_state["slot"],
                        "replacement_index": initial_state["replacement_index"],
                        "retry_index": retry_index,
                        "env_check_success_final": final_env_success,
                        "policy_success": composite_success,
                        "accepted_success": accepted_success,
                        "failure_reason": failure_reason,
                        "aborted_stage": aborted_stage,
                        "steps": stats.get("steps"),
                        "actual_close_angle_deg": stats.get("actual_close_angle_deg"),
                        "actual_insert_angle_deg": stats.get("actual_insert_angle_deg"),
                        "env_success_debug_final": final_debug,
                        "saved_episode": Path(ep_dir).name if accepted_success and ep_dir else None,
                    }
                    append_attempt(attempt_log_path, attempt_record)

                    if accepted_success:
                        state_succeeded = True
                        angle_summary["successful_trajectories"] += 1
                        angle_summary["successful_initial_states"] += 1
                        retry_histogram[str(retry_index)] += 1
                        break

                    angle_summary["failure_reasons"] = dict(failure_reasons)
                    angle_summary["aborted_stages"] = dict(aborted_stages)
                    write_summary(summary_path, summary)

                if not state_succeeded:
                    angle_summary["initial_states_replaced"] += 1
                    replacement = sample_initial_state(
                        sampler_env,
                        state_serial,
                        initial_state["slot"],
                        initial_state["replacement_index"] + 1,
                    )
                    state_serial += 1
                    angle_summary["initial_states_sampled"] += 1
                    pending.append(replacement)

                angle_summary["retry_histogram"] = dict(retry_histogram)
                angle_summary["failure_reasons"] = dict(failure_reasons)
                angle_summary["aborted_stages"] = dict(aborted_stages)
                write_summary(summary_path, summary)
                print(
                    "angle={} successes={}/{} attempts={} replacements={} state={} result={}".format(
                        angle_key,
                        angle_summary["successful_trajectories"],
                        args.rollouts_per_angle,
                        angle_summary["rollout_attempts"],
                        angle_summary["initial_states_replaced"],
                        initial_state["state_id"],
                        "success" if state_succeeded else "replaced",
                    ),
                    flush=True,
                )
    finally:
        collection_env.close()
        sampler_env.close()

    hdf5_path, hdf5_count, labels_path = write_hdf5(raw_dir, dataset_dir, env_info)
    expected_count = len(args.angles) * args.rollouts_per_angle
    if hdf5_count != expected_count:
        raise RuntimeError(f"HDF5 contains {hdf5_count} demos; expected {expected_count}")
    summary["status"] = "complete"
    summary["hdf5_path"] = str(hdf5_path)
    summary["hdf5_demonstrations"] = int(hdf5_count)
    summary["observability_labels_path"] = str(labels_path)
    summary["observability_label_rules"] = {
        "partial": [86.0, 90.0],
        "full": [93.0, 97.0],
    }
    write_summary(summary_path, summary)
    print(f"dataset={hdf5_path} demos={hdf5_count}", flush=True)


if __name__ == "__main__":
    main()
