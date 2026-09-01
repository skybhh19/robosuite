"""Collect Threading demos with bounded retries per sampled initial state.

Each requested grasp angle starts with an independently sampled pool of initial
states. A state is retried with policy randomness up to ``--max-retries`` times.
The acceptance gate is configurable. In strict environment / joint-margin
mode, only trajectories for which the final environment ``_check_success()``
is true and every measured robot joint remains safely inside its physical
limit are retained. Exhausted states are replaced until the requested number
of successful trajectories is reached.
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
    JOINT_ACTION_NOISE_SCALE_RAD,
    JOINT_TRAJECTORY_TIMING_PROFILE,
    MAX_LIFT_HEIGHT,
    OSC_TRAJECTORY_TIMING_PROFILE,
    ThreadingScriptedPolicy,
    finalize_episode,
    json_safe,
    make_controller_config,
)
from robosuite.wrappers import DataCollectionWrapper


DEFAULT_OUTPUT = REPO_ROOT / "threading_d05_bc_88_96_15_per_angle_balanced_initial_states"
DEFAULT_PARTIAL_ANGLES = (86.0, 87.0, 88.0, 89.0, 90.0)
DEFAULT_FULL_ANGLES = (93.0, 94.0, 95.0, 96.0, 97.0)

JOINT_ACTION_INFO_FIELDS = (
    "robot0_joint_pos",
    "joint_position",
    "absolute_joint_target",
    "joint_delta",
    "joint_delta_scale",
    "joint_delta_reference_scaled",
    "joint_delta_exceeds_reference_scale",
    "actions_absolute_joint_position",
    "actions_joint_delta",
)

PANDA_JOINT_MIN = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=float,
)
PANDA_JOINT_MAX = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=float,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--environment", type=str, default="Threading_D05")
    parser.add_argument("--angles", nargs="+", type=float, default=list(range(88, 97)))
    parser.add_argument(
        "--partial-angles",
        nargs="+",
        type=float,
        default=list(DEFAULT_PARTIAL_ANGLES),
    )
    parser.add_argument(
        "--full-angles",
        nargs="+",
        type=float,
        default=list(DEFAULT_FULL_ANGLES),
    )
    parser.add_argument("--rollouts-per-angle", type=int, default=15)
    parser.add_argument("--initial-states-per-angle", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted in-progress collection in --output.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("osc_pose", "joint_position"),
        default="osc_pose",
        help="Use OSC delta-pose actions or damped IK with absolute joint-position actions.",
    )
    parser.add_argument(
        "--record-joint-training-fields",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record absolute and delta joint labels. Defaults to enabled in joint-position mode.",
    )
    parser.add_argument("--joint-delta-scale", type=float, default=0.05)
    parser.add_argument(
        "--success-criterion",
        choices=("policy_composite_and_env", "final_env_and_joint_margin"),
        default="policy_composite_and_env",
        help=(
            "Retain the legacy composite+environment successes, or use only "
            "the final environment success check plus a measured joint-margin gate."
        ),
    )
    parser.add_argument(
        "--min-joint-margin-rad",
        type=float,
        default=0.05,
        help="Minimum actual Panda qpos distance from either physical limit.",
    )
    parser.add_argument(
        "--require-clean-pregrasp-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require pregrasp_contact_check.passed=true and detected=false in "
            "addition to the selected success criterion."
        ),
    )
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

    robot = wrapped_env.robots[0]
    robot.composite_controller.update_state()
    robot.composite_controller.reset()

    base_env = wrapped_env.unwrapped
    base_env._threading_initial_tripod_pos = None
    base_env._threading_max_insert_progress = -np.inf

    # DataCollectionWrapper starts the episode before this callback. Replace
    # its recorded initial state so replay begins from the sampled state.
    wrapped_env._current_task_instance_state = state.copy()
    wrapped_env.successful = False
    if wrapped_env.record_joint_position_fields:
        # The wrapper reads cached proprioception before each action. Refresh it
        # after the direct state restore so the first joint label matches the
        # first recorded simulator state.
        base_env._get_observations(force_update=True)


def measured_joint_safety(wrapped_env):
    """Measure the minimum actual Panda joint margin over the whole rollout."""
    robot = wrapped_env.robots[0]
    arm_name = robot.arms[0]
    controller = robot.composite_controller.part_controllers[arm_name]
    qpos_indexes = np.asarray(controller.qpos_index, dtype=int)

    state_arrays = []
    ep_dir = Path(wrapped_env.ep_directory) if wrapped_env.ep_directory else None
    if ep_dir is not None and ep_dir.is_dir():
        for state_path in sorted(ep_dir.glob("state_*.npz")):
            state_arrays.extend(np.load(state_path, allow_pickle=True)["states"])
    state_arrays.extend(wrapped_env.states)
    if not state_arrays:
        raise RuntimeError("Cannot measure joint margin: rollout recorded no simulator states")

    # MjSimState.flatten() stores time first, followed by qpos.
    states = np.asarray(state_arrays, dtype=float)
    qpos = states[:, 1 + qpos_indexes]
    margins = np.minimum(
        qpos - PANDA_JOINT_MIN[None, :],
        PANDA_JOINT_MAX[None, :] - qpos,
    )
    flat_index = int(np.argmin(margins))
    step_index, joint_index = np.unravel_index(flat_index, margins.shape)
    return {
        "minimum_margin_rad": float(margins[step_index, joint_index]),
        "minimum_margin_joint": int(joint_index + 1),
        "minimum_margin_state_index": int(step_index),
        "minimum_margin_by_joint_rad": np.min(margins, axis=0).tolist(),
    }


def empty_angle_summary(angle, target_successes, initial_count):
    return {
        "target_grasp_angle_deg": float(angle),
        "target_successes": int(target_successes),
        "initial_states_sampled_up_front": int(initial_count),
        "successful_trajectories": 0,
        "rollout_attempts": 0,
        "retry_attempt_success_rate": 0.0,
        "final_env_success_attempts": 0,
        "final_env_success_attempt_rate": 0.0,
        "joint_margin_pass_attempts": 0,
        "joint_margin_fail_attempts": 0,
        "pregrasp_contact_pass_attempts": 0,
        "pregrasp_contact_rejection_attempts": 0,
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
        item["final_env_success_attempt_rate"] = (
            float(item.get("final_env_success_attempts", 0) / attempts) if attempts else 0.0
        )


def write_summary(path, summary):
    update_rates(summary)
    with path.open("w") as output:
        json.dump(json_safe(summary), output, indent=2, allow_nan=False)


def append_attempt(path, record):
    with path.open("a") as output:
        output.write(json.dumps(json_safe(record), allow_nan=False) + "\n")


def load_attempts(path):
    if not path.exists():
        return []
    with path.open() as attempt_file:
        return [json.loads(line) for line in attempt_file if line.strip()]


def validate_resume_summary(summary, args, record_joint_training_fields):
    expected = {
        "environment": args.environment,
        "control_mode": args.control_mode,
        "angles_deg": [float(angle) for angle in args.angles],
        "partial_observability_angles_deg": [float(angle) for angle in args.partial_angles],
        "full_observability_angles_deg": [float(angle) for angle in args.full_angles],
        "rollouts_per_angle": int(args.rollouts_per_angle),
        "initial_states_sampled_up_front_per_angle": int(args.initial_states_per_angle),
        "max_retries_per_initial_state": int(args.max_retries),
        "record_joint_training_fields": bool(record_joint_training_fields),
        "success_criterion_mode": args.success_criterion,
        "minimum_joint_margin_rad": (
            float(args.min_joint_margin_rad)
            if args.success_criterion == "final_env_and_joint_margin"
            else None
        ),
        "require_clean_pregrasp_contact": bool(args.require_clean_pregrasp_contact),
    }
    if summary.get("status") != "in_progress":
        raise ValueError("--resume requires an in-progress collection_summary.json")
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume arguments do not match collection summary: {mismatches}")


def resume_sampler_position(summary, attempts, initial_states_per_angle):
    attempted_angles = {
        f"{float(record['target_grasp_angle_deg']):g}" for record in attempts
    }
    sampled = len(attempted_angles) * int(initial_states_per_angle)
    for angle_key in attempted_angles:
        angle_summary = summary["angles"][angle_key]
        sampled += int(angle_summary.get("initial_states_replaced", 0))
        attempted_slots = [
            int(record["slot"])
            for record in attempts
            if f"{float(record['target_grasp_angle_deg']):g}" == angle_key
        ]
        sampled += max(0, max(attempted_slots, default=-1) + 1 - initial_states_per_angle)
    return sampled


def load_raw_episode(ep_dir):
    states = []
    actions = []
    action_fields = {field: [] for field in JOINT_ACTION_INFO_FIELDS}
    successful = False
    env_name = None
    for state_path in sorted(ep_dir.glob("state_*.npz")):
        data = np.load(state_path, allow_pickle=True)
        env_name = str(data["env"])
        states.extend(data["states"])
        for info in data["action_infos"]:
            actions.append(info["actions"])
            for field in JOINT_ACTION_INFO_FIELDS:
                if field in info:
                    action_fields[field].append(info[field])
        successful = bool(successful or data["successful"])
    if not successful or not states:
        return None
    if len(states) != len(actions) + 1:
        raise ValueError(
            f"Episode {ep_dir.name} has {len(states)} states for {len(actions)} actions"
        )
    populated_action_fields = {}
    for field, values in action_fields.items():
        if values and len(values) != len(actions):
            raise ValueError(
                f"Episode {ep_dir.name} has {len(values)} {field} values for {len(actions)} actions"
            )
        if values:
            populated_action_fields[field] = np.asarray(values)
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
        "action_fields": populated_action_fields,
    }


def observability_label(
    angle,
    partial_angles=DEFAULT_PARTIAL_ANGLES,
    full_angles=DEFAULT_FULL_ANGLES,
):
    angle = float(angle)
    if angle in {float(value) for value in partial_angles}:
        return "partial"
    if angle in {float(value) for value in full_angles}:
        return "full"
    return "unlabeled"


def write_hdf5(raw_dir, dataset_dir, env_info, partial_angles, full_angles):
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
            for field, values in episode["action_fields"].items():
                demo.create_dataset(field, data=values)
            demo.attrs["model_file"] = episode["model_xml"]
            angle = float(episode["stats"]["target_grasp_angle_deg"])
            sampling = episode["stats"]["initial_state_sampling"]
            label = observability_label(angle, partial_angles, full_angles)
            demo.attrs["target_grasp_angle_deg"] = angle
            demo.attrs["observability"] = label
            demo.attrs["approach_path_profile"] = episode["stats"].get(
                "approach_path_profile", "unknown"
            )
            demo.attrs["grasp_offset_along_m"] = float(
                episode["stats"].get("grasp_offset_along", np.nan)
            )
            pregrasp_contact = episode["stats"].get("pregrasp_contact_check", {})
            demo.attrs["pregrasp_contact_passed"] = bool(pregrasp_contact.get("passed", False))
            demo.attrs["pregrasp_contact_detected"] = bool(pregrasp_contact.get("detected", True))
            approach_outcome = episode["stats"].get("stage_outcomes", {}).get("aim_continuous", {})
            demo.attrs["joint_approach_stream_waypoints"] = bool(
                approach_outcome.get("stream_waypoints", True)
            )
            demo.attrs["joint_approach_waypoint_position_tolerance_m"] = float(
                approach_outcome.get("waypoint_position_tolerance_m", np.nan)
            )
            demo.attrs["joint_approach_waypoint_orientation_tolerance_deg"] = float(
                approach_outcome.get("waypoint_orientation_tolerance_deg", np.nan)
            )
            demo.attrs["initial_state_id"] = sampling["state_id"]
            demo.attrs["retry_index"] = int(sampling["retry_index"])
            demo.attrs["raw_episode"] = ep_dir.name
            demo.attrs["action_representation"] = episode["stats"].get(
                "action_representation", "unknown"
            )
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
    if args.joint_delta_scale <= 0:
        raise ValueError("--joint-delta-scale must be positive")
    if args.min_joint_margin_rad <= 0:
        raise ValueError("--min-joint-margin-rad must be positive")
    if args.success_criterion == "final_env_and_joint_margin" and args.control_mode != "joint_position":
        raise ValueError("final_env_and_joint_margin requires --control-mode joint_position")
    partial_angles = {float(angle) for angle in args.partial_angles}
    full_angles = {float(angle) for angle in args.full_angles}
    if partial_angles & full_angles:
        raise ValueError("--partial-angles and --full-angles must not overlap")
    record_joint_training_fields = (
        args.control_mode == "joint_position"
        if args.record_joint_training_fields is None
        else args.record_joint_training_fields
    )
    if record_joint_training_fields and args.control_mode != "joint_position":
        raise ValueError("--record-joint-training-fields requires --control-mode joint_position")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")

    raw_dir = args.output / "raw"
    dataset_dir = args.output / "dataset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "collection_summary.json"
    attempt_log_path = args.output / "attempt_log.jsonl"

    controller_config = make_controller_config("Panda", args.control_mode)
    env_info = {
        "env_name": args.environment,
        "robots": ["Panda"],
        "controller_configs": controller_config,
        "control_mode": args.control_mode,
        "action_representation": (
            "absolute_joint_position" if args.control_mode == "joint_position" else "delta_eef_pose"
        ),
        "trajectory_timing_profile": (
            JOINT_TRAJECTORY_TIMING_PROFILE
            if args.control_mode == "joint_position"
            else OSC_TRAJECTORY_TIMING_PROFILE
        ),
        "joint_action_noise_scale_rad": (
            JOINT_ACTION_NOISE_SCALE_RAD if args.control_mode == "joint_position" else None
        ),
        "max_lift_height_m": MAX_LIFT_HEIGHT,
    }
    sampler_env = make_env(args.environment, controller_config, args.seed, args.horizon)
    collection_base_env = make_env(args.environment, controller_config, args.seed + 1, args.horizon)
    collection_env = DataCollectionWrapper(
        collection_base_env,
        str(raw_dir),
        collect_freq=1,
        flush_freq=args.horizon + 1,
        record_joint_position_fields=record_joint_training_fields,
        joint_delta_scale=args.joint_delta_scale,
    )
    policy = ThreadingScriptedPolicy(
        rng=np.random.RandomState(args.seed + 2),
        action_noise_std=args.action_noise_std,
        grasp_angle_range=(min(args.angles), max(args.angles)),
        control_mode=args.control_mode,
        collision_aware_threading=args.collision_aware_threading,
    )

    new_summary = {
        "status": "in_progress",
        "environment": args.environment,
        "controller": "JOINT_POSITION" if args.control_mode == "joint_position" else "OSC_POSE",
        "control_mode": args.control_mode,
        "action_representation": (
            "absolute_joint_position" if args.control_mode == "joint_position" else "delta_eef_pose"
        ),
        "trajectory_timing_profile": (
            JOINT_TRAJECTORY_TIMING_PROFILE
            if args.control_mode == "joint_position"
            else OSC_TRAJECTORY_TIMING_PROFILE
        ),
        "record_joint_training_fields": bool(record_joint_training_fields),
        "joint_delta_scale": float(args.joint_delta_scale) if record_joint_training_fields else None,
        "success_criterion": (
            "final env._check_success() AND minimum actual Panda joint margin"
            if args.success_criterion == "final_env_and_joint_margin"
            else "policy composite success AND final env._check_success()"
        ),
        "success_criterion_mode": args.success_criterion,
        "minimum_joint_margin_rad": (
            float(args.min_joint_margin_rad)
            if args.success_criterion == "final_env_and_joint_margin"
            else None
        ),
        "require_clean_pregrasp_contact": bool(args.require_clean_pregrasp_contact),
        "angles_deg": [float(angle) for angle in args.angles],
        "partial_observability_angles_deg": sorted(partial_angles),
        "full_observability_angles_deg": sorted(full_angles),
        "rollouts_per_angle": int(args.rollouts_per_angle),
        "initial_states_sampled_up_front_per_angle": int(args.initial_states_per_angle),
        "max_retries_per_initial_state": int(args.max_retries),
        "seed": int(args.seed),
        "action_noise_std": float(args.action_noise_std),
        "joint_action_noise_std_rad": (
            float(args.action_noise_std * JOINT_ACTION_NOISE_SCALE_RAD)
            if args.control_mode == "joint_position"
            else None
        ),
        "max_lift_height_m": MAX_LIFT_HEIGHT,
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
    if args.resume:
        if not summary_path.exists() or not attempt_log_path.exists() or not raw_dir.exists():
            raise FileNotFoundError(
                "--resume requires collection_summary.json, attempt_log.jsonl, and raw/"
            )
        with summary_path.open() as summary_file:
            summary = json.load(summary_file)
        validate_resume_summary(summary, args, record_joint_training_fields)
        summary["resume_count"] = int(summary.get("resume_count", 0)) + 1
        summary["resumed_after_interruption"] = True
        attempts = load_attempts(attempt_log_path)
        logged_successes = sum(bool(record.get("accepted_success")) for record in attempts)
        if logged_successes != summary["totals"]["successful_trajectories"]:
            raise ValueError(
                "Attempt log and summary disagree before resume: "
                f"{logged_successes} != {summary['totals']['successful_trajectories']}"
            )
    else:
        summary = new_summary
    write_summary(summary_path, summary)

    existing_attempts = load_attempts(attempt_log_path)
    if args.resume:
        state_serial = resume_sampler_position(
            summary, existing_attempts, args.initial_states_per_angle
        )
        for _ in range(state_serial):
            sampler_env.reset()
        summary["resume_sampler_fast_forward_resets"] = int(state_serial)
        write_summary(summary_path, summary)
    else:
        state_serial = 0
    try:
        for angle in args.angles:
            angle_key = f"{float(angle):g}"
            angle_summary = summary["angles"][angle_key]
            failure_reasons = Counter()
            aborted_stages = Counter()
            retry_histogram = Counter()
            failure_reasons.update(angle_summary.get("failure_reasons", {}))
            aborted_stages.update(angle_summary.get("aborted_stages", {}))
            retry_histogram.update(angle_summary.get("retry_histogram", {}))

            if angle_summary["successful_trajectories"] >= args.rollouts_per_angle:
                continue

            pending = deque()
            angle_attempts = [
                record
                for record in existing_attempts
                if float(record["target_grasp_angle_deg"]) == float(angle)
            ]
            next_slot = max((int(record["slot"]) for record in angle_attempts), default=-1) + 1
            remaining = args.rollouts_per_angle - angle_summary["successful_trajectories"]
            initial_pool_size = (
                remaining if angle_attempts else args.initial_states_per_angle
            )
            if angle_attempts:
                angle_summary["initial_states_resampled_after_resume"] = (
                    int(angle_summary.get("initial_states_resampled_after_resume", 0))
                    + initial_pool_size
                )
                angle_summary["initial_states_sampled"] += initial_pool_size
            for slot in range(next_slot, next_slot + initial_pool_size):
                pending.append(sample_initial_state(sampler_env, state_serial, slot, 0))
                state_serial += 1
            next_slot += initial_pool_size

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
                    joint_safety = measured_joint_safety(collection_env)
                    joint_margin_passed = bool(
                        joint_safety["minimum_margin_rad"] >= args.min_joint_margin_rad
                    )
                    pregrasp_contact = stats.get("pregrasp_contact_check", {})
                    clean_pregrasp_contact = bool(
                        pregrasp_contact.get("passed", False)
                        and not pregrasp_contact.get("detected", True)
                    )
                    if args.success_criterion == "final_env_and_joint_margin":
                        accepted_success = bool(
                            final_env_success
                            and joint_margin_passed
                            and (not args.require_clean_pregrasp_contact or clean_pregrasp_contact)
                        )
                    else:
                        accepted_success = bool(composite_success and final_env_success)
                    final_debug = dict(
                        getattr(collection_env.unwrapped, "_threading_success_debug", {})
                    )

                    angle_summary["rollout_attempts"] += 1
                    angle_summary["final_env_success_attempts"] = int(
                        angle_summary.get("final_env_success_attempts", 0)
                    ) + int(final_env_success)
                    angle_summary["joint_margin_pass_attempts"] = int(
                        angle_summary.get("joint_margin_pass_attempts", 0)
                    ) + int(joint_margin_passed)
                    angle_summary["joint_margin_fail_attempts"] = int(
                        angle_summary.get("joint_margin_fail_attempts", 0)
                    ) + int(not joint_margin_passed)
                    angle_summary["pregrasp_contact_pass_attempts"] = int(
                        angle_summary.get("pregrasp_contact_pass_attempts", 0)
                    ) + int(clean_pregrasp_contact)
                    angle_summary["pregrasp_contact_rejection_attempts"] = int(
                        angle_summary.get("pregrasp_contact_rejection_attempts", 0)
                    ) + int(not clean_pregrasp_contact)
                    stats["env_check_success_final"] = final_env_success
                    stats["env_success_debug_final"] = final_debug
                    stats["joint_safety"] = {
                        **joint_safety,
                        "required_margin_rad": float(args.min_joint_margin_rad),
                        "passed": joint_margin_passed,
                    }
                    stats["collection_success"] = accepted_success
                    stats["collection_success_source"] = (
                        (
                            "final_env_check_success_joint_margin_and_clean_pregrasp"
                            if args.require_clean_pregrasp_contact
                            else "final_env_check_success_and_joint_margin"
                        )
                        if args.success_criterion == "final_env_and_joint_margin"
                        else "policy_composite_and_final_env_check_success"
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
                        "joint_margin_rad": joint_safety["minimum_margin_rad"],
                        "joint_margin_joint": joint_safety["minimum_margin_joint"],
                        "joint_margin_state_index": joint_safety["minimum_margin_state_index"],
                        "joint_margin_passed": joint_margin_passed,
                        "pregrasp_contact_passed": bool(pregrasp_contact.get("passed", False)),
                        "pregrasp_contact_detected": bool(pregrasp_contact.get("detected", True)),
                        "clean_pregrasp_contact": clean_pregrasp_contact,
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

    hdf5_path, hdf5_count, labels_path = write_hdf5(
        raw_dir,
        dataset_dir,
        env_info,
        partial_angles,
        full_angles,
    )
    expected_count = len(args.angles) * args.rollouts_per_angle
    if hdf5_count != expected_count:
        raise RuntimeError(f"HDF5 contains {hdf5_count} demos; expected {expected_count}")
    summary["status"] = "complete"
    summary["hdf5_path"] = str(hdf5_path)
    summary["hdf5_demonstrations"] = int(hdf5_count)
    summary["observability_labels_path"] = str(labels_path)
    summary["observability_label_rules"] = {
        "partial": sorted(partial_angles),
        "full": sorted(full_angles),
    }
    write_summary(summary_path, summary)
    print(f"dataset={hdf5_path} demos={hdf5_count}", flush=True)


if __name__ == "__main__":
    main()
