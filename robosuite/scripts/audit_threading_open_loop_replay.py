"""Audit raw Threading demonstrations with deterministic open-loop action replay.

For each episode, this script loads the recorded MuJoCo XML and initial state
once, then advances the simulator only through ``env.step(action)``. It never
writes recorded states back into the simulator after the initial reset.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite


def make_joint_position_controller_config(robot):
    """Reproduce the absolute joint-position controller used for collection."""
    config = suite.load_composite_controller_config(robot=robot)
    arm_names = [
        name
        for name, part in config["body_parts"].items()
        if part.get("type", "").startswith("OSC")
    ]
    if len(arm_names) != 1:
        raise ValueError(f"Expected one OSC arm to replace, found {arm_names}")

    arm_name = arm_names[0]
    gripper_config = config["body_parts"][arm_name].get("gripper", {"type": "GRIP"})
    config["body_parts"][arm_name] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 100,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper_config,
    }
    return config


def load_episode_arrays(episode):
    state_files = sorted(episode.glob("state_*.npz"))
    if len(state_files) != 1:
        raise ValueError(f"{episode}: expected exactly one state NPZ, found {len(state_files)}")

    data = np.load(state_files[0], allow_pickle=True)
    states = np.asarray(data["states"])
    actions = np.asarray([item["actions"] for item in data["action_infos"]])
    if len(states) != len(actions) + 1:
        raise ValueError(
            f"{episode}: expected one more state than action, got "
            f"{len(states)} states and {len(actions)} actions"
        )
    return states, actions


def make_environment(environment, robot):
    return suite.make(
        environment,
        robots=[robot],
        controller_configs=make_joint_position_controller_config(robot),
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=1000,
        control_freq=20,
    )


def audit_episode(env, episode):
    ep_meta_path = episode / "ep_meta.json"
    ep_meta = json.loads(ep_meta_path.read_text()) if ep_meta_path.exists() else {}
    states, actions = load_episode_arrays(episode)

    env.unset_ep_meta()
    env.set_ep_meta(ep_meta)
    env.reset_from_xml_string((episode / "model.xml").read_text())
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    # These values are trajectory-level success state, so initialize them once
    # before action replay and let env.step update them naturally.
    env._threading_max_insert_progress = -np.inf
    env._threading_initial_tripod_pos = None

    any_success = False
    final_success = False
    exact_state_match = True
    max_state_error = 0.0
    first_divergent_step = None

    for step, action in enumerate(actions):
        env.step(action)
        final_success = bool(env._check_success())
        any_success = any_success or final_success

        replayed_state = env.sim.get_state().flatten()
        recorded_state = states[step + 1]
        if not np.array_equal(replayed_state, recorded_state):
            exact_state_match = False
            state_error = float(np.linalg.norm(replayed_state - recorded_state))
            max_state_error = max(max_state_error, state_error)
            if first_divergent_step is None:
                first_divergent_step = step

    return {
        "episode": episode.name,
        "actions": len(actions),
        "any_success": any_success,
        "final_success": final_success,
        "exact_state_match": exact_state_match,
        "max_state_error": max_state_error,
        "first_divergent_step": first_divergent_step,
    }


def audit_chunk(payload):
    episode_paths, environment, robot = payload
    env = make_environment(environment, robot)
    try:
        return [audit_episode(env, Path(path)) for path in episode_paths]
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--environment", default="Threading_D05")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    episodes = sorted(
        path
        for path in args.dataset.resolve().iterdir()
        if path.is_dir() and path.name.startswith("ep_")
    )
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise ValueError(f"No episode directories found in {args.dataset}")

    worker_count = min(args.workers, len(episodes))
    chunks = [
        [str(path) for path in episodes[index::worker_count]]
        for index in range(worker_count)
    ]
    payloads = [(chunk, args.environment, args.robot) for chunk in chunks if chunk]

    results = []
    if worker_count == 1:
        results = audit_chunk(payloads[0])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(audit_chunk, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), 1):
                chunk_results = future.result()
                results.extend(chunk_results)
                print(
                    f"completed_chunks={completed}/{len(futures)} "
                    f"episodes={len(results)}/{len(episodes)}",
                    flush=True,
                )

    results.sort(key=lambda result: result["episode"])
    summary = {
        "dataset": str(args.dataset.resolve()),
        "environment": args.environment,
        "robot": args.robot,
        "controller": {
            "type": "JOINT_POSITION",
            "input_type": "absolute",
            "control_freq": 20,
            "kp": 100,
            "damping_ratio": 1,
            "output_range": [-0.05, 0.05],
            "interpolation": None,
        },
        "episodes": len(results),
        "any_success": sum(result["any_success"] for result in results),
        "final_success": sum(result["final_success"] for result in results),
        "exact_state_match": sum(result["exact_state_match"] for result in results),
        "diverged": sum(not result["exact_state_match"] for result in results),
    }
    report = {"summary": summary, "episodes": results}
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    failures = [
        result
        for result in results
        if not (
            result["any_success"]
            and result["final_success"]
            and result["exact_state_match"]
        )
    ]
    for failure in failures:
        print("FAIL " + json.dumps(failure, sort_keys=True))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
