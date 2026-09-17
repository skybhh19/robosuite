#!/usr/bin/env python3
"""Assemble accepted YCBForkInRack raw episodes into a Robomimic HDF5."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

import robosuite as suite
from robosuite.scripts.collect_ycb_fork_joint import make_joint_position_config


REGIMES = (("full_visible", "full"), ("partial_hidden", "partial"))


def load_rows(root, source_regime):
    path = root / "raw" / source_regime / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 150 or not all(row["accepted"] for row in rows):
        raise RuntimeError(f"{source_regime}: expected 150 accepted rows, got {len(rows)}")
    return rows


def load_episode(path):
    fragments = sorted(path.glob("state_*.npz"))
    if len(fragments) != 1:
        raise RuntimeError(f"{path}: expected one state fragment, got {len(fragments)}")
    raw = np.load(fragments[0], allow_pickle=True)
    states = np.asarray(raw["states"], dtype=np.float64)
    infos = list(raw["action_infos"])
    if len(states) != len(infos) + 1:
        raise RuntimeError(f"{path}: states/actions mismatch {len(states)} != {len(infos)} + 1")
    actions = np.asarray(
        [info.get("actions_absolute_joint_position", info["actions"]) for info in infos],
        dtype=np.float64,
    )
    qpos = np.asarray([info["robot0_joint_pos"] for info in infos], dtype=np.float64)
    deltas = np.asarray([info["actions_joint_delta"] for info in infos], dtype=np.float64)
    if actions.shape != (len(infos), 8) or qpos.shape != (len(infos), 7):
        raise RuntimeError(f"{path}: invalid action or joint-position shape")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise RuntimeError(f"{path}: non-finite values")
    if np.max(np.abs(actions[:, :7] - qpos - deltas[:, :7])) >= 1e-10:
        raise RuntimeError(f"{path}: joint delta identity failed")
    return states, actions, qpos, deltas


def env_args():
    config = make_joint_position_config("Panda")
    kwargs = dict(
        robots=["Panda"], controller_configs=config, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
        hard_reset=False, reward_shaping=True, initialization_noise=None,
        control_freq=20, horizon=3000, ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env = suite.make("YCBForkInRack", **kwargs)
    try:
        return {"env_name": "YCBForkInRack", "type": 1, "env_kwargs": kwargs}
    finally:
        env.close()


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=202609173)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = []
    for source_regime, label in REGIMES:
        rows.extend((source_regime, label, index, row) for index, row in enumerate(load_rows(args.root, source_regime)))
    valid = set()
    rng = np.random.default_rng(args.split_seed)
    for _, label in REGIMES:
        valid.update((label, int(index)) for index in rng.permutation(150)[:30])

    masks = {key: [] for key in ("train", "valid", "all", "full", "partial", "fully_observable", "partially_observable")}
    temporary = args.output.with_suffix(".partial.hdf5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with h5py.File(temporary, "w") as output:
        data = output.create_group("data")
        metadata = env_args()
        # JSON serialization handles this controller dictionary without loss.
        data.attrs["env_args"] = json.dumps(metadata)
        for source_regime, label, source_index, row in rows:
            episode_path = Path(row["episode_directory"])
            states, actions, qpos, deltas = load_episode(episode_path)
            name = f"demo_{len(data)}"
            demo = data.create_group(name)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("actions_joint_delta", data=deltas)
            demo.create_dataset("states", data=states[:-1])
            demo.create_dataset("next_states", data=states[1:])
            demo.create_dataset("robot0_joint_pos", data=qpos)
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=qpos)
            rewards = np.zeros(len(actions), dtype=np.float32); rewards[-1] = 1.0
            dones = np.zeros(len(actions), dtype=np.int64); dones[-1] = 1
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            demo.attrs["model_file"] = (episode_path / "model.xml").read_text()
            demo.attrs["ep_meta"] = (episode_path / "ep_meta.json").read_text()
            demo.attrs["num_samples"] = len(actions)
            demo.attrs["control_hz"] = 20
            demo.attrs["observability"] = label
            demo.attrs["source_regime"] = source_regime
            demo.attrs["source_index"] = source_index
            demo.attrs["pair_id"] = source_index + (0 if label == "full" else 150)
            demo.attrs["pairing_protocol"] = "independent_initial_states"
            demo.attrs["action_representation"] = "absolute_joint_position_plus_gripper"
            demo.attrs["variation"] = json.dumps(row["variation"], sort_keys=True)
            masks["all"].append(name)
            masks[label].append(name)
            masks["fully_observable" if label == "full" else "partially_observable"].append(name)
            masks["valid" if (label, source_index) in valid else "train"].append(name)
            total += len(actions)
        data.attrs["total"] = total
        mask = output.create_group("mask")
        for key, names in masks.items():
            mask.create_dataset(key, data=np.asarray(names, dtype="S"))
        output.attrs["description"] = "Independent YCB fork production dataset: 150 Full + 150 Partial."
        output.attrs["pairing_protocol"] = "independent_initial_states"
        output.attrs["split_seed"] = args.split_seed
    temporary.replace(args.output)
    report = {
        "episodes": 300, "full": 150, "partial": 150, "train": 240,
        "valid": 60, "steps_20hz": total, "sha256": digest(args.output),
    }
    (args.output.parent / "state_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
