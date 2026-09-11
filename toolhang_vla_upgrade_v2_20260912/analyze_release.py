#!/usr/bin/env python3
"""Small release audit for the paired ToolHang observability dataset."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def decoded(values):
    return {value.decode() if isinstance(value, bytes) else str(value) for value in values}


def grouped_linear_r2(features, targets, groups):
    unique = np.asarray(sorted(set(groups)))
    valid_groups = set(unique[::5])
    train = np.asarray([group not in valid_groups for group in groups])
    valid = ~train
    design = np.column_stack([features[train], np.ones(train.sum())])
    valid_design = np.column_stack([features[valid], np.ones(valid.sum())])
    weights = np.linalg.lstsq(design, targets[train], rcond=None)[0]
    prediction = valid_design @ weights
    residual = np.sum((targets[valid] - prediction) ** 2)
    variance = np.sum((targets[valid] - targets[valid].mean(axis=0)) ** 2)
    return float(1.0 - residual / variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    manifest = json.loads((args.root / "manifest.json").read_text())
    selected = set(summary["selected_pairs"])
    entries = [entry for entry in manifest["entries"] if entry["pair"] in selected]

    features, absolute, delta, groups = [], [], [], []
    pair_demos = {}
    with h5py.File(args.root / "dataset_state.hdf5", "r") as dataset:
        train = decoded(dataset["mask/train"][:])
        valid = decoded(dataset["mask/valid"][:])
        for name, demo in dataset["data"].items():
            pair = int(demo.attrs["pair_id"])
            pair_demos.setdefault(pair, []).append(name)
            qpos = demo["robot0_joint_pos"][:, :7]
            action = demo["actions"][:, :7]
            features.append(qpos)
            absolute.append(action)
            delta.append(action - qpos)
            groups.extend([pair] * len(qpos))
        pair_split_isolation = all(
            len(names) == 2 and (set(names) <= train or set(names) <= valid)
            for names in pair_demos.values()
        )
        paired_initial_states = all(
            np.array_equal(
                dataset["data"][names[0]]["states"][0],
                dataset["data"][names[1]]["states"][0],
            )
            for names in pair_demos.values()
        )

    features = np.vstack(features)
    absolute = np.vstack(absolute)
    delta = np.vstack(delta)
    groups = np.asarray(groups)
    translations = np.asarray(
        [entry["initial"]["fixture_translation_m"] for entry in entries]
    )
    yaws = np.rad2deg(
        [entry["initial"]["fixture_yaw_rad"] for entry in entries]
    )
    result = {
        "passed": bool(pair_split_isolation and paired_initial_states),
        "pairs": len(pair_demos),
        "episodes": 2 * len(pair_demos),
        "pair_split_isolation": pair_split_isolation,
        "paired_initial_states_exact": paired_initial_states,
        "fixture_selected_range": {
            "x_m": [float(translations[:, 0].min()), float(translations[:, 0].max())],
            "y_m": [float(translations[:, 1].min()), float(translations[:, 1].max())],
            "yaw_deg": [float(yaws.min()), float(yaws.max())],
        },
        "episode_grouped_linear_r2": {
            "absolute_action_from_current_joint": grouped_linear_r2(
                features, absolute, groups
            ),
            "delta_action_from_current_joint": grouped_linear_r2(features, delta, groups),
        },
    }
    (args.root / "release_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
