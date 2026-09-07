#!/usr/bin/env python3
"""Episode-held-out correlation audit for commanded and executed joint deltas."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


def ordered(group):
    return sorted(group, key=lambda x: int(x.rsplit("_", 1)[-1]))


def arrays(demo, real):
    action = np.asarray(demo["actions"][:, :7], dtype=np.float64)
    if real:
        indices = np.asarray(demo["source_step_index"][:], dtype=np.int64)
        source = Path(str(demo.attrs["source_episode"])) / "trajectory.h5"
        with h5py.File(source, "r") as raw:
            q = np.asarray(raw["observation/robot_state/joint_positions"][indices], dtype=np.float64)
        return action, q
    if "robot0_joint_pos" in demo:
        return action, np.asarray(demo["robot0_joint_pos"][:], dtype=np.float64)
    if "obs" in demo and "robot0_joint_pos" in demo["obs"]:
        return action, np.asarray(demo["obs/robot0_joint_pos"][:], dtype=np.float64)
    xml = demo.attrs.get("model_file", demo.attrs.get("model_xml", ""))
    joints = list(ET.fromstring(xml).find("worldbody").iter("joint"))
    assert [x.get("name") for x in joints[:7]] == [f"robot0_joint{i}" for i in range(1, 8)]
    return action, np.asarray(demo["states"][:, 1:8], dtype=np.float64)


def ridge_r2(episodes, feature, target):
    values = []
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(len(episodes))
        train = set(order[: int(0.8 * len(episodes))])
        xx = [[], []]
        yy = [[], []]
        for index, (action, q) in enumerate(episodes):
            command_delta = action - q
            executed_delta = q[1:] - q[:-1]
            options = {
                "q_to_command_delta": (q, command_delta),
                "q_to_executed_delta": (q[:-1], executed_delta),
                "previous_executed_delta_to_next": (executed_delta[:-1], executed_delta[1:]),
                "executed_delta_to_command_delta": (executed_delta, command_delta[:-1]),
            }
            x, y = options[feature]
            side = 0 if index in train else 1
            xx[side].append(x)
            yy[side].append(y)
        x_train, x_test = map(np.concatenate, xx)
        y_train, y_test = map(np.concatenate, yy)
        mean, scale = x_train.mean(0), x_train.std(0)
        scale[scale < 1e-8] = 1
        x_train = np.column_stack([(x_train - mean) / scale, np.ones(len(x_train))])
        x_test = np.column_stack([(x_test - mean) / scale, np.ones(len(x_test))])
        y_mean, y_scale = y_train.mean(0), y_train.std(0)
        y_scale[y_scale < 1e-8] = 1
        weights = np.linalg.solve(
            x_train.T @ x_train + np.eye(x_train.shape[1]),
            x_train.T @ ((y_train - y_mean) / y_scale),
        )
        prediction = (x_test @ weights) * y_scale + y_mean
        per_dim = 1 - ((prediction - y_test) ** 2).sum(0) / np.maximum(
            ((y_test - y_test.mean(0)) ** 2).sum(0), 1e-12
        )
        values.append(float(per_dim.mean()))
    return {"mean": float(np.mean(values)), "seeds": values}


def audit(path, real):
    with h5py.File(path, "r") as dataset:
        episodes = [arrays(dataset["data"][name], real) for name in ordered(dataset["data"])]
    metrics = {}
    for key in (
        "q_to_command_delta",
        "q_to_executed_delta",
        "previous_executed_delta_to_next",
        "executed_delta_to_command_delta",
    ):
        metrics[key] = ridge_r2(episodes, key, key)
    return {"path": path, "episodes": len(episodes), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="name:path:real")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {}
    for spec in args.dataset:
        name, path, real = spec.split(":", 2)
        result[name] = audit(path, bool(int(real)))
        print(name, json.dumps(result[name]["metrics"]), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
