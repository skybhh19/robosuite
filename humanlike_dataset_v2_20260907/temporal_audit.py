#!/usr/bin/env python3
"""Compare absolute-joint command dynamics with raw DROID demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import xml.etree.ElementTree as ET


def ordered(group):
    return sorted(group, key=lambda x: int(x.rsplit("_", 1)[-1]))


def episode_arrays(demo, real):
    action = np.asarray(demo["actions"][:, :7], dtype=np.float64)
    if not real:
        if "robot0_joint_pos" in demo:
            state = np.asarray(demo["robot0_joint_pos"][:], dtype=np.float64)
        elif "obs" in demo and "robot0_joint_pos" in demo["obs"]:
            state = np.asarray(demo["obs/robot0_joint_pos"][:], dtype=np.float64)
        else:
            xml = demo.attrs.get("model_file", demo.attrs.get("model_xml", ""))
            joints = list(ET.fromstring(xml).find("worldbody").iter("joint"))
            expected = [f"robot0_joint{i}" for i in range(1, 8)]
            assert [joint.get("name") for joint in joints[:7]] == expected
            state = np.asarray(demo["states"][:, 1:8], dtype=np.float64)
        return action, state
    indices = np.asarray(demo["source_step_index"][:], dtype=np.int64)
    source = Path(str(demo.attrs["source_episode"])) / "trajectory.h5"
    with h5py.File(source, "r") as raw:
        state = np.asarray(
            raw["observation/robot_state/joint_positions"][indices], dtype=np.float64
        )
    return action, state


def features(action, state):
    delta = np.diff(action, axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    residual = np.linalg.norm(action - state, axis=1)
    acceleration = np.diff(delta, axis=0)
    accel_norm = np.linalg.norm(acceleration, axis=1)
    a, b = delta[:-1], delta[1:]
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 1e-10
    cosine = np.sum(a[valid] * b[valid], axis=1) / denominator[valid]
    return {
        "steps": float(len(action)),
        "target_delta_median": float(np.median(delta_norm)),
        "target_delta_p90": float(np.percentile(delta_norm, 90)),
        "target_hold_1e5": float(np.mean(delta_norm < 1e-5)),
        "target_hold_1e4": float(np.mean(delta_norm < 1e-4)),
        "target_residual_median": float(np.median(residual)),
        "target_residual_p90": float(np.percentile(residual, 90)),
        "target_accel_median": float(np.median(accel_norm)),
        "target_accel_p90": float(np.percentile(accel_norm, 90)),
        "direction_reverse": float(np.mean(cosine < 0)) if len(cosine) else 0.0,
        "strong_direction_reverse": float(np.mean(cosine < -0.5)) if len(cosine) else 0.0,
    }


def summarize(path, real):
    rows = []
    with h5py.File(path, "r") as dataset:
        for key in ordered(dataset["data"]):
            rows.append(features(*episode_arrays(dataset["data"][key], real)))
    keys = list(rows[0])
    return {
        "path": str(path),
        "real_robot": real,
        "episodes": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "median": float(np.median([row[key] for row in rows])),
                "p10": float(np.percentile([row[key] for row in rows], 10)),
                "p90": float(np.percentile([row[key] for row in rows], 90)),
            }
            for key in keys
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True,
                        help="name:path:real, with real equal to 0 or 1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for spec in args.dataset:
        name, path, real = spec.split(":", 2)
        result[name] = summarize(Path(path), bool(int(real)))
        print(name, json.dumps(result[name]["metrics"], sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
