"""Episode-held-out low-dimensional predictability diagnostics."""

import argparse
import glob
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def q_from_states(demo):
    if "robot0_joint_pos" in demo:
        return demo["robot0_joint_pos"][:]
    if "obs" in demo and "robot0_joint_pos" in demo["obs"]:
        return demo["obs/robot0_joint_pos"][:]
    xml = demo.attrs.get("model_file", demo.attrs.get("model_xml", ""))
    root = ET.fromstring(xml)
    joints = list(root.find("worldbody").iter("joint"))
    assert [joint.get("name") for joint in joints[:7]] == [f"robot0_joint{i}" for i in range(1, 8)]
    assert all(joint.get("type", "hinge") == "hinge" for joint in joints[:7])
    return demo["states"][:, 1:8]


def load(path, raw=False):
    episodes = []
    files = sorted(glob.glob(path + "/*/trajectory.h5")) if raw else [path]
    for filename in files:
        with h5py.File(filename) as dataset:
            if raw:
                robot = dataset["observation/robot_state"]
                episodes.append({
                    "a": dataset["action/joint_position"][:],
                    "q": robot["joint_positions"][:],
                    "p": robot["cartesian_position"][:],
                    "v": dataset["action/cartesian_velocity"][:],
                    "label": "real",
                    "id": filename,
                })
                continue
            for name in sorted(dataset["data"]):
                demo = dataset["data"][name]
                action = demo["actions"][:, :-1]
                joint = q_from_states(demo)
                assert len(action) == len(joint)
                obs = demo.get("obs", {})
                if "robot0_eef_pos" in obs:
                    pose = np.concatenate([
                        obs[key][:] for key in
                        ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos") if key in obs
                    ], axis=1)
                else:
                    pose = joint
                episodes.append({
                    "a": action,
                    "q": joint,
                    "p": pose,
                    "label": str(demo.attrs.get("observability", demo.attrs.get("label_regime", "human"))),
                    "id": name,
                })
    return episodes


def correlation(left, right):
    left = left - left.mean(0)
    right = right - right.mean(0)
    denominator = np.sqrt((left * left).sum(0) * (right * right).sum(0))
    return np.divide(
        (left * right).sum(0), denominator, out=np.zeros(left.shape[1]), where=denominator > 1e-12
    ).tolist()


def audit(episodes, absolute):
    result = {
        "episodes": len(episodes),
        "steps": sum(len(episode["a"]) for episode in episodes),
        "groups": {},
        "source_ids": [episode["id"] for episode in episodes],
    }
    for label in sorted({episode["label"] for episode in episodes}):
        group = [episode for episode in episodes if episode["label"] == label]
        action = np.concatenate([episode["a"] for episode in group])
        joint = np.concatenate([episode["q"] for episode in group])
        stats = {
            "episodes": len(group),
            "steps": len(action),
            "median_length": float(np.median([len(episode["a"]) for episode in group])),
            "previous_action_corr": correlation(
                np.concatenate([episode["a"][1:] for episode in group]),
                np.concatenate([episode["a"][:-1] for episode in group]),
            ),
        }
        if absolute:
            stats.update(
                action_q_corr=correlation(action, joint),
                residual_q_corr=correlation(action - joint, joint),
                residual_rms=float(np.sqrt(np.mean((action - joint) ** 2))),
            )
        result["groups"][label] = stats
    targets = ["a", "residual"] if absolute else ["a"]
    for target in targets:
        scores = {}
        for seed in (0, 1, 2):
            order = np.random.default_rng(seed).permutation(len(episodes))
            train = set(order[: int(0.8 * len(episodes))])
            for feature in ("q", "q_prev", "p", "history", "a_prev"):
                blocks, ys = [[], []], [[], []]
                for index, episode in enumerate(episodes):
                    joint, action, pose = episode["q"], episode["a"], episode["p"]
                    target_value = action if target == "a" else action - joint
                    features = {
                        "q": joint[1:],
                        "q_prev": joint[:-1],
                        "p": pose[1:],
                        "history": np.concatenate([joint[1:], joint[:-1], action[:-1]], axis=1),
                        "a_prev": action[:-1],
                    }[feature]
                    side = 0 if index in train else 1
                    blocks[side].append(features)
                    ys[side].append(target_value[1:])
                x, z = [np.concatenate(block) for block in blocks]
                y, test = [np.concatenate(values) for values in ys]
                mean, scale = x.mean(0), x.std(0)
                scale[scale < 1e-8] = 1
                x = np.column_stack([(x - mean) / scale, np.ones(len(x))])
                z = np.column_stack([(z - mean) / scale, np.ones(len(z))])
                y_mean, y_scale = y.mean(0), y.std(0)
                y_scale[y_scale < 1e-8] = 1
                weights = np.linalg.solve(
                    x.T @ x + np.eye(x.shape[1]), x.T @ ((y - y_mean) / y_scale)
                )
                prediction = (z @ weights) * y_scale + y_mean
                per_dim = 1 - ((prediction - test) ** 2).sum(0) / np.maximum(
                    ((test - test.mean(0)) ** 2).sum(0), 1e-12
                )
                scores.setdefault(feature, []).append(float(per_dim.mean()))
        result[target + "_heldout_ridge_r2"] = {
            key: {"mean": float(np.mean(values)), "seeds": values} for key, values in scores.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(load(args.path, args.raw), args.absolute)
    result.update(path=args.path, absolute=args.absolute)
    if args.raw:
        result["cartesian_velocity"] = audit(
            [{**episode, "a": episode["v"]} for episode in load(args.path, True)], False
        )
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(args.output, len(result["source_ids"]), flush=True)


if __name__ == "__main__":
    main()
