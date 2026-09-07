#!/usr/bin/env python3
"""Image-only BC Full-vs-All verification for the two production datasets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/iris/u/jasonyan/data/humanlike_dataset_v2_20260907")
REPO = Path("/iris/u/jasonyan/repos/demonstration-information")
IMAGES = ["agentview_image", "robot0_eye_in_hand_image"]


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def rows():
    return [
        {"kind": kind, "condition": condition, "seed": seed,
         "name": f"human_{'v8' if kind == 'threading' else 'v4'}_{kind}_{condition}_seed{seed}"}
        for kind in ("threading", "toolhang")
        for condition in ("all", "full")
        for seed in (1, 2, 3)
    ]


def prepare():
    base = json.loads((
        REPO / "robomimic/robomimic/exps/threading/policy/image/full/gmm/seed1.json"
    ).read_text())
    for row in rows():
        config = copy.deepcopy(base)
        config["experiment"]["name"] = row["name"]
        config["experiment"]["logging"]["log_wandb"] = False
        config["experiment"]["rollout"]["enabled"] = False
        config["experiment"]["save"].update(
            enabled=True, every_n_epochs=100, epochs=[],
            on_best_validation=False, on_best_rollout_return=False,
            on_best_rollout_success_rate=False,
        )
        data = (
            ROOT / "production_v8/threading/dataset_image84_14hz.hdf5"
            if row["kind"] == "threading"
            else ROOT / "production/toolhang/dataset_image84_14hz.hdf5"
        )
        mask = "all" if row["condition"] == "all" else "fully_observable"
        dataset_keys = ["actions"]
        action_keys = ["action_chunks"]
        action_config = {"action_chunks": {"normalization": "min_max"}}
        config["train"].update(
            data=str(data), output_dir=str(ROOT / "bc/outputs"),
            hdf5_filter_key=mask, hdf5_validation_filter_key=None,
            hdf5_cache_mode="all", num_data_workers=0, seq_length=1,
            batch_size=32, num_epochs=300, seed=row["seed"],
            dataset_keys=dataset_keys, action_keys=action_keys,
            action_config=action_config,
        )
        config["algo"]["gmm"].update(enabled=True, num_modes=5, low_noise_eval=True)
        config["algo"]["rnn"]["enabled"] = False
        config["observation"]["modalities"]["obs"] = {
            "low_dim": [], "rgb": IMAGES, "depth": [], "scan": []
        }
        for key in config["observation"]["modalities"].get("goal", {}):
            config["observation"]["modalities"]["goal"][key] = []
        config["experiment"]["env_meta_update_dict"] = {
            "env_kwargs": {
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "camera_heights": 84, "camera_widths": 84,
            }
        }
        write(ROOT / "bc/configs" / f"{row['name']}.json", config)
    write(ROOT / "bc/manifest.json", rows())


def train(index):
    row = rows()[index]
    config = ROOT / "bc/configs" / f"{row['name']}.json"
    output = ROOT / "bc/outputs" / row["name"]
    if (output / "TRAIN_DONE").exists():
        return
    # robomimic prompts interactively when an incomplete model directory remains.
    # Each array index owns one unique directory, so removing only its failed output
    # makes scheduler retries deterministic without touching completed runs.
    shutil.rmtree(output, ignore_errors=True)
    log = ROOT / "bc/logs" / f"train_{row['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        run([sys.executable, REPO / "robomimic/robomimic/scripts/train.py",
             "--config", config], stdout=stream, stderr=subprocess.STDOUT)
    if not list(output.rglob("model_epoch_300.pth")):
        raise RuntimeError(f"missing epoch-300 checkpoint: {output}")
    (output / "TRAIN_DONE").touch()


def smoke():
    """Run two gradient steps to catch dataset, modality, and model config errors."""
    prepare()
    shutil.rmtree(ROOT / "bc/smoke/smoke_threading_image_only", ignore_errors=True)
    shutil.rmtree(ROOT / "bc/smoke_eval", ignore_errors=True)
    source = ROOT / "bc/configs/human_v8_threading_all_seed1.json"
    config = json.loads(source.read_text())
    config["experiment"]["name"] = "smoke_threading_image_only"
    config["experiment"]["epoch_every_n_steps"] = 2
    config["experiment"]["save"].update(every_n_epochs=1)
    config["train"]["num_epochs"] = 1
    config["train"]["output_dir"] = str(ROOT / "bc/smoke")
    path = ROOT / "bc/configs/smoke_threading_image_only.json"
    write(path, config)
    run([sys.executable, REPO / "robomimic/robomimic/scripts/train.py", "--config", path])
    candidates = list((ROOT / "bc/smoke/smoke_threading_image_only").rglob("model_epoch_1.pth"))
    if len(candidates) != 1:
        raise RuntimeError(f"smoke checkpoint missing: {candidates}")
    smoke_eval = ROOT / "bc/smoke_eval"
    run([
        sys.executable, ROOT / "eval_policy_14hz.py", "--agent", candidates[0],
        "--output-dir", smoke_eval, "--n-rollouts", "2", "--horizon", "24",
        "--seed", "2026090800",
    ])
    smoke_summary = json.loads((smoke_eval / "summary.json").read_text())
    assert smoke_summary["n_rollouts"] == 2
    (ROOT / "bc/SMOKE_DONE").touch()


def checkpoint(row):
    candidates = list((ROOT / "bc/outputs" / row["name"]).rglob("model_epoch_300.pth"))
    if len(candidates) != 1:
        raise RuntimeError((row, candidates))
    return candidates[0]


def evaluate(index):
    row = rows()[index]
    output = ROOT / "bc/eval" / row["name"]
    output.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, ROOT / "eval_policy_14hz.py", "--agent", checkpoint(row),
        "--output-dir", output, "--n-rollouts", "100",
        "--horizon", "800" if row["kind"] == "threading" else "700",
        "--seed", "2026090800",
    ])
    (output / "EVAL_DONE").touch()


def report():
    import numpy as np
    records = []
    for row in rows():
        folder = ROOT / "bc/eval" / row["name"]
        value = json.loads((folder / "summary.json").read_text())
        success = float(value["success_rate"])
        records.append({**row, "epoch": 300, "rollouts": 100, "success_rate": success})
    result = {
        "runs": records,
        "groups": {},
        "criterion": "mean(full) > mean(all) at frozen epoch 300",
        "datasets": {
            "threading": str(ROOT / "production_v8/threading/dataset_image84_14hz.hdf5"),
            "toolhang": str(ROOT / "production/toolhang/dataset_image84_14hz.hdf5"),
        },
        "protocol": {
            "observations": IMAGES,
            "low_dim": [],
            "algorithm": "BC-GMM",
            "action_chunk_size": 10,
            "training_seeds": [1, 2, 3],
            "rollouts_per_policy": 100,
            "sim_hz": 20,
            "policy_target_hz": 14,
        },
    }
    for kind in ("threading", "toolhang"):
        grouped = {}
        for condition in ("all", "full"):
            values = [x["success_rate"] for x in records
                      if x["kind"] == kind and x["condition"] == condition]
            grouped[condition] = {"values": values, "mean": float(np.mean(values)),
                                  "std_population": float(np.std(values))}
        difference = grouped["full"]["mean"] - grouped["all"]["mean"]
        paired_seed_differences = []
        paired_episode_differences = []
        discordant = {"full_only": 0, "all_only": 0}
        for seed in (1, 2, 3):
            by_condition = {}
            for condition in ("all", "full"):
                name = next(
                    row["name"] for row in rows()
                    if row["kind"] == kind and row["condition"] == condition and row["seed"] == seed
                )
                summary = json.loads((ROOT / "bc/eval" / name / "summary.json").read_text())
                by_condition[condition] = np.asarray(
                    [int(item["success"]) for item in summary["records"]], dtype=np.int8
                )
            assert len(by_condition["all"]) == len(by_condition["full"]) == 100
            delta = by_condition["full"] - by_condition["all"]
            paired_seed_differences.append(float(delta.mean()))
            paired_episode_differences.append(delta)
            discordant["full_only"] += int(np.sum(delta == 1))
            discordant["all_only"] += int(np.sum(delta == -1))
        # Hierarchical paired bootstrap: resample policy seeds, then matched environment episodes.
        rng = np.random.default_rng(20260908)
        bootstrap = np.empty(50000, dtype=np.float64)
        for draw in range(len(bootstrap)):
            sampled_seeds = rng.integers(0, 3, size=3)
            bootstrap[draw] = np.mean([
                paired_episode_differences[index][rng.integers(0, 100, size=100)].mean()
                for index in sampled_seeds
            ])
        result["groups"][kind] = {**grouped, "full_minus_all": difference,
                                   "paired_seed_differences": paired_seed_differences,
                                   "paired_hierarchical_bootstrap95": np.quantile(
                                       bootstrap, [.025, .975]
                                   ).tolist(),
                                   "matched_rollout_discordance": discordant,
                                   "passed": bool(difference > 0)}
    result["passed"] = all(x["passed"] for x in result["groups"].values())
    write(ROOT / "bc/report.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Full did not beat All in both environments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "smoke", "train", "eval", "report"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "train": lambda: train(args.index),
     "eval": lambda: evaluate(args.index), "report": report}[args.command]()


if __name__ == "__main__":
    main()
