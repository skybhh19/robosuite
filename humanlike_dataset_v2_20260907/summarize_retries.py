"""Build a paired dataset from the first accepted attempt for each cell."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--attempt-dir", action="append", required=True)
parser.add_argument("--target-pairs", type=int, required=True)
args = parser.parse_args()

manifest = json.loads((args.root / "manifest.json").read_text())
version = manifest["versions"][0]
regimes = tuple(manifest["regimes"])
attempt_rows = []
attempt_lookup = []
expected = {(i, regime) for i in range(manifest["pairs"]) for regime in regimes}

for directory in args.attempt_dir:
    rows = [json.loads(path.read_text()) for path in sorted((args.root / directory).glob("*.json"))]
    lookup = {(row["pair"], row["regime"]): row for row in rows}
    if set(lookup) != expected or len(rows) != len(expected):
        raise ValueError(f"incomplete attempt {directory}: {len(rows)}/{len(expected)}")
    attempt_rows.append(rows)
    attempt_lookup.append(lookup)

for pair in range(manifest["pairs"]):
    hashes = {
        lookup[(pair, regime)]["initial_state_sha256"]
        for lookup in attempt_lookup
        for regime in regimes
    }
    if len(hashes) != 1:
        raise ValueError(f"initial state mismatch for pair {pair}: {hashes}")

chosen = {}
for pair in range(manifest["pairs"]):
    for regime in regimes:
        key = (pair, regime)
        for attempt_index, lookup in enumerate(attempt_lookup):
            if lookup[key]["accepted"]:
                chosen[key] = (attempt_index, lookup[key])
                break

eligible_pairs = [pair for pair in range(manifest["pairs"]) if all((pair, regime) in chosen for regime in regimes)]
if len(eligible_pairs) < args.target_pairs:
    raise ValueError(f"only {len(eligible_pairs)} accepted paired states; need {args.target_pairs}")
selected = eligible_pairs[: args.target_pairs]

summary = {
    "complete": True,
    "attempt_dirs": args.attempt_dir,
    "attempt_results": {},
    "any_attempt": {},
    "paired_initial_state_checks": "passed",
    "eligible_pairs": eligible_pairs,
    "selected_pairs": selected,
    "selected_episodes": 2 * len(selected),
    "selection_rule": "first accepted attempt per Full/Partial cell; both regimes accepted on the same frozen initial state; fixed pair order",
}
for attempt_index, (directory, rows) in enumerate(zip(args.attempt_dir, attempt_rows)):
    summary["attempt_results"][directory] = {}
    for regime in regimes:
        cells = [row for row in rows if row["regime"] == regime]
        summary["attempt_results"][directory][regime] = {
            "n": len(cells),
            "physical": sum(row["physical_success"] for row in cells),
            "accepted": sum(row["accepted"] for row in cells),
        }
for regime in regimes:
    physical = sum(
        any(lookup[(pair, regime)]["physical_success"] for lookup in attempt_lookup)
        for pair in range(manifest["pairs"])
    )
    accepted = sum((pair, regime) in chosen for pair in range(manifest["pairs"]))
    summary["any_attempt"][regime] = {"n": manifest["pairs"], "physical": physical, "accepted": accepted}
summary["any_attempt"]["paired_accepted"] = len(eligible_pairs)

output = args.root / "dataset_state.hdf5"
if output.exists():
    raise FileExistsError(output)
valid_pairs = set(np.random.default_rng(720).permutation(selected)[: max(1, round(0.2 * len(selected)))])
masks = {key: [] for key in ("train", "valid", "all", "full", "partial", "fully_observable", "partially_observable")}
with h5py.File(output, "w") as destination:
    data = destination.create_group("data")
    data.attrs["env_args"] = json.dumps(manifest["env_args"])
    total = 0
    for pair in selected:
        for regime in regimes:
            attempt_index, row = chosen[(pair, regime)]
            demo_name = f"demo_{len(data)}"
            source_path = args.root / args.attempt_dir[attempt_index] / f"pair_{pair:03d}_{version}_{regime}.hdf5"
            with h5py.File(source_path) as source:
                source.copy("data/demo_0", data, name=demo_name)
            demo = data[demo_name]
            demo.attrs["source_attempt"] = int(row["attempt"])
            demo.attrs["source_attempt_dir"] = args.attempt_dir[attempt_index]
            samples = len(demo["actions"])
            total += samples
            rewards = np.zeros(samples, dtype=np.float32)
            rewards[-1] = 1.0
            dones = np.zeros(samples, dtype=np.int64)
            dones[-1] = 1
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            if demo["actions"].shape != (samples, 8) or not np.isfinite(demo["actions"][:]).all():
                raise ValueError(f"invalid actions in {source_path}")
            if np.max(np.abs(demo["actions"][:, :7] - demo["robot0_joint_pos"][:] - demo["actions_joint_delta"][:, :7])) >= 1e-10:
                raise ValueError(f"joint action mismatch in {source_path}")
            if not np.array_equal(demo["states"][1:], demo["next_states"][:-1]):
                raise ValueError(f"state transition mismatch in {source_path}")
            masks["all"].append(demo_name)
            masks[regime].append(demo_name)
            masks["fully_observable" if regime == "full" else "partially_observable"].append(demo_name)
            masks["valid" if pair in valid_pairs else "train"].append(demo_name)
    data.attrs["total"] = total
    mask = destination.create_group("mask")
    for key, names in masks.items():
        mask.create_dataset(key, data=np.asarray(names, dtype="S"))
    destination.attrs["description"] = "ToolHang humanlike scripted dataset with at most one collection retry per cell."

summary["dataset_steps"] = total
summary["dataset_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
(args.root / "summary.json").write_text(json.dumps(summary, indent=2))
(args.root / "selection.json").write_text(json.dumps(selected, indent=2))
print(json.dumps(summary, indent=2))
print("DATASET", output, 2 * len(selected), total)
