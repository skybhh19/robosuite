"""Build balanced Full/Partial data from independently sampled initial states."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--attempt-dir", action="append", required=True)
parser.add_argument("--target-per-regime", type=int)
parser.add_argument("--target-full", type=int)
parser.add_argument("--target-partial", type=int)
parser.add_argument("--require-visibility-labels", action="store_true")
args = parser.parse_args()
if args.target_per_regime is not None:
    if args.target_full is not None or args.target_partial is not None:
        parser.error("use target-per-regime or target-full/target-partial")
    targets = {"full": args.target_per_regime, "partial": args.target_per_regime}
elif args.target_full is not None and args.target_partial is not None:
    targets = {"full": args.target_full, "partial": args.target_partial}
else:
    parser.error("set target-per-regime or both target-full and target-partial")
if any(value <= 0 for value in targets.values()):
    parser.error("target counts must be positive")

manifest = json.loads((args.root / "manifest.json").read_text())
version = manifest["versions"][0]
regimes = tuple(manifest["regimes"])
assignment = {int(entry["pair"]): entry["regime_assignment"] for entry in manifest["entries"]}
expected = {(pair, regime) for pair, regime in assignment.items()}
attempt_rows = []
attempt_lookup = []
for directory in args.attempt_dir:
    rows = [json.loads(path.read_text()) for path in sorted((args.root / directory).glob("*.json"))]
    lookup = {(row["pair"], row["regime"]): row for row in rows}
    if set(lookup) != expected or len(rows) != len(expected):
        raise ValueError(f"incomplete attempt {directory}: {len(rows)}/{len(expected)}")
    attempt_rows.append(rows)
    attempt_lookup.append(lookup)

for key in expected:
    hashes = {
        lookup[key]["initial_state_sha256"]
        for lookup in attempt_lookup
        if "initial_state_sha256" in lookup[key]
    }
    # An exception before the first recorded action has no state hash. Compare
    # every available recording; accepted cells are guaranteed to have one.
    if len(hashes) > 1:
        raise ValueError(f"initial state mismatch across attempts for {key}: {hashes}")

chosen = {}
for key in sorted(expected):
    for attempt_index, lookup in enumerate(attempt_lookup):
        row = lookup[key]
        visible = row.get("stats", {}).get("visibility_diagnostics", {}).get("hole_center_visible_at_preinsert")
        label_ok = not args.require_visibility_labels or visible is (key[1] == "full")
        if row["accepted"] and label_ok:
            chosen[key] = (attempt_index, lookup[key])
            break

eligible = {
    regime: [pair for pair in sorted(assignment) if assignment[pair] == regime and (pair, regime) in chosen]
    for regime in regimes
}
for regime in regimes:
    if len(eligible[regime]) < targets[regime]:
        raise ValueError(f"only {len(eligible[regime])} accepted {regime} states; need {targets[regime]}")

# Match the grasp-bin histogram exactly across observability regimes. The
# assignment itself is random, but acceptance can otherwise leave one regime
# with more examples from an easier offset bin.
entry_by_pair = {int(entry["pair"]): entry for entry in manifest["entries"]}
bins = sorted({int(entry["bin"]) for entry in manifest["entries"]})
common_capacity = {
    bin_index: min(
        sum(int(entry_by_pair[pair]["bin"]) == bin_index for pair in eligible[regime])
        for regime in regimes
    )
    for bin_index in bins
}
if targets["full"] == targets["partial"]:
    base = targets["full"] // len(bins)
    matched = {bin_index: min(base, common_capacity[bin_index]) for bin_index in bins}
    while sum(matched.values()) < targets["full"]:
        candidates = [bin_index for bin_index in bins if matched[bin_index] < common_capacity[bin_index]]
        if not candidates:
            raise ValueError(f"insufficient common grasp-bin capacity: {common_capacity}")
        bin_index = min(candidates, key=lambda value: (matched[value], value))
        matched[bin_index] += 1
    quota = {regime: matched.copy() for regime in regimes}
else:
    quota = {}
    for regime in regimes:
        target = targets[regime]
        quota[regime] = {bin_index: target // len(bins) + int(position < target % len(bins)) for position, bin_index in enumerate(bins)}
        for bin_index in bins:
            capacity = sum(int(entry_by_pair[pair]["bin"]) == bin_index for pair in eligible[regime])
            if capacity < quota[regime][bin_index]:
                raise ValueError(f"insufficient {regime} grasp-bin {bin_index}: {capacity} < {quota[regime][bin_index]}")
selected = {}
for regime in regimes:
    selected[regime] = [
        pair
        for bin_index in bins
        for pair in [
            candidate
            for candidate in eligible[regime]
            if int(entry_by_pair[candidate]["bin"]) == bin_index
        ][: quota[regime][bin_index]]
    ]
selected_cells = [(pair, regime) for regime in regimes for pair in selected[regime]]

summary = {
    "complete": True,
    "protocol": "independent_initial_states_balanced_by_observability_with_one_retry",
    "attempt_dirs": args.attempt_dir,
    "attempt_results": {},
    "any_attempt": {},
    "eligible_states": eligible,
    "selected_states": selected,
    "grasp_bin_quota": quota,
    "selected_episodes": len(selected_cells),
    "unique_initial_states": len(selected_cells),
    "selection_rule": "each initial state is assigned to exactly one observability regime; first accepted attempt; fixed state order within regime",
    "visibility_label_gate": args.require_visibility_labels,
}
for directory, rows in zip(args.attempt_dir, attempt_rows):
    summary["attempt_results"][directory] = {}
    for regime in regimes:
        cells = [row for row in rows if row["regime"] == regime]
        summary["attempt_results"][directory][regime] = {
            "n": len(cells),
            "physical": sum(row["physical_success"] for row in cells),
            "accepted": sum(row["accepted"] for row in cells),
        }
for regime in regimes:
    keys = [(pair, regime) for pair in assignment if assignment[pair] == regime]
    summary["any_attempt"][regime] = {
        "n": len(keys),
        "physical": sum(any(lookup[key]["physical_success"] for lookup in attempt_lookup) for key in keys),
        "accepted": len(eligible[regime]),
    }

output = args.root / "dataset_state.hdf5"
if output.exists():
    raise FileExistsError(output)
rng = np.random.default_rng(720)
valid_cells = set()
for regime in regimes:
    valid_cells.update((pair, regime) for pair in rng.permutation(selected[regime])[: round(0.2 * targets[regime])])
masks = {key: [] for key in ("train", "valid", "all", "full", "partial", "fully_observable", "partially_observable")}
with h5py.File(output, "w") as destination:
    data = destination.create_group("data")
    data.attrs["env_args"] = json.dumps(manifest["env_args"])
    total = 0
    for pair, regime in selected_cells:
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
        masks["valid" if (pair, regime) in valid_cells else "train"].append(demo_name)
    data.attrs["total"] = total
    mask = destination.create_group("mask")
    for key, names in masks.items():
        mask.create_dataset(key, data=np.asarray(names, dtype="S"))
    destination.attrs["description"] = "ToolHang training dataset with independent initial states for Full and Partial, following the Threading collection protocol."

summary["dataset_steps"] = total
summary["dataset_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
(args.root / "summary.json").write_text(json.dumps(summary, indent=2))
(args.root / "selection.json").write_text(json.dumps(selected, indent=2))
print(json.dumps(summary, indent=2))
print("DATASET", output, len(selected_cells), total)
