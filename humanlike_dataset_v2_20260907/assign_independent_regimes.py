"""Add a deterministic balanced Full/Partial assignment to a prepared manifest."""

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--seed", type=int, default=202609132)
parser.add_argument("--full-count", type=int)
args = parser.parse_args()

manifest = json.loads(args.manifest.read_text())
entries = manifest["entries"]
full_count = len(entries) // 2 if args.full_count is None else args.full_count
if not 0 < full_count < len(entries):
    raise ValueError("full-count must leave at least one state in each regime")
if args.full_count is None and len(entries) % 2:
    raise ValueError("balanced assignment requires an even number of states")
rng = np.random.default_rng(args.seed)
bins = sorted({int(entry["bin"]) for entry in entries})
quotas = {bin_index: full_count * sum(int(entry["bin"]) == bin_index for entry in entries) // len(entries) for bin_index in bins}
while sum(quotas.values()) < full_count:
    remaining = [bin_index for bin_index in bins if quotas[bin_index] < sum(int(entry["bin"]) == bin_index for entry in entries)]
    bin_index = min(remaining, key=lambda value: (quotas[value], value))
    quotas[bin_index] += 1
for bin_index in bins:
    matching = [entry for entry in entries if int(entry["bin"]) == bin_index]
    selected = set(rng.choice(len(matching), size=quotas[bin_index], replace=False).tolist())
    for index, entry in enumerate(matching):
        entry["regime_assignment"] = "full" if index in selected else "partial"
labels = [entry["regime_assignment"] for entry in entries]
manifest["protocol"] = "independent_initial_states_stratified_by_grasp_bin_with_one_retry"
manifest["regime_assignment_seed"] = args.seed
manifest["assigned_counts"] = {label: labels.count(label) for label in ("full", "partial")}
args.manifest.write_text(json.dumps(manifest, indent=2))
print(manifest["assigned_counts"])
