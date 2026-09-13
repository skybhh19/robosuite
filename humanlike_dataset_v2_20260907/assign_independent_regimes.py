"""Add a deterministic balanced Full/Partial assignment to a prepared manifest."""

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--seed", type=int, default=202609132)
args = parser.parse_args()

manifest = json.loads(args.manifest.read_text())
entries = manifest["entries"]
if len(entries) % 2:
    raise ValueError("balanced assignment requires an even number of states")
labels = np.asarray(["full"] * (len(entries) // 2) + ["partial"] * (len(entries) // 2))
np.random.default_rng(args.seed).shuffle(labels)
for entry, label in zip(entries, labels.tolist()):
    entry["regime_assignment"] = label
manifest["protocol"] = "independent_initial_states_balanced_by_observability_with_one_retry"
manifest["regime_assignment_seed"] = args.seed
manifest["assigned_counts"] = {label: labels.tolist().count(label) for label in ("full", "partial")}
args.manifest.write_text(json.dumps(manifest, indent=2))
print(manifest["assigned_counts"])
