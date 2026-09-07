"""Audit paired completion rates under a fixed maximum attempt budget."""
import argparse
import json
from pathlib import Path

import numpy as np


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--version", required=True)
p.add_argument("--retry-dir", default="retry1")
p.add_argument("--output", type=Path)
a = p.parse_args()

manifest = json.loads((a.root / "manifest.json").read_text())
n = manifest["pairs"]
first = {}
completed = {}
recoveries = {}
for regime in manifest["regimes"]:
    first[regime] = []
    completed[regime] = []
    recoveries[regime] = 0
    for pair in range(n):
        name = f"pair_{pair:03d}_{a.version}_{regime}.json"
        row0 = json.loads((a.root / "trials" / name).read_text())
        ok0 = bool(row0["physical_success"])
        retry_path = a.root / a.retry_dir / name
        ok1 = bool(json.loads(retry_path.read_text())["physical_success"]) if retry_path.exists() else False
        first[regime].append(ok0)
        completed[regime].append(ok0 or ok1)
        recoveries[regime] += int((not ok0) and ok1)


def comparison(values):
    x = np.asarray(values["full"], dtype=int) - np.asarray(values["partial"], dtype=int)
    boot = np.random.default_rng(721).choice(x, size=(20000, len(x)), replace=True).mean(1)
    return {
        "full": int(np.sum(values["full"])),
        "partial": int(np.sum(values["partial"])),
        "n_pairs": len(x),
        "full_rate": float(np.mean(values["full"])),
        "partial_rate": float(np.mean(values["partial"])),
        "full_minus_partial": float(np.mean(x)),
        "paired_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "discordant": int(np.count_nonzero(x)),
    }


result = {
    "protocol": "same frozen initial states; at most two independently seeded attempts per regime; stop after success",
    "first_attempt": comparison(first),
    "within_two_attempts": comparison(completed),
    "conditional_retry_recoveries": recoveries,
    "retry_files_expected_only_for_first_attempt_failures": True,
}
out = a.output or (a.root / "retry_completion_audit.json")
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
