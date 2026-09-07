#!/usr/bin/env python3
"""Replace early pilot wording with the frozen production motion version."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    versions = [value for value in manifest["versions"] if value != "baseline"]
    assert len(versions) == 1
    version = versions[0]
    description = (
        f"Human-calibrated scripted production dataset ({version}). "
        "Privileged-geometry expert with paired Full/Partial initial states; "
        "see summary.json for fixed-budget success rates."
    )
    for filename in ("dataset_state.hdf5", "dataset_image84.hdf5", "dataset_image84_14hz.hdf5"):
        path = args.root / filename
        if not path.exists():
            continue
        with h5py.File(path, "r+") as dataset:
            dataset.attrs["description"] = description
            dataset.attrs["generation_version"] = version
            dataset.attrs["motion_profile_pairing"] = "shared by Full and Partial within each pair"
            if filename == "dataset_image84_14hz.hdf5":
                for demo in dataset["data"].values():
                    if "action_chunks" in demo:
                        continue
                    actions = np.asarray(demo["actions"][:], dtype=np.float32)
                    future = np.minimum(
                        np.arange(len(actions))[:, None] + np.arange(10)[None, :],
                        len(actions) - 1,
                    )
                    demo.create_dataset("action_chunks", data=actions[future], compression="lzf")
    summary_path = args.root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["generation_version"] = version
    summary["selection_rule"] = (
        f"both {version} regimes accepted on the same frozen initial state; "
        "fixed pair order; no smoothness ranking or score selection"
    )
    checksum_keys = {
        "dataset_state.hdf5": "dataset_sha256",
        "dataset_image84.hdf5": "image_dataset_sha256",
        "dataset_image84_14hz.hdf5": "human_rate_dataset_sha256",
    }
    checksums = {}
    for filename, key in checksum_keys.items():
        path = args.root / filename
        if path.exists():
            checksums[filename] = sha256(path)
            summary[key] = checksums[filename]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    (args.root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {filename}\n" for filename, digest in checksums.items())
    )
    print(json.dumps({"root": str(args.root), "generation_version": version}))


if __name__ == "__main__":
    main()
