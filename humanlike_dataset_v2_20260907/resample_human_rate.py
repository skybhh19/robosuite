#!/usr/bin/env python3
"""Create a 14 Hz training view from 20 Hz robosuite trajectories."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def copy_group(source, target, indices, original_length):
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            copy_group(item, target.create_group(name), indices, original_length)
            continue
        kwargs = {}
        if item.compression is not None:
            kwargs["compression"] = item.compression
            kwargs["compression_opts"] = item.compression_opts
        if item.ndim and item.shape[0] == original_length:
            values = item[indices]
            dataset = target.create_dataset(name, data=values, **kwargs)
        else:
            dataset = target.create_dataset(name, data=item[:], **kwargs)
        for key, value in item.attrs.items():
            dataset.attrs[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    source_path = args.root / "dataset_image84.hdf5"
    output_path = args.root / "dataset_image84_14hz.hdf5"
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_suffix(".partial.hdf5")
    audit = {"source_hz": 20, "target_hz": 14, "episodes": [], "rule": "floor(k*20/14), include terminal", "action_chunk_size": 10}
    with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["source_control_hz"] = 20
        output.attrs["training_control_hz"] = 14
        output.attrs["resampling_rule"] = audit["rule"]
        data = output.create_group("data")
        for key, value in source["data"].attrs.items():
            data.attrs[key] = value
        total = 0
        for name in source["data"]:
            source_demo = source["data"][name]
            length = len(source_demo["actions"])
            count = int(np.ceil((length - 1) * 14 / 20)) + 1
            indices = np.unique(np.floor(np.arange(count) * 20 / 14).astype(int))
            indices = indices[indices < length]
            if indices[-1] != length - 1:
                indices = np.r_[indices, length - 1]
            demo = data.create_group(name)
            copy_group(source_demo, demo, indices, length)
            if "source_step_index" in demo:
                del demo["source_step_index"]
            demo.create_dataset("source_step_index", data=indices.astype(np.int32))
            actions = np.asarray(demo["actions"][:], dtype=np.float32)
            future = np.minimum(
                np.arange(len(actions))[:, None] + np.arange(10)[None, :], len(actions) - 1
            )
            if "action_chunks" in demo:
                del demo["action_chunks"]
            demo.create_dataset(
                "action_chunks", data=actions[future].reshape(len(actions), -1), compression="lzf"
            )
            demo.attrs["num_samples"] = len(indices)
            demo.attrs["control_hz"] = 14
            demo["rewards"][:] = 0
            demo["rewards"][-1] = 1
            demo["dones"][:] = 0
            demo["dones"][-1] = 1
            total += len(indices)
            audit["episodes"].append({"demo": name, "source_steps": length, "steps_14hz": len(indices)})
        data.attrs["total"] = total
        source.copy("mask", output)
    temporary.replace(output_path)
    audit["episodes_count"] = len(audit["episodes"])
    audit["total_steps_14hz"] = total
    (args.root / "human_rate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "episodes": len(audit["episodes"]), "steps": total}))


if __name__ == "__main__":
    main()
