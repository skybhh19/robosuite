#!/usr/bin/env python3
"""Deterministically resample the 20 Hz fork dataset to 14 Hz."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def copy_group(source, target, indices, original_length):
    for key, value in source.attrs.items(): target.attrs[key] = value
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            copy_group(item, target.create_group(name), indices, original_length)
        else:
            values = item[indices] if item.ndim and item.shape[0] == original_length else item[:]
            kwargs = {"compression": item.compression} if item.compression else {}
            created = target.create_dataset(name, data=values, **kwargs)
            for key, value in item.attrs.items(): created.attrs[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    audit = {"source_hz": 20, "target_hz": 14, "rule": "floor(k*20/14), include terminal", "episodes": []}
    temporary = args.output.with_suffix(".partial.hdf5")
    with h5py.File(args.source) as source, h5py.File(temporary, "w") as output:
        for key, value in source.attrs.items(): output.attrs[key] = value
        output.attrs["source_control_hz"] = 20; output.attrs["training_control_hz"] = 14
        data = output.create_group("data")
        for key, value in source["data"].attrs.items(): data.attrs[key] = value
        total = 0
        for name in sorted(source["data"], key=lambda x: int(x.rsplit("_", 1)[1])):
            src = source["data"][name]; length = len(src["actions"])
            count = int(np.ceil((length - 1) * 14 / 20)) + 1
            indices = np.unique(np.floor(np.arange(count) * 20 / 14).astype(int)); indices = indices[indices < length]
            if indices[-1] != length - 1: indices = np.r_[indices, length - 1]
            demo = data.create_group(name); copy_group(src, demo, indices, length)
            demo.create_dataset("source_step_index", data=indices.astype(np.int32))
            actions = np.asarray(demo["actions"], dtype=np.float32)
            future = np.minimum(np.arange(len(actions))[:, None] + np.arange(10)[None, :], len(actions) - 1)
            demo.create_dataset("action_chunks", data=actions[future].reshape(len(actions), -1), compression="lzf")
            demo.attrs["num_samples"] = len(indices); demo.attrs["control_hz"] = 14
            demo["rewards"][:] = 0; demo["rewards"][-1] = 1; demo["dones"][:] = 0; demo["dones"][-1] = 1
            total += len(indices); audit["episodes"].append({"demo": name, "source_steps": length, "steps_14hz": len(indices)})
        data.attrs["total"] = total; source.copy("mask", output)
    temporary.replace(args.output)
    audit.update(episodes_count=len(audit["episodes"]), total_steps_14hz=total)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit | {"episodes": 300}, indent=2))


if __name__ == "__main__": main()
