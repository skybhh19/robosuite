#!/usr/bin/env python3
"""Clone a training HDF5 exactly while replacing its two RGB observations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def copy_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def copy_dataset(source, target, name):
    kwargs = {}
    if source.compression is not None:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    if source.chunks is not None:
        kwargs["chunks"] = source.chunks
    dataset = target.create_dataset(name, data=source[...], **kwargs)
    copy_attrs(source, dataset)


def clone_group(source, target, path=""):
    copy_attrs(source, target)
    for name, item in source.items():
        item_path = f"{path}/{name}"
        if item_path.endswith("/obs/agentview_image") or item_path.endswith("/obs/robot0_eye_in_hand_image"):
            continue
        if isinstance(item, h5py.Group):
            clone_group(item, target.create_group(name), item_path)
        else:
            copy_dataset(item, target, name)


def merge(source, shards, output, height, width):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".tmp.", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            clone_group(src, dst)
            for name in src["data"]:
                shard_path = shards / f"{name}.hdf5"
                if not shard_path.exists():
                    raise FileNotFoundError(shard_path)
                with h5py.File(shard_path, "r") as shard:
                    if shard.attrs["demo"] != name:
                        raise RuntimeError(f"{shard_path}: demo mismatch")
                    obs = dst["data"][name]["obs"]
                    length = len(dst["data"][name]["actions"])
                    for key in CAMERA_KEYS:
                        values = shard[key]
                        if values.shape != (length, height, width, 3) or values.dtype != np.uint8:
                            raise RuntimeError(f"{shard_path}:{key} invalid {values.shape} {values.dtype}")
                        obs.create_dataset(
                            key,
                            data=values,
                            compression="lzf",
                            chunks=(1, height, width, 3),
                        )
            dst.attrs["native_image_height"] = height
            dst.attrs["native_image_width"] = width
            dst.attrs["native_image_source"] = "stored pre-action simulator states"
            dst.flush()
        os.replace(temporary, output)
        hasher = hashlib.sha256()
        with output.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        print(json.dumps({"output": str(output), "sha256": digest}, indent=2))
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    merge(args.source.resolve(), args.shards.resolve(), args.output.resolve(), args.height, args.width)


if __name__ == "__main__":
    main()
