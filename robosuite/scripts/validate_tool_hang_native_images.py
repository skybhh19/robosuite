#!/usr/bin/env python3
"""Validate exact non-image parity and native RGB structure for a ToolHang release."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def equal_attrs(left, right):
    if set(left.attrs) - {"native_image_height", "native_image_width", "native_image_source"} != set(right.attrs):
        return False
    for key in right.attrs:
        a, b = left.attrs[key], right.attrs[key]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                return False
        elif a != b:
            return False
    return True


def compare_tree(output, source, path=""):
    if not equal_attrs(output, source):
        raise RuntimeError(f"attribute mismatch: {path or '/'}")
    expected = set(source)
    if set(output) != expected:
        raise RuntimeError(f"member mismatch: {path or '/'}")
    for name in expected:
        item_path = f"{path}/{name}"
        left, right = output[name], source[name]
        if isinstance(right, h5py.Group):
            compare_tree(left, right, item_path)
        elif item_path.endswith("/obs/agentview_image") or item_path.endswith("/obs/robot0_eye_in_hand_image"):
            continue
        elif left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left[...], right[...]):
            raise RuntimeError(f"dataset mismatch: {item_path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_mask(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate(source, output, expected_episodes, expected_full, expected_partial, height, width, report):
    result = {"source": str(source), "output": str(output)}
    with h5py.File(source, "r") as src, h5py.File(output, "r") as out:
        compare_tree(out, src)
        names = list(out["data"])
        if len(names) != expected_episodes:
            raise RuntimeError(f"episodes {len(names)} != {expected_episodes}")
        full = [n for n in names if out["data"][n].attrs["observability"] == "full"]
        partial = [n for n in names if out["data"][n].attrs["observability"] == "partial"]
        if (len(full), len(partial)) != (expected_full, expected_partial):
            raise RuntimeError("Full/Partial count mismatch")
        image_arrays = 0
        total_frames = 0
        min_std = float("inf")
        for name in names:
            demo = out["data"][name]
            length = len(demo["actions"])
            total_frames += length
            for key in CAMERA_KEYS:
                array = demo["obs"][key]
                if array.shape != (length, height, width, 3) or array.dtype != np.uint8:
                    raise RuntimeError(f"{name}/{key}: {array.shape} {array.dtype}")
                # uint8 is intrinsically finite. Check every frame is nonblank.
                for start in range(0, length, 32):
                    values = array[start : start + 32]
                    deviations = np.std(values, axis=(1, 2, 3))
                    min_std = min(min_std, float(deviations.min()))
                    if np.any(deviations <= 1):
                        raise RuntimeError(f"blank image in {name}/{key}")
                image_arrays += 1
        train = set(decode_mask(out["mask/train"][:]))
        valid = set(decode_mask(out["mask/valid"][:]))
        train_pairs = {int(out["data"][n].attrs["pair_id"]) for n in train}
        valid_pairs = {int(out["data"][n].attrs["pair_id"]) for n in valid}
        if train_pairs & valid_pairs:
            raise RuntimeError("train/valid pair leakage")
        all_names = set(decode_mask(out["mask/all"][:]))
        if all_names != set(names):
            raise RuntimeError("all mask does not cover every episode")
        result.update(
            passed=True,
            episodes=len(names),
            full=len(full),
            partial=len(partial),
            train=len(train),
            valid=len(valid),
            pair_isolation=True,
            non_image_exact_parity=True,
            cameras=list(CAMERA_KEYS),
            image_arrays=image_arrays,
            image_shape=[height, width, 3],
            image_dtype="uint8",
            total_frames=total_frames,
            minimum_frame_std=min_std,
        )
    result["sha256"] = sha256(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--expected-full", required=True, type=int)
    parser.add_argument("--expected-partial", required=True, type=int)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    validate(
        args.source.resolve(), args.output.resolve(), args.expected_episodes,
        args.expected_full, args.expected_partial, args.height, args.width,
        args.report.resolve(),
    )


if __name__ == "__main__":
    main()
