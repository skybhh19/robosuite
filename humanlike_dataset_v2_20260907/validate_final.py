#!/usr/bin/env python3
"""Validate the final 20 Hz and 14 Hz image datasets."""

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


def decoded(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    source_path = args.root / "dataset_image84.hdf5"
    rate_path = args.root / "dataset_image84_14hz.hdf5"
    result = {
        "passed": False,
        "files": {},
        "mask_counts": {},
        "episodes": 0,
        "steps_20hz": 0,
        "steps_14hz": 0,
    }
    with h5py.File(source_path, "r") as source, h5py.File(rate_path, "r") as rate:
        source_names = set(source["data"])
        rate_names = set(rate["data"])
        assert source_names == rate_names and len(source_names) == 400
        result["episodes"] = len(source_names)
        expected_masks = {
            "all": 400,
            "fully_observable": 200,
            "partially_observable": 200,
            "train": 320,
            "valid": 80,
        }
        for name, expected in expected_masks.items():
            assert name in source["mask"] and name in rate["mask"]
            source_values = decoded(source["mask"][name][:])
            rate_values = decoded(rate["mask"][name][:])
            assert source_values == rate_values
            assert len(source_values) == expected
            assert len(set(source_values)) == expected
            assert set(source_values) <= source_names
            result["mask_counts"][name] = expected
        full = set(decoded(source["mask/fully_observable"][:]))
        partial = set(decoded(source["mask/partially_observable"][:]))
        train = set(decoded(source["mask/train"][:]))
        valid = set(decoded(source["mask/valid"][:]))
        assert not full & partial and full | partial == source_names
        assert not train & valid and train | valid == source_names
        for name in sorted(source_names):
            src = source["data"][name]
            dst = rate["data"][name]
            n_source = len(src["actions"])
            indices = dst["source_step_index"][:]
            assert indices.ndim == 1 and len(indices) == len(dst["actions"])
            assert indices[0] == 0 and indices[-1] == n_source - 1
            assert np.all(np.diff(indices) > 0)
            for key in ("actions", "states", "rewards", "dones"):
                assert len(src[key]) == n_source
                assert len(dst[key]) == len(indices)
                assert np.isfinite(dst[key][:]).all()
            assert np.array_equal(dst["actions"][:], src["actions"][indices])
            assert np.array_equal(dst["states"][:], src["states"][indices])
            assert dst["action_chunks"].shape == (len(indices), 80)
            future = np.minimum(
                np.arange(len(indices))[:, None] + np.arange(10)[None, :], len(indices) - 1
            )
            assert np.array_equal(
                dst["action_chunks"][:],
                dst["actions"][:][future].astype(np.float32).reshape(len(indices), 80),
            )
            assert dst["rewards"][-1] == 1 and dst["dones"][-1] == 1
            for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                src_images = src["obs"][camera]
                dst_images = dst["obs"][camera]
                assert src_images.shape == (n_source, 84, 84, 3)
                assert dst_images.shape == (len(indices), 84, 84, 3)
                assert src_images.dtype == np.uint8 and dst_images.dtype == np.uint8
                # Compare every resampled image without loading both datasets at once.
                for start in range(0, len(indices), 128):
                    stop = min(start + 128, len(indices))
                    assert np.array_equal(dst_images[start:stop], src_images[indices[start:stop]])
            result["steps_20hz"] += n_source
            result["steps_14hz"] += len(indices)
        assert int(source["data"].attrs["total"]) == result["steps_20hz"]
        assert int(rate["data"].attrs["total"]) == result["steps_14hz"]
    result["files"] = {
        source_path.name: {"bytes": source_path.stat().st_size, "sha256": sha256(source_path)},
        rate_path.name: {"bytes": rate_path.stat().st_size, "sha256": sha256(rate_path)},
    }
    result["passed"] = True
    (args.root / "final_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
