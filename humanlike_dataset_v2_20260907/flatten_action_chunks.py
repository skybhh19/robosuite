"""Convert legacy (T, 10, 8) chunks to robomimic-compatible (T, 80)."""
import argparse
from pathlib import Path

import h5py


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
path = a.root / "dataset_image84_14hz.hdf5"
changed = 0
with h5py.File(path, "r+") as f:
    for demo in f["data"].values():
        source = demo["action_chunks"]
        if source.ndim == 2:
            assert source.shape[1] == 80
            continue
        assert source.ndim == 3 and source.shape[1:] == (10, 8)
        values = source[:].reshape(len(source), 80)
        del demo["action_chunks"]
        demo.create_dataset("action_chunks", data=values, compression="lzf")
        changed += 1
print({"path": str(path), "episodes_changed": changed})
