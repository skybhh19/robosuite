"""Export the ToolHang Full/Partial episode labels next to a validated HDF5."""

import argparse
import csv
import hashlib
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-full", type=int, required=True)
    parser.add_argument("--expected-partial", type=int, required=True)
    args = parser.parse_args()

    rows = []
    with h5py.File(args.dataset, "r") as dataset:
        names = sorted(dataset["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        for name in names:
            demo = dataset["data"][name]
            state_id = int(demo.attrs["pair_id"])
            rows.append({
                "ep_idx": int(name.rsplit("_", 1)[1]),
                "demo_id": name,
                "observability": demo.attrs["observability"],
                "source_state_id": state_id,
                "pair_id": state_id,
                "num_steps": len(demo["actions"]),
                "dataset_path": str(args.dataset.resolve()),
            })
    counts = {regime: sum(row["observability"] == regime for row in rows) for regime in ("full", "partial")}
    if counts != {"full": args.expected_full, "partial": args.expected_partial}:
        raise ValueError(f"unexpected observability counts: {counts}")
    if len({row["source_state_id"] for row in rows}) != len(rows):
        raise ValueError("source initial states are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print({"counts": counts, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
