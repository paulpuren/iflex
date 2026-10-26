#!/usr/bin/env python3
"""Pack one NS_KT result directory into the HDF5 format used by Interp-DM."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def physical_field(array, resolution):
    array = np.asarray(array)
    if array.shape == (resolution + 5, resolution + 5):
        return array[2 : resolution + 2, 2 : resolution + 2]
    if array.shape == (resolution + 1, resolution + 1):
        return array[:-1, :-1]
    if array.shape == (resolution, resolution):
        return array
    raise ValueError(f"Unexpected solver field shape {array.shape} for resolution {resolution}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--reynolds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=("u", "v", "w"),
        default=("w",),
        help="Fields to persist. Held-out interpolation evaluation only needs w.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=140,
        help=(
            "Keep only the first N snapshots (default: 140, sufficient for the "
            "current 2000-candidate, T=20 evaluation protocol). Use 0 for all."
        ),
    )
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")

    fields = tuple(dict.fromkeys(args.fields))
    paths = {
        name: sorted(
            (args.result_dir / name).glob("field_*.npz"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for name in fields
    }
    available_counts = {name: len(value) for name, value in paths.items()}
    if args.max_frames and any(
        count < args.max_frames for count in available_counts.values()
    ):
        raise RuntimeError(
            f"Requested {args.max_frames} frames, but found {available_counts}"
        )
    if args.max_frames:
        paths = {name: value[: args.max_frames] for name, value in paths.items()}
    counts = {name: len(value) for name, value in paths.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise RuntimeError(f"Incomplete trajectory fields: {counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with h5py.File(partial, "w") as handle:
        handle.attrs.update({
            "reynolds": args.reynolds,
            "seed": args.seed,
            "resolution": args.resolution,
            "stored_fields": json.dumps(fields),
            "stored_frames": next(iter(counts.values())),
            "simulation_config": args.config.read_text(),
        })
        for name in fields:
            dataset = handle.create_dataset(
                name,
                shape=(counts[name], args.resolution, args.resolution),
                dtype="float32",
                chunks=(1, args.resolution, args.resolution),
            )
            for index, path in enumerate(paths[name]):
                with np.load(path) as item:
                    dataset[index] = physical_field(item[name], args.resolution).astype(np.float32)
    partial.replace(args.output)
    print(json.dumps({"output": str(args.output), "frames": next(iter(counts.values())), "fields": counts}))


if __name__ == "__main__":
    main()
