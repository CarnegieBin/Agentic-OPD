#!/usr/bin/env python3
"""Create a deterministic Search-R1 train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_NAMES = {
    "searchR1_nq": "nq",
    "searchR1_hotpotqa": "hotpotqa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(input_path: Path, output_dir: Path, seed: int, n_per_source: int) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    data = pd.read_parquet(input_path)
    if "data_source" not in data.columns:
        raise KeyError("train parquet must contain a data_source column")

    counts = data["data_source"].value_counts(dropna=False).to_dict()
    selected_positions: list[int] = []
    selected_counts: dict[str, int] = {}
    source_values = data["data_source"].to_numpy()
    for source, short_name in SOURCE_NAMES.items():
        source_positions = np.flatnonzero(source_values == source)
        if len(source_positions) < n_per_source:
            raise ValueError(
                f"{source} has {len(source_positions)} rows, fewer than requested {n_per_source}"
            )
        source_rng = np.random.default_rng(seed)
        source_rng.shuffle(source_positions)
        selected_positions.extend(source_positions[:n_per_source].tolist())
        selected_counts[short_name] = n_per_source

    selected_mask = np.zeros(len(data), dtype=bool)
    selected_mask[selected_positions] = True
    val = data.iloc[selected_positions].copy()
    train = data.iloc[~selected_mask].copy()

    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    manifest_path = output_dir / "manifest.json"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    manifest = {
        "input_path": str(input_path),
        "seed": seed,
        "n_per_source": n_per_source,
        "input_rows": len(data),
        "train_rows": len(train),
        "val_rows": len(val),
        "input_source_counts": {str(key): int(value) for key, value in counts.items()},
        "train_source_counts": {
            str(key): int(value) for key, value in train["data_source"].value_counts().items()
        },
        "val_source_counts": {
            str(key): int(value) for key, value in val["data_source"].value_counts().items()
        },
        "selected_counts": selected_counts,
        "train_sha256": sha256(train_path),
        "val_sha256": sha256(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-per-source", type=int, default=1000)
    args = parser.parse_args()
    manifest = build_split(args.input, args.output_dir, args.seed, args.n_per_source)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
