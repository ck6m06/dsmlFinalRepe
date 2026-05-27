#!/usr/bin/env python3
"""Slice existing hidden-state vectors into train/val/test subsets.

This script expects:
- a split JSON with train_idx / val_idx / test_idx
- an extracted-vector directory containing metadata.json, indices_correct.npy,
  indices_incorrect.npy, and the per-group vector .npy files

It writes three subdirectories under the output dir:
- train/
- val/
- test/

Each split directory contains the same .npy files as the source directory,
but sliced to the selected samples only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_indices(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def build_position_map(indices: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for position, raw_index in enumerate(indices.tolist()):
        mapping[int(raw_index)] = int(position)
    return mapping


def slice_array(array: np.ndarray, positions: list[int]) -> np.ndarray:
    if array.ndim == 0:
        return array
    return array[positions]


def save_group_split(
    source_dir: Path,
    output_dir: Path,
    split_name: str,
    selected_indices: list[int],
    correct_map: dict[int, int],
    incorrect_map: dict[int, int],
) -> None:
    metadata = load_json(source_dir / "metadata.json")

    correct_selected_indices = [idx for idx in selected_indices if idx in correct_map]
    incorrect_selected_indices = [idx for idx in selected_indices if idx in incorrect_map]
    correct_positions = [correct_map[idx] for idx in correct_selected_indices]
    incorrect_positions = [incorrect_map[idx] for idx in incorrect_selected_indices]

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    correct_files = metadata["files"]["correct"]
    incorrect_files = metadata["files"]["incorrect"]

    # Slice correct group arrays.
    for key, file_name in correct_files.items():
        source_path = source_dir / file_name
        if not source_path.exists():
            continue
        array = np.load(source_path, allow_pickle=True)
        sliced = slice_array(array, correct_positions) if key != "indices" else np.asarray(correct_selected_indices, dtype=array.dtype)
        np.save(split_dir / file_name, sliced)

    # Slice incorrect group arrays.
    for key, file_name in incorrect_files.items():
        source_path = source_dir / file_name
        if not source_path.exists():
            continue
        array = np.load(source_path, allow_pickle=True)
        sliced = slice_array(array, incorrect_positions) if key != "indices" else np.asarray(incorrect_selected_indices, dtype=array.dtype)
        np.save(split_dir / file_name, sliced)

    split_meta = {
        "source_dir": str(source_dir),
        "split_name": split_name,
        "selected_count": len(selected_indices),
        "correct_count": len(correct_selected_indices),
        "incorrect_count": len(incorrect_selected_indices),
        "correct_position_map_size": len(correct_map),
        "incorrect_position_map_size": len(incorrect_map),
        "note": "indices_correct.npy / indices_incorrect.npy in each split directory remain aligned to original sample indices.",
    }
    with (split_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(split_meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice extracted hidden-state vectors by train/val/test split.")
    parser.add_argument(
        "--split_json",
        type=str,
        required=True,
        help="Path to open_book_eval_results_splits.json containing train_idx / val_idx / test_idx",
    )
    parser.add_argument(
        "--vector_dir",
        type=str,
        required=True,
        help="Directory containing extracted .npy vectors and metadata.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where train/val/test subdirectories will be written",
    )
    args = parser.parse_args()

    split_payload = load_json(Path(args.split_json))
    train_idx = [int(x) for x in split_payload.get("train_idx", [])]
    val_idx = [int(x) for x in split_payload.get("val_idx", [])]
    test_idx = [int(x) for x in split_payload.get("test_idx", [])]

    source_dir = Path(args.vector_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_indices = load_indices(source_dir / "indices_correct.npy")
    incorrect_indices = load_indices(source_dir / "indices_incorrect.npy")
    correct_map = build_position_map(correct_indices)
    incorrect_map = build_position_map(incorrect_indices)

    for split_name, selected_indices in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        save_group_split(
            source_dir=source_dir,
            output_dir=output_dir,
            split_name=split_name,
            selected_indices=selected_indices,
            correct_map=correct_map,
            incorrect_map=incorrect_map,
        )

    print(
        "saved splits:",
        {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "to",
        str(output_dir),
    )
    print(
        "source distribution:",
        Counter({"correct": len(correct_indices), "incorrect": len(incorrect_indices)}),
    )


if __name__ == "__main__":
    main()
