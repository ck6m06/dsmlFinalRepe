"""Compute subtraction/addition directions from open-book hidden-state vectors.

This script reads the arrays produced by
`extract_open_book_hidden_states.py`, groups them into `correct` and
`incorrect`, and computes layer-wise direction vectors for each vector type.

Example:
    python experiment/compute_open_book_directions.py \
        --input_dir experiment/type2_results/all445/meta-llama_Llama-3.2-1B-Instruct \
        --method mean_diff

python experiment/compute_open_book_directions.py --input_dir experiment/type2_results/all445/meta-llama_Llama-3.2-1B-Instruct --method mean_diff
                
Supported vector types:
- mlp
- attention
- residual
- heads

For each vector type, the script saves one direction per layer. For `mean_diff`,
the direction is `mean(correct) - mean(incorrect)`. For `pca`, the first
principal component is oriented so that the correct samples have the larger
mean projection when possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


VECTOR_FILE_PATTERNS = {
    "mlp": ("all_mlp_vector_correct.npy", "all_mlp_vector_incorrect.npy"),
    "attention": ("all_attention_vector_correct.npy", "all_attention_vector_incorrect.npy"),
    "residual": ("all_residual_vectors_correct.npy", "all_residual_vectors_incorrect.npy"),
    "heads": ("heads_vectors_correct_no_projection.npy", "heads_vectors_incorrect_no_projection.npy"),
}


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def flatten_layer(arr: np.ndarray, layer_idx: int) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, layer_idx, :]
    if arr.ndim == 4:
        return arr[:, layer_idx, :, :].reshape(arr.shape[0], -1)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def fit_direction_vector(
    positive: np.ndarray,
    negative: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    if positive.size == 0:
        raise ValueError("Positive group is empty")
    if negative.size == 0:
        raise ValueError("Negative group is empty")

    if method == "mean_diff":
        direction = positive.mean(axis=0) - negative.mean(axis=0)
    elif method == "pca":
        pca = PCA(n_components=1, random_state=seed)
        x = np.vstack([positive, negative])
        pca.fit(x)
        direction = pca.components_[0]

        pos_proj = positive @ direction
        neg_proj = negative @ direction
        if pos_proj.mean() < neg_proj.mean():
            direction = -direction
    else:
        raise ValueError(f"Unsupported direction method: {method}")

    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Direction vector has zero norm")
    return direction / norm


def compute_layer_directions(
    correct: np.ndarray,
    incorrect: np.ndarray,
    method: str,
    seed: int,
) -> dict[str, np.ndarray]:
    if correct.ndim not in (3, 4):
        raise ValueError(f"Unsupported correct array shape: {correct.shape}")
    if incorrect.ndim != correct.ndim:
        raise ValueError(f"correct shape {correct.shape} != incorrect shape {incorrect.shape}")

    n_layers = correct.shape[1]
    directions = []
    reversed_directions = []
    norms = []

    for layer_idx in range(n_layers):
        pos = flatten_layer(correct, layer_idx)
        neg = flatten_layer(incorrect, layer_idx)
        direction = fit_direction_vector(pos, neg, method=method, seed=seed)
        directions.append(direction)
        reversed_directions.append(-direction)
        norms.append(float(np.linalg.norm(direction)))

    return {
        "correct_minus_incorrect": np.asarray(directions),
        "incorrect_minus_correct": np.asarray(reversed_directions),
        "norms": np.asarray(norms),
    }


def save_outputs(output_dir: Path, vector_type: str, method: str, payload: dict[str, np.ndarray]) -> None:
    np.save(output_dir / f"direction_{vector_type}_{method}_correct_minus_incorrect.npy", payload["correct_minus_incorrect"])
    np.save(output_dir / f"direction_{vector_type}_{method}_incorrect_minus_correct.npy", payload["incorrect_minus_correct"])
    np.save(output_dir / f"direction_{vector_type}_{method}_norms.npy", payload["norms"])


def parse_vector_types(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(VECTOR_FILE_PATTERNS.keys())
    vector_types = [item.strip().lower() for item in value.split(",") if item.strip()]
    for vector_type in vector_types:
        if vector_type not in VECTOR_FILE_PATTERNS:
            raise ValueError(f"Unsupported vector type: {vector_type}")
    return vector_types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute layer-wise direction vectors from correct vs incorrect open-book hidden states."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the .npy files produced by extract_open_book_hidden_states.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save the computed direction vectors. Defaults to <input_dir>/directions.",
    )
    parser.add_argument(
        "--vector_types",
        type=str,
        default="all",
        help="Comma-separated vector types or 'all'. Supported: mlp, attention, residual, heads.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mean_diff", "pca", "both"],
        default="mean_diff",
        help="Direction fitting method.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (input_dir / "directions")
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_types = parse_vector_types(args.vector_types)
    methods = ["mean_diff", "pca"] if args.method == "both" else [args.method]

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"vector_types: {vector_types}")
    print(f"methods: {methods}")

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "vector_types": vector_types,
        "methods": methods,
        "files": {},
    }

    for vector_type in vector_types:
        correct_name, incorrect_name = VECTOR_FILE_PATTERNS[vector_type]
        correct = load_array(input_dir / correct_name)
        incorrect = load_array(input_dir / incorrect_name)

        print(f"\n[{vector_type}] correct shape={correct.shape}, incorrect shape={incorrect.shape}")

        summary["files"][vector_type] = {
            "correct": correct_name,
            "incorrect": incorrect_name,
            "shape_correct": list(correct.shape),
            "shape_incorrect": list(incorrect.shape),
        }

        for method in methods:
            payload = compute_layer_directions(
                correct=correct,
                incorrect=incorrect,
                method=method,
                seed=args.seed,
            )
            save_outputs(output_dir, vector_type, method, payload)
            summary["files"][vector_type][method] = {
                "correct_minus_incorrect": f"direction_{vector_type}_{method}_correct_minus_incorrect.npy",
                "incorrect_minus_correct": f"direction_{vector_type}_{method}_incorrect_minus_correct.npy",
                "norms": f"direction_{vector_type}_{method}_norms.npy",
            }
            print(
                f"[{vector_type}/{method}] saved layer-wise directions: "
                f"{payload['correct_minus_incorrect'].shape}"
            )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nSaved metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
