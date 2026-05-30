#!/usr/bin/env python3
"""Train a logistic-regression probe on train split hidden states, select the best
layer on val, and evaluate the selected layer on test.

This script treats `correct=False` as the positive class by default, so the learned
direction points toward the samples you would intervene on for CAA.

Inputs:
- split hidden vectors produced by `slice_hidden_vectors_by_split.py`

Outputs:
- per-layer metrics JSON
- best-layer summary JSON
- LR direction arrays for all layers and the selected layer

Usage example:
    python experiment2/train_caa_svm.py \
        --split_root experiment2/hidden_vectors/open_book_eval_results \
        --vector_type all \
        --output_dir experiment2/caa_svm/open_book_eval_results

python experiment2/train_caa_svm.py --split_root experiment2/hidden_vectors/open_book_eval_results --vector_type all --output_dir experiment2/caa_lr/open_book_eval_results --positive_label incorrect --decision_threshold 0.5
        """
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


VECTOR_FILE_PATTERNS = {
    "mlp": ("all_mlp_vector_correct.npy", "all_mlp_vector_incorrect.npy"),
    "attention": ("all_attention_vector_correct.npy", "all_attention_vector_incorrect.npy"),
    "residual": ("all_residual_vectors_correct.npy", "all_residual_vectors_incorrect.npy"),
    "heads": ("heads_vectors_correct_no_projection.npy", "heads_vectors_incorrect_no_projection.npy"),
}


def load_arrays(split_dir: Path, vector_type: str) -> dict[str, np.ndarray]:
    correct_name, incorrect_name = VECTOR_FILE_PATTERNS[vector_type]
    correct = np.load(split_dir / correct_name, allow_pickle=True)
    incorrect = np.load(split_dir / incorrect_name, allow_pickle=True)
    return {"correct": correct, "incorrect": incorrect}


def extract_layer_matrix(arr: np.ndarray, layer_idx: int) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, layer_idx, :]
    if arr.ndim == 4:
        return arr[:, layer_idx, :, :].reshape(arr.shape[0], -1)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def make_xy(arrays: dict[str, np.ndarray], layer_idx: int, positive_class: str = "incorrect") -> tuple[np.ndarray, np.ndarray]:
    if positive_class not in {"correct", "incorrect"}:
        raise ValueError(f"Unsupported positive_class: {positive_class}")
    negative_class = "correct" if positive_class == "incorrect" else "incorrect"
    x_pos = extract_layer_matrix(arrays[positive_class], layer_idx)
    x_neg = extract_layer_matrix(arrays[negative_class], layer_idx)
    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([
        np.ones(len(x_pos), dtype=np.int64),
        np.zeros(len(x_neg), dtype=np.int64),
    ])
    return x, y


def fit_layer_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression, np.ndarray]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        class_weight="balanced",
        random_state=seed,
        max_iter=2000,
        solver="liblinear",
    )
    clf.fit(x_train_scaled, y_train)
    direction = np.asarray(clf.coef_[0], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("LR direction has zero norm")
    direction = direction / norm
    return scaler, clf, direction


def evaluate_layer(
    scaler: StandardScaler,
    clf: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    x_scaled = scaler.transform(x)
    scores = clf.decision_function(x_scaled)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x_scaled)[:, 1]
    else:
        probs = 1.0 / (1.0 + np.exp(-scores))
    preds = (probs >= decision_threshold).astype(np.int64)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "decision_threshold": float(decision_threshold),
        "positive_rate": float(preds.mean()) if len(preds) else 0.0,
        "mean_positive_probability": float(probs.mean()) if len(probs) else 0.0,
    }
    try:
        metrics["auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        metrics["auc"] = None
    return metrics


def load_split(split_root: Path, split_name: str, vector_type: str) -> dict[str, np.ndarray]:
    return load_arrays(split_root / split_name, vector_type)


def split_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(len(arr)) for name, arr in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LR directions on split hidden vectors and select the best layer on val.")
    parser.add_argument("--split_root", type=str, required=True, help="Directory containing train/val/test subdirectories")
    parser.add_argument("--vector_type", type=str, default="residual", choices=list(VECTOR_FILE_PATTERNS.keys()) + ["all"], help="Which vector type to use")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write LR outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive_label",
        type=str,
        default="incorrect",
        choices=["correct", "incorrect"],
        help="Which class should be treated as the positive class for the LR probe.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for deciding whether to intervene.",
    )
    args = parser.parse_args()

    split_root = Path(args.split_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_types = list(VECTOR_FILE_PATTERNS.keys()) if args.vector_type == "all" else [args.vector_type]
    positive_class = args.positive_label

    summary: dict[str, Any] = {
        "split_root": str(split_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "positive_label": args.positive_label,
        "vector_types": vector_types,
        "results": {},
    }

    for vector_type in vector_types:
        train_arrays = load_split(split_root, "train", vector_type)
        val_arrays = load_split(split_root, "val", vector_type)
        test_arrays = load_split(split_root, "test", vector_type)

        train_counts = split_counts(train_arrays)
        val_counts = split_counts(val_arrays)
        test_counts = split_counts(test_arrays)

        n_layers = train_arrays["correct"].shape[1]
        layer_rows: list[dict[str, Any]] = []
        direction_rows: list[np.ndarray] = []

        for layer_idx in range(n_layers):
            x_train, y_train = make_xy(train_arrays, layer_idx, positive_class=positive_class)
            x_val, y_val = make_xy(val_arrays, layer_idx, positive_class=positive_class)

            scaler, clf, direction = fit_layer_lr(x_train, y_train, seed=args.seed)
            train_metrics = evaluate_layer(scaler, clf, x_train, y_train, decision_threshold=args.decision_threshold)
            val_metrics = evaluate_layer(scaler, clf, x_val, y_val, decision_threshold=args.decision_threshold)

            direction_rows.append(direction)
            layer_rows.append(
                {
                    "layer": layer_idx,
                    "direction_norm": float(np.linalg.norm(direction)),
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

        best_layer_idx = max(
            range(len(layer_rows)),
            key=lambda i: (
                layer_rows[i]["val"]["accuracy"],
                layer_rows[i]["val"]["auc"] if layer_rows[i]["val"]["auc"] is not None else -1.0,
            ),
        )

        x_train_best, y_train_best = make_xy(train_arrays, best_layer_idx, positive_class=positive_class)
        x_val_best, y_val_best = make_xy(val_arrays, best_layer_idx, positive_class=positive_class)
        x_test_best, y_test_best = make_xy(test_arrays, best_layer_idx, positive_class=positive_class)
        scaler_best, clf_best, direction_best = fit_layer_lr(x_train_best, y_train_best, seed=args.seed)

        train_best_metrics = evaluate_layer(scaler_best, clf_best, x_train_best, y_train_best, decision_threshold=args.decision_threshold)
        val_best_metrics = evaluate_layer(scaler_best, clf_best, x_val_best, y_val_best, decision_threshold=args.decision_threshold)
        test_best_metrics = evaluate_layer(scaler_best, clf_best, x_test_best, y_test_best, decision_threshold=args.decision_threshold)

        all_directions = np.stack(direction_rows, axis=0)
        np.save(output_dir / f"direction_{vector_type}_lr_all_layers.npy", all_directions)
        np.save(output_dir / f"direction_{vector_type}_lr_best_layer.npy", direction_best)

        vector_dir = output_dir / vector_type
        vector_dir.mkdir(parents=True, exist_ok=True)
        with (vector_dir / "layer_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "vector_type": vector_type,
                    "train_counts": train_counts,
                    "val_counts": val_counts,
                    "test_counts": test_counts,
                    "decision_threshold": args.decision_threshold,
                    "layers": layer_rows,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        with (vector_dir / "best_layer_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "vector_type": vector_type,
                    "best_layer": best_layer_idx,
                    "positive_label": args.positive_label,
                    "decision_threshold": args.decision_threshold,
                    "train": train_best_metrics,
                    "val": val_best_metrics,
                    "test": test_best_metrics,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        with (vector_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "split_root": str(split_root),
                    "vector_type": vector_type,
                    "positive_label": args.positive_label,
                    "decision_threshold": args.decision_threshold,
                    "best_layer": best_layer_idx,
                    "direction_file": f"direction_{vector_type}_lr_best_layer.npy",
                    "all_directions_file": f"direction_{vector_type}_lr_all_layers.npy",
                    "train_counts": train_counts,
                    "val_counts": val_counts,
                    "test_counts": test_counts,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        summary["results"][vector_type] = {
            "best_layer": best_layer_idx,
            "best_layer_metrics": {
                "train": train_best_metrics,
                "val": val_best_metrics,
                "test": test_best_metrics,
            },
            "layer_metrics_file": str(vector_dir / "layer_metrics.json"),
            "best_layer_summary_file": str(vector_dir / "best_layer_summary.json"),
        }

        print(f"[{vector_type}] best_layer={best_layer_idx} val_acc={val_best_metrics['accuracy']:.4f} test_acc={test_best_metrics['accuracy']:.4f}")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
