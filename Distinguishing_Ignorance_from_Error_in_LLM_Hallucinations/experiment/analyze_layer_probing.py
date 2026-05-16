import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ["hallucinate", "nonhallucinate", "general"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_results_dir() -> Path:
    return repo_root() / "results"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def build_result_dir(
    results_dir: Path,
    model_name: str,
    dataset_name: str,
    threshold: str,
    concat_answer: bool,
    dataset_size: int,
    non_static: bool,
) -> Path:
    base = (
        results_dir
        / model_slug(model_name)
        / dataset_name
        / threshold
        / f"concat_answer{concat_answer}_size{dataset_size}"
    )
    if non_static:
        base = base / "non_static"
    return base


def vector_filename(vector_type: str, class_name: str) -> str:
    if vector_type == "mlp":
        return f"_all_mlp_vector_{class_name}.npy"
    if vector_type == "attention":
        return f"_all_attention_vector_{class_name}.npy"
    if vector_type == "residual":
        return f"_all_residual_vectors_{class_name}.npy"
    if vector_type == "heads":
        return f"_heads_vectors_{class_name}_no_projection.npy"
    raise ValueError(f"Unsupported vector type: {vector_type}")


def load_three_class_arrays(result_dir: Path, vector_type: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    missing = []

    for class_name in CLASS_NAMES:
        file_path = result_dir / vector_filename(vector_type, class_name)
        if not file_path.exists():
            missing.append(str(file_path))
            continue
        arrays[class_name] = np.load(file_path, allow_pickle=True)

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    return arrays


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def probe_layer_classification(
    arrays: dict[str, np.ndarray],
    task: str = "hall_vs_nonhall",
    max_samples_per_class: int = 500,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train linear probing classifier for each layer with cross-validation.
    Handles class imbalance with balanced class weights.
    
    Args:
        arrays: dict with "hallucinate", "nonhallucinate", "general" arrays
        task: "hall_vs_nonhall" or "gen_vs_nonhall"
        max_samples_per_class: max samples to use per class
        n_splits: number of cross-validation folds
    
    Returns:
        accuracy_per_layer: shape (n_layers,) - mean CV accuracy
        f1_per_layer: shape (n_layers,) - mean CV F1
        accuracy_std: shape (n_layers,) - std of CV accuracy
        f1_std: shape (n_layers,) - std of CV F1
    """
    arr_hall = arrays["hallucinate"]
    arr_nonhall = arrays["nonhallucinate"]
    arr_gen = arrays["general"]

    n_layers = arr_hall.shape[1]
    accuracy_per_layer = []
    f1_per_layer = []
    accuracy_std = []
    f1_std = []

    # Select class pairs based on task
    if task == "hall_vs_nonhall":
        class_a, class_b = arr_hall, arr_nonhall
        label_a, label_b = 0, 1
    elif task == "gen_vs_nonhall":
        class_a, class_b = arr_gen, arr_nonhall
        label_a, label_b = 0, 1
    else:
        raise ValueError(f"Unknown task: {task}")

    # Subsample if too many
    idx_a = np.random.choice(len(class_a), min(max_samples_per_class, len(class_a)), replace=False)
    idx_b = np.random.choice(len(class_b), min(max_samples_per_class, len(class_b)), replace=False)
    class_a = class_a[idx_a]
    class_b = class_b[idx_b]

    print(f"Probing {task}: {len(class_a)} samples + {len(class_b)} samples, {n_splits}-fold CV")

    for layer_idx in range(n_layers):
        # Extract layer representations
        layer_a = class_a[:, layer_idx, :]  # (n_samples_a, ...)
        layer_b = class_b[:, layer_idx, :]  # (n_samples_b, ...)

        # Flatten all dimensions except first
        if layer_a.ndim > 1:
            layer_a = layer_a.reshape(layer_a.shape[0], -1)
            layer_b = layer_b.reshape(layer_b.shape[0], -1)

        # Build dataset
        X = np.vstack([layer_a, layer_b])
        y = np.concatenate([np.full(len(layer_a), label_a), np.full(len(layer_b), label_b)])

        # Use StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        acc_scores = []
        f1_scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Normalize: fit scaler only on train set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train with regularization and class balancing
            clf = LogisticRegression(
                max_iter=1000,
                random_state=42,
                penalty="l2",
                C=0.1,  # Stronger regularization
                class_weight="balanced",  # Handle class imbalance
            )
            clf.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = clf.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        accuracy_per_layer.append(np.mean(acc_scores))
        f1_per_layer.append(np.mean(f1_scores))
        accuracy_std.append(np.std(acc_scores))
        f1_std.append(np.std(f1_scores))

    return (
        np.array(accuracy_per_layer),
        np.array(f1_per_layer),
        np.array(accuracy_std),
        np.array(f1_std),
    )


def probe_per_head_classification(
    arrays: dict[str, np.ndarray],
    task: str = "hall_vs_nonhall",
    max_samples_per_class: int = 500,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train linear probing classifier for each (layer, head) pair with cross-validation.
    Only for heads vector type (4D tensor).
    
    Returns:
        accuracy_per_head: shape (n_layers, n_heads)
        f1_per_head: shape (n_layers, n_heads)
        accuracy_std: shape (n_layers, n_heads)
        f1_std: shape (n_layers, n_heads)
    """
    arr_hall = arrays["hallucinate"]
    arr_nonhall = arrays["nonhallucinate"]
    arr_gen = arrays["general"]

    if arr_hall.ndim != 4:
        raise ValueError(f"Expected 4D shape for heads, got {arr_hall.shape}")

    n_layers = arr_hall.shape[1]
    n_heads = arr_hall.shape[2]

    # Select class pairs
    if task == "hall_vs_nonhall":
        class_a, class_b = arr_hall, arr_nonhall
        label_a, label_b = 0, 1
    elif task == "gen_vs_nonhall":
        class_a, class_b = arr_gen, arr_nonhall
        label_a, label_b = 0, 1
    else:
        raise ValueError(f"Unknown task: {task}")

    # Subsample
    idx_a = np.random.choice(len(class_a), min(max_samples_per_class, len(class_a)), replace=False)
    idx_b = np.random.choice(len(class_b), min(max_samples_per_class, len(class_b)), replace=False)
    class_a = class_a[idx_a]
    class_b = class_b[idx_b]

    print(f"Probing per-head {task}: {len(class_a)} + {len(class_b)} samples, {n_splits}-fold CV")

    accuracy_per_head = np.zeros((n_layers, n_heads))
    f1_per_head = np.zeros((n_layers, n_heads))
    accuracy_std = np.zeros((n_layers, n_heads))
    f1_std = np.zeros((n_layers, n_heads))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Extract (layer, head)
            head_a = class_a[:, layer_idx, head_idx, :]  # (n_samples_a, head_dim)
            head_b = class_b[:, layer_idx, head_idx, :]  # (n_samples_b, head_dim)

            # Build dataset
            X = np.vstack([head_a, head_b])
            y = np.concatenate([np.full(len(head_a), label_a), np.full(len(head_b), label_b)])

            acc_scores = []
            f1_scores = []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Normalize: fit scaler only on train set
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train with regularization and class balancing
                clf = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    penalty="l2",
                    C=0.1,
                    class_weight="balanced",
                )
                clf.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = clf.predict(X_test)
                acc_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred))

            accuracy_per_head[layer_idx, head_idx] = np.mean(acc_scores)
            f1_per_head[layer_idx, head_idx] = np.mean(f1_scores)
            accuracy_std[layer_idx, head_idx] = np.std(acc_scores)
            f1_std[layer_idx, head_idx] = np.std(f1_scores)

    return accuracy_per_head, f1_per_head, accuracy_std, f1_std


def save_layer_probing_plot(
    accuracy: np.ndarray,
    f1: np.ndarray,
    accuracy_std: np.ndarray,
    f1_std: np.ndarray,
    task: str,
    out_file: Path,
) -> None:
    """Save layer-wise probing accuracy/F1 plot with error bars."""
    n_layers = len(accuracy)
    layers = np.arange(n_layers)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.errorbar(layers, accuracy, yerr=accuracy_std, marker="o", linewidth=2, 
                 markersize=6, color="#1f78b4", capsize=3, label="Mean ± Std")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Linear Probing Accuracy (5-fold CV) - {task}")
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.legend()

    ax2.errorbar(layers, f1, yerr=f1_std, marker="s", linewidth=2, 
                 markersize=6, color="#d7301f", capsize=3, label="Mean ± Std")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("F1 Score")
    ax2.set_title(f"Linear Probing F1 (5-fold CV) - {task}")
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def save_head_probing_heatmap(
    accuracy: np.ndarray,
    f1: np.ndarray,
    accuracy_std: np.ndarray,
    f1_std: np.ndarray,
    task: str,
    out_dir: Path,
) -> None:
    """Save per-head probing heatmaps with uncertainty."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Accuracy heatmap
    im1 = axes[0, 0].imshow(accuracy, aspect="auto", cmap="viridis", vmin=0.5, vmax=1)
    axes[0, 0].set_xlabel("Head Index")
    axes[0, 0].set_ylabel("Layer Index")
    axes[0, 0].set_title(f"Linear Probing Accuracy (Mean) - {task}")
    plt.colorbar(im1, ax=axes[0, 0], label="Accuracy")

    # Accuracy std heatmap
    im2 = axes[0, 1].imshow(accuracy_std, aspect="auto", cmap="hot")
    axes[0, 1].set_xlabel("Head Index")
    axes[0, 1].set_ylabel("Layer Index")
    axes[0, 1].set_title(f"Linear Probing Accuracy (Std) - {task}")
    plt.colorbar(im2, ax=axes[0, 1], label="Std")

    # F1 heatmap
    im3 = axes[1, 0].imshow(f1, aspect="auto", cmap="viridis", vmin=0.5, vmax=1)
    axes[1, 0].set_xlabel("Head Index")
    axes[1, 0].set_ylabel("Layer Index")
    axes[1, 0].set_title(f"Linear Probing F1 (Mean) - {task}")
    plt.colorbar(im3, ax=axes[1, 0], label="F1")

    # F1 std heatmap
    im4 = axes[1, 1].imshow(f1_std, aspect="auto", cmap="hot")
    axes[1, 1].set_xlabel("Head Index")
    axes[1, 1].set_ylabel("Layer Index")
    axes[1, 1].set_title(f"Linear Probing F1 (Std) - {task}")
    plt.colorbar(im4, ax=axes[1, 1], label="Std")

    plt.tight_layout()
    out_file = out_dir / f"heads_probing_heatmap_{task}.png"
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear probing analysis for hallucination detection."
    )
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trivia_qa_no_context")
    parser.add_argument("--threshold", type=str, default="1.0")
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--concat_answer", action="store_true")
    parser.add_argument("--non_static", action="store_true")
    parser.add_argument(
        "--vector_type",
        type=str,
        choices=["mlp", "attention", "residual", "heads"],
        default="mlp",
    )
    parser.add_argument("--max_samples_per_class", type=int, default=500)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_root() / "experiment" / "probing_outputs"),
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    target_dir = build_result_dir(
        results_dir=results_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        threshold=args.threshold,
        concat_answer=args.concat_answer,
        dataset_size=args.dataset_size,
        non_static=args.non_static,
    )

    if not target_dir.exists():
        raise FileNotFoundError(f"Target result directory not found: {target_dir}")

    arrays = load_three_class_arrays(target_dir, args.vector_type)

    print(f"target_dir: {target_dir}")
    print(f"vector_type: {args.vector_type}")
    for k, v in arrays.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")

    np.random.seed(42)

    if args.vector_type == "heads":
        print(f"\n{'='*60}")
        print(f"Task 1: Hallucinate vs NonHallucinate (per-head)")
        print(f"{'='*60}")
        acc_h_vs_nh, f1_h_vs_nh, acc_std_h, f1_std_h = probe_per_head_classification(
            arrays, task="hall_vs_nonhall", max_samples_per_class=args.max_samples_per_class, n_splits=5
        )
        
        print(f"Accuracy: Mean={np.mean(acc_h_vs_nh):.4f} ± {np.mean(acc_std_h):.4f}, Max={np.max(acc_h_vs_nh):.4f}, Min={np.min(acc_h_vs_nh):.4f}")
        print(f"F1 Score: Mean={np.mean(f1_h_vs_nh):.4f} ± {np.mean(f1_std_h):.4f}, Max={np.max(f1_h_vs_nh):.4f}, Min={np.min(f1_h_vs_nh):.4f}")
        
        # Find best head
        best_idx = np.unravel_index(np.argmax(acc_h_vs_nh), acc_h_vs_nh.shape)
        print(f"Best head: Layer {best_idx[0]}, Head {best_idx[1]} (Acc={acc_h_vs_nh[best_idx]:.4f} ± {acc_std_h[best_idx]:.4f})")

        print(f"\n{'='*60}")
        print(f"Task 2: General vs NonHallucinate (per-head)")
        print(f"{'='*60}")
        acc_g_vs_nh, f1_g_vs_nh, acc_std_g, f1_std_g = probe_per_head_classification(
            arrays, task="gen_vs_nonhall", max_samples_per_class=args.max_samples_per_class, n_splits=5
        )
        
        print(f"Accuracy: Mean={np.mean(acc_g_vs_nh):.4f} ± {np.mean(acc_std_g):.4f}, Max={np.max(acc_g_vs_nh):.4f}, Min={np.min(acc_g_vs_nh):.4f}")
        print(f"F1 Score: Mean={np.mean(f1_g_vs_nh):.4f} ± {np.mean(f1_std_g):.4f}, Max={np.max(f1_g_vs_nh):.4f}, Min={np.min(f1_g_vs_nh):.4f}")
        
        best_idx = np.unravel_index(np.argmax(acc_g_vs_nh), acc_g_vs_nh.shape)
        print(f"Best head: Layer {best_idx[0]}, Head {best_idx[1]} (Acc={acc_g_vs_nh[best_idx]:.4f} ± {acc_std_g[best_idx]:.4f})")

        # Save heatmaps
        save_head_probing_heatmap(acc_h_vs_nh, f1_h_vs_nh, acc_std_h, f1_std_h, "hall_vs_nonhall", output_dir)
        save_head_probing_heatmap(acc_g_vs_nh, f1_g_vs_nh, acc_std_g, f1_std_g, "gen_vs_nonhall", output_dir)
        print(f"\nSaved heatmaps to {output_dir}/")
    else:
        print(f"\n{'='*60}")
        print(f"Task 1: Hallucinate vs NonHallucinate (per-layer)")
        print(f"{'='*60}")
        acc_h_vs_nh, f1_h_vs_nh, acc_std_h, f1_std_h = probe_layer_classification(
            arrays, task="hall_vs_nonhall", max_samples_per_class=args.max_samples_per_class, n_splits=5
        )
        
        print("\nLayer | Accuracy (Mean±Std) | F1 Score (Mean±Std)")
        print("------|---------------------|-------------------")
        for i in range(len(acc_h_vs_nh)):
            print(f"{i:5d} | {acc_h_vs_nh[i]:8.4f}±{acc_std_h[i]:6.4f} | {f1_h_vs_nh[i]:8.4f}±{f1_std_h[i]:6.4f}")
        
        best_layer = np.argmax(acc_h_vs_nh)
        print(f"\nBest layer: {best_layer} (Acc={acc_h_vs_nh[best_layer]:.4f}±{acc_std_h[best_layer]:.4f}, F1={f1_h_vs_nh[best_layer]:.4f}±{f1_std_h[best_layer]:.4f})")

        print(f"\n{'='*60}")
        print(f"Task 2: General vs NonHallucinate (per-layer)")
        print(f"{'='*60}")
        acc_g_vs_nh, f1_g_vs_nh, acc_std_g, f1_std_g = probe_layer_classification(
            arrays, task="gen_vs_nonhall", max_samples_per_class=args.max_samples_per_class, n_splits=5
        )
        
        print("\nLayer | Accuracy (Mean±Std) | F1 Score (Mean±Std)")
        print("------|---------------------|-------------------")
        for i in range(len(acc_g_vs_nh)):
            print(f"{i:5d} | {acc_g_vs_nh[i]:8.4f}±{acc_std_g[i]:6.4f} | {f1_g_vs_nh[i]:8.4f}±{f1_std_g[i]:6.4f}")
        
        best_layer = np.argmax(acc_g_vs_nh)
        print(f"\nBest layer: {best_layer} (Acc={acc_g_vs_nh[best_layer]:.4f}±{acc_std_g[best_layer]:.4f}, F1={f1_g_vs_nh[best_layer]:.4f}±{f1_std_g[best_layer]:.4f})")

        # Save plots
        out_file_h = output_dir / f"probing_{args.vector_type}_hall_vs_nonhall_concat{args.concat_answer}_nonstatic{args.non_static}.png"
        out_file_g = output_dir / f"probing_{args.vector_type}_gen_vs_nonhall_concat{args.concat_answer}_nonstatic{args.non_static}.png"
        save_layer_probing_plot(acc_h_vs_nh, f1_h_vs_nh, acc_std_h, f1_std_h, "hall_vs_nonhall", out_file_h)
        save_layer_probing_plot(acc_g_vs_nh, f1_g_vs_nh, acc_std_g, f1_std_g, "gen_vs_nonhall", out_file_g)
        print(f"\nSaved plots:")
        print(f"  {out_file_h}")
        print(f"  {out_file_g}")


if __name__ == "__main__":
    main()
