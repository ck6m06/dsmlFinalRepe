import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
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


def sample_per_class(arr: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    if arr.shape[0] <= max_per_class:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=max_per_class, replace=False)
    return arr[idx]


def pool_features(arr: np.ndarray, pool_mode: str) -> np.ndarray:
    """Convert one sample tensor into a 2D feature matrix of shape (n_samples, n_features)."""
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)

    if pool_mode == "flatten":
        return arr.reshape(arr.shape[0], -1)

    if arr.ndim < 3:
        return arr.reshape(arr.shape[0], -1)

    if pool_mode == "mean_layers":
        return np.mean(arr, axis=1)
    if pool_mode == "max_layers":
        return np.max(arr, axis=1)
    if pool_mode == "mean_features":
        return np.mean(arr, axis=2)
    if pool_mode == "max_features":
        return np.max(arr, axis=2)

    raise ValueError(
        "Unsupported pool mode: {pool_mode}. Use flatten, mean_layers, max_layers, mean_features, or max_features."
    )


def build_xy(
    arrays: dict[str, np.ndarray],
    max_per_class: int,
    pool_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs = []
    ys = []
    labels = []

    for label_id, class_name in enumerate(CLASS_NAMES):
        arr = sample_per_class(arrays[class_name], max_per_class=max_per_class, seed=seed)
        x = pool_features(arr, pool_mode=pool_mode)
        xs.append(x)
        ys.append(np.full(x.shape[0], label_id, dtype=np.int64))
        labels.extend([class_name] * x.shape[0])

    x_all = np.vstack(xs).astype(np.float32)
    y_all = np.concatenate(ys)
    return x_all, y_all, labels


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pca_scatter(pca_2d: np.ndarray, y: np.ndarray, out_file: Path) -> None:
    colors = ["#d7301f", "#1f78b4", "#33a02c"]
    names = CLASS_NAMES

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(names):
        mask = y == i
        plt.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            s=12,
            alpha=0.65,
            c=colors[i],
            label=class_name,
        )
    plt.title("PCA (2D) of Hidden State Vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def save_explained_variance_plot(explained_ratio: np.ndarray, out_file: Path) -> None:
    cum = np.cumsum(explained_ratio)
    x = np.arange(1, len(explained_ratio) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, cum, marker="o", linewidth=1.8)
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether 3 hidden-state classes are separable by PCA and clustering."
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
    parser.add_argument(
        "--pool",
        type=str,
        choices=["flatten", "mean_layers", "max_layers", "mean_features", "max_features"],
        default="mean_layers",
        help="How to reduce each sample before PCA/KMeans.",
    )
    parser.add_argument("--max_per_class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_root() / "experiment" / "cluster_analysis_outputs"),
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

    x, y, _ = build_xy(
        arrays=arrays,
        max_per_class=args.max_per_class,
        pool_mode=args.pool,
        seed=args.seed,
    )

    print(f"\ncombined feature matrix: {x.shape}")
    print(f"pool mode: {args.pool}")

    if not args.no_scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        print("feature scaling: enabled")
    else:
        print("feature scaling: disabled")

    pca_2 = PCA(n_components=2, random_state=args.seed)
    x_pca_2 = pca_2.fit_transform(x)

    pca_20 = PCA(n_components=min(20, x.shape[1]), random_state=args.seed)
    pca_20.fit(x)

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=args.seed)
    pred = kmeans.fit_predict(x)

    sil = silhouette_score(x, y)
    ari = adjusted_rand_score(y, pred)

    exp_ratio_2 = pca_2.explained_variance_ratio_
    print("\n=== PCA Summary ===")
    print(f"PC1 explained variance ratio: {exp_ratio_2[0]:.4f}")
    print(f"PC2 explained variance ratio: {exp_ratio_2[1]:.4f}")
    print(f"PC1+PC2 total: {(exp_ratio_2[0] + exp_ratio_2[1]):.4f}")

    print("\n=== Clustering Summary ===")
    print(f"Silhouette score (true labels): {sil:.4f}")
    print(f"Adjusted Rand Index (KMeans vs true): {ari:.4f}")

    scatter_file = output_dir / f"pca_scatter_{args.vector_type}_concat{args.concat_answer}_nonstatic{args.non_static}.png"
    var_file = output_dir / f"pca_explained_{args.vector_type}_concat{args.concat_answer}_nonstatic{args.non_static}.png"

    save_pca_scatter(x_pca_2, y, scatter_file)
    save_explained_variance_plot(pca_20.explained_variance_ratio_, var_file)

    print("\nSaved plots:")
    print(scatter_file)
    print(var_file)


if __name__ == "__main__":
    main()
