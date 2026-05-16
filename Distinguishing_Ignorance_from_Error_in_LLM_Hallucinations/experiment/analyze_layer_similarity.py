import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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


def compute_layer_cosine_similarities(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cosine similarities for each layer.
    Handles both simple (n_samples, n_layers, hidden_dim) and complex shapes 
    like (n_samples, n_layers, n_heads, head_dim) by flattening extra dimensions.
    
    Returns:
    - cos_hall_vs_nonhall: shape (n_layers,)
    - cos_gen_vs_nonhall: shape (n_layers,)
    - n_layers: number of layers
    """
    arr_hall = arrays["hallucinate"]
    arr_nonhall = arrays["nonhallucinate"]
    arr_gen = arrays["general"]

    n_layers = arr_hall.shape[1]
    
    cos_hall_vs_nonhall = []
    cos_gen_vs_nonhall = []

    for layer_idx in range(n_layers):
        # Extract layer representations
        layer_hall = arr_hall[:, layer_idx, :]  # (n_samples_hall, ...)
        layer_nonhall = arr_nonhall[:, layer_idx, :]  # (n_samples_nonhall, ...)
        layer_gen = arr_gen[:, layer_idx, :]  # (n_samples_gen, ...)

        # Flatten all dimensions except first (sample dimension)
        if layer_hall.ndim > 1:
            layer_hall = layer_hall.reshape(layer_hall.shape[0], -1)
            layer_nonhall = layer_nonhall.reshape(layer_nonhall.shape[0], -1)
            layer_gen = layer_gen.reshape(layer_gen.shape[0], -1)

        # Compute mean vectors per class
        mean_hall = np.mean(layer_hall, axis=0, keepdims=True)  # (1, flattened_dim)
        mean_nonhall = np.mean(layer_nonhall, axis=0, keepdims=True)  # (1, flattened_dim)
        mean_gen = np.mean(layer_gen, axis=0, keepdims=True)  # (1, flattened_dim)

        # Compute cosine similarity
        sim_hall_vs_nonhall = cosine_similarity(mean_hall, mean_nonhall)[0, 0]
        sim_gen_vs_nonhall = cosine_similarity(mean_gen, mean_nonhall)[0, 0]

        cos_hall_vs_nonhall.append(sim_hall_vs_nonhall)
        cos_gen_vs_nonhall.append(sim_gen_vs_nonhall)

    return np.array(cos_hall_vs_nonhall), np.array(cos_gen_vs_nonhall), n_layers


def compute_per_head_similarities(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute cosine similarities per (layer, head) for heads vector type.
    Only works for shape (n_samples, n_layers, n_heads, head_dim).
    
    Returns:
    - cos_hall_vs_nonhall: shape (n_layers, n_heads)
    - cos_gen_vs_nonhall: shape (n_layers, n_heads)
    - n_layers, n_heads
    """
    arr_hall = arrays["hallucinate"]  # (n_samples_hall, 16, 32, 64)
    arr_nonhall = arrays["nonhallucinate"]
    arr_gen = arrays["general"]

    if arr_hall.ndim != 4:
        raise ValueError(f"Expected shape (n_samples, n_layers, n_heads, head_dim), got {arr_hall.shape}")

    n_layers = arr_hall.shape[1]
    n_heads = arr_hall.shape[2]
    
    cos_hall_vs_nonhall = np.zeros((n_layers, n_heads))
    cos_gen_vs_nonhall = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Extract (layer, head) representations
            head_hall = arr_hall[:, layer_idx, head_idx, :]  # (n_samples_hall, head_dim)
            head_nonhall = arr_nonhall[:, layer_idx, head_idx, :]  # (n_samples_nonhall, head_dim)
            head_gen = arr_gen[:, layer_idx, head_idx, :]  # (n_samples_gen, head_dim)

            # Compute mean vectors per class
            mean_hall = np.mean(head_hall, axis=0, keepdims=True)  # (1, head_dim)
            mean_nonhall = np.mean(head_nonhall, axis=0, keepdims=True)  # (1, head_dim)
            mean_gen = np.mean(head_gen, axis=0, keepdims=True)  # (1, head_dim)

            # Compute cosine similarity
            sim_hall_vs_nonhall = cosine_similarity(mean_hall, mean_nonhall)[0, 0]
            sim_gen_vs_nonhall = cosine_similarity(mean_gen, mean_nonhall)[0, 0]

            cos_hall_vs_nonhall[layer_idx, head_idx] = sim_hall_vs_nonhall
            cos_gen_vs_nonhall[layer_idx, head_idx] = sim_gen_vs_nonhall

    return cos_hall_vs_nonhall, cos_gen_vs_nonhall, n_layers, n_heads


def save_similarity_plot(
    cos_hall_vs_nonhall: np.ndarray,
    cos_gen_vs_nonhall: np.ndarray,
    out_file: Path,
) -> None:
    """Save layer-wise cosine similarity plot."""
    n_layers = len(cos_hall_vs_nonhall)
    layers = np.arange(n_layers)

    plt.figure(figsize=(10, 6))
    plt.plot(
        layers,
        cos_hall_vs_nonhall,
        marker="o",
        linewidth=2,
        label="cos(hallucinate, nonhallucinate)",
        color="#d7301f",
    )
    plt.plot(
        layers,
        cos_gen_vs_nonhall,
        marker="s",
        linewidth=2,
        label="cos(general, nonhallucinate)",
        color="#1f78b4",
    )
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Layer-wise Cosine Similarity with Nonhallucinate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def save_heatmap(similarity_matrix: np.ndarray, title: str, out_file: Path) -> None:
    """Save heatmap of cosine similarities for (layer, head) pairs."""
    plt.figure(figsize=(14, 6))
    im = plt.imshow(similarity_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, label="Cosine Similarity")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.title(f"{title} - Per (Layer, Head) Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise cosine similarity between classes."
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
        "--output_dir",
        type=str,
        default=str(repo_root() / "experiment" / "layer_similarity_outputs"),
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

    # Special handling for heads
    if args.vector_type == "heads":
        cos_hall_vs_nonhall, cos_gen_vs_nonhall, n_layers, n_heads = compute_per_head_similarities(arrays)
        
        print(f"\n=== Per-Head Cosine Similarity ===")
        print(f"Layers: {n_layers}, Heads per layer: {n_heads}")
        
        # Find best and worst heads
        min_hall_idx = np.unravel_index(np.argmin(cos_hall_vs_nonhall), cos_hall_vs_nonhall.shape)
        max_hall_idx = np.unravel_index(np.argmax(cos_hall_vs_nonhall), cos_hall_vs_nonhall.shape)
        min_gen_idx = np.unravel_index(np.argmin(cos_gen_vs_nonhall), cos_gen_vs_nonhall.shape)
        max_gen_idx = np.unravel_index(np.argmax(cos_gen_vs_nonhall), cos_gen_vs_nonhall.shape)
        
        print(f"\nHallucinate vs NonHallucinate:")
        print(f"  Best (lowest similarity): Layer {min_hall_idx[0]}, Head {min_hall_idx[1]} = {cos_hall_vs_nonhall[min_hall_idx]:.4f}")
        print(f"  Worst (highest similarity): Layer {max_hall_idx[0]}, Head {max_hall_idx[1]} = {cos_hall_vs_nonhall[max_hall_idx]:.4f}")
        
        print(f"\nGeneral vs NonHallucinate:")
        print(f"  Best (lowest similarity): Layer {min_gen_idx[0]}, Head {min_gen_idx[1]} = {cos_gen_vs_nonhall[min_gen_idx]:.4f}")
        print(f"  Worst (highest similarity): Layer {max_gen_idx[0]}, Head {max_gen_idx[1]} = {cos_gen_vs_nonhall[max_gen_idx]:.4f}")
        
        print(f"\nMean similarity across all (layer, head) pairs:")
        print(f"  Hall vs NonHall: {np.mean(cos_hall_vs_nonhall):.4f} ± {np.std(cos_hall_vs_nonhall):.4f}")
        print(f"  Gen vs NonHall: {np.mean(cos_gen_vs_nonhall):.4f} ± {np.std(cos_gen_vs_nonhall):.4f}")
        
        # Save heatmaps
        out_file_hall = output_dir / f"heads_heatmap_hall_vs_nonhall_concat{args.concat_answer}_nonstatic{args.non_static}.png"
        out_file_gen = output_dir / f"heads_heatmap_gen_vs_nonhall_concat{args.concat_answer}_nonstatic{args.non_static}.png"
        save_heatmap(cos_hall_vs_nonhall, "Hallucinate vs NonHallucinate", out_file_hall)
        save_heatmap(cos_gen_vs_nonhall, "General vs NonHallucinate", out_file_gen)
        
        print(f"\nSaved heatmaps:")
        print(f"  {out_file_hall}")
        print(f"  {out_file_gen}")
    else:
        cos_hall_vs_nonhall, cos_gen_vs_nonhall, n_layers = compute_layer_cosine_similarities(arrays)

        print(f"\n=== Layer-wise Cosine Similarity ===")
        print(f"Layers: {n_layers}")
        print("\nLayer | Hall vs NonHall | Gen vs NonHall")
        print("------|-----------------|---------------")
        for i in range(n_layers):
            print(f"{i:5d} | {cos_hall_vs_nonhall[i]:15.4f} | {cos_gen_vs_nonhall[i]:14.4f}")

        print(f"\nMean (Hall vs NonHall): {np.mean(cos_hall_vs_nonhall):.4f}")
        print(f"Mean (Gen vs NonHall): {np.mean(cos_gen_vs_nonhall):.4f}")
        print(f"Std (Hall vs NonHall): {np.std(cos_hall_vs_nonhall):.4f}")
        print(f"Std (Gen vs NonHall): {np.std(cos_gen_vs_nonhall):.4f}")

        out_file = output_dir / f"layer_similarity_{args.vector_type}_concat{args.concat_answer}_nonstatic{args.non_static}.png"
        save_similarity_plot(cos_hall_vs_nonhall, cos_gen_vs_nonhall, out_file)

        print(f"\nSaved plot: {out_file}")


if __name__ == "__main__":
    main()
