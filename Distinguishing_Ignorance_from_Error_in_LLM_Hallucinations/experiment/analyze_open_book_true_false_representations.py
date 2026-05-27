"""Analyze true/false open-book samples with cosine similarity, entropy, and logit lens.

This script is designed for the two-readable-dataset workflow:

- `--true_results`  -> samples treated as `correct=True`
- `--false_results` -> samples treated as `correct=False`

It reuses the same hidden-state extraction wrapper as the existing pipeline,
then:

1. Computes layer-wise cosine similarity between each sample representation and
   the learned direction vector for the matching vector type.
2. Plots the distribution of those cosine similarities for true vs false
   samples.
3. Computes per-layer entropy from a simple logit lens over the residual stream
   and plots those distributions as well.
4. Saves logit-lens top-k token predictions for each sample and layer.

The output is standalone and does not require changes to downstream scripts.

Example:
    python experiment/analyze_open_book_true_false_representations.py \
      --true_results datasets/NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json \
      --false_results datasets/GeneralTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json \
      --directions_dir experiment/eval_results_type1/meta-llama_Llama-3.2-1B-Instruct/directions \
      --model_name meta-llama/Llama-3.2-1B-Instruct \
      --vector_type attention \
      --method mean_diff \
      --direction_variant correct_minus_incorrect
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make repository-root modules importable when running as
# `python experiment/analyze_open_book_true_false_representations.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from InfoModelUsingWrapper import InnerStatesUsingWrapper


VECTOR_FILE_PATTERNS = {
    "mlp": "direction_mlp_{method}_{variant}.npy",
    "attention": "direction_attention_{method}_{variant}.npy",
    "residual": "direction_residual_{method}_{variant}.npy",
    "heads": "direction_heads_{method}_{variant}.npy",
}


def load_rows(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        model_name = payload.get("model_name")
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise ValueError("Expected 'results' to be a list in JSON payload")
        return model_name, rows

    if isinstance(payload, list):
        return None, payload

    raise ValueError(f"Unsupported JSON payload type: {type(payload)}")


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int, sample_mode: str) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if sample_mode == "first":
        return rows[:limit]
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:limit]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def flatten_hidden(arr: np.ndarray, vector_type: str) -> np.ndarray:
    if vector_type == "heads":
        if arr.ndim != 3:
            raise ValueError(f"Expected heads array with ndim=3, got shape {arr.shape}")
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected vector array with ndim=2, got shape {arr.shape}")
    return arr


def load_direction_stack(directions_dir: Path, vector_type: str, method: str, variant: str) -> np.ndarray:
    file_name = VECTOR_FILE_PATTERNS[vector_type].format(method=method, variant=variant)
    direction_path = directions_dir / file_name
    if not direction_path.exists():
        raise FileNotFoundError(direction_path)

    direction_stack = np.load(direction_path, allow_pickle=True)
    if direction_stack.ndim < 2:
        raise ValueError(f"Expected direction stack with ndim >= 2, got {direction_stack.shape}")
    return direction_stack


def get_hidden_representations(wrapper: InnerStatesUsingWrapper, prompt: str) -> dict[str, np.ndarray]:
    _, _, mlp_vec, attention_vec, heads_vec, residual_vec = wrapper.run_model([prompt], wrapper.model)
    return {
        "mlp": np.asarray(mlp_vec),
        "attention": np.asarray(attention_vec),
        "heads": np.asarray(heads_vec),
        "residual": np.asarray(residual_vec),
    }


def get_output_embeddings(model) -> torch.nn.Module:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None and hasattr(model, "lm_head"):
        output_embeddings = model.lm_head
    if output_embeddings is None:
        raise ValueError("Could not find output embeddings / lm_head on model")
    return output_embeddings


def select_last_token_hidden(hidden: np.ndarray) -> np.ndarray:
    hidden_arr = np.asarray(hidden)
    if hidden_arr.ndim == 1:
        return hidden_arr
    if hidden_arr.ndim == 2:
        return hidden_arr[-1]
    if hidden_arr.ndim >= 3:
        return hidden_arr[0, -1]
    raise ValueError(f"Unsupported hidden shape: {hidden_arr.shape}")


def compute_logit_lens_stats(
    model,
    tokenizer,
    hidden_by_layer: np.ndarray,
    reference_answer: str,
    topk: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    output_embeddings = get_output_embeddings(model)
    answer_token_ids = tokenizer.encode(reference_answer, add_special_tokens=False)
    answer_first_token_id = int(answer_token_ids[0]) if answer_token_ids else None

    entropies = []
    layer_rows: list[dict[str, Any]] = []

    for layer_idx, hidden in enumerate(hidden_by_layer):
        hidden_vec = select_last_token_hidden(hidden)
        hidden_t = torch.tensor(hidden_vec, device=device, dtype=dtype).unsqueeze(0)
        if final_norm is not None:
            hidden_t = final_norm(hidden_t)

        logits = output_embeddings(hidden_t).float()
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = float((-(probs * log_probs)).sum(dim=-1).item())
        entropies.append(entropy)

        top_probs, top_ids = torch.topk(probs, k=topk, dim=-1)
        top_ids_list = top_ids[0].tolist()
        top_probs_list = top_probs[0].tolist()
        top_tokens = [tokenizer.decode([token_id]).replace("\n", "\\n") for token_id in top_ids_list]
        top1_token = top_tokens[0] if top_tokens else ""
        top1_token_id = int(top_ids_list[0]) if top_ids_list else None
        top1_correct = bool(answer_first_token_id is not None and top1_token_id == answer_first_token_id)
        topk_hit = bool(answer_first_token_id is not None and answer_first_token_id in top_ids_list)

        layer_rows.append(
            {
                "layer": layer_idx,
                "entropy": entropy,
                "answer_first_token_id": answer_first_token_id,
                "answer_first_token": tokenizer.decode([answer_first_token_id]).replace("\n", "\\n") if answer_first_token_id is not None else "",
                "top1_token_id": top1_token_id,
                "top1_token": top1_token,
                "top1_correct": top1_correct,
                "topk_hit": topk_hit,
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token": token_text,
                        "prob": float(prob),
                    }
                    for rank, (token_id, token_text, prob) in enumerate(zip(top_ids_list, top_tokens, top_probs_list))
                ],
            }
        )

    return np.asarray(entropies, dtype=float), layer_rows


def attach_tokenizer_to_embeddings(wrapper: InnerStatesUsingWrapper) -> None:
    # Kept for compatibility with the wrapper object lifecycle.
    return None


def make_boxplot_by_layer(
    true_values: list[np.ndarray],
    false_values: list[np.ndarray],
    out_file: Path,
    title: str,
    ylabel: str,
) -> None:
    n_layers = len(true_values[0]) if true_values else len(false_values[0])
    fig, ax = plt.subplots(figsize=(14, 6))

    positions_true = np.arange(n_layers) * 2.0 - 0.2
    positions_false = np.arange(n_layers) * 2.0 + 0.2

    bp_true = ax.boxplot(
        [np.asarray([row[layer_idx] for row in true_values]) for layer_idx in range(n_layers)],
        positions=positions_true,
        widths=0.32,
        patch_artist=True,
        showfliers=False,
    )
    bp_false = ax.boxplot(
        [np.asarray([row[layer_idx] for row in false_values]) for layer_idx in range(n_layers)],
        positions=positions_false,
        widths=0.32,
        patch_artist=True,
        showfliers=False,
    )

    for patch in bp_true["boxes"]:
        patch.set_facecolor("#1b9e77")
        patch.set_alpha(0.35)
    for patch in bp_false["boxes"]:
        patch.set_facecolor("#d95f02")
        patch.set_alpha(0.35)

    ax.set_xticks(np.arange(n_layers) * 2.0)
    ax.set_xticklabels([str(i) for i in range(n_layers)], rotation=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    ax.legend([bp_true["boxes"][0], bp_false["boxes"][0]], ["true", "false"], loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def make_lineplot_by_layer(
    true_values: list[np.ndarray],
    false_values: list[np.ndarray],
    out_file: Path,
    title: str,
    ylabel: str,
) -> None:
    n_layers = len(true_values[0]) if true_values else len(false_values[0])
    true_arr = np.asarray(true_values, dtype=float)
    false_arr = np.asarray(false_values, dtype=float)
    true_mean = np.nanmean(true_arr, axis=0)
    false_mean = np.nanmean(false_arr, axis=0)
    true_std = np.nanstd(true_arr, axis=0)
    false_std = np.nanstd(false_arr, axis=0)

    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, true_mean, color="#1b9e77", marker="o", linewidth=2.0, label="true")
    ax.plot(x, false_mean, color="#d95f02", marker="o", linewidth=2.0, label="false")
    ax.fill_between(x, true_mean - true_std, true_mean + true_std, color="#1b9e77", alpha=0.18)
    ax.fill_between(x, false_mean - false_std, false_mean + false_std, color="#d95f02", alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)], rotation=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cosine similarity, entropy, and logit lens for true/false open-book samples."
    )
    parser.add_argument("--true_results", type=str, required=False, default="", help="Readable dataset treated as correct=True (omit if using --merged_results)")
    parser.add_argument("--false_results", type=str, required=False, default="", help="Readable dataset treated as correct=False (omit if using --merged_results)")
    parser.add_argument(
        "--merged_results",
        type=str,
        required=False,
        default="",
        help="A single JSON containing rows with 'correct': true/false. If provided, this file will be split into true/false groups.",
    )
    parser.add_argument("--directions_dir", type=str, required=True, help="Directory containing direction_*.npy files")
    parser.add_argument("--model_name", type=str, default="", help="Model for hidden-state analysis")
    parser.add_argument("--true_prompt_field", type=str, default="prompt")
    parser.add_argument("--false_prompt_field", type=str, default="prompt")
    parser.add_argument("--vector_type", type=str, choices=["mlp", "attention", "residual", "heads"], default="attention")
    parser.add_argument("--method", type=str, choices=["mean_diff", "pca"], default="mean_diff")
    parser.add_argument(
        "--direction_variant",
        type=str,
        choices=["correct_minus_incorrect", "incorrect_minus_correct"],
        default="correct_minus_incorrect",
    )
    parser.add_argument("--limit_true", type=int, default=0, help="0 means use all true samples")
    parser.add_argument("--limit_false", type=int, default=0, help="0 means use all false samples")
    parser.add_argument(
        "--sample_mode",
        choices=["random", "first"],
        default="first",
        help="How to sample rows when limits are set",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=5, help="Top-k tokens to save for the logit lens")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "experiment" / "representation_analysis_outputs"),
        help="Directory for plots and JSON outputs",
    )
    args = parser.parse_args()

    true_path = Path(args.true_results).resolve() if args.true_results else None
    false_path = Path(args.false_results).resolve() if args.false_results else None
    merged_path = Path(args.merged_results).resolve() if args.merged_results else None
    directions_dir = Path(args.directions_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if merged_path:
        merged_model_name, merged_rows = load_rows(merged_path)
        true_model_name = merged_model_name
        false_model_name = merged_model_name
        true_rows = [r for r in merged_rows if bool(r.get("correct", False))]
        false_rows = [r for r in merged_rows if not bool(r.get("correct", False))]
        # For traceability set the path vars
        true_path = merged_path
        false_path = merged_path
    else:
        if true_path is None or false_path is None:
            raise ValueError("Either --merged_results or both --true_results and --false_results must be provided")
        true_model_name, true_rows = load_rows(true_path)
        false_model_name, false_rows = load_rows(false_path)

    model_name = args.model_name.strip() or (true_model_name or false_model_name or "")
    if not model_name:
        raise ValueError("Model name is required. Pass --model_name or include model_name in the JSON payload.")

    true_rows = sample_rows(true_rows, args.limit_true, args.seed, args.sample_mode)
    false_rows = sample_rows(false_rows, args.limit_false, args.seed + 1, args.sample_mode)

    direction_stack = load_direction_stack(directions_dir, args.vector_type, args.method, args.direction_variant)

    print(f"true_results: {true_path}")
    print(f"false_results: {false_path}")
    print(f"model_name: {model_name}")
    print(f"vector_type: {args.vector_type}")
    print(f"direction_shape: {direction_stack.shape}")
    print(f"true samples: {len(true_rows)}")
    print(f"false samples: {len(false_rows)}")
    print(f"output_dir: {output_dir}")

    wrapper = InnerStatesUsingWrapper(MODEL_NAME=model_name)
    attach_tokenizer_to_embeddings(wrapper)

    true_cosines: list[np.ndarray] = []
    false_cosines: list[np.ndarray] = []
    true_entropies: list[np.ndarray] = []
    false_entropies: list[np.ndarray] = []
    true_top1_accuracies: list[np.ndarray] = []
    false_top1_accuracies: list[np.ndarray] = []
    true_topk_hits: list[np.ndarray] = []
    false_topk_hits: list[np.ndarray] = []
    logit_lens_rows: list[dict[str, Any]] = []

    for group_name, rows, label, cosine_store, entropy_store in [
        ("true", true_rows, True, true_cosines, true_entropies),
        ("false", false_rows, False, false_cosines, false_entropies),
    ]:
        for index, row in enumerate(rows, start=1):
            prompt = str(row.get(args.true_prompt_field if label else args.false_prompt_field, "")).strip()
            reference_answer = str(row.get("reference_answer", "")).strip()
            if not prompt:
                raise ValueError(f"{group_name} record {index} is missing prompt")
            if not reference_answer:
                raise ValueError(f"{group_name} record {index} is missing reference_answer")

            hidden = get_hidden_representations(wrapper, prompt)
            sample_hidden = hidden[args.vector_type]
            sample_hidden_flat = flatten_hidden(sample_hidden, args.vector_type)

            if sample_hidden_flat.shape[0] != direction_stack.shape[0]:
                raise ValueError(
                    f"Layer mismatch for sample {index}: hidden layers {sample_hidden_flat.shape[0]} vs direction layers {direction_stack.shape[0]}"
                )

            cos_by_layer = np.asarray(
                [cosine_similarity(sample_hidden_flat[layer_idx], direction_stack[layer_idx]) for layer_idx in range(sample_hidden_flat.shape[0])],
                dtype=float,
            )
            cosine_store.append(cos_by_layer)

            residual_hidden = hidden["residual"]
            entropy_by_layer, layer_rows = compute_logit_lens_stats(
                wrapper.model,
                wrapper.tok,
                residual_hidden,
                reference_answer,
                topk=args.topk,
            )
            entropy_store.append(entropy_by_layer)

            top1_accuracy_by_layer = np.asarray([1.0 if layer_row["top1_correct"] else 0.0 for layer_row in layer_rows], dtype=float)
            topk_hit_by_layer = np.asarray([1.0 if layer_row["topk_hit"] else 0.0 for layer_row in layer_rows], dtype=float)
            if label:
                true_top1_accuracies.append(top1_accuracy_by_layer)
                true_topk_hits.append(topk_hit_by_layer)
            else:
                false_top1_accuracies.append(top1_accuracy_by_layer)
                false_topk_hits.append(topk_hit_by_layer)

            for layer_row in layer_rows:
                logit_lens_rows.append(
                    {
                        "index": int(row.get("index", index)),
                        "group": group_name,
                        "correct": label,
                        "prompt": prompt,
                        "reference_answer": reference_answer,
                        "layer": layer_row["layer"],
                        "entropy": layer_row["entropy"],
                        "top1_token_id": layer_row["top1_token_id"],
                        "top1_token": layer_row["top1_token"],
                        "top1_correct": layer_row["top1_correct"],
                        "topk_hit": layer_row["topk_hit"],
                        "top_tokens": layer_row["top_tokens"],
                    }
                )

            if index % 10 == 0:
                print(f"[{group_name}] processed {index}/{len(rows)}")

    true_cosines_arr = np.asarray(true_cosines, dtype=float)
    false_cosines_arr = np.asarray(false_cosines, dtype=float)
    true_entropies_arr = np.asarray(true_entropies, dtype=float)
    false_entropies_arr = np.asarray(false_entropies, dtype=float)

    cosine_plot = output_dir / f"cosine_similarity_{args.vector_type}_{args.method}_{args.direction_variant}.png"
    entropy_plot = output_dir / f"entropy_{args.vector_type}_{args.method}_{args.direction_variant}.png"
    logit_lens_accuracy_plot = output_dir / f"logit_lens_accuracy_{args.vector_type}_{args.method}_{args.direction_variant}.png"

    make_boxplot_by_layer(
        true_values=true_cosines,
        false_values=false_cosines,
        out_file=cosine_plot,
        title=f"Layer-wise cosine similarity vs direction ({args.vector_type})",
        ylabel="Cosine similarity",
    )
    make_boxplot_by_layer(
        true_values=true_entropies,
        false_values=false_entropies,
        out_file=entropy_plot,
        title="Layer-wise logit-lens entropy",
        ylabel="Entropy",
    )
    make_lineplot_by_layer(
        true_values=true_top1_accuracies,
        false_values=false_top1_accuracies,
        out_file=logit_lens_accuracy_plot,
        title="Layer-wise logit-lens top-1 accuracy",
        ylabel="Accuracy",
    )

    cosine_summary = {
        "true": {
            "mean_by_layer": np.nanmean(true_cosines_arr, axis=0).tolist(),
            "std_by_layer": np.nanstd(true_cosines_arr, axis=0).tolist(),
        },
        "false": {
            "mean_by_layer": np.nanmean(false_cosines_arr, axis=0).tolist(),
            "std_by_layer": np.nanstd(false_cosines_arr, axis=0).tolist(),
        },
    }
    entropy_summary = {
        "true": {
            "mean_by_layer": np.nanmean(true_entropies_arr, axis=0).tolist(),
            "std_by_layer": np.nanstd(true_entropies_arr, axis=0).tolist(),
        },
        "false": {
            "mean_by_layer": np.nanmean(false_entropies_arr, axis=0).tolist(),
            "std_by_layer": np.nanstd(false_entropies_arr, axis=0).tolist(),
        },
    }
    logit_lens_accuracy_summary = {
        "true": {
            "top1_mean_by_layer": np.nanmean(np.asarray(true_top1_accuracies, dtype=float), axis=0).tolist() if true_top1_accuracies else [],
            "topk_hit_mean_by_layer": np.nanmean(np.asarray(true_topk_hits, dtype=float), axis=0).tolist() if true_topk_hits else [],
        },
        "false": {
            "top1_mean_by_layer": np.nanmean(np.asarray(false_top1_accuracies, dtype=float), axis=0).tolist() if false_top1_accuracies else [],
            "topk_hit_mean_by_layer": np.nanmean(np.asarray(false_topk_hits, dtype=float), axis=0).tolist() if false_topk_hits else [],
        },
    }

    summary = {
        "true_results": str(true_path),
        "false_results": str(false_path),
        "directions_dir": str(directions_dir),
        "model_name": model_name,
        "vector_type": args.vector_type,
        "method": args.method,
        "direction_variant": args.direction_variant,
        "prompt_fields": {
            "true": args.true_prompt_field,
            "false": args.false_prompt_field,
        },
        "sampling": {
            "limit_true": args.limit_true,
            "limit_false": args.limit_false,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
        },
        "counts": {
            "true": int(len(true_rows)),
            "false": int(len(false_rows)),
        },
        "cosine_plot": cosine_plot.name,
        "entropy_plot": entropy_plot.name,
        "logit_lens_accuracy_plot": logit_lens_accuracy_plot.name,
        "cosine_summary": cosine_summary,
        "entropy_summary": entropy_summary,
        "logit_lens_accuracy_summary": logit_lens_accuracy_summary,
        "logit_lens_topk": args.topk,
        "logit_lens_rows": len(logit_lens_rows),
        "logit_lens_file": "logit_lens_predictions.jsonl",
    }

    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (output_dir / "logit_lens_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in logit_lens_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved: {output_dir / 'analysis_summary.json'}")
    print(f"saved: {output_dir / 'logit_lens_predictions.jsonl'}")
    print(f"saved: {cosine_plot}")
    print(f"saved: {entropy_plot}")
    print(f"saved: {logit_lens_accuracy_plot}")

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
