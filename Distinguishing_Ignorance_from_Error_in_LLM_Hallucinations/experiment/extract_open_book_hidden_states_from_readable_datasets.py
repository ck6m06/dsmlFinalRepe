"""Extract hidden-state vectors from two readable datasets treated as true/false.

This script is a compatibility-oriented variant of
`extract_open_book_hidden_states.py`.

Instead of reading a single eval-results JSON and splitting rows by `correct`,
it reads two readable dataset files and treats:

- `--true_results`  -> exported as the `correct` group
- `--false_results` -> exported as the `incorrect` group

The output schema is intentionally kept identical to the existing extractor so
that downstream scripts such as `compute_open_book_directions.py` and
`run_open_book_intervention.py` do not need to change.

Example:
    python experiment/extract_open_book_hidden_states_from_readable_datasets.py \
        --true_results datasets/NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json \
        --false_results datasets/GeneralTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --output_dir experiment/eval_results_experiment_without_instruction
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make repository-root modules importable when running as
# `python experiment/extract_open_book_hidden_states_from_readable_datasets.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from InfoModelUsingWrapper import InnerStatesUsingWrapper


def load_rows(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        model_name = payload.get("model_name")
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise ValueError("Expected 'results' to be a list in JSON")
        return model_name, rows

    if isinstance(payload, list):
        return None, payload

    raise ValueError(f"Unsupported JSON format: {type(payload)}")


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int, sample_mode: str) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if sample_mode == "first":
        return rows[:limit]
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:limit]


def extract_group_vectors(
    wrapper: InnerStatesUsingWrapper,
    rows: list[dict[str, Any]],
    prompt_field: str,
    tag: str,
) -> dict[str, np.ndarray]:
    all_mlp_vector = []
    all_attention_vector = []
    all_heads_vector = []
    all_residual_vector = []
    used_indices = []

    for idx, row in enumerate(rows, start=1):
        prompt = str(row.get(prompt_field, "")).strip()
        if not prompt:
            continue

        _, _, mlp_vec, attention_vec, heads_vec, residual_vec = wrapper.generate_interactive(
            prompt=prompt,
            paraphraze_prompt=prompt,
        )

        all_mlp_vector.append(mlp_vec)
        all_attention_vector.append(attention_vec)
        all_heads_vector.append(heads_vec)
        all_residual_vector.append(residual_vec)
        used_indices.append(int(row.get("index", idx)))

        if idx % 10 == 0:
            print(f"[{tag}] processed {idx}/{len(rows)}")

    return {
        "all_mlp_vector": np.array(all_mlp_vector),
        "all_attention_vector": np.array(all_attention_vector),
        "all_heads_vector": np.array(all_heads_vector),
        "all_residual_vector": np.array(all_residual_vector),
        "used_indices": np.array(used_indices),
    }


def save_group(output_dir: Path, group_name: str, vectors: dict[str, np.ndarray]) -> None:
    np.save(output_dir / f"all_mlp_vector_{group_name}.npy", vectors["all_mlp_vector"])
    np.save(output_dir / f"all_attention_vector_{group_name}.npy", vectors["all_attention_vector"])
    np.save(output_dir / f"heads_vectors_{group_name}_no_projection.npy", vectors["all_heads_vector"])
    np.save(output_dir / f"all_residual_vectors_{group_name}.npy", vectors["all_residual_vector"])
    np.save(output_dir / f"indices_{group_name}.npy", vectors["used_indices"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hidden-state vectors from two readable datasets and save them as correct/incorrect groups."
    )
    parser.add_argument("--true_results", type=str, required=True, help="Path to the readable dataset treated as correct=True")
    parser.add_argument("--false_results", type=str, required=True, help="Path to the readable dataset treated as correct=False")
    parser.add_argument("--model_name", type=str, default="", help="Model for hidden-state extraction")
    parser.add_argument("--true_prompt_field", type=str, default="prompt", help="Prompt field used in the true dataset")
    parser.add_argument("--false_prompt_field", type=str, default="prompt", help="Prompt field used in the false dataset")
    parser.add_argument("--output_dir", type=str, default="experiment/eval_results_experiment_without_instruction", help="Directory for output .npy files")
    parser.add_argument("--limit_true", type=int, default=1000, help="Optional cap for true samples")
    parser.add_argument("--limit_false", type=int, default=1000, help="Optional cap for false samples")
    parser.add_argument(
        "--sample_mode",
        choices=["random", "first"],
        default="random",
        help="How to select rows when limits are set",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    true_path = Path(args.true_results).resolve()
    false_path = Path(args.false_results).resolve()

    true_model_name, true_rows = load_rows(true_path)
    false_model_name, false_rows = load_rows(false_path)

    model_name = args.model_name.strip() or (true_model_name or false_model_name or "")
    if not model_name:
        raise ValueError("Model name is required. Pass --model_name or include model_name in the JSON payload.")

    true_rows = sample_rows(true_rows, args.limit_true, args.seed, args.sample_mode)
    false_rows = sample_rows(false_rows, args.limit_false, args.seed + 1, args.sample_mode)

    base_output = Path(args.output_dir).resolve() if args.output_dir else (true_path.parent / "open_book_vectors")
    output_dir = base_output / model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"true_results: {true_path}")
    print(f"false_results: {false_path}")
    print(f"model_name: {model_name}")
    print(f"true samples: {len(true_rows)}")
    print(f"false samples: {len(false_rows)}")
    print(f"output_dir: {output_dir}")

    wrapper = InnerStatesUsingWrapper(MODEL_NAME=model_name)

    vectors_true = extract_group_vectors(
        wrapper=wrapper,
        rows=true_rows,
        prompt_field=args.true_prompt_field,
        tag="correct",
    )
    vectors_false = extract_group_vectors(
        wrapper=wrapper,
        rows=false_rows,
        prompt_field=args.false_prompt_field,
        tag="incorrect",
    )

    save_group(output_dir, "correct", vectors_true)
    save_group(output_dir, "incorrect", vectors_false)

    meta = {
        "true_results": str(true_path),
        "false_results": str(false_path),
        "model_name": model_name,
        "true_prompt_field": args.true_prompt_field,
        "false_prompt_field": args.false_prompt_field,
        "prompt_field": args.true_prompt_field if args.true_prompt_field == args.false_prompt_field else args.true_prompt_field,
        "prompt_fields": {
            "correct": args.true_prompt_field,
            "incorrect": args.false_prompt_field,
        },
        "sampling": {
            "limit_true": args.limit_true,
            "limit_false": args.limit_false,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
        },
        "correct_count": int(len(vectors_true["used_indices"])),
        "incorrect_count": int(len(vectors_false["used_indices"])),
        "files": {
            "correct": {
                "mlp": "all_mlp_vector_correct.npy",
                "attention": "all_attention_vector_correct.npy",
                "heads": "heads_vectors_correct_no_projection.npy",
                "residual": "all_residual_vectors_correct.npy",
                "indices": "indices_correct.npy",
            },
            "incorrect": {
                "mlp": "all_mlp_vector_incorrect.npy",
                "attention": "all_attention_vector_incorrect.npy",
                "heads": "heads_vectors_incorrect_no_projection.npy",
                "residual": "all_residual_vectors_incorrect.npy",
                "indices": "indices_incorrect.npy",
            },
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Saved vector files and metadata.json")

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()