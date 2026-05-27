"""Build an open_book_eval_results-style JSON from two readable datasets.

This script does not run model generation. It prepares a payload compatible with
`run_open_book_intervention.py` by merging two dataset files into one `results`
list with fixed labels:

- true dataset rows  -> correct=True
- false dataset rows -> correct=False

Expected downstream-compatible row keys:
- index
- prompt
- reference_answer
- correct

Example:
    python experiment/build_open_book_eval_results_from_readable_datasets.py \
      --true_results datasets/NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json \
      --false_results datasets/GeneralTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json \
      --model_name meta-llama/Llama-3.2-1B-Instruct \
      --output_json experiment/eval_results_type1/open_book_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


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


def pick_rows(rows: list[dict[str, Any]], limit: int, seed: int, sample_mode: str) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if sample_mode == "first":
        return rows[:limit]
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:limit]


def normalize_row(row: dict[str, Any], prompt_field: str, idx: int, assumed_correct: bool, source_name: str) -> dict[str, Any]:
    prompt = str(row.get(prompt_field, "")).strip()
    reference_answer = str(row.get("reference_answer", "")).strip()
    if not prompt:
        raise ValueError(f"Row {idx} missing prompt field: {prompt_field}")
    if not reference_answer:
        raise ValueError(f"Row {idx} missing reference_answer")

    return {
        "index": idx,
        "prompt": prompt,
        "reference_answer": reference_answer,
        "correct": bool(assumed_correct),
        "source_dataset": source_name,
        # Keep optional metadata if present for debugging/traceability.
        "rank_diff": row.get("rank_diff"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build open_book_eval_results.json-style payload from two readable datasets with fixed labels."
    )
    parser.add_argument("--true_results", type=str, required=True, help="Readable dataset treated as correct=True")
    parser.add_argument("--false_results", type=str, required=True, help="Readable dataset treated as correct=False")
    parser.add_argument("--true_prompt_field", type=str, default="prompt")
    parser.add_argument("--false_prompt_field", type=str, default="prompt")
    parser.add_argument("--model_name", type=str, default="", help="Model name to store in output metadata")
    parser.add_argument("--limit_true", type=int, default=0, help="0 means use all true rows")
    parser.add_argument("--limit_false", type=int, default=0, help="0 means use all false rows")
    parser.add_argument(
        "--sample_mode",
        choices=["random", "first"],
        default="random",
        help="How to select rows when limit_true/limit_false are set",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_merged", action="store_true", help="Shuffle merged results after labeling")
    parser.add_argument("--output_json", type=str, required=True, help="Output path for open_book_eval_results-style JSON")
    args = parser.parse_args()

    true_path = Path(args.true_results).resolve()
    false_path = Path(args.false_results).resolve()
    out_path = Path(args.output_json).resolve()

    true_model_name, true_rows = load_rows(true_path)
    false_model_name, false_rows = load_rows(false_path)

    model_name = args.model_name.strip() or (true_model_name or false_model_name or "")

    true_rows = pick_rows(true_rows, args.limit_true, args.seed, args.sample_mode)
    false_rows = pick_rows(false_rows, args.limit_false, args.seed + 1, args.sample_mode)

    merged: list[dict[str, Any]] = []
    for row in true_rows:
        merged.append(row | {"_assumed_correct": True, "_source": str(true_path.name)})
    for row in false_rows:
        merged.append(row | {"_assumed_correct": False, "_source": str(false_path.name)})

    if args.shuffle_merged:
        rng = random.Random(args.seed + 999)
        rng.shuffle(merged)

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(merged, start=1):
        results.append(
            normalize_row(
                row=row,
                prompt_field=args.true_prompt_field if row.get("_assumed_correct") else args.false_prompt_field,
                idx=idx,
                assumed_correct=bool(row.get("_assumed_correct", False)),
                source_name=str(row.get("_source", "unknown")),
            )
        )

    payload = {
        "dataset": {
            "true_results": str(true_path),
            "false_results": str(false_path),
        },
        "model_name": model_name,
        "prompt_field": "prompt",
        "prompt_fields": {
            "true": args.true_prompt_field,
            "false": args.false_prompt_field,
        },
        "sampling": {
            "limit_true": args.limit_true,
            "limit_false": args.limit_false,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "shuffle_merged": bool(args.shuffle_merged),
        },
        "samples": len(results),
        "correct": int(sum(1 for r in results if r.get("correct"))),
        "accuracy": (sum(1 for r in results if r.get("correct")) / len(results)) if results else float("nan"),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"saved: {out_path}")
    print(f"samples: {payload['samples']}")
    print(f"assumed correct: {payload['correct']}")


if __name__ == "__main__":
    main()
