#!/usr/bin/env python3
"""Balance or subsample an open-book eval JSON.

By default this keeps all records with `correct == False` and takes the first
N records with `correct == True`, where N is the number of incorrect records.

You can also set `--target_total` to create a fixed-size balanced split. For
example, `--target_total 50` will keep 25 correct and 25 incorrect records in
the test split, and the remaining records will be written to the train split.

The output preserves the original JSON structure when the input is a dict with a
`results` list, while updating summary fields such as `samples`, `correct`, and
`accuracy`.

Example:
    python experiment2/balance_open_book_eval_results.py \
        --input experiment/type2_results/open_book_eval_results.json \
        --output experiment/type2_results/open_book_eval_results_balanced.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> tuple[dict[str, Any] | list[Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Expected 'results' to be a list in {path}")
        if not all(isinstance(row, dict) for row in results):
            raise ValueError(f"Expected every item in {path}['results'] to be an object")
        return payload, results

    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Expected every item in {path} to be an object")
        return payload, payload

    raise ValueError(f"Unsupported JSON format in {path}: {type(payload)}")


def balance_records(
    records: list[dict[str, Any]],
    correct_mode: str,
    seed: int,
    target_total: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    incorrect_records = [row for row in records if row.get("correct") is False]
    correct_records = [row for row in records if row.get("correct") is True]

    if target_total is not None:
        if target_total <= 0:
            raise ValueError("target_total must be positive")
        if target_total % 2 != 0:
            raise ValueError("target_total must be even to balance correct/incorrect equally")
        target_correct = target_total // 2
        target_incorrect = target_total // 2
    else:
        target_correct = len(incorrect_records)
        target_incorrect = len(incorrect_records)

    if len(incorrect_records) < target_incorrect:
        raise ValueError(
            f"Not enough incorrect records to sample: need {target_incorrect}, got {len(incorrect_records)}"
        )
    if target_correct == 0:
        raise ValueError("No incorrect records found; nothing to balance against.")
    if len(correct_records) < target_correct:
        raise ValueError(
            f"Not enough correct records to balance: need {target_correct}, got {len(correct_records)}"
        )

    if correct_mode == "first":
        selected_correct = correct_records[:target_correct]
    elif correct_mode == "random":
        import random

        rng = random.Random(seed)
        selected_correct = rng.sample(correct_records, target_correct)
    else:
        raise ValueError(f"Unsupported correct_mode: {correct_mode}")

    if correct_mode == "first":
        selected_incorrect = incorrect_records[:target_incorrect]
    elif correct_mode == "random":
        import random

        rng = random.Random(seed + 1)
        selected_incorrect = rng.sample(incorrect_records, target_incorrect)
    else:
        raise ValueError(f"Unsupported correct_mode: {correct_mode}")

    selected_ids = {id(row) for row in selected_correct}
    selected_ids.update(id(row) for row in selected_incorrect)
    train_records = [row for row in records if id(row) not in selected_ids]
    balanced = [*selected_correct, *selected_incorrect]

    summary = {
        "total": len(balanced),
        "correct": len(selected_correct),
        "incorrect": len(selected_incorrect),
        "train_total": len(train_records),
        "source_total": len(records),
    }
    return balanced, train_records, summary


def update_payload(
    payload: dict[str, Any] | list[Any],
    balanced_records: list[dict[str, Any]],
    summary: dict[str, int],
) -> dict[str, Any] | list[Any]:
    if isinstance(payload, dict):
        updated = dict(payload)
        updated["results"] = balanced_records
        updated["samples"] = summary["total"]
        updated["correct"] = summary["correct"]
        updated["incorrect"] = summary["incorrect"]
        updated["accuracy"] = summary["correct"] / summary["total"] if summary["total"] else 0.0
        updated["balanced"] = True
        updated["balanced_to"] = summary["incorrect"]
        return updated

    return balanced_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance open-book eval results by class count.")
    parser.add_argument("--input", type=str, required=True, help="Input open_book_eval_results JSON")
    parser.add_argument("--output", type=str, required=True, help="Output test/balanced JSON")
    parser.add_argument(
        "--train_output",
        type=str,
        default="",
        help="Optional output path for the remaining train split when --target_total is set.",
    )
    parser.add_argument(
        "--correct_mode",
        type=str,
        choices=["first", "random"],
        default="first",
        help="How to choose the matching correct records.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --correct_mode random")
    parser.add_argument(
        "--target_total",
        type=int,
        default=0,
        help="Optional fixed balanced total to sample. Use 0 to match the smaller class count.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    payload, records = load_payload(input_path)
    target_total = args.target_total if args.target_total > 0 else None
    balanced_records, train_records, summary = balance_records(
        records,
        correct_mode=args.correct_mode,
        seed=args.seed,
        target_total=target_total,
    )
    updated_payload = update_payload(payload, balanced_records, summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(updated_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    train_output_path = None
    if args.train_output:
        train_output_path = Path(args.train_output).resolve()
        train_payload = update_payload(payload, train_records, {
            "total": len(train_records),
            "correct": sum(1 for row in train_records if row.get("correct") is True),
            "incorrect": sum(1 for row in train_records if row.get("correct") is False),
            "source_total": summary["source_total"],
        })
        if isinstance(train_payload, dict):
            train_payload["balanced"] = False
            train_payload["split"] = "train"
            train_payload["source_split"] = "remaining"
        with train_output_path.open("w", encoding="utf-8") as handle:
            json.dump(train_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print(f"input_total={summary['source_total']}")
    print(f"balanced_total={summary['total']}")
    print(f"correct={summary['correct']}")
    print(f"incorrect={summary['incorrect']}")
    print(f"train_total={summary['train_total']}")
    if target_total is not None:
        print(f"target_total={target_total}")
    print(f"saved={output_path}")
    if train_output_path is not None:
        print(f"saved_train={train_output_path}")


if __name__ == "__main__":
    main()