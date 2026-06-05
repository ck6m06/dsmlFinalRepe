"""Post-process intervention result JSON files to add source-based metrics.

This script reads result JSON files that already contain per-sample
`baseline_rows` and `intervention_rows`, then adds:

- source_accuracy
- intervention_accuracy_source
- source_total
- source_true_count
- source_false_count
- source_correct_count
- source_fault_count
- intervention_source_correct_count
- intervention_source_fault_count

It does NOT require rerunning intervention, but it can only work for JSON files
that include per-sample rows. Aggregate layer summary files that only contain
top-level counts cannot be retrofitted with source metrics.

Usage:
    python experiment/augment_result_source_metrics.py \
      --input experiment/type1_results/all239/result239.json \
      --output experiment/type1_results/all239/result239_with_source.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    values = [1.0 if bool(row.get(key, False)) else 0.0 for row in rows]
    return sum(values) / len(values)


def augment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = payload.get("baseline_rows")
    intervention_rows = payload.get("intervention_rows")
    if not isinstance(baseline_rows, list) or not isinstance(intervention_rows, list):
        raise ValueError("payload must contain baseline_rows and intervention_rows lists")
    if len(baseline_rows) != len(intervention_rows):
        raise ValueError("baseline_rows and intervention_rows must have the same length")

    source_rows = [row for row in baseline_rows if bool(row.get("source_correct", False))]
    intervention_source_rows = [i_row for b_row, i_row in zip(baseline_rows, intervention_rows) if bool(b_row.get("source_correct", False))]

    source_total = len(source_rows)
    source_true_count = sum(1 for row in baseline_rows if bool(row.get("source_correct", False)) is True)
    source_false_count = sum(1 for row in baseline_rows if bool(row.get("source_correct", False)) is False)
    source_correct_count = int(sum(1 for row in source_rows if bool(row.get("correct", False))))
    source_fault_count = source_total - source_correct_count
    intervention_source_correct_count = int(sum(1 for row in intervention_source_rows if bool(row.get("correct", False))))
    intervention_source_fault_count = source_total - intervention_source_correct_count

    payload = dict(payload)
    payload["source_accuracy"] = summarize(source_rows, "correct")
    payload["intervention_accuracy_source"] = summarize(intervention_source_rows, "correct")
    payload["source_total"] = source_total
    payload["source_true_count"] = source_true_count
    payload["source_false_count"] = source_false_count
    payload["source_correct_count"] = source_correct_count
    payload["source_fault_count"] = source_fault_count
    payload["intervention_source_correct_count"] = intervention_source_correct_count
    payload["intervention_source_fault_count"] = intervention_source_fault_count
    return payload


def process_file(input_path: Path, output_path: Path | None) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")

    augmented = augment_payload(payload)
    if output_path is None:
        output_path = input_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(augmented, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add source-based metrics to existing intervention result JSON files.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file or directory")
    parser.add_argument("--output", type=str, default="", help="Output JSON file or directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input files when output is omitted")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    if input_path.is_file():
        if output_path is None and not args.overwrite:
            raise ValueError("For a single file, provide --output or --overwrite")
        final_output = output_path if output_path is not None else input_path
        written = process_file(input_path, final_output)
        print(f"saved: {written}")
        return

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for json_path in sorted(input_path.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "baseline_rows" not in payload or "intervention_rows" not in payload:
                skipped += 1
                continue
            target = json_path if output_path is None else output_path / json_path.relative_to(input_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(augment_payload(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            processed += 1
        except Exception as exc:
            print(f"skip: {json_path} ({exc})")
            skipped += 1

    print(f"processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
