"""Add no-instruction prompt fields to an existing open-book JSON file.

This script does not call a model. It reads an existing JSON list of records,
reconstructs a prompt that removes the instruction preamble, and writes the
augmented records back out.

Example:
    python experiment/add_without_instruction_prompt.py --input experiment/eval_results_experiment/open_book_eval_results.json --output experiment/eval_results_experiment/open_book_eval_results_with_without_instruction.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_payload(input_path: Path) -> Any:
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(payload: Any, input_path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if "results" not in payload:
            raise ValueError(
                f"Expected {input_path} to contain a top-level 'results' list, got keys: {sorted(payload.keys())}"
            )
        records = payload["results"]
    else:
        raise ValueError(f"Expected a JSON list or object in {input_path}, got {type(payload)}")

    if not isinstance(records, list):
        raise ValueError(f"Expected records to be a list in {input_path}, got {type(records)}")
    if not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected each item in {input_path} to be an object/dict")
    return records


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def answer_mentioned(context: str, answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)
    return bool(normalized_answer) and normalized_answer in normalized_context


def extract_question(prompt_text: str) -> str:
    if "\nanswer:" in prompt_text:
        return prompt_text.split("\nanswer:", 1)[0].strip()
    return prompt_text.strip()


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


def extract_few_shots(record: dict[str, Any], question: str) -> str:
    few_shot_prompt = str(record.get("few_shot_prompt", ""))
    if not few_shot_prompt:
        return ""
    question_pos = few_shot_prompt.rfind(question)
    if question_pos == -1:
        return few_shot_prompt.strip()
    return few_shot_prompt[:question_pos].strip()


def build_without_instruction_prompt(record: dict[str, Any]) -> str:
    context = str(record.get("generated_context", "")).strip()
    prompt_text = str(record.get("prompt", ""))
    question = extract_question(prompt_text)
    answer = str(record.get("reference_answer", "")).strip()
    few_shots = extract_few_shots(record, question)

    sentences = split_sentences(context)
    if sentences:
        chosen_sentence = None
        for sentence in sentences:
            if answer_mentioned(sentence, answer):
                chosen_sentence = sentence
                break
        if chosen_sentence is None:
            chosen_sentence = sentences[0]
        context_block = chosen_sentence.strip()
    else:
        context_block = context

    parts: list[str] = []
    if context_block:
        parts.append(context_block)
    if few_shots:
        parts.append(few_shots)
    parts.append(f"{question}\nanswer:")
    return "\n".join(parts)


def build_full_without_instruction_prompt(record: dict[str, Any]) -> str:
    context = str(record.get("generated_context", "")).strip()
    prompt_text = str(record.get("prompt", ""))
    question = extract_question(prompt_text)
    few_shots = extract_few_shots(record, question)

    parts: list[str] = []
    if context:
        parts.append(context)
    if few_shots:
        parts.append(few_shots)
    parts.append(f"{question}\nanswer:")
    return "\n".join(parts)


def augment_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for record in records:
        augmented_record = dict(record)
        augmented_record["without_instruction_prompt"] = build_without_instruction_prompt(record)
        augmented_record["full_without_instruct_prompt"] = build_full_without_instruction_prompt(record)
        augmented.append(augmented_record)
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add without_instruction_prompt to an existing open-book JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with existing open-book records.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file. Required unless --inplace is used.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of writing a separate output file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick testing. Use 0 to process the whole file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if args.inplace:
        output_path = input_path
    else:
        if not args.output:
            raise SystemExit("--output is required unless --inplace is set")
        output_path = Path(args.output).resolve()

    print(f"Loading records from {input_path}...")
    payload = load_payload(input_path)
    records = load_records(payload, input_path)
    print(f"Loaded {len(records)} records")
    if args.limit and args.limit > 0:
        records = records[: args.limit]
        print(f"Limited to first {len(records)} records")

    if isinstance(payload, dict):
        augmented_payload = dict(payload)
        augmented_payload["results"] = augment_records(records)
    else:
        augmented_payload = augment_records(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(augmented_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
