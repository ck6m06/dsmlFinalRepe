"""Evaluate open-book prompts with a configurable prompt field.

This script loads a JSON dataset containing a prompt field and
`reference_answer`, runs the specified causal language model on each prompt,
and reports accuracy based on whether the model output contains the reference
answer.

Example:
    python experiment/evaluate_open_book_prompt.py --dataset datasets/HallucinateTrivia_qa_with_context_meta-llama_Llama-3.1-8b-Instruct.json --model_name meta-llama/Llama-3.2-1B-Instruct --limit 10 --output_json experiment/open_book_eval_results.json 
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected each item in {path} to be a JSON object")
    return data


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def answer_matches(prediction: str, reference_answer: str) -> bool:
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference_answer)
    return bool(normalized_reference) and normalized_reference in normalized_prediction


def get_model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            # dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a prompt field's accuracy against reference_answer."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Input JSON dataset file.")
    parser.add_argument("--model_name", type=str, required=True, help="Causal LM to evaluate.")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="full_without_instruct_prompt",
        help="Record field to evaluate, such as generated_context_prompt or without_instruction_prompt.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit. Use 0 for all records.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Optional path to save per-sample results as JSONL.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save the full evaluation results as a single JSON file.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    records = load_records(dataset_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    results: list[dict[str, Any]] = []
    correct = 0

    for index, record in enumerate(records, start=1):
        prompt = str(record.get(args.prompt_field, "")).strip()
        reference_answer = str(record.get("reference_answer", "")).strip()

        if not prompt:
            raise ValueError(f"Record {index} is missing {args.prompt_field}")
        if not reference_answer:
            raise ValueError(f"Record {index} is missing reference_answer")

        prediction = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        is_correct = answer_matches(prediction, reference_answer)
        correct += int(is_correct)

        result = {
            "index": index,
            "prompt": prompt,
            "reference_answer": reference_answer,
            "prediction": prediction,
            "correct": is_correct,
        }
        results.append(result)

        print(
            json.dumps(
                {
                    "index": index,
                    "correct": is_correct,
                    "reference_answer": reference_answer,
                    "prediction": prediction,
                },
                ensure_ascii=False,
            )
        )

    accuracy = correct / len(records) if records else float("nan")
    summary = {
        "dataset": str(dataset_path),
        "model_name": args.model_name,
        "prompt_field": args.prompt_field,
        "samples": len(records),
        "correct": correct,
        "accuracy": accuracy,
    }

    print("\n=== Summary ===")
    print(f"dataset: {dataset_path}")
    print(f"model_name: {args.model_name}")
    print(f"prompt_field: {args.prompt_field}")
    print(f"samples: {len(records)}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.4f}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved: {output_path}")

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **summary,
            "results": results,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()