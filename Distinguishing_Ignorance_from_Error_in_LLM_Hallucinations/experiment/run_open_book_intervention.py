"""Run open-book intervention using a learned direction vector.

This script loads evaluation rows containing `generated_context_prompt` and
`reference_answer`, runs the base model once, then runs the same prompts again
with a hidden-state direction injected at a chosen transformer layer.

The direction is expected to come from
`compute_open_book_directions.py` and is applied as an additive perturbation:

    hidden = hidden + alpha * direction

for the selected token position.

Example:
    python experiment/run_open_book_intervention.py \
        --eval_results experiment/open_book_eval_results.json \
        --directions_dir experiment/eval_results_experiment/meta-llama_Llama-3.2-1B-Instruct/directions \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --vector_type attention \
        --method mean_diff \
        --layer 0 \
        --alpha 4.0

python experiment/run_open_book_intervention.py --eval_results experiment/eval_results_experiment/open_book_eval_results.json --directions_dir experiment/eval_results_experiment/meta-llama_Llama-3.2-1B-Instruct/directions --model_name meta-llama/Llama-3.2-1B-Instruct --vector_type attention --method mean_diff --direction_variant correct_minus_incorrect --layer auto --layer_search_metric delta_accuracy --layer_search_limit 32 --alpha 4.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VECTOR_FILE_PATTERNS = {
    "mlp": "direction_mlp_{method}_{variant}.npy",
    "attention": "direction_attention_{method}_{variant}.npy",
    "residual": "direction_residual_{method}_{variant}.npy",
    "heads": "direction_heads_{method}_{variant}.npy",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise ValueError("Expected 'results' to be a list in eval JSON")
        return rows

    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported eval file format: {type(payload)}")


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
            dtype=torch.float16,
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


def add_direction_hook(
    model,
    layer_idx: int,
    direction: np.ndarray,
    alpha: float,
    token_position: str,
):
    device = get_model_input_device(model)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    layer_module = model.model.layers[layer_idx]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        hidden = hidden.clone()
        direction_cast = direction_t.to(hidden.dtype)
        if token_position == "last":
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * direction_cast
        elif token_position == "all":
            hidden = hidden + alpha * direction_cast.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported token_position: {token_position}")

        if rest is None:
            return hidden
        return (hidden, *rest)

    return layer_module.register_forward_hook(hook)


def load_direction(
    directions_dir: Path,
    vector_type: str,
    method: str,
    variant: str,
) -> np.ndarray:
    file_name = VECTOR_FILE_PATTERNS[vector_type].format(method=method, variant=variant)
    direction_path = directions_dir / file_name
    if not direction_path.exists():
        raise FileNotFoundError(direction_path)

    directions = np.load(direction_path, allow_pickle=True)
    if directions.ndim < 2:
        raise ValueError(f"Expected layer-wise directions with ndim >= 2, got {directions.shape}")
    return directions


def load_lr_probe(directions_dir: Path, vector_type: str) -> tuple[np.ndarray, int]:
    summary_path = directions_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    results = summary.get("results", {}) if isinstance(summary, dict) else {}
    vector_summary = results.get(vector_type)
    if not isinstance(vector_summary, dict):
        raise ValueError(f"Missing vector_type '{vector_type}' in {summary_path}")

    best_layer = int(vector_summary["best_layer"])
    direction_path = directions_dir / f"direction_{vector_type}_lr_best_layer.npy"
    if not direction_path.exists():
        raise FileNotFoundError(direction_path)

    direction = np.load(direction_path, allow_pickle=True)
    if direction.ndim != 1:
        raise ValueError(f"Expected a 1D direction vector, got {direction.shape}")
    return direction, best_layer


def summarize(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

def build_combined_rows(
    baseline_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    combined_rows = []

    for baseline_row, intervention_row in zip(
        baseline_rows,
        intervention_rows,
    ):
        combined_rows.append(
            {
                "index": baseline_row["index"],

                # source
                "source_correct": baseline_row["source_correct"],

                # baseline
                "baseline_response": baseline_row["response"],
                "baseline_correct": baseline_row["correct"],

                # intervention
                "intervention_response": intervention_row["response"],
                "intervention_correct": intervention_row["correct"],

                # metadata
                "prompt": baseline_row["prompt"],
                "reference_answer": baseline_row["reference_answer"],
            }
        )

    return combined_rows

def build_run_summary(
    baseline_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    layer_idx: int,
    alpha_value: float,
) -> dict[str, Any]:

    combined_rows = build_combined_rows(baseline_rows, intervention_rows)

    # source true false 各別的 rows
    source_true_rows = [r for r in combined_rows if bool(r.get("source_correct", False))]
    source_false_rows = [r for r in combined_rows if not bool(r.get("source_correct", False))]
    # baseline true false 各別的 rows
    baseline_true_rows= [r for r in combined_rows if bool(r.get("baseline_correct", False))]
    baseline_false_rows= [r for r in combined_rows if not bool(r.get("baseline_correct", False))]
    

    # accuracy    
    source_accuracy = summarize(combined_rows, "source_correct")
    baseline_accuracy = summarize(combined_rows, "baseline_correct")
    intervention_accuracy = summarize(combined_rows, "intervention_correct")

    ## ==== source ===
    # source true false 各有幾筆
    source_true_count = int(np.sum([1 if r.get("source_correct") else 0 for r in combined_rows]))
    source_false_count = int(np.sum([1 if not r.get("source_correct") else 0 for r in combined_rows]))

    # source true false 各別的 rows
    source_true_rows = [row for row in combined_rows if bool(row.get("source_correct", False))]
    source_false_rows = [row for row in combined_rows if not bool(row.get("source_correct", False))]

    # acc for source true false 各自的 intervention
    intervention_accuracy_source_true = summarize(source_true_rows, "intervention_correct") if source_true_rows else float("nan")
    intervention_accuracy_source_false = summarize(source_false_rows, "intervention_correct") if source_false_rows else float("nan")

    ## ===== baseline ======
    # baseline true false 各有幾筆
    baseline_true_count = int(np.sum([1 if r.get("baseline_correct") else 0 for r in combined_rows]))
    baseline_false_count = int(np.sum([1 if not r.get("baseline_correct") else 0 for r in combined_rows]))
    
    # baseline true false 各別的 rows
    baseline_true_rows = [row for row in combined_rows if bool(row.get("baseline_correct", False))]
    baseline_false_rows = [row for row in combined_rows if not bool(row.get("baseline_correct", False))]

    # acc for baseline true false 各自的 intervention
    intervention_accuracy_baseline_true = summarize(baseline_true_rows, "intervention_correct") if baseline_true_rows else float("nan")
    intervention_accuracy_baseline_false = summarize(baseline_false_rows, "intervention_correct") if baseline_false_rows else float("nan")

    # ==== intervention count ====
    # intervention true false 各有幾筆
    intervention_true_count = int(np.sum([1 if r.get("intervention_correct") else 0 for r in combined_rows]))
    intervention_false_count = int(np.sum([1 if not r.get("intervention_correct") else 0 for r in combined_rows]))



    return {
        "layer": layer_idx,
        "alpha": alpha_value,
        "source_accuracy": source_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "intervention_accuracy": intervention_accuracy,
        "source_true_count":source_true_count,
        "source_false_count": source_false_count,
        "intervention_true_count": intervention_true_count,
        "intervention_false_count": intervention_false_count,
        "intervention_accuracy_source_true": intervention_accuracy_source_true,
        "intervention_accuracy_source_false": intervention_accuracy_source_false,
        
        "baseline_true_count": baseline_true_count,
        "baseline_false_count": baseline_false_count,
        "intervention_accuracy_baseline_true": intervention_accuracy_baseline_true,
        "intervention_accuracy_baseline_false": intervention_accuracy_baseline_false,
        # "combined_rows": combined_rows,

    }


def run_generation_rows(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    prompt_field: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    direction: np.ndarray | None = None,
    layer_idx: int | None = None,
    alpha: float = 0.0,
    token_position: str = "last",
    progress_label: str = "",
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hook_handle = None
    total_records = len(records)

    if progress_label:
        print(f"[{progress_label}] start: {total_records} samples")

    if direction is not None:
        if layer_idx is None:
            raise ValueError("layer_idx is required when direction is provided")
        hook_handle = add_direction_hook(
            model=model,
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            token_position=token_position,
        )

    try:
        for index, record in enumerate(records, start=1):
            prompt = str(record.get(prompt_field, "")).strip()
            reference_answer = str(record.get("reference_answer", "")).strip()
            source_correct = bool(record.get("correct", False))

            if not prompt:
                raise ValueError(f"Record {index} is missing {prompt_field}")
            if not reference_answer:
                raise ValueError(f"Record {index} is missing reference_answer")

            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            rows.append(
                {
                    "index": index,
                    "source_correct": source_correct,
                    "prompt": prompt,
                    "reference_answer": reference_answer,
                    "response": response,
                    "correct": answer_matches(response, reference_answer),
                }
            )

            if progress_label and (index == 1 or index == total_records or index % max(1, progress_every) == 0):
                print(f"[{progress_label}] processed {index}/{total_records}")
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if progress_label:
        print(f"[{progress_label}] done: {len(rows)}/{total_records}")

    return rows


def score_rows(baseline_rows: list[dict[str, Any]], intervention_rows: list[dict[str, Any]], metric: str) -> float:
    baseline_accuracy = summarize(baseline_rows, "correct")
    intervention_accuracy = summarize(intervention_rows, "correct")
    recovery_rate = float(np.mean([not b["correct"] and i["correct"] for b, i in zip(baseline_rows, intervention_rows)])) if baseline_rows else float("nan")
    regression_rate = float(np.mean([b["correct"] and not i["correct"] for b, i in zip(baseline_rows, intervention_rows)])) if baseline_rows else float("nan")
    baseline_correct_count = int(np.sum([1 if r.get("correct") else 0 for r in baseline_rows]))
    intervention_correct_count = int(np.sum([1 if r.get("correct") else 0 for r in intervention_rows]))

    # Subgroup statistics: split by baseline correctness (baseline == True/False)
    recovered_indices: list[int] = []
    regressed_indices: list[int] = []
    baseline_true_count = 0
    baseline_false_count = 0
    baseline_true_intervention_correct = 0
    baseline_false_intervention_correct = 0

    for b_row, i_row in zip(baseline_rows, intervention_rows):
        b_corr = bool(b_row.get("correct", False))
        i_corr = bool(i_row.get("correct", False))
        idx = int(b_row.get("index", 0))
        if b_corr:
            baseline_true_count += 1
            if not i_corr:
                regressed_indices.append(idx)
            else:
                baseline_true_intervention_correct += 1
        else:
            baseline_false_count += 1
            if i_corr:
                recovered_indices.append(idx)
                baseline_false_intervention_correct += 1

    baseline_true_intervention_accuracy = (
        baseline_true_intervention_correct / baseline_true_count if baseline_true_count > 0 else float("nan")
    )
    baseline_false_intervention_accuracy = (
        baseline_false_intervention_correct / baseline_false_count if baseline_false_count > 0 else float("nan")
    )

    if metric == "delta_accuracy":
        return intervention_accuracy - baseline_accuracy
    if metric == "intervention_accuracy":
        return intervention_accuracy
    if metric == "recovery_minus_regression":
        return recovery_rate - regression_rate
    raise ValueError(f"Unsupported metric: {metric}")


def search_best_layer(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    prompt_field: str,
    direction_stack: np.ndarray,
    alpha: float,
    token_position: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    metric: str,
) -> tuple[int, list[dict[str, Any]], dict[str, float]]:
    baseline_rows = run_generation_rows(
        model=model,
        tokenizer=tokenizer,
        records=records,
        prompt_field=prompt_field,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    best_layer = 0
    best_score = float("-inf")
    best_summary: dict[str, float] = {}

    for layer_idx in range(direction_stack.shape[0]):
        intervention_rows = run_generation_rows(
            model=model,
            tokenizer=tokenizer,
            records=records,
            prompt_field=prompt_field,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            direction=direction_stack[layer_idx],
            layer_idx=layer_idx,
            alpha=alpha,
            token_position=token_position,
        )
        score = score_rows(baseline_rows, intervention_rows, metric=metric)
        summary = build_run_summary(
            baseline_rows=baseline_rows,
            intervention_rows=intervention_rows,
            layer_idx=layer_idx,
            alpha_value=alpha,
        )
        summary["score"] = score
        print(
            json.dumps(
                {
                    "layer": layer_idx,
                    "baseline_accuracy": summary["baseline_accuracy"],
                    "source_accuracy": summary["source_accuracy"],
                    "baseline_accuracy(source)": summary["baseline_accuracy_source"],
                    "baseline_count": f"{summary['baseline_correct_count']}/{summary['baseline_total']}",
                    "intervention_accuracy": summary["intervention_accuracy"],
                    "intervention_accuracy(source)": summary["intervention_accuracy_source"],
                    "intervention_count": f"{summary['intervention_correct_count']}/{summary['intervention_total']}",
                    "delta_accuracy": summary["delta_accuracy"],
                    "recovery_rate": summary["recovery_rate"],
                    "regression_rate": summary["regression_rate"],
                    "score": score,
                },
                ensure_ascii=False,
            )
        )
        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_summary = summary

    return best_layer, baseline_rows, best_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run open-book intervention with a learned direction vector.")
    parser.add_argument("--eval_results", type=str, required=True, help="Path to open_book_eval_results.json")
    parser.add_argument("--model_name", type=str, default="", help="Causal LM to evaluate")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="prompt",
        help="Field containing the open-book prompt.",
    )
    parser.add_argument(
        "--directions_dir",
        type=str,
        default="",
        help="Directory containing direction_*.npy files. Defaults to <eval_results_dir>/directions.",
    )
    parser.add_argument(
        "--probe_summary",
        type=str,
        default="",
        help="Optional summary.json from experiment2/train_caa_svm.py for LR-based intervention.",
    )
    parser.add_argument("--vector_type", type=str, choices=["mlp", "attention", "residual", "heads"], default="attention")
    parser.add_argument("--method", type=str, choices=["mean_diff", "pca"], default="mean_diff")
    parser.add_argument(
        "--direction_variant",
        type=str,
        choices=["correct_minus_incorrect", "incorrect_minus_correct"],
        default="correct_minus_incorrect",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="auto",
        help="Layer index to inject, 'auto' to search for the best layer, or 'sweep' to evaluate every layer.",
    )
    parser.add_argument(
        "--layer_search_metric",
        type=str,
        choices=["delta_accuracy", "intervention_accuracy", "recovery_minus_regression"],
        default="delta_accuracy",
        help="Metric used when --layer auto is enabled.",
    )
    parser.add_argument(
        "--layer_search_limit",
        type=int,
        default=32,
        help="Optional cap on samples used to search for the best layer. Use 0 to search on all samples.",
    )
    parser.add_argument(
        "--layer_search_subset",
        type=str,
        choices=["all", "incorrect", "correct"],
        default="incorrect",
        help="Which subset of records to use when --layer auto: 'incorrect' uses only records with correct==false.",
    )
    parser.add_argument("--alpha", type=float, default=4.0, help="Direction scale factor.")
    parser.add_argument("--token_position", type=str, choices=["last", "all"], default="last")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit. Use 0 for all records.")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print progress every N samples while generating baseline/intervention rows.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save the full intervention results as a single JSON file.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Optional path to save per-sample intervention results as JSONL.",
    )
    parser.add_argument(
        "--extra_test",
        type=str,
        default="",
        help="Optional additional test JSON to run the saved best_layer on (will run baseline+intervention).",
    )
    parser.add_argument(
        "--extra_output_json",
        type=str,
        default="",
        help="Optional path to save the extra test results JSON. Defaults next to extra test file.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_results).resolve()
    records = load_records(eval_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    # print(f"Loaded {len(records)} records from {eval_path}")
    # print(records[0] if records else "No records found in eval results.")

    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    model_name = args.model_name.strip() or (eval_payload.get("model_name") if isinstance(eval_payload, dict) else "")
    if not model_name:
        raise ValueError("Model name is required. Pass --model_name or include model_name in eval JSON.")

    directions_dir = Path(args.directions_dir).resolve() if args.directions_dir else (eval_path.parent / "directions")
    probe_summary_path = Path(args.probe_summary).resolve() if args.probe_summary else None
    if probe_summary_path is not None:
        direction, loaded_best_layer = load_lr_probe(
            directions_dir=directions_dir,
            vector_type=args.vector_type,
        )
        direction_stack = np.expand_dims(direction, axis=0)
    else:
        loaded_best_layer = None
        direction_stack = load_direction(
            directions_dir=directions_dir,
            vector_type=args.vector_type,
            method=args.method,
            variant=args.direction_variant,
        )

    layer_arg = args.layer.strip().lower()
    search_records: list[dict[str, Any]] = []
    if layer_arg in {"auto", "sweep"}:
        # choose subset for layer search
        subset = args.layer_search_subset.lower()
        if subset == "incorrect":
            search_records = [r for r in records if not bool(r.get("correct", False))]
        elif subset == "correct":
            search_records = [r for r in records if bool(r.get("correct", False))]
        else:
            search_records = list(records)

        if args.layer_search_limit and args.layer_search_limit > 0:
            search_records = search_records[: args.layer_search_limit]

    print(f"eval_results: {eval_path}")
    print(f"model_name: {model_name}")
    print(f"directions_dir: {directions_dir}")
    print(f"probe_summary: {probe_summary_path if probe_summary_path is not None else 'none'}")
    print(f"vector_type: {args.vector_type}")
    print(f"method: {args.method}")
    print(f"direction_variant: {args.direction_variant}")
    print(f"layer: {'from_probe' if probe_summary_path is not None else ('auto' if layer_arg == 'auto' else ('sweep' if layer_arg == 'sweep' else args.layer))}")
    print(f"direction_shape: {direction_stack.shape}")
    print(f"samples: {len(records)}")

    model, tokenizer = load_model_and_tokenizer(model_name)

    baseline_rows = run_generation_rows(
        model=model,
        tokenizer=tokenizer,
        records=records,
        prompt_field=args.prompt_field,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        progress_label="baseline",
        progress_every=args.progress_every,
    )

    best_search_summary: dict[str, float] = {}
    layer_sweep_summaries: list[dict[str, Any]] = []
    layer_sweep_rows: dict[tuple[int, float], list[dict[str, Any]]] = {}
    if probe_summary_path is not None:
        best_layer = int(loaded_best_layer)
    elif layer_arg == "sweep":
        train_start_time = time.time()
        print(f"search_records: {len(records)} (full records sweep)")
        sweep_alphas = [1.0, -1.0]
        for layer_idx in range(direction_stack.shape[0]):
            for alpha_sign in sweep_alphas:
                alpha_value = float(alpha_sign)
                intervention_rows = run_generation_rows(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    prompt_field=args.prompt_field,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    direction=direction_stack[layer_idx],
                    layer_idx=layer_idx,
                    alpha=alpha_value,
                    token_position=args.token_position,
                )
                # print("baselineeeeeeeeee")
                # print(baseline_rows[0] if baseline_rows else "No baseline rows")
                run_summary = build_run_summary(
                    baseline_rows=baseline_rows,
                    intervention_rows=intervention_rows,
                    layer_idx=layer_idx,
                    alpha_value=alpha_value,
                )
                layer_sweep_summaries.append(run_summary)
                # layer_sweep_rows[(layer_idx, alpha_value)] = intervention_rows
                combined_rows = build_combined_rows(
                    baseline_rows,
                    intervention_rows,
                )

                layer_sweep_rows[(layer_idx, alpha_value)] = combined_rows
                print(
                    json.dumps(
                        {
                            "layer": layer_idx,
                            "alpha": alpha_value,
                            "source_accuracy": run_summary["source_accuracy"],
                            "baseline_accuracy": run_summary["baseline_accuracy"],
                            "intervention_accuracy": run_summary["intervention_accuracy"],
                            # "delta_accuracy": run_summary["delta_accuracy"],
                            # "recovery_rate": run_summary["recovery_rate"],
                            # "regression_rate": run_summary["regression_rate"],
                            "source_true_count": run_summary["source_true_count"],
                            "source_false_count": run_summary["source_false_count"],
                            "intervention_true_count": run_summary["intervention_true_count"],
                            "intervention_false_count": run_summary["intervention_false_count"],
                            "intervention_accuracy_source_true": run_summary["intervention_accuracy_source_true"],
                            "intervention_accuracy_source_false": run_summary["intervention_accuracy_source_false"],
                            "baseline_true_count": run_summary["baseline_true_count"],
                            "baseline_false_count": run_summary["baseline_false_count"],
                            "intervention_accuracy_baseline_true": run_summary["intervention_accuracy_baseline_true"],
                            "intervention_accuracy_baseline_false": run_summary["intervention_accuracy_baseline_false"],

                            # "baseline_false_intervention_correct": run_summary["baseline_false_intervention_correct"],
                            # "baseline_false_intervention_accuracy": run_summary["baseline_false_intervention_accuracy"],
                            # "baseline_true_count": run_summary["baseline_true_count"],
                            # "baseline_true_intervention_correct": run_summary["baseline_true_intervention_correct"],
                            # "baseline_true_intervention_accuracy": run_summary["baseline_true_intervention_accuracy"],
                            # "recovered_count": run_summary["recovered_count"],
                            # "regressed_count": run_summary["regressed_count"],
                        },
                        ensure_ascii=False,
                    )
                )
        
        train_end_time = time.time()
        print(f"Layer sweep completed in {train_end_time - train_start_time:.2f} seconds")

        # 確認 資料夾
        if args.output_json:
            output_path = Path(args.output_json).resolve()

            sweep_dir = (
                output_path.parent
                / f"{output_path.stem}_layer_summaries"
            )

        for layer_summary in layer_sweep_summaries:

            layer_key = (
                layer_summary["layer"],
                layer_summary["alpha"],
            )

            layer_path = (
                sweep_dir
                / f"layer_{layer_summary['layer']:03d}_alpha_{layer_summary['alpha']:+.0f}.json"
            )

            write_json_file(
                layer_path,
                {
                    "summary": layer_summary,
                    "rows": layer_sweep_rows[layer_key],
                },
            )

            # write_json_file(
            #     sweep_dir / "layer_sweep_summary.json",
            #     {
            #         "runs": layer_sweep_summaries
            #     },
            # )

        best_run_index = int(max(range(len(layer_sweep_summaries)), key=lambda i: layer_sweep_summaries[i]["intervention_accuracy"]))
        best_search_summary = layer_sweep_summaries[best_run_index]
        best_layer = int(best_search_summary["layer"])

        # print(
        #     f"Best layer selected by intervention_accuracy: {best_layer} "
        #     f"(best summary{best_search_summary}, "
        #     f"(intervention_accuracy={best_search_summary.get('intervention_accuracy', float('nan')):.4f})"
        # )

    elif layer_arg == "auto":
        print(f"search_records: {len(search_records)}")
        best_layer, _, best_search_summary = search_best_layer(
            model=model,
            tokenizer=tokenizer,
            records=search_records,
            prompt_field=args.prompt_field,
            direction_stack=direction_stack,
            alpha=args.alpha,
            token_position=args.token_position,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            metric=args.layer_search_metric,
        )
        # print(
        #     f"Best layer selected by {args.layer_search_metric}: {best_layer} "
        #     f"(score={best_search_summary.get('score', float('nan')):.4f})"
        # )
    else:
        best_layer = int(layer_arg)

    # intervention_rows: list[dict[str, Any]] = []
    intervention_direction = direction_stack[0] if probe_summary_path is not None else direction_stack[best_layer]

    # if layer_arg == "sweep":
    #     intervention_rows = layer_sweep_rows[(best_layer, best_search_summary.get("alpha", 1.0))]
    # else:
    #     intervention_rows = run_generation_rows(
    #         model=model,
    #         tokenizer=tokenizer,
    #         records=records,
    #         prompt_field=args.prompt_field,
    #         max_new_tokens=args.max_new_tokens,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #         direction=intervention_direction,
    #         layer_idx=best_layer,
    #         alpha=args.alpha,
    #         token_position=args.token_position,
    #         progress_label="intervention",
    #         progress_every=args.progress_every,
    #     )

    selected_alpha = float(best_search_summary.get("alpha", 1.0)) if layer_arg == "sweep" else float(args.alpha)

    # summary = build_run_summary(
    #     baseline_rows=baseline_rows,
    #     intervention_rows=intervention_rows,
    #     layer_idx=best_layer,
    #     alpha_value=selected_alpha,
    # )
    # summary.update({
    #     "dataset": str(eval_path),
    #     "model_name": model_name,
    #     "directions_dir": str(directions_dir),
    #     "probe_summary": str(probe_summary_path) if probe_summary_path is not None else "",
    #     "vector_type": args.vector_type,
    #     "method": args.method,
    #     "direction_variant": args.direction_variant,
    #     "layer_search_metric": args.layer_search_metric if layer_arg == "auto" else "manual",
    #     "layer_search_limit": args.layer_search_limit if layer_arg == "auto" else 0,
    #     "layer_search_subset": args.layer_search_subset if layer_arg == "auto" else "manual",
    #     "samples": len(records),
    #     # "baseline_rows": baseline_rows,
    #     # "intervention_rows": intervention_rows,
    #     # "layer_search_summary": best_search_summary if layer_arg == "auto" else {},
    #     # "layer_sweep_summaries": layer_sweep_summaries if layer_arg == "sweep" else [],
    #     # "combined_rows": combined_rows,
    # })

    # baseline_accuracy = summary["baseline_accuracy"]
    # source_accuracy = summary["source_accuracy"]
    # baseline_accuracy_source = summary["baseline_accuracy_source"]
    # intervention_accuracy = summary["intervention_accuracy"]
    # intervention_accuracy_source = summary["intervention_accuracy_source"]
    # recovery_rate = summary["recovery_rate"]
    # regression_rate = summary["regression_rate"]
    # baseline_false_count = summary["baseline_false_count"]
    # baseline_false_intervention_correct = summary["baseline_false_intervention_correct"]
    # baseline_false_intervention_accuracy = summary["baseline_false_intervention_accuracy"]
    # baseline_true_count = summary["baseline_true_count"]
    # regressed_indices = summary["regressed_indices"]
    # baseline_true_intervention_accuracy = summary["baseline_true_intervention_accuracy"]
    # source_total = summary["source_total"]
    # source_correct_count = summary["source_correct_count"]
    # intervention_source_correct_count = summary["intervention_source_correct_count"]

    # baseline_accuracy_source_incorrect = summary["baseline_accuracy_source_incorrect"]
    # intervention_accuracy_source_incorrect = summary["intervention_accuracy_source_incorrect"]
    # baseline_vs_source_delta = summary["baseline_vs_source_delta"]
    # intervention_vs_source_delta = summary["intervention_vs_source_delta"]
    # source_incorrect_total = summary["source_incorrect_total"]
    # baseline_source_incorrect_correct_count = summary["baseline_source_incorrect_correct_count"]
    # intervention_source_incorrect_correct_count = summary["intervention_source_incorrect_correct_count"]

    # intervention_vs_source_delta_source_correct = summary["intervention_vs_source_delta_source_correct"]
    # intervention_vs_source_delta_source_incorrect = summary["intervention_vs_source_delta_source_incorrect"]
    total = len(baseline_rows)

    print("\n=== Summary ===")
    print(f"Best Layer: {best_layer}  |  Total Samples: {total} ")
    print()


    if layer_arg == "sweep":
        print(f"sweep_runs: {len(layer_sweep_summaries)}")
        print(f"best_layer_by_intervention_accuracy: {best_layer}")
        print(f"best_alpha: {best_search_summary.get('alpha', 1.0)}")
        # print(f"baseline_false_count: {baseline_false_count} recovered: {baseline_false_intervention_correct} (acc {baseline_false_intervention_accuracy:.4f} if not nan)")
        # print(f"baseline_true_count: {baseline_true_count} regressed: {len(regressed_indices)} (acc after {baseline_true_intervention_accuracy:.4f} if not nan)")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for baseline_row, intervention_row in zip(baseline_rows, intervention_rows):
                handle.write(
                    json.dumps(
                        {
                            "index": baseline_row["index"],
                            "source_correct": baseline_row["source_correct"],
                            "prompt": baseline_row["prompt"],
                            "reference_answer": baseline_row["reference_answer"],
                            "baseline_response": baseline_row["response"],
                            "baseline_correct": baseline_row["correct"],
                            "intervention_response": intervention_row["response"],
                            "intervention_correct": intervention_row["correct"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"saved: {output_path}")

    # if args.output_json:
    #     output_path = Path(args.output_json).resolve()
    #     write_json_file(output_path, summary)
    #     print(f"saved: {output_path}")

    # If provided, run the same best-layer intervention on an extra test JSON
    if args.extra_test:
        extra_eval_path = Path(args.extra_test).resolve()
        print(f"extra_eval_results: {extra_eval_path}")
        extra_records = load_records(extra_eval_path)
        if args.limit and args.limit > 0:
            extra_records = extra_records[: args.limit]

        # Run baseline and intervention on the extra test using the selected best layer
        print(f"Running baseline on extra test: {len(extra_records)} samples")
        baseline_extra = run_generation_rows(
            model=model,
            tokenizer=tokenizer,
            records=extra_records,
            prompt_field=args.prompt_field,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            progress_label="baseline_extra",
            progress_every=args.progress_every,
        )

        print(f"Running intervention on extra test with best_layer={best_layer}")
        print(f"Using alpha={selected_alpha} for extra test intervention")
        test_intervention_time_start = time.time()
        intervention_extra = run_generation_rows(
            model=model,
            tokenizer=tokenizer,
            records=extra_records,
            prompt_field=args.prompt_field,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            direction=intervention_direction,
            layer_idx=best_layer,
            alpha=selected_alpha,
            token_position=args.token_position,
            progress_label="intervention_extra",
            progress_every=args.progress_every,
        )
        test_intervention_time_end = time.time()
        print(f"test intervention completed in {test_intervention_time_end - test_intervention_time_start:.2f} seconds")

        extra_summary = {
            "dataset": str(extra_eval_path),
            "model_name": model_name,
            "directions_dir": str(directions_dir),
            "probe_summary": str(probe_summary_path) if probe_summary_path is not None else "",
            "vector_type": args.vector_type,
            "method": args.method,
            "direction_variant": args.direction_variant,
            "layer": best_layer,
            "alpha": selected_alpha,
            "samples": len(extra_records),
            "source_accuracy": summarize(baseline_extra, "source_correct"),
            "baseline_accuracy": summarize(baseline_extra, "correct"),
            "intervention_accuracy": summarize(intervention_extra, "correct"),
            # "delta_accuracy": summarize(intervention_extra, "correct") - summarize(baseline_extra, "correct"),
            "source_true_count": int(np.sum([1 if r.get("source_correct") else 0 for r in baseline_extra])),
            "source_false_count": int(np.sum([1 if not r.get("source_correct") else 0 for r in baseline_extra])),
            "baseline_correct_count": int(np.sum([1 if r.get("correct") else 0 for r in baseline_extra])),
            "baseline_false_count": int(np.sum([1 if not r.get("correct") else 0 for r in baseline_extra])),
            "intervention_correct_count": int(np.sum([1 if r.get("correct") else 0 for r in intervention_extra])),
            "intervention_false_count": int(np.sum([1 if not r.get("correct") else 0 for r in intervention_extra])),
            "baseline_rows": baseline_extra,
            "intervention_rows": intervention_extra,

        }   

        if args.extra_output_json:
            extra_output_path = Path(args.extra_output_json).resolve()
        else:
            extra_output_path = extra_eval_path.parent / f"{extra_eval_path.stem}_best_layer_{best_layer}_results.json"

        write_json_file(extra_output_path, extra_summary)
        print(f"saved extra test results: {extra_output_path}")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
