import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ["hallucinate", "nonhallucinate", "general"]
TASK_PAIRS = {
    "hall_vs_nonhall": ("hallucinate", "nonhallucinate"),
    "general_vs_nonhall": ("general", "nonhallucinate"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_dir() -> Path:
    return repo_root() / "datasets"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def load_records(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data)}")
    return data


def find_dataset_file(
    dataset_dir: Path,
    class_name: str,
    dataset_name: str,
    threshold: str,
    model_name: str,
) -> Path:
    target_terms = [
        class_name.lower(),
        dataset_name.lower(),
        f"threshold{threshold.lower()}",
        model_slug(model_name).lower(),
    ]
    candidates = []

    for path in dataset_dir.rglob("*.json"):
        lower_name = path.name.lower()
        score = sum(1 for term in target_terms if term in lower_name)
        if score >= 3:
            candidates.append((score, len(lower_name), path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset file for {class_name} under {dataset_dir}. "
            f"Try passing explicit --dataset_dir or rename files to include {target_terms}."
        )

    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    return candidates[0][2]


def record_to_preview(record: list, prompt_field: int) -> dict:
    prompt = str(record[prompt_field]) if len(record) > prompt_field else ""
    gold = str(record[1]) if len(record) > 1 else ""
    wrong = str(record[2]) if len(record) > 2 else ""
    return {
        "prompt_preview": prompt[:180].replace("\n", " "),
        "gold": gold,
        "wrong": wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug which dataset files and samples are selected for each class/task."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(default_dataset_dir()))
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trivia_qa_no_context")
    parser.add_argument("--threshold", type=str, default="1.0")
    parser.add_argument("--eval_samples_per_class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        type=str,
        choices=["hall_vs_nonhall", "general_vs_nonhall", "both", "all_classes"],
        default="both",
    )
    parser.add_argument(
        "--raw_prompt",
        action="store_true",
        help="Use prompt field index 0. Default uses bad-shot prompt field index 5 when present.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Optional path to write sampled debug rows as JSONL.",
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    prompt_field = 0 if args.raw_prompt else 5

    classes_to_check = []
    if args.task == "all_classes":
        classes_to_check = CLASS_NAMES
    elif args.task == "both":
        classes_to_check = ["hallucinate", "nonhallucinate", "general"]
    else:
        pos, neg = TASK_PAIRS[args.task]
        classes_to_check = [pos, neg]

    # Keep order stable while removing duplicates.
    seen = set()
    deduped = []
    for class_name in classes_to_check:
        if class_name not in seen:
            seen.add(class_name)
            deduped.append(class_name)

    rows = []
    print("=== Dataset sampling debug ===")
    print(f"dataset_dir: {dataset_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"threshold: {args.threshold}")
    print(f"model_name: {args.model_name}")
    print(f"task: {args.task}")
    print(f"eval_samples_per_class: {args.eval_samples_per_class}")
    print(f"seed: {args.seed}")
    print("")

    for offset, class_name in enumerate(deduped):
        class_seed = args.seed + offset
        dataset_file = find_dataset_file(
            dataset_dir=dataset_dir,
            class_name=class_name,
            dataset_name=args.dataset_name,
            threshold=args.threshold,
            model_name=args.model_name,
        )
        records = load_records(dataset_file)
        idx = sample_indices(len(records), args.eval_samples_per_class, class_seed)

        print(f"[{class_name}]")
        print(f"  dataset_file: {dataset_file}")
        print(f"  total_records: {len(records)}")
        print(f"  sampled_count: {len(idx)}")
        print(f"  sampled_indices: {idx.tolist()}")

        for sample_rank, i in enumerate(idx.tolist(), start=1):
            preview = record_to_preview(records[i], prompt_field=prompt_field)
            row = {
                "class_name": class_name,
                "class_seed": class_seed,
                "dataset_file": str(dataset_file),
                "sample_rank": sample_rank,
                "sample_index": i,
                **preview,
            }
            rows.append(row)
            print(
                "  "
                f"sample#{sample_rank} idx={i} | "
                f"gold={preview['gold']} | wrong={preview['wrong']} | "
                f"prompt={preview['prompt_preview']}"
            )
        print("")

    if args.output_jsonl:
        out_path = Path(args.output_jsonl).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved debug rows: {out_path}")


if __name__ == "__main__":
    main()