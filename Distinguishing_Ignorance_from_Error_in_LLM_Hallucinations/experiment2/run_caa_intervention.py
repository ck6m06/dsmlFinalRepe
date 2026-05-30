#!/usr/bin/env python3
"""Run CAA intervention from experiment2 using the LR probe output.

This is a thin launcher around `experiment/run_open_book_intervention.py`.
It sets convenient defaults so you can run the intervention directly from
`experiment2` after training the LR probe.

Default wiring:
- probe summary: `experiment2/caa_lr/open_book_eval_results/summary.json`
- directions dir: `experiment2/caa_lr/open_book_eval_results`
Run one split:
    python experiment2/run_caa_intervention.py --split test --model_name meta-llama/Llama-3.1-8b-Instruct

Run all splits:

python experiment2/run_caa_intervention.py --split all --model_name meta-llama/Llama-3.2-1B-Instruct --probe_summary experiment2/caa_lr/open_book_eval_results/summary.json --directions_dir experiment2/caa_lr/open_book_eval_results --vector_type residual --layer auto --alpha 4.0 --output_json experiment2/caa_lr/open_book_eval_results/intervention_results.json
    python experiment2/run_caa_intervention.py --split all --model_name meta-llama/Llama-3.1-8b-Instruct

The launcher will use the matching `experiment2/datasets/open_book_eval_results_*.json`
file for each split.
You can still override any argument on the command line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.run_open_book_intervention import main as run_intervention_main


def _ensure_default_args(argv: list[str], repo_root: Path) -> list[str]:
    layer_value = ""
    if "--layer" in argv:
        layer_index = argv.index("--layer")
        if layer_index + 1 < len(argv):
            layer_value = argv[layer_index + 1].strip().lower()

    defaults = {
        "--eval_results": str(repo_root / "experiment2" / "datasets" / "open_book_eval_results_test.json"),
        "--directions_dir": str(repo_root / "experiment2" / "caa_lr" / "open_book_eval_results"),
        "--probe_summary": str(repo_root / "experiment2" / "caa_lr" / "open_book_eval_results" / "summary.json"),
        "--vector_type": "residual",
        "--layer": "auto",
        "--layer_search_metric": "delta_accuracy",
        "--layer_search_subset": "incorrect",
        "--alpha": "4.0",
        "--token_position": "last",
        "--temperature": "0.0",
        "--top_p": "1.0",
    }

    present_flags = {arg for arg in argv if arg.startswith("--")}
    if layer_value == "sweep" and "--probe_summary" not in present_flags:
        defaults.pop("--probe_summary", None)
    for flag, value in defaults.items():
        if flag not in present_flags:
            argv.extend([flag, value])
    return argv


def _with_split_suffix(path_text: str, split_name: str) -> str:
    path = Path(path_text)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_{split_name}{path.suffix}"))
    return str(path.with_name(f"{path.name}_{split_name}"))


def _set_or_replace_arg(argv: list[str], flag: str, value: str) -> list[str]:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            argv[index + 1] = value
            return argv
    argv.extend([flag, value])
    return argv


def _run_single_split(argv: list[str], split_name: str, repo_root: Path) -> None:
    split_eval = repo_root / "experiment2" / "datasets" / f"open_book_eval_results_{split_name}.json"
    split_argv = list(argv)
    split_argv = _set_or_replace_arg(split_argv, "--eval_results", str(split_eval))

    if "--output_json" in split_argv:
        idx = split_argv.index("--output_json")
        split_argv[idx + 1] = _with_split_suffix(split_argv[idx + 1], split_name)
    if "--output_jsonl" in split_argv:
        idx = split_argv.index("--output_jsonl")
        split_argv[idx + 1] = _with_split_suffix(split_argv[idx + 1], split_name)

    sys.argv = [sys.argv[0], *split_argv]
    print(f"=== Running split: {split_name} ===")
    run_intervention_main()


def main() -> None:
    repo_root = REPO_ROOT
    split_parser = argparse.ArgumentParser(add_help=False)
    split_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    split_args, remaining_argv = split_parser.parse_known_args(sys.argv[1:])

    argv = _ensure_default_args(remaining_argv, repo_root)

    splits: Iterable[str]
    if split_args.split == "all":
        splits = ("train", "val", "test")
    else:
        splits = (split_args.split,)

    for split_name in splits:
        _run_single_split(argv, split_name, repo_root)


if __name__ == "__main__":
    main()
