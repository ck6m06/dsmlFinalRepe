"""Analyze vector shifts before vs after intervention at a target layer.

Given a result JSON from run_open_book_intervention.py (or compatible), this script:
1) Loads baseline/intervention per-sample rows.
2) Replays each prompt through the model at the target layer:
   - baseline pass (no hook)
   - intervention pass (direction hook enabled)
3) Extracts last-token hidden vectors at the selected layer and computes:
   - cosine(before, after)
   - L2 shift ||after-before||
   - direction projection shift (after-before projected on direction)
4) Saves per-sample CSV, summary JSON, and visualization PNGs.

Example:
    python experiment/analyze_intervention_vector_shift.py \
      --result_json experiment/type1_results/all239/result239.json \
      --out_dir experiment/plots/result239/vector_shift
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.run_open_book_intervention import (  # noqa: E402
    VECTOR_FILE_PATTERNS,
    add_direction_hook,
    get_model_input_device,
    load_model_and_tokenizer,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _group_name(baseline_correct: bool, intervention_correct: bool) -> str:
    if (not baseline_correct) and intervention_correct:
        return "recovered"
    if baseline_correct and (not intervention_correct):
        return "regressed"
    if baseline_correct and intervention_correct:
        return "stable_true"
    return "stable_false"


def _load_direction(result_payload: dict[str, Any], layer: int) -> np.ndarray:
    directions_dir = Path(str(result_payload["directions_dir"]))
    vector_type = str(result_payload.get("vector_type", "residual"))
    method = str(result_payload.get("method", "mean_diff"))
    variant = str(result_payload.get("direction_variant", "correct_minus_incorrect"))
    file_name = VECTOR_FILE_PATTERNS[vector_type].format(method=method, variant=variant)
    direction_path = directions_dir / file_name
    if not direction_path.exists():
        raise FileNotFoundError(f"Direction file not found: {direction_path}")
    direction_all = np.load(direction_path, allow_pickle=True)
    if direction_all.ndim != 2:
        raise ValueError(f"Expected 2D direction array [layers, hidden], got {direction_all.shape}")
    if layer < 0 or layer >= direction_all.shape[0]:
        raise ValueError(f"Layer {layer} out of range for direction shape {direction_all.shape}")
    return np.asarray(direction_all[layer], dtype=np.float32)


def _extract_layer_vec(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    max_seq_len: int,
) -> np.ndarray:
    captured: dict[str, np.ndarray] = {}

    layer_module = model.model.layers[layer_idx]

    def capture_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if torch.is_tensor(hidden) and hidden.ndim == 3:
            captured["vec"] = hidden[0, -1, :].detach().float().cpu().numpy()
        return output

    handle = layer_module.register_forward_hook(capture_hook)

    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if "vec" not in captured:
        raise RuntimeError(f"Failed to capture layer output at layer {layer_idx}")
    return captured["vec"]


def analyze(
    result_json: Path,
    out_dir: Path,
    limit: int,
    max_seq_len: int,
) -> None:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    baseline_rows = payload.get("baseline_rows", [])
    intervention_rows = payload.get("intervention_rows", [])
    if not baseline_rows or not intervention_rows:
        raise ValueError("result_json must contain baseline_rows and intervention_rows")
    if len(baseline_rows) != len(intervention_rows):
        raise ValueError("baseline_rows and intervention_rows length mismatch")

    if limit > 0:
        baseline_rows = baseline_rows[:limit]
        intervention_rows = intervention_rows[:limit]

    model_name = str(payload["model_name"])
    layer = int(payload["layer"])
    alpha = float(payload["alpha"])
    direction = _load_direction(payload, layer)
    direction_unit = direction / (np.linalg.norm(direction) + 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_name)
    rows_out: list[dict[str, Any]] = []

    for b, i in zip(baseline_rows, intervention_rows):
        prompt = str(b.get("prompt", ""))
        if not prompt:
            continue
        b_correct = bool(b.get("correct", False))
        i_correct = bool(i.get("correct", False))
        group = _group_name(b_correct, i_correct)

        vec_before = _extract_layer_vec(model, tokenizer, prompt, layer, max_seq_len)

        hook = add_direction_hook(
            model=model,
            layer_idx=layer,
            direction=direction,
            alpha=alpha,
            token_position="last",
        )
        try:
            vec_after = _extract_layer_vec(model, tokenizer, prompt, layer, max_seq_len)
        finally:
            hook.remove()

        diff = vec_after - vec_before
        proj_before = float(np.dot(vec_before, direction_unit))
        proj_after = float(np.dot(vec_after, direction_unit))

        rows_out.append(
            {
                "index": int(b.get("index", -1)),
                "group": group,
                "baseline_correct": b_correct,
                "intervention_correct": i_correct,
                "cosine_before_after": _cosine(vec_before, vec_after),
                "l2_shift": float(np.linalg.norm(diff)),
                "proj_before": proj_before,
                "proj_after": proj_after,
                "proj_shift": proj_after - proj_before,
            }
        )

    df = pd.DataFrame(rows_out)
    if df.empty:
        raise ValueError("No usable rows for analysis")

    csv_path = out_dir / "vector_shift_per_sample.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "result_json": str(result_json),
        "model_name": model_name,
        "layer": layer,
        "alpha": alpha,
        "n": int(len(df)),
        "overall": {
            "cosine_mean": float(df["cosine_before_after"].mean()),
            "l2_shift_mean": float(df["l2_shift"].mean()),
            "proj_shift_mean": float(df["proj_shift"].mean()),
        },
        "by_group": {},
    }

    for group_name, sub in df.groupby("group"):
        summary["by_group"][group_name] = {
            "n": int(len(sub)),
            "cosine_mean": float(sub["cosine_before_after"].mean()),
            "cosine_std": float(sub["cosine_before_after"].std(ddof=0)),
            "l2_shift_mean": float(sub["l2_shift"].mean()),
            "proj_shift_mean": float(sub["proj_shift"].mean()),
        }

    summary_path = out_dir / "vector_shift_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    order = ["recovered", "regressed", "stable_true", "stable_false"]

    # Plot 1: cosine distribution by group
    plt.figure(figsize=(9, 5))
    data_for_plot = [
        df.loc[df["group"] == g, "cosine_before_after"].dropna().values for g in order if g in set(df["group"])
    ]
    labels = [g for g in order if g in set(df["group"])]
    if data_for_plot:
        plt.boxplot(data_for_plot, labels=labels, showmeans=True)
    plt.ylabel("cosine(before, after)")
    plt.title(f"Best-layer vector similarity before/after intervention (layer={layer}, alpha={alpha})")
    plt.tight_layout()
    p1 = out_dir / "vector_shift_cosine_by_group.png"
    plt.savefig(p1, dpi=150)
    plt.close()

    # Plot 2: projection shift by group
    plt.figure(figsize=(9, 5))
    data_for_plot = [
        df.loc[df["group"] == g, "proj_shift"].dropna().values for g in order if g in set(df["group"])
    ]
    labels = [g for g in order if g in set(df["group"])]
    if data_for_plot:
        plt.boxplot(data_for_plot, labels=labels, showmeans=True)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.ylabel("projection shift on direction")
    plt.title("Direction projection shift by transition group")
    plt.tight_layout()
    p2 = out_dir / "vector_shift_projection_by_group.png"
    plt.savefig(p2, dpi=150)
    plt.close()

    # Plot 2b: projection before intervention by group
    plt.figure(figsize=(9, 5))
    data_for_plot = [
        df.loc[df["group"] == g, "proj_before"].dropna().values for g in order if g in set(df["group"])
    ]
    labels = [g for g in order if g in set(df["group"])]
    if data_for_plot:
        plt.boxplot(data_for_plot, labels=labels, showmeans=True)
    plt.ylabel("projection before intervention")
    plt.title("Pre-intervention projection by transition group")
    plt.tight_layout()
    p2b = out_dir / "vector_shift_projection_before_by_group.png"
    plt.savefig(p2b, dpi=150)
    plt.close()

    # Plot 3: before vs after projection scatter
    plt.figure(figsize=(6, 6))
    color_map = {
        "recovered": "tab:green",
        "regressed": "tab:red",
        "stable_true": "tab:blue",
        "stable_false": "tab:orange",
    }
    for g, sub in df.groupby("group"):
        plt.scatter(sub["proj_before"], sub["proj_after"], s=12, alpha=0.6, label=g, color=color_map.get(g, None))
    lo = float(min(df["proj_before"].min(), df["proj_after"].min()))
    hi = float(max(df["proj_before"].max(), df["proj_after"].max()))
    if math.isfinite(lo) and math.isfinite(hi):
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("projection before")
    plt.ylabel("projection after")
    plt.title("Direction projection: before vs after")
    plt.legend()
    plt.tight_layout()
    p3 = out_dir / "vector_shift_projection_scatter.png"
    plt.savefig(p3, dpi=150)
    plt.close()

    print(f"saved: {csv_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {p2b}")
    print(f"saved: {p3}")
    print("summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze best-layer vector shift before/after intervention")
    parser.add_argument("--result_json", type=str, required=True, help="Result JSON containing baseline_rows/intervention_rows")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV/JSON/plots")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit (0 means all)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Tokenizer truncation length")
    args = parser.parse_args()

    analyze(
        result_json=Path(args.result_json).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        limit=int(args.limit),
        max_seq_len=int(args.max_seq_len),
    )


if __name__ == "__main__":
    main()
