"""Aggregate layer/alpha JSON summaries and produce visualizations.

Searches a results directory for files named like ``layer_XXX_alpha_Y.json``,
reads metrics and builds a DataFrame, then creates:

- heatmap of `delta_accuracy` (layers x alphas)
- line plots of `delta_accuracy` across layers for each alpha

Usage:
    python experiment/visualize_results.py --results_dir experiment/type1_results --out_dir experiment/plots

Requires: pandas, matplotlib, seaborn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def collect_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("layer_*alpha*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        name = path.stem
        # attempt to parse layer and alpha from filename
        layer = None
        alpha = None
        try:
            # filename like layer_000_alpha_-1
            parts = name.split("_")
            li = parts.index("layer") if "layer" in parts else 0
        except Exception:
            parts = name.split("_")
        # basic parsing
        try:
            # find numeric layer (first int-like part after 'layer')
            for i, p in enumerate(parts):
                if p.startswith("layer"):
                    # next part likely the number
                    layer = int(parts[i + 1])
                    break
                if p == "layer":
                    layer = int(parts[i + 1])
                    break
        except Exception:
            layer = None
        try:
            # find 'alpha' token and parse following value
            if "alpha" in parts:
                ai = parts.index("alpha")
                alpha = float(parts[ai + 1])
            else:
                # fallback: try extracting last numeric token
                for p in reversed(parts):
                    try:
                        alpha = float(p)
                        break
                    except Exception:
                        continue
        except Exception:
            alpha = None

        # metrics may be in top-level or under 'summary'
        summary = data.get("summary") if isinstance(data, dict) else None
        if summary is None and isinstance(data, dict):
            summary = {k: v for k, v in data.items() if isinstance(v, (int, float, bool))}

        def g(key: str):
            val = None
            if isinstance(summary, dict) and key in summary:
                val = summary.get(key)
            elif isinstance(data, dict) and key in data:
                val = data.get(key)
            return val

        row = {
            "file": str(path),
            "layer": layer if layer is not None else -1,
            "alpha": alpha if alpha is not None else 0.0,
            "baseline_accuracy": g("baseline_accuracy") or g("baseline_acc") or g("baseline_correct") or None,
            "intervention_accuracy": g("intervention_accuracy") or g("intervention_acc") or None,
            "delta_accuracy": g("delta_accuracy") if g("delta_accuracy") is not None else None,
            "recovery_rate": g("recovery_rate") or g("recovery") or None,
            "regression_rate": g("regression_rate") or g("regression") or None,
            "baseline_false_count": g("baseline_false_count") or None,
            "baseline_false_intervention_correct": g("baseline_false_intervention_correct") or None,
            "baseline_false_intervention_accuracy": g("baseline_false_intervention_accuracy") or None,
            "baseline_true_count": g("baseline_true_count") or None,
            "baseline_true_intervention_correct": g("baseline_true_intervention_correct") or None,
            "regressed_count": g("regressed_count") or None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # ensure numeric types
    if not df.empty:
        df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype("Int64")
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
        for col in ["baseline_accuracy", "intervention_accuracy", "delta_accuracy", "recovery_rate", "regression_rate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        print("No data to plot.")
        return

    pivot = df.pivot_table(index="layer", columns="alpha", values="delta_accuracy", aggfunc="mean")
    plt.figure(figsize=(12, max(4, pivot.shape[0] * 0.2)))
    sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".3f")
    plt.title("Delta Accuracy (intervention - baseline) by Layer and Alpha")
    out = out_dir / "heatmap_delta_accuracy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved heatmap to {out}")


def plot_lines(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    alphas = sorted(df["alpha"].dropna().unique())
    plt.figure(figsize=(10, 6))
    for a in alphas:
        sub = df[df["alpha"] == a].sort_values("layer")
        if sub.empty:
            continue
        plt.plot(sub["layer"].astype(int), sub["delta_accuracy"], marker="o", label=f"alpha={a}")
    plt.xlabel("Layer")
    plt.ylabel("Delta Accuracy")
    plt.title("Delta Accuracy across Layers by Alpha")
    plt.legend()
    out = out_dir / "lines_delta_accuracy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved line plot to {out}")


def plot_baseline_false_and_true_regression(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    # focus on alpha values separately; if multiple alphas present, plot for each
    alphas = sorted(df["alpha"].dropna().unique())
    for a in alphas:
        sub = df[df["alpha"] == a].sort_values("layer")
        if sub.empty:
            continue
        layers = sub["layer"].astype(int)
        # compute rates
        bf_acc = sub.get("baseline_false_intervention_accuracy")
        # regression rate shown as negative (baseline-true regress goes downward)
        regression_rate = None
        if "regressed_count" in sub.columns and "baseline_true_count" in sub.columns:
            regression_rate = -(sub["regressed_count"].astype(float) / sub["baseline_true_count"].astype(float)).replace([pd.NA, float("inf")], pd.NA)

        plt.figure(figsize=(10, 5))
        if bf_acc is not None and not bf_acc.isna().all():
            plt.plot(layers, bf_acc, marker="o", label="baseline_false->intervention_correct (accuracy)")
        if regression_rate is not None and not regression_rate.isna().all():
            plt.plot(layers, regression_rate, marker="x", label="-baseline_true->regressed (rate)")
        plt.xlabel("Layer")
        plt.ylabel("Rate / Accuracy")
        plt.title(f"Baseline-false intervention-correct and Baseline-true regression (alpha={a})")
        plt.legend()
        out = out_dir / f"baseline_false_vs_true_regression_alpha_{str(a).replace('.', '_')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved comparison plot to {out}")


def plot_requested_counts(df: pd.DataFrame, out_dir: Path, normalize: bool = False, combine_alphas: bool = False, ylog: bool = False) -> None:
    """Plot baseline_false_intervention_correct and (baseline_true_intervention_correct - baseline_true_count).

    The second metric is a signed difference (usually negative when some baseline-true samples regressed).
    """
    if df.empty:
        return
    alphas = sorted(df["alpha"].dropna().unique())
    # If combining alphas, produce a single plot per metric with multiple alpha lines
    if combine_alphas:
        plt.figure(figsize=(10, 6))
        any_plotted = False
        for a in alphas:
            sub = df[df["alpha"] == a].sort_values("layer")
            if sub.empty:
                continue
            layers = sub["layer"].astype(int)
            if "baseline_false_intervention_correct" in sub.columns:
                y = sub["baseline_false_intervention_correct"].astype(float)
                if normalize:
                    if "baseline_false_count" in sub.columns:
                        denom = sub["baseline_false_count"].astype(float)
                        y = (y / denom).replace([pd.NA, float("inf")], pd.NA)
                plt.plot(layers, y, marker="o", label=f"bf_correct alpha={a}")
                any_plotted = True
        if any_plotted:
            plt.xlabel("Layer")
            plt.ylabel("Normalized rate" if normalize else "Count")
            plt.title("Baseline-false intervention correct across layers (combined alphas)")
            if ylog:
                plt.yscale("log")
            plt.legend()
            out = out_dir / "requested_counts_combined_baseline_false.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Saved combined baseline-false plot to {out}")

        plt.figure(figsize=(10, 6))
        any_plotted = False
        for a in alphas:
            sub = df[df["alpha"] == a].sort_values("layer")
            if sub.empty:
                continue
            layers = sub["layer"].astype(int)
            if "baseline_true_intervention_correct" in sub.columns and "baseline_true_count" in sub.columns:
                val = sub["baseline_true_intervention_correct"].astype(float) - sub["baseline_true_count"].astype(float)
                if normalize:
                    denom = sub["baseline_true_count"].astype(float)
                    val = (val / denom).replace([pd.NA, float("inf")], pd.NA)
                plt.plot(layers, val, marker="x", label=f"delta_true alpha={a}")
                any_plotted = True
        if any_plotted:
            plt.xlabel("Layer")
            plt.ylabel("Normalized delta" if normalize else "Signed count delta")
            plt.title("Baseline-true intervention correct delta across layers (combined alphas)")
            if ylog:
                plt.yscale("symlog")
            plt.legend()
            out = out_dir / "requested_counts_combined_delta_true.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Saved combined delta-true plot to {out}")

        return

    # Not combining alphas: generate per-alpha plots
    for a in alphas:
        sub = df[df["alpha"] == a].sort_values("layer")
        if sub.empty:
            continue
        layers = sub["layer"].astype(int)

        bf_correct = None
        if "baseline_false_intervention_correct" in sub.columns:
            bf_correct = sub["baseline_false_intervention_correct"].astype(float)

        delta_true = None
        if "baseline_true_intervention_correct" in sub.columns and "baseline_true_count" in sub.columns:
            delta_true = (sub["baseline_true_intervention_correct"].astype(float) - sub["baseline_true_count"].astype(float))

        if normalize and bf_correct is not None and "baseline_false_count" in sub.columns:
            bf_correct = (bf_correct / sub["baseline_false_count"].astype(float)).replace([pd.NA, float("inf")], pd.NA)
        if normalize and delta_true is not None and "baseline_true_count" in sub.columns:
            delta_true = (delta_true / sub["baseline_true_count"].astype(float)).replace([pd.NA, float("inf")], pd.NA)

        plt.figure(figsize=(10, 5))
        plotted = False
        if bf_correct is not None and not bf_correct.isna().all():
            plt.plot(layers, bf_correct, marker="o", label="baseline_false_intervention_correct")
            plotted = True
        if delta_true is not None and not delta_true.isna().all():
            plt.plot(layers, delta_true, marker="x", label="baseline_true_intervention_correct - baseline_true_count")
            plotted = True

        if not plotted:
            continue

        plt.xlabel("Layer")
        plt.ylabel("Normalized rate" if normalize else "Count (signed)")
        plt.title(f"Baseline-false correct and delta baseline-true correct (alpha={a})")
        if ylog:
            plt.yscale("symlog")
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.legend()
        out = out_dir / f"requested_counts_alpha_{str(a).replace('.', '_')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved requested counts plot to {out}")


def plot_single_figure_counts_combined_alphas(df: pd.DataFrame, out_dir: Path, ylog: bool = False) -> None:
    """Plot raw counts in one figure:

    - baseline_false_intervention_correct
    - baseline_true regression count (regressed_count)

    with alpha values combined as separate lines.
    """
    if df.empty:
        return

    alphas = sorted(df["alpha"].dropna().unique())
    plt.figure(figsize=(11, 6))
    any_plotted = False

    for a in alphas:
        sub = df[df["alpha"] == a].sort_values("layer")
        if sub.empty:
            continue
        layers = sub["layer"].astype(int)

        if "baseline_false_intervention_correct" in sub.columns:
            bf = sub["baseline_false_intervention_correct"].astype(float)
            if not bf.isna().all():
                plt.plot(layers, bf, marker="o", label=f"alpha={a} baseline_false_intervention_correct")
                any_plotted = True

        if "regressed_count" in sub.columns:
            rg = -sub["regressed_count"].astype(float)
            if not rg.isna().all():
                plt.plot(layers, rg, marker="x", linestyle="--", label=f"alpha={a} -baseline_true_regressed_count")
                any_plotted = True

    if not any_plotted:
        plt.close()
        return

    plt.xlabel("Layer")
    plt.ylabel("Count")
    plt.title("Combined alphas: baseline_false intervention-correct count and baseline_true regressed count")
    if ylog:
        plt.yscale("log")
    plt.legend()
    out = out_dir / "requested_counts_single_figure_combined_alphas.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved single-figure combined counts plot to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize layer/alpha intervention results")
    parser.add_argument("--results_dir", type=str, required=True, help="Root folder containing layer_*.json files")
    parser.add_argument("--out_dir", type=str, default="experiment/plots", help="Directory to write plot PNGs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--normalize", action="store_true", help="Plot normalized rates instead of raw counts for requested counts plot")
    parser.add_argument("--combine_alphas", action="store_true", help="Combine multiple alphas onto a single plot for requested counts")
    parser.add_argument("--ylog", action="store_true", help="Use log scale for Y axis on requested counts plots")
    parser.add_argument(
        "--single_figure_counts",
        action="store_true",
        help="Create one figure combining alpha lines for baseline_false_intervention_correct and regressed_count (raw counts)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_metrics(results_dir)
    if df.empty:
        print("No result JSON files found under", results_dir)
        return

    df.to_csv(out_dir / "aggregated_results.csv", index=False)
    print(f"Wrote aggregated CSV to {out_dir / 'aggregated_results.csv'}")

    plot_heatmap(df, out_dir)
    plot_lines(df, out_dir)
    plot_baseline_false_and_true_regression(df, out_dir)
    plot_requested_counts(df, out_dir, normalize=args.normalize, combine_alphas=args.combine_alphas, ylog=args.ylog)
    if args.single_figure_counts:
        plot_single_figure_counts_combined_alphas(df, out_dir, ylog=args.ylog)

    if args.show:
        import subprocess
        try:
            # attempt to open the out directory (works on Windows)
            subprocess.run(["explorer", str(out_dir)])
        except Exception:
            pass


if __name__ == "__main__":
    main()
