import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def analyze_and_plot_separately(json_path):
    # ----------------------------------------------------
    # 1. 讀取與初始化資料
    # ----------------------------------------------------
    data_file = Path(json_path)
    if not data_file.exists():
        raise FileNotFoundError(f"找不到結果檔案: {data_file}")

    with data_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("baseline_rows", [])

    idx_to_source_true = {}
    baseline_correct_indices = set()

    for i, row in enumerate(rows):
        idx = row.get("index", i)
        idx_to_source_true[idx] = 1 if row.get("source_correct") is True else 0
        if row.get("correct") is True:
            baseline_correct_indices.add(idx)

    sorted_indices = sorted(idx_to_source_true.keys())
    y_true = [idx_to_source_true[idx] for idx in sorted_indices]

    den_true = sum(1 for v in idx_to_source_true.values() if v == 1) or 1
    den_false = sum(1 for v in idx_to_source_true.values() if v == 0) or 1
    total_source = den_true + den_false

    print("=================== 原始數據基準 ===================")
    print(f"原始標註 source_correct 為 True 的總數 (分母): {den_true}")
    print(f"原始標註 source_correct 為 False 的總數 (分母): {den_false}")
    print(f"模型在 Baseline (干預前) 整體答對的總數: {len(baseline_correct_indices)}")
    print("====================================================\n")

    # ----------------------------------------------------
    # 2. 遍歷數據並收集指標
    # ----------------------------------------------------
    layer_summaries = data.get("layer_sweep_summaries", [])
    if not layer_summaries:
        print('警告: 找不到 "layer_sweep_summaries"，請確認 JSON 結構。')
        return

    plot_records = []
    best_layer = None
    best_alpha = None
    best_acc = -1.0
    best_intervention_correct_set = set()

    print("--- 開始計算每一層干預後的真實原始 Source 準確度 ---")
    print(
        f"{'Layer':<6} {'Alpha':<8} {'Source_True_Acc':<18} {'Source_False_Acc':<18} {'Intervention_Acc':<15}"
    )
    print("-" * 75)

    for summary in layer_summaries:
        layer = summary.get("layer")
        alpha = summary.get("alpha")

        recovered = set(summary.get("recovered_indices", []))
        regressed = set(summary.get("regressed_indices", []))

        # 當前層干預後的最終答對集合
        intervention_correct = (baseline_correct_indices - regressed) | recovered

        # 計算分子數量
        source_true_correct_count = sum(
            1 for idx in intervention_correct if idx_to_source_true.get(idx) == 1
        )
        source_false_correct_count = sum(
            1 for idx in intervention_correct if idx_to_source_true.get(idx) == 0
        )

        source_true_acc = source_true_correct_count / den_true
        source_false_acc = source_false_correct_count / den_false

        total_intervention_correct = (
            source_true_correct_count + source_false_correct_count
        )
        intervention_acc = (
            total_intervention_correct / total_source if total_source else 0.0
        )

        print(
            f"{layer:<6} {alpha:<8} {source_true_acc:.4f} ({source_true_correct_count}/{den_true}) {source_false_acc:.4f} ({source_false_correct_count}/{den_false})    {intervention_acc:.4f}"
        )

        # True 答錯/退步空間數量 = 總數 - 答對數
        true_missed_count = source_true_correct_count- den_true
        plot_records.append(
            {
                "Layer": layer,
                "Alpha": alpha,
                "False_Correct_Src": source_false_correct_count,
                "True_Missed_Src": true_missed_count,
            }
        )

        # 尋找全局最優解
        if intervention_acc > best_acc:
            best_acc = intervention_acc
            best_layer = layer
            best_alpha = alpha
            best_intervention_correct_set = intervention_correct

    print("\n" + "=" * 50)
    print(f"🔥 最佳干預表現：Layer {best_layer} (Alpha: {best_alpha})")
    print(f"✨ 最高整體準確率: {best_acc:.4f}")
    print("=" * 50 + "\n")

    df_plot = pd.DataFrame(plot_records)
    df_plot.save_csv("layer_summary_source_metrics.csv", index=False)

    # ----------------------------------------------------
    # 3. 獨立輸出圖表 1: 四線折線圖
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))

    df_alpha_pos = df_plot[df_plot["Alpha"] == 1.0].sort_values("Layer")
    df_alpha_neg = df_plot[df_plot["Alpha"] == -1.0].sort_values("Layer")

    # False Correct 折線
    plt.plot(
        df_alpha_pos["Layer"],
        df_alpha_pos["False_Correct_Src"],
        marker="o",
        color="#1f77b4",
        linestyle="-",
        linewidth=2,
        label="False Correct (Alpha = 1.0)",
    )
    plt.plot(
        df_alpha_neg["Layer"],
        df_alpha_neg["False_Correct_Src"],
        marker="o",
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="False Correct (Alpha = -1.0)",
    )

    # True Missed 折線
    plt.plot(
        df_alpha_pos["Layer"],
        df_alpha_pos["True_Missed_Src"],
        marker="s",
        color="#d62728",
        linestyle="-",
        linewidth=2,
        label="True Missed (Alpha = 1.0)",
    )
    plt.plot(
        df_alpha_neg["Layer"],
        df_alpha_neg["True_Missed_Src"],
        marker="s",
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="True Missed (Alpha = -1.0)",
    )

    plt.title("HK+ Layer Comparison", fontsize=13, pad=15)
    plt.xlabel("Layer", fontsize=11)
    plt.ylabel("Number of Samples (Counts)", fontsize=11)
    plt.xticks(range(df_plot["Layer"].min(), df_plot["Layer"].max() + 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=8, frameon=True, shadow=True)

    plt.tight_layout()
    line_chart_name = "intervention_line_chart_type1.png"
    plt.savefig(line_chart_name, dpi=300)
    print(f"📈 圖表 1 已儲存至: {line_chart_name}")
    plt.show()  # 會在這裡暫停，關閉視窗後才會繼續畫下一張

    # ----------------------------------------------------
    # 4. 獨立輸出圖表 2: 混淆矩陣圖
    # ----------------------------------------------------
    plt.figure(figsize=(6.5, 5.5))

    y_pred = [
        1 if idx in best_intervention_correct_set else 0 for idx in sorted_indices
    ]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, None]
    group_percentages = [
        "{0:.2%}".format(value) for value in cm_percentage.flatten()
    ]

    labels = [
        f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Correct", "Incorrect"],
        yticklabels=["Grounded", "Hallucinated"],
    )

    plt.title(
        f"Confusion Matrix at Best Intervention\n(Layer {best_layer}, Alpha {best_alpha}, Acc: {best_acc:.4f})",
        fontsize=12,
        pad=15,
    )
    plt.xlabel("Prediction outcome", fontsize=11, labelpad=8)
    plt.ylabel("Input type", fontsize=11, labelpad=8)

    plt.tight_layout()
    cm_chart_name = "best_intervention_confusion_matrix_type1.png"
    plt.savefig(cm_chart_name, dpi=300)
    print(f"🧩 圖表 2 已儲存至: {cm_chart_name}")
    plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    json_path = (
        base_dir / "type2_results" / "all445" / "result445.json"
    )

    analyze_and_plot_separately(json_path)