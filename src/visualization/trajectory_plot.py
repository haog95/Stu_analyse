"""
行为轨迹时序图 - 展示学生在时间维度上的行为变化趋势
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.student_profile import StudentProfile
from src.visualization.font_config import chinese_available, L


# 指标中英文映射
_METRIC_LABELS = {
    "行为偏移分值": "Behavior Drift",
    "沉迷指数": "Net Addiction",
    "异常得分": "Anomaly Score",
    "final_score": "Study Score",
    "procrastination_index": "Procrastination",
    "ZYNJ_Score": "Competitiveness",
}


def _metric_label(col: str) -> str:
    """根据字体可用性返回指标标签"""
    if chinese_available:
        return col
    return _METRIC_LABELS.get(col, col)


def plot_behavior_trajectory(merged_df: pd.DataFrame, student_id: str,
                             feature_cols: list = None,
                             save_path: str = None):
    """绘制学生行为轨迹时序图

    展示学生在关键行为指标上的时间变化趋势。

    Args:
        merged_df: 合并后的 DataFrame
        student_id: 学生ID
        feature_cols: 要展示的特征列列表
        save_path: 保存路径
    """
    row = merged_df[merged_df["student_id"] == student_id]
    if row.empty:
        return

    # 默认展示的轨迹指标（如果存在的话）
    if feature_cols is None:
        feature_cols = [
            "行为偏移分值", "沉迷指数", "异常得分",
            "final_score", "procrastination_index",
            "ZYNJ_Score",
        ]

    available = [c for c in feature_cols if c in merged_df.columns]
    if not available:
        return

    # 获取该学生的值和全校统计
    student_vals = {}
    group_means = {}
    group_stds = {}

    group_col = "Group_Profile" if "Group_Profile" in merged_df.columns else None
    student_group = None
    if group_col:
        student_group = row.iloc[0].get(group_col)

    for col in available:
        val = row.iloc[0].get(col, 0)
        try:
            student_vals[col] = float(val)
        except (ValueError, TypeError):
            student_vals[col] = 0.0

        # 全校均值
        group_means[col] = merged_df[col].mean()
        group_stds[col] = merged_df[col].std()

    # 绘制对比雷达式柱状图
    n_features = len(available)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 5))

    if n_features == 1:
        axes = [axes]

    bar_labels = [L("该生"), L("全校均值")]
    for ax, col in zip(axes, available):
        val = student_vals[col]
        mean_val = group_means[col]
        std_val = group_stds[col]

        # 画学生值和全校均值
        bars = ax.bar(bar_labels, [val, mean_val],
                       color=["#3498db", "#bdc3c7"], edgecolor="white", width=0.5)

        # 标注偏差
        deviation = val - mean_val
        dev_text = f"{L('偏差')}: {deviation:+.3f}"
        color = "#e74c3c" if abs(deviation) > std_val else "#2ecc71"
        ax.set_title(_metric_label(col), fontsize=10, fontweight="bold")
        ax.text(0.5, 0.95, dev_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=color)

        # 添加标准差范围
        ax.axhspan(mean_val - std_val, mean_val + std_val,
                    alpha=0.1, color="#f39c12", label="±1σ")

        for bar, v in zip(bars, [val, mean_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    if chinese_available:
        suptitle = f"学生 {student_id} — 关键行为指标与全校对比"
    else:
        suptitle = f"Student {student_id} — Key Metrics vs School Average"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_group_trajectory_comparison(merged_df: pd.DataFrame,
                                     student_id: str,
                                     metrics: list = None,
                                     save_path: str = None):
    """绘制个体与群体在各维度上的对比图

    Args:
        merged_df: 合并后的 DataFrame
        student_id: 学生ID
        metrics: 对比的指标列表
        save_path: 保存路径
    """
    if metrics is None:
        metrics = [
            "final_score", "procrastination_index", "沉迷指数",
            "ZYNJ_Score", "行为偏移分值", "total_active_days",
        ]

    available = [m for m in metrics if m in merged_df.columns]
    if not available:
        return

    row = merged_df[merged_df["student_id"] == student_id]
    if row.empty:
        return

    group_col = "Group_Profile" if "Group_Profile" in merged_df.columns else None
    if not group_col:
        return

    student_group = row.iloc[0].get(group_col)
    group_df = merged_df[merged_df[group_col] == student_group]

    # 归一化到 0-100
    fig, ax = plt.subplots(figsize=(12, 6))

    # 计算各指标的归一化值
    student_normalized = []
    group_mean_normalized = []
    labels = []

    for m in available:
        val = row.iloc[0].get(m, 0)
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        col_min = merged_df[m].min()
        col_max = merged_df[m].max()
        col_range = col_max - col_min if col_max != col_min else 1

        s_norm = (val - col_min) / col_range * 100
        g_norm = (group_df[m].mean() - col_min) / col_range * 100

        student_normalized.append(s_norm)
        group_mean_normalized.append(g_norm)
        labels.append(_metric_label(m))

    x = np.arange(len(labels))
    width = 0.35

    student_label = L("该生")
    group_label = (f"{student_group[:10]}...{L('同群均值')}" if chinese_available
                   else "Group Avg")

    bars1 = ax.bar(x - width / 2, student_normalized, width,
                    label=student_label, color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, group_mean_normalized, width,
                    label=group_label, color="#95a5a6", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(L("归一化得分"), fontsize=11)

    if chinese_available:
        ax.set_title(f"学生 {student_id} vs 群体均值对比", fontsize=13)
    else:
        ax.set_title(f"Student {student_id} vs Group Average", fontsize=13)

    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
