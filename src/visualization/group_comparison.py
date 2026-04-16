"""
群体对比柱状图 - 个体 vs 群体均值 vs 全校均值的多维对比
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.student_profile import StudentProfile
from src.explanation.feature_catalog import get_feature_cn_name
from src.visualization.font_config import chinese_available, L


# 用于群体对比的关键指标及其含义
COMPARISON_METRICS = {
    "final_score": {"name": "学习投入", "name_en": "Study Score", "higher_better": True},
    "procrastination_index": {"name": "拖延指数", "name_en": "Procrast.", "higher_better": False},
    "沉迷指数": {"name": "网络沉迷", "name_en": "Net Addict", "higher_better": False},
    "ZYNJ_Score": {"name": "综合竞争力", "name_en": "Competitiv.", "higher_better": True},
    "行为偏移分值": {"name": "行为偏移", "name_en": "Drift", "higher_better": False},
    "total_active_days": {"name": "活跃天数", "name_en": "Active Days", "higher_better": True},
    "avg_job_rate": {"name": "作业完成率", "name_en": "Job Rate", "higher_better": True},
    "lib_visit_count": {"name": "图书馆到访", "name_en": "Lib Visit", "higher_better": True},
    "Schol_Total_Score": {"name": "奖学金得分", "name_en": "Scholarship", "higher_better": True},
    "Comp_Count": {"name": "竞赛参与", "name_en": "Competition", "higher_better": True},
}


def _metric_name(info: dict) -> str:
    """根据字体可用性返回指标名"""
    return info["name"] if chinese_available else info["name_en"]


def plot_group_comparison(merged_df: pd.DataFrame, student_id: str,
                          save_path: str = None):
    """绘制个体 vs 群体均值 vs 全校均值的对比图

    Args:
        merged_df: 合并后的 DataFrame
        student_id: 学生ID
        save_path: 保存路径
    """
    row = merged_df[merged_df["student_id"] == student_id]
    if row.empty:
        return

    group_col = "Group_Profile" if "Group_Profile" in merged_df.columns else None
    if not group_col:
        return

    student_group = row.iloc[0].get(group_col)
    group_df = merged_df[merged_df[group_col] == student_group]

    # 筛选可用指标
    available_metrics = []
    for col, info in COMPARISON_METRICS.items():
        if col in merged_df.columns and merged_df[col].notna().any():
            try:
                val = float(row.iloc[0].get(col, 0))
                if not np.isnan(val):
                    available_metrics.append((col, info))
            except (ValueError, TypeError):
                pass

    if not available_metrics:
        return

    # 归一化并计算三组值
    labels = []
    student_vals = []
    group_vals = []
    school_vals = []

    for col, info in available_metrics:
        col_min = merged_df[col].min()
        col_max = merged_df[col].max()
        col_range = col_max - col_min if col_max != col_min else 1

        s_val = float(row.iloc[0].get(col, 0))
        g_val = group_df[col].mean()
        sc_val = merged_df[col].mean()

        # 归一化到 0-100
        s_norm = (s_val - col_min) / col_range * 100
        g_norm = (g_val - col_min) / col_range * 100
        sc_norm = (sc_val - col_min) / col_range * 100

        labels.append(_metric_name(info))
        student_vals.append(s_norm)
        group_vals.append(g_norm)
        school_vals.append(sc_norm)

    # 绘制分组柱状图
    n_metrics = len(labels)
    x = np.arange(n_metrics)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.2), 6))

    student_label = L("该生")
    group_label = L("同群均值")
    school_label = L("全校均值_短")

    bars1 = ax.bar(x - width, student_vals, width,
                    label=student_label, color="#3498db", alpha=0.85)
    bars2 = ax.bar(x, group_vals, width,
                    label=group_label, color="#f39c12", alpha=0.75)
    bars3 = ax.bar(x + width, school_vals, width,
                    label=school_label, color="#95a5a6", alpha=0.65)

    # 标注该生数值
    for bar, val in zip(bars1, student_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7, color="#3498db")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(L("归一化得分"), fontsize=11)

    if chinese_available:
        ax.set_title(f"学生 {student_id} — 个体/群体/全校三维对比", fontsize=13)
    else:
        ax.set_title(f"Student {student_id} — vs Group vs School", fontsize=13)

    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_group_distribution(merged_df: pd.DataFrame, student_id: str,
                            feature_col: str = "final_score",
                            save_path: str = None):
    """绘制个体在群体分布中的位置标注图

    Args:
        merged_df: 合并后的 DataFrame
        student_id: 学生ID
        feature_col: 要展示分布的特征列
        save_path: 保存路径
    """
    row = merged_df[merged_df["student_id"] == student_id]
    if row.empty or feature_col not in merged_df.columns:
        return

    try:
        student_val = float(row.iloc[0].get(feature_col, 0))
    except (ValueError, TypeError):
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    # 绘制全校分布直方图
    all_vals = merged_df[feature_col].dropna()
    ax.hist(all_vals, bins=40, color="#bdc3c7", edgecolor="white",
            alpha=0.7, label=L("全校分布"))

    # 标注该生的位置
    ax.axvline(x=student_val, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"{L('该生')}: {student_val:.3f}")

    # 标注均值和中位数
    mean_val = all_vals.mean()
    median_val = all_vals.median()
    ax.axvline(x=mean_val, color="#3498db", linewidth=1.5, linestyle="-",
               label=f"{L('均值')}: {mean_val:.3f}")
    ax.axvline(x=median_val, color="#2ecc71", linewidth=1.5, linestyle="-.",
               label=f"{L('中位数')}: {median_val:.3f}")

    # 计算百分位
    percentile = (all_vals < student_val).mean() * 100
    feature_display = get_feature_cn_name(feature_col)

    pct_label = "百分位" if chinese_available else "Percentile"
    if chinese_available:
        ax.set_title(f"学生 {student_id} 在「{feature_display}」"
                     f"上的位置 ({pct_label}: {percentile:.1f}%)", fontsize=12)
    else:
        ax.set_title(f"Student {student_id} — {feature_col} "
                     f"({pct_label}: {percentile:.1f}%)", fontsize=12)

    ax.legend(fontsize=9)
    ax.set_xlabel(feature_display, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
