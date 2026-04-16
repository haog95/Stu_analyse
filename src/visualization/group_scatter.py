"""
群体聚类散点图 - 基于 PCA 分量展示群体分布
"""
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.font_config import chinese_available


GROUP_COLORS = {
    "积极主导与规律成就型 (Active Achievers)": "#2ecc71",
    "稳健应付与中庸型 (Passive Casuals)": "#3498db",
    "行为轨迹时序退化型 (Degrading At-risk)": "#f39c12",
    "隐性逃课与全面游离型 (Disconnected Riskers)": "#e74c3c",
    "极端异常离群群体 (Outliers)": "#9b59b6",
}

# 中英文图例名
GROUP_LEGEND = {
    "积极主导与规律成就型 (Active Achievers)": "Active Achievers",
    "稳健应付与中庸型 (Passive Casuals)": "Passive Casuals",
    "行为轨迹时序退化型 (Degrading At-risk)": "Degrading At-risk",
    "隐性逃课与全面游离型 (Disconnected Riskers)": "Disconnected",
    "极端异常离群群体 (Outliers)": "Outliers",
}


def plot_group_scatter(merged_df, highlight_id: str = None, save_path: str = None):
    """绘制群体聚类 PCA 散点图

    Args:
        merged_df: 合并后的 DataFrame（需含 PCA_Component_1, PCA_Component_2, Group_Profile）
        highlight_id: 高亮的学生ID
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if "PCA_Component_1" not in merged_df.columns:
        print("数据中没有 PCA 分量列")
        return

    # 按群体分组绘制
    for group_name, color in GROUP_COLORS.items():
        mask = merged_df["Group_Profile"] == group_name
        if mask.any():
            legend_label = (group_name[:15] + "..." if chinese_available
                            else GROUP_LEGEND.get(group_name, group_name[:15]))
            ax.scatter(
                merged_df.loc[mask, "PCA_Component_1"],
                merged_df.loc[mask, "PCA_Component_2"],
                c=color, label=legend_label, alpha=0.5, s=20,
            )

    # 高亮指定学生
    if highlight_id and "student_id" in merged_df.columns:
        student = merged_df[merged_df["student_id"] == highlight_id]
        if not student.empty:
            highlight_label = (f"目标: {highlight_id[:10]}" if chinese_available
                               else f"Target: {highlight_id[:10]}")
            ax.scatter(
                student.iloc[0]["PCA_Component_1"],
                student.iloc[0]["PCA_Component_2"],
                c="red", s=200, marker="*", edgecolors="black", linewidths=1.5,
                zorder=10, label=highlight_label,
            )

    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    title = "学生群体聚类分布" if chinese_available else "Student Group Clustering"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
