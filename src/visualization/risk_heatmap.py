"""
风险等级热力图 - 展示学生在各维度的风险等级分布
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.student_profile import RiskLevel
from src.visualization.font_config import chinese_available, RISK_LABELS, RISK_LABELS_EN


RISK_LEVEL_NUMERIC = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.SEVERE: 3,
}


def plot_risk_heatmap(profiles, save_path: str = None):
    """绘制多学生风险等级热力图

    Args:
        profiles: StudentProfile 列表
        save_path: 保存路径
    """
    if not profiles:
        return

    # 收集维度名称
    dim_names = list(profiles[0].dimensions.keys())
    student_ids = [p.student_id[:12] for p in profiles]  # 截断ID

    # 构建矩阵
    matrix = np.zeros((len(profiles), len(dim_names)))
    for i, profile in enumerate(profiles):
        for j, dim_name in enumerate(dim_names):
            dim = profile.get_dimension(dim_name)
            if dim:
                matrix[i, j] = RISK_LEVEL_NUMERIC.get(dim.risk_level, 0)

    fig, ax = plt.subplots(figsize=(14, max(6, len(profiles) * 0.4)))

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels(dim_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(student_ids)))
    ax.set_yticklabels(student_ids, fontsize=8)

    # 选择标签语言
    risk_labels = RISK_LABELS if chinese_available else RISK_LABELS_EN

    # 添加数值标注
    for i in range(len(profiles)):
        for j in range(len(dim_names)):
            text = risk_labels.get(int(matrix[i, j]), "")
            color = "white" if matrix[i, j] >= 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    cbar_label = "风险等级" if chinese_available else "Risk Level"
    plt.colorbar(im, ax=ax, label=cbar_label, ticks=[0, 1, 2, 3],
                 format=plt.FuncFormatter(lambda x, _: risk_labels.get(int(x), "")))

    title = "学生多维度风险等级热力图" if chinese_available else "Multi-Dim Risk Heatmap"
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
