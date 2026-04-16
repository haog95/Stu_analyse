"""
学生画像雷达图 - 展示学生在各维度的评分
"""
import matplotlib.pyplot as plt
import numpy as np

from src.core.student_profile import StudentProfile, RiskLevel
from src.visualization.font_config import chinese_available, L, RADAR_DIMENSIONS


# 雷达图的维度和标准化函数
RADAR_DIMENSIONS_DEF = {
    "课堂参与": {"dim": "课堂参与度", "score_fn": lambda d: {"沉浸": 100, "一般": 60, "游离": 20}.get(d.label, 50)},
    "学习投入": {"dim": "学习投入度", "score_fn": lambda d: {"主动学习型": 100, "普通型": 60, "被动应付型": 20}.get(d.label, 50)},
    "风险控制": {"dim": "挂科退学预警", "score_fn": lambda d: max(0, 100 - d.features.get("挂科退学概率", 0) * 100)},
    "生活规律": {"dim": "规律性生活", "score_fn": lambda d: {"晨间规律型": 100, "常规混合型": 70, "深夜活跃型": 50, "高波动不规律型": 20}.get(d.label, 50)},
    "身体素质": {"dim": "身体素质与意志力", "score_fn": lambda d: {"钻石意志": 100, "铂金意志": 85, "黄金意志": 70, "白银意志": 55, "青铜意志": 40, "待提升": 20}.get(d.label, 50)},
    "网络自控": {"dim": "网络依赖", "score_fn": lambda d: {"正常": 100, "轻度沉迷": 70, "边缘关注": 60, "中度沉迷": 35, "重度沉迷": 10}.get(d.label, 50)},
    "综合竞争力": {"dim": "综合竞争力", "score_fn": lambda d: {"优秀免检 (安全)": 100, "学业平稳 (低风险)": 75, "学业关注 (潜在风险)": 45, "学业预警 (高危风险)": 15}.get(d.label, 50)},
}


def plot_student_radar(profile: StudentProfile, save_path: str = None):
    """绘制学生画像雷达图

    Args:
        profile: 学生画像
        save_path: 保存路径，None则显示
    """
    # 计算各维度分数
    categories_cn = list(RADAR_DIMENSIONS_DEF.keys())
    scores = []
    for cat, info in RADAR_DIMENSIONS_DEF.items():
        dim = profile.get_dimension(info["dim"])
        if dim:
            scores.append(info["score_fn"](dim))
        else:
            scores.append(50)

    # 根据字体可用性选择标签
    categories = [RADAR_DIMENSIONS.get(c, c) if not chinese_available else c
                  for c in categories_cn]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, scores_plot, alpha=0.25, color="#4C72B0")
    ax.plot(angles, scores_plot, "o-", linewidth=2, color="#4C72B0")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    # 标题
    risk_text = profile.overall_risk.value
    title = (f"学生画像雷达图\nID: {profile.student_id} | 风险: {risk_text}"
             if chinese_available
             else f"Student Portrait Radar\nID: {profile.student_id} | Risk: {risk_text}")
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
