"""
SHAP 瀑布图 - 展示单个学生的特征贡献排名
"""
import matplotlib.pyplot as plt
import numpy as np

from src.core.student_profile import StudentProfile
from src.explanation.feature_catalog import get_feature_cn_name
from src.visualization.font_config import chinese_available, L


def plot_shap_waterfall(shap_data: dict, student_id: str,
                        top_k: int = 10, save_path: str = None):
    """绘制 SHAP 瀑布图

    展示推高风险和保护性降低风险的 Top-K 特征贡献。

    Args:
        shap_data: 包含 shap_top3 和 shap_top3_protective 的字典
        student_id: 学生ID
        top_k: 显示的特征数量
        save_path: 保存路径
    """
    risk_factors = shap_data.get("shap_top3", [])
    protective_factors = shap_data.get("shap_top3_protective", [])

    # 合并并排序
    all_factors = []
    for f in risk_factors:
        all_factors.append({
            "name": get_feature_cn_name(f.get("feature", "")),
            "value": f.get("contribution", 0),
            "type": "risk",
        })
    for f in protective_factors:
        all_factors.append({
            "name": get_feature_cn_name(f.get("feature", "")),
            "value": -f.get("contribution", 0),  # 保护性因子取负值
            "type": "protective",
        })

    # 按绝对贡献度排序，取 top_k
    all_factors.sort(key=lambda x: abs(x["value"]), reverse=True)
    all_factors = all_factors[:top_k]

    if not all_factors:
        return

    # 绘制水平条形图（瀑布图风格）
    names = [f["name"] for f in all_factors]
    values = [f["value"] for f in all_factors]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_factors) * 0.6)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)

    # 标注数值
    for bar, val in zip(bars, values):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{abs(val):.4f}", va="center", ha=ha, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)

    xlabel = L("SHAP 贡献度")
    ax.set_xlabel(xlabel, fontsize=11)

    if chinese_available:
        title = (f"学生 {student_id} — 特征贡献瀑布图\n"
                 "(红色=推高风险  绿色=保护性因子)")
    else:
        title = (f"Student {student_id} — SHAP Waterfall\n"
                 "(Red=Risk Factor  Green=Protective)")
    ax.set_title(title, fontsize=12)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label=L("风险推高因子")),
        Patch(facecolor="#2ecc71", label=L("保护性因子")),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
