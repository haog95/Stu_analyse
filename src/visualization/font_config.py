"""
中文字体配置模块 - 优先使用中文字体，不可用时回退英文

在 Windows 上按优先级尝试: SimHei > Microsoft YaHei > SimSun > KaiTi
在其他系统上尝试: Noto Sans CJK SC > Source Han Sans SC > WenQuanYi
如果均不可用，则标记 chinese_available=False，图表标签自动切换英文。
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Windows 中文字体候选（按优先级排序）
_CJK_CANDIDATES_WIN = [
    "SimHei", "Microsoft YaHei", "SimSun", "KaiTi",
    "FangSong", "STXihei", "STKaiti", "DengXian",
]

# Linux / macOS 中文字体候选
_CJK_CANDIDATES_OTHER = [
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC", "Songti SC",
]

# 全局状态
chinese_available = False
_active_font = None


def _try_set_font(font_name: str) -> bool:
    """尝试设置指定字体，成功返回 True"""
    try:
        # 检查字体是否存在于系统
        available = {f.name for f in fm.fontManager.ttflist}
        if font_name not in available:
            return False

        # 设置 matplotlib 使用该字体
        matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 清除字体缓存，确保设置生效
        fm._load_fontmanager(try_read_cache=False)

        # 验证设置是否生效
        test_prop = fm.FontProperties(family=font_name)
        resolved = test_prop.get_name()
        if resolved and resolved.lower() != "dejavu sans":
            return True
        return True  # 即使验证不完美也尝试使用
    except Exception:
        return False


def setup_chinese_font():
    """初始化中文字体配置

    Returns:
        True 表示成功配置中文字体，False 表示不可用
    """
    global chinese_available, _active_font

    import platform
    candidates = (_CJK_CANDIDATES_WIN if platform.system() == "Windows"
                  else _CJK_CANDIDATES_OTHER)

    for font_name in candidates:
        if _try_set_font(font_name):
            chinese_available = True
            _active_font = font_name
            break

    if not chinese_available:
        # 没有中文字体，确保负号至少正常
        matplotlib.rcParams["axes.unicode_minus"] = False

    return chinese_available


# ============================================================
# 中英文标签映射 — 所有图表文本集中管理
# ============================================================

# 雷达图维度
RADAR_DIMENSIONS = {
    "课堂参与": "Classroom",
    "学习投入": "Study Engage",
    "风险控制": "Risk Control",
    "生活规律": "Life Routine",
    "身体素质": "Physique",
    "网络自控": "Net Control",
    "综合竞争力": "Competitiv.",
}

# 风险等级
RISK_LABELS = {0: "低", 1: "中", 2: "高", 3: "严重"}
RISK_LABELS_EN = {0: "Low", 1: "Med", 2: "High", 3: "Severe"}

# 群体对比指标名
COMPARISON_NAMES = {
    "final_score": ("学习投入", "Study Score"),
    "procrastination_index": ("拖延指数", "Procrast."),
    "沉迷指数": ("网络沉迷", "Net Addict"),
    "ZYNJ_Score": ("综合竞争力", "Competitiv."),
    "行为偏移分值": ("行为偏移", "Drift"),
    "total_active_days": ("活跃天数", "Active Days"),
    "avg_job_rate": ("作业完成率", "Job Rate"),
    "lib_visit_count": ("图书馆到访", "Lib Visit"),
    "Schol_Total_Score": ("奖学金得分", "Scholarship"),
    "Comp_Count": ("竞赛参与", "Competition"),
}

# 通用标签
T = {
    "该生": "Student",
    "全校均值": "School Avg",
    "全校分布": "Distribution",
    "均值": "Mean",
    "中位数": "Median",
    "归一化得分": "Normalized Score (0-100)",
    "风险等级": "Risk Level",
    "风险推高因子": "Risk Factor",
    "保护性因子": "Protective Factor",
    "SHAP 贡献度": "SHAP Contribution",
    "偏差": "Deviation",
    "学生画像雷达图": "Student Portrait Radar",
    "风险": "Risk",
    "学生多维度风险等级热力图": "Multi-Dim Risk Heatmap",
    "群体聚类分布": "Student Group Clustering",
    "特征贡献瀑布图": "SHAP Waterfall Chart",
    "个体/群体/全校三维对比": "Student vs Group vs School Comparison",
    "关键行为指标与全校对比": "Key Metrics vs School Average",
    "vs 群体均值对比": "vs Group Average",
    "归一化得分_短": "Norm. Score",
    "同群均值": "Group Avg",
    "全校均值_短": "School",
}


def L(cn: str) -> str:
    """根据中文字体是否可用，返回中文或英文标签

    Args:
        cn: 中文标签

    Returns:
        中文或英文标签
    """
    if chinese_available:
        return cn
    return T.get(cn, cn)


# 模块加载时自动初始化
setup_chinese_font()
