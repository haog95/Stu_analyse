"""
特征目录 - 特征名称到中文描述的映射
"""

FEATURE_CATALOG = {
    # 画像2：学习投入度
    "avg_job_rate": {"cn": "平均作业完成率", "direction": -1, "desc": "越高越好"},
    "total_special_time": {"cn": "专题学习时长", "direction": -1, "desc": "学习时间投入"},
    "early_bird_rate": {"cn": "早起学习率", "direction": -1, "desc": "早起学习习惯"},
    "night_owl_rate": {"cn": "深夜学习率", "direction": 1, "desc": "深夜学习习惯(可能不健康)"},
    "lib_visit_count": {"cn": "图书馆到访次数", "direction": -1, "desc": "自主学习频率"},
    "procrastination_index": {"cn": "拖延指数", "direction": 1, "desc": "越高越拖延"},
    "delay_stability": {"cn": "延迟稳定性", "direction": 1, "desc": "延迟行为的一致性"},
    "final_score": {"cn": "学习投入综合得分", "direction": -1, "desc": "学习投入综合评估"},
    "overall_rank_pct": {"cn": "综合排名百分位", "direction": 1, "desc": "数值越高排名越靠后"},
    # 画像3：风险预测
    "预测概率": {"cn": "挂科退学预测概率", "direction": 1, "desc": "XGBoost预测的挂科概率"},
    # 画像4：退化预警
    "行为偏移分值": {"cn": "行为偏移分值", "direction": 1, "desc": "学习轨迹偏移程度"},
    "干预建议权重": {"cn": "干预建议权重", "direction": 1, "desc": "需要干预的程度"},
    # 画像6：规律性生活
    "total_active_days": {"cn": "总活跃天数", "direction": -1, "desc": "校园活动活跃度"},
    # 画像8：网络依赖
    "异常得分": {"cn": "网络行为异常得分", "direction": 1, "desc": "上网行为偏离正常范围"},
    "沉迷指数": {"cn": "网络沉迷指数", "direction": 1, "desc": "网络依赖程度"},
    # 画像9：综合竞争力
    "ZYNJ_Score": {"cn": "综合竞争力评分", "direction": -1, "desc": "综合能力评估"},
    "CJ_Score": {"cn": "成绩评分", "direction": -1, "desc": "学业成绩评分"},
    "SZF_Score": {"cn": "素质评分", "direction": -1, "desc": "综合素质评分"},
    "NLF_Score": {"cn": "能力评分", "direction": -1, "desc": "实践能力评分"},
    "Schol_Count": {"cn": "奖学金获得次数", "direction": -1, "desc": "学术表现"},
    "Schol_Max_Score": {"cn": "最高奖学金分值", "direction": -1, "desc": "学术成就"},
    "Schol_Total_Score": {"cn": "奖学金总分", "direction": -1, "desc": "学术成就累计"},
    "Comp_Count": {"cn": "竞赛参与次数", "direction": -1, "desc": "学科竞赛活跃度"},
    "Comp_Max_Score": {"cn": "最高竞赛分值", "direction": -1, "desc": "竞赛成就"},
    "Comp_Total_Score": {"cn": "竞赛总分", "direction": -1, "desc": "竞赛成就累计"},
}


def get_feature_cn_name(feature_name: str) -> str:
    """获取特征的中文名称"""
    if feature_name in FEATURE_CATALOG:
        return FEATURE_CATALOG[feature_name]["cn"]
    return feature_name


def get_feature_direction(feature_name: str) -> int:
    """获取特征的风险方向

    Returns:
        1: 值越高风险越高
        -1: 值越高风险越低
    """
    if feature_name in FEATURE_CATALOG:
        return FEATURE_CATALOG[feature_name]["direction"]
    return 0


def get_feature_description(feature_name: str) -> str:
    """获取特征描述"""
    if feature_name in FEATURE_CATALOG:
        return FEATURE_CATALOG[feature_name]["desc"]
    return ""


def get_risk_direction_text(feature_name: str, shap_value: float) -> str:
    """根据SHAP值和特征方向生成风险描述

    Args:
        feature_name: 特征名
        shap_value: SHAP值（正值增加风险，负值降低风险）

    Returns:
        风险描述文本
    """
    cn_name = get_feature_cn_name(feature_name)
    direction = get_feature_direction(feature_name)

    if shap_value > 0:
        if direction == 1:
            return f"{cn_name}偏高"
        elif direction == -1:
            return f"{cn_name}偏低"
        else:
            return f"{cn_name}异常"
    else:
        if direction == 1:
            return f"{cn_name}控制良好"
        elif direction == -1:
            return f"{cn_name}表现优秀"
        else:
            return f"{cn_name}正常"
