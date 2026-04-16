"""
报告分类器 - 根据学生画像自动匹配报告类型
"""
from src.core.student_profile import StudentProfile, RiskLevel


# 报告类型定义
REPORT_TYPES = {
    "motivation": {
        "name": "学习动机归因分析报告",
        "template": "report_type_motivation.txt",
        "description": "分析学习投入度不足的根本原因",
    },
    "addiction": {
        "name": "网络依赖对学业风险的深度归因诊断",
        "template": "report_type_addiction.txt",
        "description": "分析网络依赖与学业风险的因果关系",
    },
    "success": {
        "name": "面向学困生的成功经验参考蓝本",
        "template": "report_type_success.txt",
        "description": "通过对比优秀学生提供成功路径",
    },
    "career": {
        "name": "跨区域人才流动的职业规划建议信",
        "template": "report_type_career.txt",
        "description": "个性化职业发展建议",
    },
    "comprehensive": {
        "name": "综合预警干预报告",
        "template": "report_type_comprehensive.txt",
        "description": "多维度综合诊断与分级干预",
    },
}


def classify_report_types(profile: StudentProfile) -> list[str]:
    """根据学生画像判断需要生成哪些报告

    Args:
        profile: 学生画像

    Returns:
        需要生成的报告类型ID列表
    """
    reports = []

    # 统计高风险维度数
    high_risk_dims = profile.get_high_risk_dimensions()
    n_high_risk = len(high_risk_dims)

    # 综合预警：多个高风险维度 OR 群体属于高风险类型
    if n_high_risk >= 3 or profile.overall_risk in (RiskLevel.SEVERE,):
        reports.append("comprehensive")
        return reports  # 综合报告已覆盖所有内容

    # 网瘾学业风险诊断：网络依赖 + 高挂科风险
    network_dim = profile.get_dimension("网络依赖")
    risk_dim = profile.get_dimension("挂科退学预警")
    if network_dim and risk_dim:
        if network_dim.risk_level in (RiskLevel.HIGH, RiskLevel.SEVERE):
            if risk_dim.risk_level in (RiskLevel.HIGH, RiskLevel.SEVERE):
                reports.append("addiction")

    # 学习动机归因：被动应付型 OR 学业退化
    engagement_dim = profile.get_dimension("学习投入度")
    degradation_dim = profile.get_dimension("学习轨迹退化")
    if engagement_dim and engagement_dim.label == "被动应付型":
        reports.append("motivation")
    if degradation_dim and degradation_dim.risk_level in (RiskLevel.HIGH, RiskLevel.SEVERE):
        if "motivation" not in reports:
            reports.append("motivation")

    # 成功经验参考：综合竞争力高危
    comp_dim = profile.get_dimension("综合竞争力")
    if comp_dim and comp_dim.risk_level == RiskLevel.SEVERE:
        if "comprehensive" not in reports:
            reports.append("success")

    # 职业规划建议：就业选择期
    career_dim = profile.get_dimension("就业竞争力")
    if career_dim and career_dim.risk_level != RiskLevel.SEVERE:
        # 不给高危学生推荐职业规划
        if "comprehensive" not in reports:
            reports.append("career")

    # 默认：如果没有任何匹配，生成学习动机报告
    if not reports:
        if engagement_dim and engagement_dim.label == "普通型":
            reports.append("motivation")
        else:
            reports.append("comprehensive")

    return reports


def get_report_info(report_type: str) -> dict:
    """获取报告类型信息"""
    return REPORT_TYPES.get(report_type, REPORT_TYPES["comprehensive"])
