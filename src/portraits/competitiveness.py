"""
画像9：综合竞争力
- 综合竞争力评分 ZYNJ_Score
- 成绩/素质/能力评分
- 奖学金、竞赛指标
- 预警标签：学业预警(高危)/学业关注(潜在)/学业平稳(低)/优秀免检(安全)
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class CompetitivenessPortrait(PortraitDimension):
    """综合竞争力画像"""

    def __init__(self):
        super().__init__()
        self._name = "综合竞争力"
        self._description = "综合竞争力多维度评估"
        self._source_file = "画像9_综合竞争力.csv"
        self._label_column = "Warning_Label"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("Warning_Label", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        mapping = {
            "优秀免检 (安全)": RiskLevel.LOW,
            "学业平稳 (低风险)": RiskLevel.LOW,
            "学业关注 (潜在风险)": RiskLevel.MEDIUM,
            "学业预警 (高危风险)": RiskLevel.SEVERE,
        }
        return mapping.get(label, RiskLevel.LOW)

    def get_numeric_features(self, student_id: str) -> dict:
        row = self.get_student(student_id)
        if row is None:
            return {}
        features = {}
        for col in [
            "ZYNJ_Score", "CJ_Score", "SZF_Score", "NLF_Score",
            "Schol_Count", "Schol_Max_Score", "Schol_Total_Score",
            "Comp_Count", "Comp_Max_Score", "Comp_Total_Score",
        ]:
            if col in row.index:
                try:
                    features[col] = round(float(row[col]), 4)
                except (ValueError, TypeError):
                    pass
        return features

    def get_warning_label(self, student_id: str) -> str:
        """获取预警标签"""
        return self.get_label(student_id) or "未知"

    def _get_extra_info(self, row) -> dict:
        return {"预警标签": str(row.get("Warning_Label", ""))}
