"""
画像3：挂科退学高危预警
- XGBoost预测概率
- 风险分级
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class RiskPredictionPortrait(PortraitDimension):
    """挂科退学高危预警画像"""

    def __init__(self):
        super().__init__()
        self._name = "挂科退学预警"
        self._description = "基于XGBoost的挂科退学高危预警"
        self._source_file = "画像3_挂科退学高危预警.csv"
        self._label_column = "预测概率"

    def get_label(self, student_id: str) -> Optional[str]:
        prob = self.get_fail_probability(student_id)
        if prob is None:
            return None
        if prob >= 0.9:
            return "极高危"
        elif prob >= 0.5:
            return "高危"
        elif prob >= 0.2:
            return "中危"
        else:
            return "低危"

    def get_risk_level(self, student_id: str) -> RiskLevel:
        prob = self.get_fail_probability(student_id)
        if prob is None:
            return RiskLevel.LOW
        if prob >= 0.9:
            return RiskLevel.SEVERE
        elif prob >= 0.5:
            return RiskLevel.HIGH
        elif prob >= 0.2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def get_numeric_features(self, student_id: str) -> dict:
        prob = self.get_fail_probability(student_id)
        if prob is None:
            return {}
        return {"挂科退学概率": round(prob, 4)}

    def get_fail_probability(self, student_id: str) -> Optional[float]:
        """获取挂科概率"""
        row = self.get_student(student_id)
        if row is None:
            return None
        val = row.get("预测概率")
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _get_extra_info(self, row) -> dict:
        prob = row.get("预测概率", 0)
        try:
            prob = float(prob)
        except (ValueError, TypeError):
            prob = 0
        return {"挂科退学概率": round(prob, 4)}
