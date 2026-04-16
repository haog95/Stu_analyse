"""
画像6：规律性生活
- 行为模式：晨间规律型/常规混合型/深夜活跃型/高波动不规律型
- 总活跃天数
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class LifestylePortrait(PortraitDimension):
    """规律性生活画像"""

    def __init__(self):
        super().__init__()
        self._name = "规律性生活"
        self._description = "生活规律性与作息模式分析"
        self._source_file = "画像6_规律性生活.csv"
        self._label_column = "Behavioral_Pattern"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("Behavioral_Pattern", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        mapping = {
            "晨间规律型": RiskLevel.LOW,
            "常规混合型": RiskLevel.LOW,
            "深夜活跃型": RiskLevel.MEDIUM,
            "高波动不规律型": RiskLevel.HIGH,
            "样本量不足": RiskLevel.MEDIUM,
        }
        return mapping.get(label, RiskLevel.LOW)

    def get_numeric_features(self, student_id: str) -> dict:
        row = self.get_student(student_id)
        if row is None:
            return {}
        features = {}
        try:
            features["total_active_days"] = float(row.get("total_active_days", 0))
        except (ValueError, TypeError):
            pass
        return features

    def _get_extra_info(self, row) -> dict:
        return {"作息类型": str(row.get("Behavioral_Pattern", ""))}
