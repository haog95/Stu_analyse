"""
画像4：学习轨迹"退化"预警
- 行为偏移分值
- 画像描述
- 干预建议权重
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class DegradationPortrait(PortraitDimension):
    """学习轨迹退化预警画像"""

    def __init__(self):
        super().__init__()
        self._name = "学习轨迹退化"
        self._description = "学习轨迹退化预警与行为偏移检测"
        self._source_file = "画像4_学习轨迹退化预警画像.csv"
        self._label_column = "当前画像描述"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("当前画像描述", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        score = self.get_deviation_score(student_id)
        if score is None:
            return RiskLevel.LOW
        if score >= 0.8:
            return RiskLevel.SEVERE
        elif score >= 0.5:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def get_numeric_features(self, student_id: str) -> dict:
        score = self.get_deviation_score(student_id)
        weight = self.get_intervention_weight(student_id)
        features = {}
        if score is not None:
            features["行为偏移分值"] = round(score, 4)
        if weight is not None:
            features["干预建议权重"] = round(weight, 4)
        return features

    def get_deviation_score(self, student_id: str) -> Optional[float]:
        """获取行为偏移分值"""
        row = self.get_student(student_id)
        if row is None:
            return None
        try:
            return float(row.get("行为偏移分值", 0))
        except (ValueError, TypeError):
            return None

    def get_intervention_weight(self, student_id: str) -> Optional[float]:
        """获取干预建议权重"""
        row = self.get_student(student_id)
        if row is None:
            return None
        try:
            return float(row.get("干预建议权重", 0))
        except (ValueError, TypeError):
            return None

    def _get_extra_info(self, row) -> dict:
        desc = str(row.get("当前画像描述", ""))
        return {"画像描述": desc}
