"""
画像1：课堂参与度与专注力
- 听课质量评级：沉浸/一般/游离
- 是否隐性逃课
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class ClassroomPortrait(PortraitDimension):
    """课堂参与度与专注力画像"""

    def __init__(self):
        super().__init__()
        self._name = "课堂参与度"
        self._description = "课堂参与度与专注力评级"
        self._source_file = "画像1_课堂参与度与专注力评级结果.csv"
        self._label_column = "听课质量评级"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return row.get("听课质量评级")

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        mapping = {
            "沉浸": RiskLevel.LOW,
            "一般": RiskLevel.MEDIUM,
            "游离": RiskLevel.HIGH,
        }
        return mapping.get(label, RiskLevel.LOW)

    def get_numeric_features(self, student_id: str) -> dict:
        row = self.get_student(student_id)
        if row is None:
            return {}
        hidden_truant = row.get("是否隐性逃课", "否")
        return {
            "听课质量评级": row.get("听课质量评级", ""),
            "是否隐性逃课": hidden_truant,
        }

    def is_hidden_truant(self, student_id: str) -> bool:
        """判断是否隐性逃课"""
        row = self.get_student(student_id)
        if row is None:
            return False
        return str(row.get("是否隐性逃课", "否")) == "是"

    def _get_extra_info(self, row) -> dict:
        return {
            "是否隐性逃课": str(row.get("是否隐性逃课", "否")),
        }
