"""
画像10：就业竞争力与匹配
- 推荐去向1、推荐去向2
- 包含：保研/考研/境外留学/直接就业/待业其他/科研行政过渡
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class CareerPortrait(PortraitDimension):
    """就业竞争力与匹配画像"""

    CAREER_PATHS = [
        "保研", "考研", "境外留学", "直接就业", "待业/其他", "科研/行政过渡",
    ]

    def __init__(self):
        super().__init__()
        self._name = "就业竞争力"
        self._description = "就业竞争力评估与去向匹配"
        self._source_file = "画像10_就业竞争力与匹配.csv"
        self._label_column = "推荐去向1"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("推荐去向1", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        if label in ["待业/其他"]:
            return RiskLevel.HIGH
        elif label in ["科研/行政过渡"]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def get_numeric_features(self, student_id: str) -> dict:
        return {}

    def get_recommendations(self, student_id: str) -> list[str]:
        """获取推荐去向列表"""
        row = self.get_student(student_id)
        if row is None:
            return []
        recs = []
        for col in ["推荐去向1", "推荐去向2"]:
            val = row.get(col)
            if val and str(val) != "nan":
                recs.append(str(val))
        return recs

    def _get_extra_info(self, row) -> dict:
        return {
            "推荐去向1": str(row.get("推荐去向1", "")),
            "推荐去向2": str(row.get("推荐去向2", "")),
        }
