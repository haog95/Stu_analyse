"""
画像7：身体素质与意志力
- 锻炼习惯：运动达人/规律锻炼/偶尔锻炼
- 体型：标准体型/偏瘦体型/偏胖体型
- 意志力梯队：钻石/铂金/黄金/白银/青铜/待提升
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class PhysicalPortrait(PortraitDimension):
    """身体素质与意志力画像"""

    def __init__(self):
        super().__init__()
        self._name = "身体素质与意志力"
        self._description = "身体素质、锻炼习惯与意志力分析"
        self._source_file = "画像7_身体素质与意志力.csv"
        self._label_column = "意志力梯队"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("意志力梯队", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        mapping = {
            "钻石意志": RiskLevel.LOW,
            "铂金意志": RiskLevel.LOW,
            "黄金意志": RiskLevel.LOW,
            "白银意志": RiskLevel.MEDIUM,
            "青铜意志": RiskLevel.MEDIUM,
            "待提升": RiskLevel.HIGH,
        }
        return mapping.get(label, RiskLevel.LOW)

    def get_numeric_features(self, student_id: str) -> dict:
        return {}

    def _get_extra_info(self, row) -> dict:
        return {
            "锻炼习惯": str(row.get("锻炼习惯", "")),
            "体型": str(row.get("体型", "")),
            "锻炼稳定性": str(row.get("锻炼稳定性", "")),
            "意志力梯队": str(row.get("意志力梯队", "")),
        }
