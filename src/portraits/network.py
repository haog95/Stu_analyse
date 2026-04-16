"""
画像8：网络依赖与消遣
- 沉迷等级：正常/轻度沉迷/边缘关注/中度沉迷/重度沉迷
- SVM标签：正常模式/超长待机
- 异常得分、沉迷指数
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class NetworkPortrait(PortraitDimension):
    """网络依赖画像"""

    def __init__(self):
        super().__init__()
        self._name = "网络依赖"
        self._description = "网络依赖与沉迷风险分析"
        self._source_file = "画像8_网络依赖.csv"
        self._label_column = "沉迷等级"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("沉迷等级", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        mapping = {
            "正常": RiskLevel.LOW,
            "轻度沉迷": RiskLevel.MEDIUM,
            "边缘关注": RiskLevel.MEDIUM,
            "中度沉迷": RiskLevel.HIGH,
            "重度沉迷": RiskLevel.SEVERE,
        }
        return mapping.get(label, RiskLevel.LOW)

    def get_numeric_features(self, student_id: str) -> dict:
        row = self.get_student(student_id)
        if row is None:
            return {}
        features = {}
        for col in ["异常得分", "沉迷指数"]:
            if col in row.index:
                try:
                    features[col] = round(float(row[col]), 4)
                except (ValueError, TypeError):
                    pass
        return features

    def get_addiction_level(self, student_id: str) -> str:
        """获取沉迷等级"""
        return self.get_label(student_id) or "未知"

    def get_svm_label(self, student_id: str) -> str:
        """获取SVM标签"""
        row = self.get_student(student_id)
        if row is None:
            return "未知"
        return str(row.get("SVM标签", "未知"))

    def _get_extra_info(self, row) -> dict:
        return {
            "沉迷等级": str(row.get("沉迷等级", "")),
            "SVM标签": str(row.get("SVM标签", "")),
        }
