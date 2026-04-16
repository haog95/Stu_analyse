"""
群体聚类结果
- 5个群体分类
- PCA分量
- 整合画像维度信息
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class GroupClusteringPortrait(PortraitDimension):
    """群体聚类画像"""

    GROUP_PROFILES = {
        0: "积极主导与规律成就型 (Active Achievers)",
        1: "稳健应付与中庸型 (Passive Casuals)",
        2: "行为轨迹时序退化型 (Degrading At-risk)",
        3: "隐性逃课与全面游离型 (Disconnected Riskers)",
        -1: "极端异常离群群体 (Outliers)",
    }

    def __init__(self):
        super().__init__()
        self._name = "群体聚类"
        self._description = "基于多维画像的群体聚类分析"
        self._source_file = "群体画像最终聚类结果.csv"
        self._label_column = "Group_Profile"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return str(row.get("Group_Profile", ""))

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        if "积极主导" in (label or ""):
            return RiskLevel.LOW
        elif "稳健应付" in (label or ""):
            return RiskLevel.LOW
        elif "退化" in (label or ""):
            return RiskLevel.HIGH
        elif "游离" in (label or ""):
            return RiskLevel.SEVERE
        elif "极端" in (label or ""):
            return RiskLevel.SEVERE
        return RiskLevel.LOW

    def get_numeric_features(self, student_id: str) -> dict:
        row = self.get_student(student_id)
        if row is None:
            return {}
        features = {}
        for col in [
            "PCA_Component_1", "PCA_Component_2",
            "overall_rank_pct", "procrastination_index",
            "预测概率", "行为偏移分值",
            "ZYNJ_Score", "CJ_Score",
            "total_active_days",
        ]:
            if col in row.index:
                try:
                    features[col] = round(float(row[col]), 4)
                except (ValueError, TypeError):
                    pass
        return features

    def get_cluster_id(self, student_id: str) -> Optional[int]:
        """获取聚类ID"""
        row = self.get_student(student_id)
        if row is None:
            return None
        try:
            return int(row.get("Cluster", -1))
        except (ValueError, TypeError):
            return -1

    def _get_extra_info(self, row) -> dict:
        return {"群体画像": str(row.get("Group_Profile", ""))}
