"""
画像2：学习投入度与自律性
- 标签：主动学习型/普通型/被动应付型
- 核心指标：拖延指数、图书馆打卡、作业完成率等
"""
from typing import Optional

from .base import PortraitDimension
from src.core.student_profile import RiskLevel


class StudyEngagementPortrait(PortraitDimension):
    """学习投入度与自律性画像"""

    def __init__(self):
        super().__init__()
        self._name = "学习投入度"
        self._description = "学习投入度与自律性分析"
        self._source_file = "画像2_学习投入度与自律性.csv"
        self._label_column = "target_label"

    def get_label(self, student_id: str) -> Optional[str]:
        row = self.get_student(student_id)
        if row is None:
            return None
        return row.get("target_label")

    def get_risk_level(self, student_id: str) -> RiskLevel:
        label = self.get_label(student_id)
        mapping = {
            "主动学习型": RiskLevel.LOW,
            "普通型": RiskLevel.MEDIUM,
            "被动应付型": RiskLevel.HIGH,
        }
        return mapping.get(label, RiskLevel.LOW)

    def get_numeric_features(self, student_id: str) -> dict:
        row = self.get_student(student_id)
        if row is None:
            return {}
        features = {}
        for col in [
            "avg_job_rate", "total_special_time", "early_bird_rate",
            "night_owl_rate", "lib_visit_count", "procrastination_index",
            "delay_stability", "final_score", "overall_rank_pct",
        ]:
            if col in row.index:
                val = row[col]
                features[col] = float(val) if _is_number(val) else 0.0
        return features

    def get_procrastination_index(self, student_id: str) -> float:
        """获取拖延指数"""
        row = self.get_student(student_id)
        if row is None:
            return 0.0
        val = row.get("procrastination_index", 0)
        return float(val) if _is_number(val) else 0.0

    def get_diagnostic_report(self, student_id: str) -> str:
        """获取诊断报告文本"""
        row = self.get_student(student_id)
        if row is None:
            return ""
        return str(row.get("diagnostic_report", ""))

    def _get_extra_info(self, row) -> dict:
        return {
            "拖延指数": row.get("procrastination_index", 0),
            "诊断报告": str(row.get("diagnostic_report", "")),
        }


def _is_number(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False
