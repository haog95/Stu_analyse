"""
学生画像数据类 - 表示单个学生的完整画像信息
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    SEVERE = "严重风险"


@dataclass
class DimensionData:
    """单个画像维度的数据"""
    name: str  # 维度名称
    label: Optional[str] = None  # 主要标签
    risk_level: RiskLevel = RiskLevel.LOW
    features: dict = field(default_factory=dict)  # 数值特征
    extra: dict = field(default_factory=dict)  # 其他信息


@dataclass
class StudentProfile:
    """学生完整画像"""
    student_id: str
    dimensions: dict[str, DimensionData] = field(default_factory=dict)
    group: Optional[str] = None  # 群体聚类标签
    group_cluster: Optional[int] = None

    # 汇总信息
    overall_risk: RiskLevel = RiskLevel.LOW
    fail_probability: float = 0.0
    warning_label: Optional[str] = None

    def add_dimension(self, dim: DimensionData):
        """添加一个画像维度"""
        self.dimensions[dim.name] = dim

    def get_dimension(self, name: str) -> Optional[DimensionData]:
        """获取指定维度"""
        return self.dimensions.get(name)

    def get_all_labels(self) -> dict[str, str]:
        """获取所有维度的标签"""
        return {name: dim.label for name, dim in self.dimensions.items() if dim.label}

    def get_high_risk_dimensions(self) -> list[str]:
        """获取高风险维度列表"""
        return [
            name for name, dim in self.dimensions.items()
            if dim.risk_level in (RiskLevel.HIGH, RiskLevel.SEVERE)
        ]

    def to_context_json(self) -> dict:
        """转换为KERAG第二层的结构化JSON"""
        portraits = {}
        for name, dim in self.dimensions.items():
            portraits[name] = {
                "label": dim.label,
                "risk_level": dim.risk_level.value,
                **dim.features,
                **dim.extra,
            }

        return {
            "student_id": self.student_id,
            "group": self.group,
            "overall_risk": self.overall_risk.value,
            "fail_probability": round(self.fail_probability, 4),
            "warning_label": self.warning_label,
            "portraits": portraits,
        }

    def get_risk_keywords(self) -> list[str]:
        """提取风险关键词，用于知识库检索"""
        keywords = []

        risk_keywords_map = {
            "沉迷": ["网络依赖", "沉迷", "网瘾"],
            "游离": ["隐性逃课", "课堂游离", "听课质量差"],
            "退化": ["学业退化", "学习轨迹退化", "成绩下滑"],
            "被动应付": ["拖延", "学习被动", "缺乏动力"],
            "高危": ["挂科风险", "退学风险", "学业危机"],
            "不规律": ["作息紊乱", "熬夜", "生活不规律"],
        }

        labels = self.get_all_labels()
        all_text = " ".join(str(v) for v in labels.values())

        for key, words in risk_keywords_map.items():
            if key in all_text:
                keywords.extend(words)

        # 基于数值特征的额外关键词
        if self.fail_probability > 0.5:
            keywords.extend(["挂科预警", "高风险"])

        return list(set(keywords))

    def __repr__(self):
        n_dims = len(self.dimensions)
        n_risk = len(self.get_high_risk_dimensions())
        return (
            f"StudentProfile(id={self.student_id}, "
            f"dims={n_dims}, risk={self.overall_risk.value}, "
            f"high_risk_dims={n_risk})"
        )
