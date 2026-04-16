"""
画像维度注册表 - 统一访问所有画像维度
"""
from typing import Optional

from src.core.student_profile import StudentProfile, RiskLevel

from .base import PortraitDimension
from .classroom import ClassroomPortrait
from .study_engagement import StudyEngagementPortrait
from .risk_prediction import RiskPredictionPortrait
from .degradation import DegradationPortrait
from .lifestyle import LifestylePortrait
from .physical import PhysicalPortrait
from .network import NetworkPortrait
from .competitiveness import CompetitivenessPortrait
from .career import CareerPortrait
from .group_clustering import GroupClusteringPortrait


class PortraitRegistry:
    """画像维度注册表

    统一管理所有画像维度，提供便捷的访问接口。
    """

    def __init__(self):
        self._dimensions: dict[str, PortraitDimension] = {}
        self._initialized = False

    def initialize(self, data_dir=None):
        """初始化并加载所有画像数据"""
        dimension_classes = [
            ClassroomPortrait,
            StudyEngagementPortrait,
            RiskPredictionPortrait,
            DegradationPortrait,
            LifestylePortrait,
            PhysicalPortrait,
            NetworkPortrait,
            CompetitivenessPortrait,
            CareerPortrait,
            GroupClusteringPortrait,
        ]

        for dim_cls in dimension_classes:
            dim = dim_cls()
            try:
                dim.load(data_dir)
                self._dimensions[dim.name] = dim
                print(f"  [OK] {dim.name}: {dim.get_loaded_count()} 条记录")
            except Exception as e:
                print(f"  [FAIL] {dim.name}: {e}")

        self._initialized = True

    def get_dimension(self, name: str) -> Optional[PortraitDimension]:
        """按名称获取画像维度"""
        return self._dimensions.get(name)

    def get_all_dimensions(self) -> dict[str, PortraitDimension]:
        """获取所有画像维度"""
        return self._dimensions.copy()

    def get_all_for_student(self, student_id: str) -> StudentProfile:
        """获取某个学生在所有画像维度的完整数据

        Args:
            student_id: 学生ID

        Returns:
            StudentProfile 对象
        """
        profile = StudentProfile(student_id=student_id)

        # 从各画像维度收集数据
        fail_prob = 0.0
        for name, dim in self._dimensions.items():
            dim_data = dim.get_dimension_data(student_id)
            if dim_data is not None:
                profile.add_dimension(dim_data)

            # 收集挂科概率
            if name == "挂科退学预警":
                fail_prob_row = dim.get_student(student_id)
                if fail_prob_row is not None:
                    try:
                        fail_prob = float(fail_prob_row.get("预测概率", 0))
                    except (ValueError, TypeError):
                        pass

            # 收集群体信息
            if name == "群体聚类":
                from .group_clustering import GroupClusteringPortrait
                if isinstance(dim, GroupClusteringPortrait):
                    profile.group = dim.get_label(student_id)
                    profile.group_cluster = dim.get_cluster_id(student_id)

            # 收集预警标签
            if name == "综合竞争力":
                profile.warning_label = dim.get_label(student_id)

        profile.fail_probability = fail_prob
        profile.overall_risk = self._compute_overall_risk(profile)

        return profile

    def get_student_ids(self) -> set[str]:
        """获取所有画像维度的学生ID交集"""
        if not self._dimensions:
            return set()

        all_id_sets = []
        for dim in self._dimensions.values():
            ids = set(dim.get_all_student_ids())
            if ids:
                all_id_sets.append(ids)

        if not all_id_sets:
            return set()

        # 返回所有学生的并集
        result = all_id_sets[0]
        for s in all_id_sets[1:]:
            result = result | s
        return result

    def get_common_student_ids(self) -> set[str]:
        """获取所有画像维度共有的学生ID"""
        if not self._dimensions:
            return set()

        all_id_sets = []
        for dim in self._dimensions.values():
            ids = set(dim.get_all_student_ids())
            if ids:
                all_id_sets.append(ids)

        if not all_id_sets:
            return set()

        result = all_id_sets[0]
        for s in all_id_sets[1:]:
            result = result & s
        return result

    def _compute_overall_risk(self, profile: StudentProfile) -> RiskLevel:
        """根据各维度风险等级计算综合风险"""
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.SEVERE]

        max_risk_idx = 0
        for dim in profile.dimensions.values():
            idx = risk_order.index(dim.risk_level) if dim.risk_level in risk_order else 0
            max_risk_idx = max(max_risk_idx, idx)

        # 高挂科概率直接提升风险等级
        if profile.fail_probability >= 0.9:
            max_risk_idx = max(max_risk_idx, 3)
        elif profile.fail_probability >= 0.5:
            max_risk_idx = max(max_risk_idx, 2)

        return risk_order[max_risk_idx]

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def dimension_count(self) -> int:
        return len(self._dimensions)


# 全局单例
_registry = None


def get_registry(data_dir=None) -> PortraitRegistry:
    """获取全局画像注册表"""
    global _registry
    if _registry is None or not _registry.is_initialized:
        _registry = PortraitRegistry()
        _registry.initialize(data_dir)
    return _registry
