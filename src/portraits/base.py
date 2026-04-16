"""
画像维度抽象基类
"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from src.core.student_profile import DimensionData, RiskLevel


class PortraitDimension(ABC):
    """画像维度抽象基类

    每个画像维度继承此类，实现标准化的数据访问接口。
    """

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._name: str = ""
        self._description: str = ""
        self._source_file: str = ""
        self._label_column: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data

    def load(self, data_dir=None):
        """加载画像数据"""
        from src.core.data_loader import load_portrait_file
        self._data = load_portrait_file(self._source_file, data_dir)
        return self

    def get_student(self, student_id: str) -> Optional[pd.Series]:
        """获取某个学生的画像数据"""
        if self._data is None:
            return None
        mask = self._data["student_id"] == student_id
        rows = self._data[mask]
        return rows.iloc[0] if len(rows) > 0 else None

    @abstractmethod
    def get_label(self, student_id: str) -> Optional[str]:
        """获取主要分类标签"""
        pass

    @abstractmethod
    def get_risk_level(self, student_id: str) -> RiskLevel:
        """获取风险等级"""
        pass

    @abstractmethod
    def get_numeric_features(self, student_id: str) -> dict:
        """获取数值特征字典"""
        pass

    def get_dimension_data(self, student_id: str) -> Optional[DimensionData]:
        """获取完整的维度数据对象"""
        row = self.get_student(student_id)
        if row is None:
            return None

        return DimensionData(
            name=self._name,
            label=self.get_label(student_id),
            risk_level=self.get_risk_level(student_id),
            features=self.get_numeric_features(student_id),
            extra=self._get_extra_info(row),
        )

    def _get_extra_info(self, row) -> dict:
        """获取额外的描述信息，子类可覆盖"""
        return {}

    def get_loaded_count(self) -> int:
        """返回已加载的学生数量"""
        return len(self._data) if self._data is not None else 0

    def get_all_student_ids(self) -> list[str]:
        """返回所有学生ID列表"""
        if self._data is None:
            return []
        return self._data["student_id"].tolist()
