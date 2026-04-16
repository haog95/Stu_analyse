"""
风险归因模块 - 将SHAP值转化为人类可读的风险归因陈述
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .feature_catalog import get_feature_cn_name, get_risk_direction_text
from .shap_analyzer import SHAPAnalyzer


class RiskAttribution:
    """风险归因分析

    将 SHAP 值转化为结构化的风险归因 JSON，
    用于注入 KERAG 第二层的 Context JSON。
    """

    def __init__(self, shap_analyzer: SHAPAnalyzer, merged_df: pd.DataFrame):
        self.analyzer = shap_analyzer
        self.merged_df = merged_df
        self._student_id_to_idx = {}
        self._build_index()

    def _build_index(self):
        """构建学生ID到数据索引的映射"""
        if "student_id" in self.merged_df.columns:
            ids = self.merged_df["student_id"].values
            for i, sid in enumerate(ids):
                self._student_id_to_idx[str(sid)] = i

    def get_attribution(self, student_id: str, top_k: int = 3) -> dict:
        """获取学生的风险归因

        Args:
            student_id: 学生ID
            top_k: Top-K特征数

        Returns:
            结构化的风险归因字典
        """
        idx = self._student_id_to_idx.get(student_id)
        if idx is None:
            return self._empty_attribution(student_id)

        # 获取SHAP风险/保护因子
        factors = self.analyzer.get_student_risk_factors(idx, top_k)
        risk_factors = factors["risk_factors"]
        protective_factors = factors["protective_factors"]

        # 获取学生实际特征值
        row = self.merged_df.iloc[idx]

        # 转化为可读的风险因子描述
        shap_top3 = []
        for f in risk_factors:
            feature = f["feature"]
            value = f["value"]
            cn_name = get_feature_cn_name(feature)

            # 获取实际值
            actual_val = row.get(feature, "N/A")
            try:
                actual_val = round(float(actual_val), 4)
            except (ValueError, TypeError):
                actual_val = str(actual_val)

            direction_text = get_risk_direction_text(feature, value)

            shap_top3.append({
                "factor": f"{direction_text}({cn_name}={actual_val})",
                "feature": feature,
                "actual_value": actual_val,
                "contribution": round(abs(value), 4),
            })

        shap_top3_protective = []
        for f in protective_factors:
            feature = f["feature"]
            value = f["value"]
            cn_name = get_feature_cn_name(feature)

            actual_val = row.get(feature, "N/A")
            try:
                actual_val = round(float(actual_val), 4)
            except (ValueError, TypeError):
                actual_val = str(actual_val)

            direction_text = get_risk_direction_text(feature, value)

            shap_top3_protective.append({
                "factor": f"{direction_text}({cn_name}={actual_val})",
                "feature": feature,
                "actual_value": actual_val,
                "contribution": round(abs(value), 4),
            })

        return {
            "student_id": student_id,
            "shap_top3": shap_top3,
            "shap_top3_protective": shap_top3_protective,
        }

    def get_attribution_json(self, student_id: str, top_k: int = 3) -> str:
        """获取格式化的JSON字符串"""
        attr = self.get_attribution(student_id, top_k)
        return json.dumps(attr, ensure_ascii=False, indent=2)

    def batch_generate(self, student_ids: list[str] = None, top_k: int = 3) -> list[dict]:
        """批量生成风险归因

        Args:
            student_ids: 学生ID列表，None则处理全部
            top_k: Top-K特征数

        Returns:
            风险归因列表
        """
        if student_ids is None:
            student_ids = list(self._student_id_to_idx.keys())

        results = []
        for sid in student_ids:
            results.append(self.get_attribution(sid, top_k))

        return results

    def save_attributions(self, output_path: str, student_ids: list[str] = None):
        """保存风险归因到文件"""
        attributions = self.batch_generate(student_ids)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(attributions, f, ensure_ascii=False, indent=2)

        print(f"风险归因已保存到 {output_path} ({len(attributions)} 名学生)")

    def _empty_attribution(self, student_id: str) -> dict:
        return {
            "student_id": student_id,
            "shap_top3": [],
            "shap_top3_protective": [],
        }
