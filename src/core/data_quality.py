"""
数据质量评估模块 - 检测零值、缺失值、异常值，生成数据质量报告
"""
import pandas as pd
import numpy as np
from typing import Optional


class DataQualityAssessor:
    """数据质量评估器

    分析学生数据中的零值比例、缺失率、异常值，
    为报告提供数据质量上下文，避免将数据缺失误读为行为缺失。
    """

    # 可能因数据采集覆盖不全而产生零值的特征
    ZERO_SENSITIVE_FEATURES = [
        "avg_job_rate", "lib_visit_count", "early_bird_rate",
        "night_owl_rate", "total_special_time",
        "Comp_Count", "Comp_Total_Score",
        "Schol_Count", "Schol_Total_Score",
    ]

    def __init__(self, merged_df: pd.DataFrame):
        self.merged_df = merged_df
        self._quality_cache = {}

    def assess_student(self, student_id: str) -> dict:
        """评估单个学生的数据质量

        Args:
            student_id: 学生ID

        Returns:
            数据质量报告字典
        """
        if student_id in self._quality_cache:
            return self._quality_cache[student_id]

        row = self.merged_df[self.merged_df["student_id"] == student_id]
        if row.empty:
            return {"student_id": student_id, "data_available": False}

        row = row.iloc[0]
        report = {
            "student_id": student_id,
            "data_available": True,
            "total_features": 0,
            "zero_features": [],
            "zero_count": 0,
            "zero_ratio": 0.0,
            "missing_features": [],
            "missing_count": 0,
            "quality_level": "good",
            "data_quality_notes": [],
        }

        # 检查零值敏感特征
        for col in self.ZERO_SENSITIVE_FEATURES:
            if col in self.merged_df.columns:
                val = row.get(col)
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    report["missing_features"].append(col)
                    continue

                report["total_features"] += 1
                if val == 0.0:
                    # 判断是"真实的零"还是"数据缺失导致的零"
                    col_zero_rate = (self.merged_df[col] == 0).mean()
                    is_likely_missing = bool(col_zero_rate > 0.8)  # 超过80%为零，很可能是数据问题

                    report["zero_features"].append({
                        "feature": col,
                        "value": 0.0,
                        "likely_missing": is_likely_missing,
                        "population_zero_rate": round(col_zero_rate, 3),
                    })

        # 计算总体指标
        report["zero_count"] = len(report["zero_features"])
        n_checked = report["total_features"]
        report["zero_ratio"] = round(report["zero_count"] / max(n_checked, 1), 3)
        report["missing_count"] = len(report["missing_features"])

        # 生成质量等级
        if report["zero_ratio"] > 0.7:
            report["quality_level"] = "low"
        elif report["zero_ratio"] > 0.4:
            report["quality_level"] = "medium"
        else:
            report["quality_level"] = "good"

        # 生成数据质量说明
        notes = []
        for zf in report["zero_features"]:
            if zf["likely_missing"]:
                notes.append(
                    f"「{zf['feature']}」为零值，但全校{zf['population_zero_rate']*100:.0f}%"
                    f"的学生该指标也为零，可能为数据采集覆盖不全所致"
                )
            else:
                notes.append(
                    f"「{zf['feature']}」为零值，全校仅{zf['population_zero_rate']*100:.0f}%"
                    f"的学生为零，较可能反映真实行为特征"
                )

        report["data_quality_notes"] = notes

        self._quality_cache[student_id] = report
        return report

    def get_summary_for_context(self, student_id: str) -> dict:
        """生成用于 KERAG 上下文注入的数据质量摘要

        Args:
            student_id: 学生ID

        Returns:
            精简的数据质量摘要
        """
        report = self.assess_student(student_id)

        if not report.get("data_available"):
            return {"data_quality": "unavailable"}

        return {
            "quality_level": report["quality_level"],
            "zero_value_features": [
                {"feature": zf["feature"],
                 "likely_data_gap": zf["likely_missing"],
                 "population_zero_rate": zf["population_zero_rate"]}
                for zf in report["zero_features"]
            ],
            "data_quality_notes": report["data_quality_notes"],
        }
