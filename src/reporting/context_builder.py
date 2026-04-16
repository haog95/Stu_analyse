"""
上下文构建器 - KERAG 第二层：将学生画像数据结构化为 JSON
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.core.student_profile import StudentProfile
from src.explanation.risk_attribution import RiskAttribution

# 全谱系聚类标签（供报告对比叙事使用）
CLUSTER_TAXONOMY = {
    0: "积极主导与规律成就型",
    1: "稳健应付与中庸型",
    2: "行为轨迹时序退化型",
    3: "隐性逃课与全面游离型",
    -1: "极端异常离群群体",
}

# 系统发现的行为模式（4类+1类异常）
DISCOVERED_PATTERNS = [
    {
        "pattern_id": "P1",
        "name": "积极主导与规律成就型",
        "behavioral_signature": "高课堂投入、规律作息、主动学习、竞赛参与度高",
        "population_ratio": "41.1%",
        "risk_profile": "低风险为主，偶有单维度中等风险",
        "key_features": ["final_score > 0", "procrastination_index < 0.3", "early_bird_rate > 0"],
    },
    {
        "pattern_id": "P2",
        "name": "稳健应付与中庸型",
        "behavioral_signature": "能完成基本学业但投入有限，缺乏主动拓展，网络依赖偏高",
        "population_ratio": "43.2%",
        "risk_profile": "中低风险，但存在网络依赖和意志力短板的隐性风险",
        "key_features": ["-0.5 < final_score < 0.5", "procrastination_index 0.3-0.6", "沉迷指数 > 0.5"],
    },
    {
        "pattern_id": "P3",
        "name": "行为轨迹时序退化型",
        "behavioral_signature": "曾有良好基础但出现行为退化和学习投入下滑",
        "population_ratio": "7.8%",
        "risk_profile": "高风险，行为偏移分值显著升高",
        "key_features": ["行为偏移分值 > 0.3", "final_score 下降趋势", "干预建议权重 > 0.5"],
    },
    {
        "pattern_id": "P4",
        "name": "隐性逃课与全面游离型",
        "behavioral_signature": "课堂参与极低、网络依赖严重、学习投入全面崩塌",
        "population_ratio": "6.9%",
        "risk_profile": "严重风险，挂科退学概率显著升高",
        "key_features": ["听课质量评级=游离", "沉迷指数 > 0.7", "挂科退学概率 > 0.3"],
    },
]

# 8-10 个学生行为数据分析成果清单
ANALYSIS_RESULTS_CATALOG = [
    {
        "id": "A1",
        "name": "课堂参与度与专注力画像",
        "dimension": "课堂参与度",
        "method": "无监督聚类",
        "outputs": "听课质量评级、是否隐性逃课",
        "description": "基于抬头率、低头率、前排率、签到率等多维课堂行为数据，评估学生的课堂沉浸程度。",
    },
    {
        "id": "A2",
        "name": "学习投入度与自律性画像",
        "dimension": "学习投入度",
        "method": "无监督聚类 + 综合评分",
        "outputs": "投入类型标签、拖延指数、学习投入综合得分",
        "description": "融合作业完成率、图书馆打卡、早起/夜间学习率等18个特征，评估学习过程性投入质量。",
    },
    {
        "id": "A3",
        "name": "挂科退学高危预警模型",
        "dimension": "挂科退学预警",
        "method": "XGBoost 有监督预测",
        "outputs": "挂科退学概率、风险等级",
        "description": "基于课程成绩、考勤状态、学籍异动等特征，预测学生挂科退学风险。",
    },
    {
        "id": "A4",
        "name": "学习轨迹退化预警模型",
        "dimension": "学习轨迹退化",
        "method": "LSTM-Autoencoder 异常检测",
        "outputs": "行为偏移分值、退化画像描述",
        "description": "通过时序行为重构误差检测行为突变拐点，识别学业退化早期信号。",
    },
    {
        "id": "A5",
        "name": "规律性生活画像",
        "dimension": "规律性生活",
        "method": "无监督聚类",
        "outputs": "作息类型、总活跃天数",
        "description": "基于校园活动时序数据分析学生作息规律性，识别深夜活跃型等风险作息。",
    },
    {
        "id": "A6",
        "name": "身体素质与意志力画像",
        "dimension": "身体素质与意志力",
        "method": "规则引擎 + 统计分析",
        "outputs": "锻炼习惯、体型、意志力梯队",
        "description": "结合跑步打卡、运动频次、BMI等数据评估身体素质与自我管理能力。",
    },
    {
        "id": "A7",
        "name": "网络依赖与沉迷画像",
        "dimension": "网络依赖",
        "method": "SVM 异常检测 + 综合评分",
        "outputs": "沉迷指数、沉迷等级、SVM标签",
        "description": "分析上网时长、时段分布、行为模式，评估网络依赖程度及对学业的影响。",
    },
    {
        "id": "A8",
        "name": "综合竞争力画像",
        "dimension": "综合竞争力",
        "method": "TOPSIS 综合评价 + 规则引擎",
        "outputs": "综合竞争力评分、预警标签、13维子评分",
        "description": "整合成绩、素质、能力、奖学金、竞赛等维度，全面评估学生综合竞争力。",
    },
    {
        "id": "A9",
        "name": "就业竞争力画像",
        "dimension": "就业竞争力",
        "method": "有监督分类",
        "outputs": "推荐去向（保研/考研/留学/就业/待业）",
        "description": "基于综合竞争力、学业表现、能力发展等预测学生最适合的发展去向。",
    },
    {
        "id": "A10",
        "name": "群体聚类与生态位定位",
        "dimension": "群体聚类",
        "method": "PCA + 聚类算法",
        "outputs": "群体标签、PCA坐标",
        "description": "将全校学生划分为四大典型群体+异常离群群体，定位个体在同辈中的生态位。",
    },
]


class ContextBuilder:
    """构建学生画像的结构化 JSON 上下文

    将 PortraitRegistry、SHAP 风险归因、群体统计、数据质量评估
    整合为 KERAG 第二层所需的结构化数据，供 LLM 理解和推理。
    """

    def __init__(self, risk_attribution: RiskAttribution = None,
                 model_metrics: dict = None,
                 merged_df: pd.DataFrame = None):
        self.risk_attribution = risk_attribution
        self.model_metrics = model_metrics or {}
        self.merged_df = merged_df
        self._group_stats_cache = None

    def build(self, profile: StudentProfile) -> dict:
        """构建学生的完整上下文 JSON

        Args:
            profile: 学生画像

        Returns:
            结构化的上下文字典
        """
        context = {
            "student_id": profile.student_id,
            "group": profile.group,
            "overall_risk": profile.overall_risk.value,
            "fail_probability": round(profile.fail_probability, 4),
            "warning_label": profile.warning_label,
            "high_risk_dimensions": profile.get_high_risk_dimensions(),
            "portraits": {},
        }

        # 收集各画像维度数据
        for name, dim in profile.dimensions.items():
            context["portraits"][name] = {
                "label": dim.label,
                "risk_level": dim.risk_level.value,
            }
            # 添加数值特征
            if dim.features:
                for k, v in dim.features.items():
                    if isinstance(v, (int, float)):
                        context["portraits"][name][k] = round(v, 4) if isinstance(v, float) else v
                    else:
                        context["portraits"][name][k] = str(v)
            # 添加额外信息
            if dim.extra:
                for k, v in dim.extra.items():
                    if k not in context["portraits"][name]:
                        context["portraits"][name][k] = str(v)

        # 添加 SHAP 风险归因（增强版：Top 5 风险因子 + Top 5 保护因子）
        if self.risk_attribution is not None:
            shap_data = self.risk_attribution.get_attribution(profile.student_id, top_k=5)
            if shap_data.get("shap_top3"):
                context["shap_risk_factors"] = shap_data["shap_top3"]
            if shap_data.get("shap_top3_protective"):
                context["shap_protective_factors"] = shap_data["shap_top3_protective"]

        # 注入模型评估指标（AUC、交叉验证等）
        if self.model_metrics:
            context["model_performance"] = self.model_metrics

        # 注入全谱系聚类信息
        context["cluster_taxonomy"] = CLUSTER_TAXONOMY

        # 注入系统发现的行为模式（4类）
        context["discovered_patterns"] = DISCOVERED_PATTERNS

        # 注入分析成果清单（10项）
        context["analysis_results_catalog"] = ANALYSIS_RESULTS_CATALOG

        # 注入群体统计对比数据
        if self.merged_df is not None:
            context["group_statistics"] = self._build_group_stats(profile)

        # 注入数据质量评估
        if self.merged_df is not None:
            context["data_quality"] = self._build_data_quality(profile)

        return context

    def build_json_string(self, profile: StudentProfile) -> str:
        """构建 JSON 字符串"""
        return json.dumps(self.build(profile), ensure_ascii=False, indent=2)

    def _build_group_stats(self, profile: StudentProfile) -> dict:
        """构建个体 vs 群体 vs 全校的统计对比"""
        if self.merged_df is None:
            return {}

        row = self.merged_df[self.merged_df["student_id"] == profile.student_id]
        if row.empty:
            return {}

        group_col = "Group_Profile" if "Group_Profile" in self.merged_df.columns else None
        if not group_col:
            return {}

        student_group = row.iloc[0].get(group_col)
        group_df = self.merged_df[self.merged_df[self.merged_df.columns[0]].notna()]
        if group_col in self.merged_df.columns:
            group_df = self.merged_df[self.merged_df[group_col] == student_group]

        # 关键指标的对比
        comparison_metrics = [
            "final_score", "procrastination_index", "沉迷指数",
            "ZYNJ_Score", "行为偏移分值", "total_active_days",
            "avg_job_rate", "lib_visit_count",
        ]

        comparisons = []
        for metric in comparison_metrics:
            if metric not in self.merged_df.columns:
                continue

            try:
                student_val = float(row.iloc[0].get(metric, np.nan))
                group_mean = float(group_df[metric].mean())
                group_std = float(group_df[metric].std())
                school_mean = float(self.merged_df[metric].mean())

                # 计算百分位
                all_vals = self.merged_df[metric].dropna()
                percentile = float((all_vals < student_val).mean() * 100) if not np.isnan(student_val) else None

                # 计算偏差（以标准差为单位）
                deviation_sd = float((student_val - group_mean) / max(group_std, 1e-6))

                comparisons.append({
                    "metric": metric,
                    "student_value": round(student_val, 4) if not np.isnan(student_val) else None,
                    "group_mean": round(group_mean, 4),
                    "group_std": round(group_std, 4),
                    "school_mean": round(school_mean, 4),
                    "deviation_sd": round(deviation_sd, 2),
                    "percentile": round(percentile, 1) if percentile is not None else None,
                })
            except (ValueError, TypeError):
                continue

        return {
            "student_group": student_group,
            "group_size": len(group_df),
            "group_population_ratio": round(len(group_df) / len(self.merged_df) * 100, 1),
            "total_students": len(self.merged_df),
            "comparisons": comparisons,
        }

    def _build_data_quality(self, profile: StudentProfile) -> dict:
        """构建数据质量评估"""
        if self.merged_df is None:
            return {}

        from src.core.data_quality import DataQualityAssessor
        assessor = DataQualityAssessor(self.merged_df)
        return assessor.get_summary_for_context(profile.student_id)
