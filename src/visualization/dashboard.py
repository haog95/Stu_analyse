"""
Streamlit 可视化仪表盘（增强版 - 含 SHAP 瀑布图、群体对比、轨迹图）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.core.config import load_config
from src.portraits.registry import PortraitRegistry
from src.core.data_merger import merge_all_portraits
from src.visualization.student_radar import plot_student_radar
from src.visualization.group_scatter import plot_group_scatter
from src.visualization.risk_heatmap import plot_risk_heatmap
from src.visualization.shap_waterfall import plot_shap_waterfall
from src.visualization.group_comparison import (
    plot_group_comparison,
    plot_group_distribution,
)
from src.visualization.trajectory_plot import (
    plot_behavior_trajectory,
    plot_group_trajectory_comparison,
)

# 确保中文字体已初始化（各模块 import 时已自动执行）
from src.visualization.font_config import chinese_available


st.set_page_config(page_title="学生多维画像分析系统", layout="wide")
st.title("学生多维画像分析与个性化干预报告系统")


@st.cache_resource
def load_data():
    """加载数据（缓存）"""
    config = load_config()
    registry = PortraitRegistry()
    registry.initialize()
    merged_df = merge_all_portraits()
    return registry, merged_df


registry, merged_df = load_data()

# 侧边栏
st.sidebar.header("功能导航")
page = st.sidebar.radio("选择页面", [
    "数据总览", "学生画像查询", "群体分析", "风险热力图", "SHAP归因分析", "群体对比"
])

if page == "数据总览":
    st.header("数据总览")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("学生总数", f"{len(merged_df)}")
    with col2:
        if "预测概率" in merged_df.columns:
            high_risk = (merged_df["预测概率"] > 0.5).sum()
            st.metric("高风险学生", f"{high_risk}")
    with col3:
        if "Group_Profile" in merged_df.columns:
            n_groups = merged_df["Group_Profile"].nunique()
            st.metric("群体类别", f"{n_groups}")

    # 群体分布
    if "Group_Profile" in merged_df.columns:
        st.subheader("群体分布")
        group_counts = merged_df["Group_Profile"].value_counts()
        st.bar_chart(group_counts)

    # 数据概览表
    st.subheader("数据概览")
    st.dataframe(merged_df.describe(), use_container_width=True)

elif page == "学生画像查询":
    st.header("学生画像查询")
    student_ids = merged_df["student_id"].tolist()
    selected_id = st.selectbox("选择学生", student_ids, index=0)

    if selected_id:
        profile = registry.get_all_for_student(selected_id)
        st.subheader(f"学生 {selected_id} 画像")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**群体**: {profile.group}")
            st.write(f"**综合风险**: {profile.overall_risk.value}")
            st.write(f"**挂科概率**: {profile.fail_probability:.4f}")
            st.write(f"**预警标签**: {profile.warning_label}")

        with col2:
            plot_student_radar(profile)
            st.pyplot(plt.gcf())

        # 各维度详情
        st.subheader("各维度详情")
        for name, dim in profile.dimensions.items():
            with st.expander(f"{name} - {dim.label} ({dim.risk_level.value})"):
                if dim.features:
                    st.json({k: str(v) for k, v in dim.features.items()})
                if dim.extra:
                    st.json({k: str(v) for k, v in dim.extra.items()})

        # 行为轨迹对比
        st.subheader("行为指标对比")
        plot_behavior_trajectory(merged_df, selected_id)
        st.pyplot(plt.gcf())

elif page == "群体分析":
    st.header("群体聚类分析")
    plot_group_scatter(merged_df)
    st.pyplot(plt.gcf())

    if "Group_Profile" in merged_df.columns:
        st.subheader("群体统计")
        for group in merged_df["Group_Profile"].unique():
            group_df = merged_df[merged_df["Group_Profile"] == group]
            with st.expander(f"{group} ({len(group_df)}人)"):
                st.dataframe(group_df.describe(), use_container_width=True)

elif page == "风险热力图":
    st.header("风险热力图")
    n_students = st.slider("显示学生数", 5, 50, 20)
    # 取前N个学生
    sample_ids = merged_df["student_id"].tolist()[:n_students]
    profiles = [registry.get_all_for_student(sid) for sid in sample_ids]
    plot_risk_heatmap(profiles)
    st.pyplot(plt.gcf())

elif page == "SHAP归因分析":
    st.header("SHAP 特征归因分析")

    student_ids = merged_df["student_id"].tolist()
    selected_id = st.selectbox("选择学生", student_ids, index=0, key="shap_student")

    if selected_id:
        # 训练代理模型并计算 SHAP
        with st.spinner("正在计算 SHAP 值..."):
            from src.explanation.surrogate_model import SurrogateModel
            from src.explanation.shap_analyzer import SHAPAnalyzer

            surrogate = SurrogateModel()
            surrogate.train(merged_df)
            X, _ = surrogate.prepare_features(merged_df)
            analyzer = SHAPAnalyzer(surrogate)
            analyzer.compute_shap_values(X)

            from src.explanation.risk_attribution import RiskAttribution
            risk_attr = RiskAttribution(analyzer, merged_df)
            shap_data = risk_attr.get_attribution(selected_id, top_k=5)

        # 展示 SHAP 瀑布图
        st.subheader("SHAP 特征贡献瀑布图")
        plot_shap_waterfall(shap_data, selected_id, top_k=10)
        st.pyplot(plt.gcf())

        # 展示风险因子详情
        st.subheader("风险因子详情")
        for f in shap_data.get("shap_top3", []):
            st.write(f"- **{f['factor']}** (贡献度: {f['contribution']:.4f})")

        st.subheader("保护性因子详情")
        for f in shap_data.get("shap_top3_protective", []):
            st.write(f"- **{f['factor']}** (贡献度: {f['contribution']:.4f})")

        # 模型性能指标
        st.subheader("模型性能")
        metrics = surrogate.get_metrics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("测试集准确率", f"{metrics.get('test_accuracy', 0):.2%}")
        with col2:
            st.metric("AUC (OvR)", f"{metrics.get('auc_ovr', 0):.4f}")
        with col3:
            st.metric("五折CV", f"{metrics.get('cv_5fold_mean', 0):.2%}")
        with col4:
            st.metric("特征数", f"{metrics.get('n_features', 0)}")

elif page == "群体对比":
    st.header("个体-群体-全校三维对比")

    student_ids = merged_df["student_id"].tolist()
    selected_id = st.selectbox("选择学生", student_ids, index=0, key="group_comp")

    if selected_id:
        st.subheader("多指标三维对比图")
        plot_group_comparison(merged_df, selected_id)
        st.pyplot(plt.gcf())

        st.subheader("群体均值对比")
        plot_group_trajectory_comparison(merged_df, selected_id)
        st.pyplot(plt.gcf())

        # 单指标分布位置
        st.subheader("单指标分布位置")
        metric_options = [c for c in merged_df.columns
                          if merged_df[c].dtype in ["float64", "int64"]
                          and merged_df[c].notna().sum() > 100]
        selected_metric = st.selectbox("选择指标", metric_options, index=0)
        if selected_metric:
            plot_group_distribution(merged_df, selected_id, selected_metric)
            st.pyplot(plt.gcf())

if __name__ == "__main__":
    pass
