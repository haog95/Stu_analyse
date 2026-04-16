"""
端到端集成测试
"""
import json
from src.core.data_merger import merge_all_portraits
from src.explanation.surrogate_model import SurrogateModel
from src.explanation.shap_analyzer import SHAPAnalyzer
from src.explanation.risk_attribution import RiskAttribution
from src.reporting.context_builder import ContextBuilder
from src.reporting.prompt_assembler import PromptAssembler
from src.reporting.report_classifier import classify_report_types


class TestEndToEnd:
    """端到端集成测试：数据加载 → 画像构建 → SHAP → 报告组装"""

    def test_full_pipeline(self, merged_df, registry, sample_student_id):
        # 1. 构建画像
        profile = registry.get_all_for_student(sample_student_id)
        assert profile is not None
        assert len(profile.dimensions) == 10

        # 2. 训练代理模型
        surrogate = SurrogateModel()
        scores = surrogate.train(merged_df)
        assert scores["test_score"] > 0.9

        # 3. 计算 SHAP
        X, _ = surrogate.prepare_features(merged_df)
        X_small = X.iloc[:5]
        analyzer = SHAPAnalyzer(surrogate)
        shap_values = analyzer.compute_shap_values(X_small)
        assert shap_values is not None

        # 4. 风险归因
        attribution = RiskAttribution(analyzer, merged_df.iloc[:5])
        attr = attribution.get_attribution(sample_student_id)
        assert "student_id" in attr
        assert "shap_top3" in attr

        # 5. 上下文构建
        context_builder = ContextBuilder(attribution)
        context = context_builder.build(profile)
        assert "portraits" in context
        assert "shap_top3" in context

        # 6. 提示组装
        assembler = PromptAssembler()
        context_json = json.dumps(context, ensure_ascii=False)
        system, user = assembler.assemble(context_json, [], "motivation")
        assert len(system) > 100
        assert len(user) > 200
        assert sample_student_id in user

    def test_report_classification(self, registry, merged_df):
        # 测试不同类型学生的报告分类
        risk_df = merged_df[merged_df["预测概率"] > 0.99]
        if len(risk_df) > 0:
            high_risk_id = str(risk_df.iloc[0]["student_id"])
            profile = registry.get_all_for_student(high_risk_id)
            types = classify_report_types(profile)
            assert "comprehensive" in types or "addiction" in types or "success" in types
