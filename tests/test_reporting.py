"""
KERAG 报告生成测试
"""
import json

from src.reporting.report_classifier import classify_report_types, get_report_info
from src.reporting.context_builder import ContextBuilder
from src.reporting.prompt_assembler import PromptAssembler
from src.reporting.knowledge_retriever import KnowledgeRetriever
from src.core.student_profile import RiskLevel


class TestReportClassifier:
    def test_classify_normal_student(self, registry, sample_student_id):
        profile = registry.get_all_for_student(sample_student_id)
        types = classify_report_types(profile)
        assert isinstance(types, list)
        assert len(types) > 0

    def test_get_report_info(self):
        info = get_report_info("motivation")
        assert info["name"] == "学习动机归因分析报告"
        assert "template" in info


class TestContextBuilder:
    def test_build_context(self, registry, sample_student_id):
        profile = registry.get_all_for_student(sample_student_id)
        builder = ContextBuilder()
        context = builder.build(profile)
        assert "student_id" in context
        assert "portraits" in context
        assert context["student_id"] == sample_student_id

    def test_build_json_string(self, registry, sample_student_id):
        profile = registry.get_all_for_student(sample_student_id)
        builder = ContextBuilder()
        json_str = builder.build_json_string(profile)
        parsed = json.loads(json_str)
        assert "student_id" in parsed


class TestPromptAssembler:
    def test_assemble(self):
        assembler = PromptAssembler()
        context = '{"student_id": "test"}'
        system, user = assembler.assemble(context, [], "motivation")
        assert len(system) > 0
        assert len(user) > 0
        assert "test" in user

    def test_assemble_with_knowledge(self):
        assembler = PromptAssembler()
        context = '{"student_id": "test"}'
        knowledge = ["知识片段1", "知识片段2"]
        system, user = assembler.assemble(context, knowledge, "comprehensive")
        assert "知识片段1" in user


class TestKnowledgeRetriever:
    def test_retrieve(self):
        retriever = KnowledgeRetriever()
        results = retriever.retrieve(["网络依赖", "沉迷"])
        assert isinstance(results, list)

    def test_retrieve_career(self):
        retriever = KnowledgeRetriever()
        results = retriever.retrieve(["职业", "就业"])
        assert isinstance(results, list)
