"""
画像维度模块测试
"""
from src.core.student_profile import RiskLevel, StudentProfile


class TestRegistry:
    def test_initialize(self, registry):
        assert registry.is_initialized
        assert registry.dimension_count == 10

    def test_get_all_for_student(self, registry, sample_student_id):
        profile = registry.get_all_for_student(sample_student_id)
        assert isinstance(profile, StudentProfile)
        assert profile.student_id == sample_student_id
        assert len(profile.dimensions) == 10

    def test_classroom_portrait(self, registry, sample_student_id):
        classroom = registry.get_dimension("课堂参与度")
        label = classroom.get_label(sample_student_id)
        assert label in ("沉浸", "一般", "游离")

    def test_risk_prediction(self, registry, sample_student_id):
        risk_dim = registry.get_dimension("挂科退学预警")
        prob = risk_dim.get_fail_probability(sample_student_id)
        assert prob is not None
        assert 0 <= prob <= 1

    def test_network_portrait(self, registry, sample_student_id):
        network_dim = registry.get_dimension("网络依赖")
        label = network_dim.get_label(sample_student_id)
        assert label is not None

    def test_context_json(self, registry, sample_student_id):
        profile = registry.get_all_for_student(sample_student_id)
        json_data = profile.to_context_json()
        assert "student_id" in json_data
        assert "portraits" in json_data
        assert len(json_data["portraits"]) == 10
