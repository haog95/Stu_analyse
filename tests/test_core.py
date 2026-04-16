"""
核心数据模块测试
"""
import pandas as pd
from src.core.config import load_config, get_data_dir
from src.core.id_mapper import get_id_column, normalize_student_id
from src.core.data_loader import load_csv, load_portrait_file, get_data_summary
from src.core.data_merger import merge_all_portraits


class TestConfig:
    def test_load_config(self):
        config = load_config()
        assert "data" in config
        assert "id_column_mapping" in config
        assert "risk_thresholds" in config

    def test_get_data_dir(self):
        data_dir = get_data_dir()
        assert data_dir.exists()


class TestIdMapper:
    def test_get_id_column_student_id(self):
        col = get_id_column("画像1_课堂参与度与专注力评级结果.csv")
        assert col == "student_id"

    def test_get_id_column_xh(self):
        col = get_id_column("画像3_挂科退学高危预警.csv")
        assert col == "XH"

    def test_get_id_column_sid(self):
        col = get_id_column("画像10_就业竞争力与匹配.csv")
        assert col == "SID"

    def test_normalize_student_id(self):
        assert normalize_student_id(" ABC ") == "abc"
        assert normalize_student_id(None) == ""
        assert normalize_student_id(123) == "123"


class TestDataLoader:
    def test_load_csv(self, data_dir):
        files = list(data_dir.glob("*.csv"))
        assert len(files) > 0
        df = load_csv(files[0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_portrait_file(self, data_dir):
        df = load_portrait_file("画像1_课堂参与度与专注力评级结果.csv", data_dir)
        assert "student_id" in df.columns
        assert len(df) == 2500

    def test_get_data_summary(self, data_dir):
        summary = get_data_summary(data_dir)
        assert len(summary) == 10
        for name, info in summary.items():
            assert "rows" in info
            assert "columns" in info


class TestDataMerger:
    def test_merge_all_portraits(self, merged_df):
        assert isinstance(merged_df, pd.DataFrame)
        assert len(merged_df) > 2000
        assert "student_id" in merged_df.columns
        assert len(merged_df.columns) > 30

    def test_merged_has_key_columns(self, merged_df):
        key_cols = ["Group_Profile", "Cluster", "预测概率"]
        for col in key_cols:
            assert col in merged_df.columns, f"Missing column: {col}"
