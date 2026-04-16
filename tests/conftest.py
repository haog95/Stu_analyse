"""
Pytest 测试配置
"""
import sys
from pathlib import Path

import pytest

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def data_dir():
    """数据目录路径"""
    return PROJECT_ROOT / "data"


@pytest.fixture
def config():
    """加载配置"""
    from src.core.config import load_config
    return load_config()


@pytest.fixture
def merged_df(data_dir):
    """合并后的数据"""
    from src.core.data_merger import merge_all_portraits
    return merge_all_portraits(data_dir)


@pytest.fixture
def registry(data_dir):
    """画像注册表"""
    from src.portraits.registry import PortraitRegistry
    reg = PortraitRegistry()
    reg.initialize(data_dir)
    return reg


@pytest.fixture
def sample_student_id(merged_df):
    """测试用的学生ID"""
    return str(merged_df.iloc[0]["student_id"])
