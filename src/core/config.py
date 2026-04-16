"""
全局配置模块 - 加载和管理系统配置
"""
import os
from pathlib import Path

import yaml


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 默认配置
DEFAULT_CONFIG = {
    "data": {
        "portrait_dir": "data/",
        "output_dir": "output/",
        "reports_dir": "reports/",
    },
    "unified_id_column": "student_id",
    "id_column_mapping": {},
    "risk_thresholds": {
        "fail_probability": {"severe": 0.9, "high": 0.5, "medium": 0.2},
        "degradation_score": {"severe": 0.8, "high": 0.5, "medium": 0.3},
        "addiction_index": {"severe": 0.8, "high": 0.6, "medium": 0.3},
    },
    "llm": {
        "provider": "openai",
        "openai": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
    },
    "knowledge_base": {
        "embedding_model": "shibing624/text2vec-base-chinese",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 3,
    },
    "report": {"min_word_count": 800, "batch_size": 10, "max_workers": 3},
}


def load_config(config_path=None):
    """加载配置文件，合并默认配置"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "settings.yaml"
    else:
        config_path = Path(config_path)

    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
            if user_config:
                _deep_merge(config, user_config)

    return config


def _deep_merge(base, override):
    """深度合并字典"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_data_dir(config=None):
    """获取数据目录的绝对路径"""
    if config is None:
        config = load_config()
    data_dir = config["data"]["portrait_dir"]
    if not os.path.isabs(data_dir):
        return PROJECT_ROOT / data_dir
    return Path(data_dir)


# 全局单例
_config = None


def get_config():
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
