"""
学生ID映射模块 - 处理不同CSV文件中学生ID列名不一致的问题
"""
import re
from pathlib import Path

from .config import get_config


def get_id_column(filename):
    """根据文件名获取对应的学生ID列名

    Args:
        filename: CSV文件名或完整路径

    Returns:
        该文件对应的学生ID列名
    """
    config = get_config()
    mapping = config.get("id_column_mapping", {})

    # 提取纯文件名
    fname = Path(filename).name

    # 精确匹配
    if fname in mapping:
        return mapping[fname]

    # 模糊匹配（处理文件名中的特殊字符）
    fname_normalized = fname.replace('"', '').replace('"', '')
    for key, col in mapping.items():
        key_normalized = key.replace('"', '').replace('"', '')
        if key_normalized in fname_normalized or fname_normalized in key_normalized:
            return col

    return config.get("unified_id_column", "student_id")


def normalize_student_id(raw_id):
    """标准化学生ID格式

    Args:
        raw_id: 原始学生ID（可能是字符串、数字等）

    Returns:
        标准化后的字符串ID
    """
    if raw_id is None:
        return ""
    sid = str(raw_id).strip()
    # 去除可能的空白字符和换行
    sid = re.sub(r"\s+", "", sid)
    return sid.lower()


def validate_id_consistency(dataframes_with_ids):
    """验证多个DataFrame之间学生ID的一致性

    Args:
        dataframes_with_ids: list of (name, set_of_ids) 元组

    Returns:
        dict with 'common_count', 'total_unique', 'per_file' stats
    """
    all_ids = set()
    for name, ids in dataframes_with_ids:
        all_ids.update(ids)

    common_ids = None
    for name, ids in dataframes_with_ids:
        if common_ids is None:
            common_ids = set(ids)
        else:
            common_ids &= ids

    per_file = {}
    for name, ids in dataframes_with_ids:
        per_file[name] = {
            "count": len(ids),
            "missing_from_common": len(ids - (common_ids or set())),
        }

    return {
        "common_count": len(common_ids) if common_ids else 0,
        "total_unique": len(all_ids),
        "per_file": per_file,
    }
