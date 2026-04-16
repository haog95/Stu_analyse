"""
统一数据加载器 - 处理CSV文件加载、编码和表头检测
"""
import os
from pathlib import Path

import pandas as pd

from .config import get_config, get_data_dir
from .id_mapper import get_id_column


def load_csv(filepath, **kwargs):
    """加载单个CSV文件，自动处理编码和表头

    Args:
        filepath: CSV文件路径
        **kwargs: 传递给 pandas.read_csv 的额外参数

    Returns:
        pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    # 尝试多种编码
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None:
        raise ValueError(f"无法读取文件 {filepath}，尝试了所有编码")

    # 清理列名：去除前后空格
    df.columns = df.columns.str.strip()

    return df


def load_portrait_file(filename, data_dir=None):
    """加载指定画像数据文件

    Args:
        filename: 画像文件名
        data_dir: 数据目录路径（默认使用配置中的路径）

    Returns:
        pd.DataFrame，已标准化学生ID列
    """
    if data_dir is None:
        data_dir = get_data_dir()

    filepath = _find_file(filename, data_dir)
    df = load_csv(filepath)

    # 标准化学生ID列名
    id_col = get_id_column(filename)
    if id_col in df.columns:
        df = df.rename(columns={id_col: "student_id"})
        df["student_id"] = df["student_id"].astype(str).str.strip()

    # 去除全为空的行
    df = df.dropna(how="all")

    return df


def load_all_portraits(data_dir=None):
    """加载所有画像数据文件

    Args:
        data_dir: 数据目录路径

    Returns:
        dict: {画像名称: DataFrame}
    """
    if data_dir is None:
        data_dir = get_data_dir()

    config = get_config()
    mapping = config.get("id_column_mapping", {})

    portraits = {}
    for filename in mapping.keys():
        try:
            name = _portrait_name_from_filename(filename)
            portraits[name] = load_portrait_file(filename, data_dir)
            print(f"  已加载: {filename} ({len(portraits[name])} 条记录)")
        except Exception as e:
            print(f"  加载失败: {filename} - {e}")

    return portraits


def _find_file(filename, data_dir):
    """在数据目录中查找文件（处理文件名中的特殊字符）"""
    data_dir = Path(data_dir)

    # 精确匹配
    if (data_dir / filename).exists():
        return data_dir / filename

    # 模糊匹配
    filename_clean = filename.replace('"', '').replace('"', '')
    for f in data_dir.glob("*.csv"):
        f_clean = f.name.replace('"', '').replace('"', '')
        if f_clean == filename_clean:
            return f

    # 前缀匹配
    for f in data_dir.glob("*.csv"):
        if filename[:6] in f.name:
            return f

    raise FileNotFoundError(f"在 {data_dir} 中找不到文件: {filename}")


def _portrait_name_from_filename(filename):
    """从文件名提取画像名称"""
    name = Path(filename).stem
    # 去除引号
    name = name.replace('"', '').replace('"', '')
    return name


def get_data_summary(data_dir=None):
    """获取数据目录中所有CSV文件的摘要信息"""
    if data_dir is None:
        data_dir = get_data_dir()

    data_dir = Path(data_dir)
    summary = {}

    for f in sorted(data_dir.glob("*.csv")):
        try:
            df = load_csv(f)
            id_col = get_id_column(f.name)
            summary[f.name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "id_column": id_col,
                "file_size": f.stat().st_size,
            }
        except Exception as e:
            summary[f.name] = {"error": str(e)}

    return summary
