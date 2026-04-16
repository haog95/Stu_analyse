"""
数据合并模块 - 将多个画像数据文件合并为统一的学生数据集
"""
import pandas as pd

from .config import get_config
from .data_loader import load_portrait_file


def merge_all_portraits(data_dir=None):
    """合并所有画像数据文件为统一的DataFrame

    以群体聚类结果为基底，依次连接其他画像文件。
    所有文件的学生ID列名已由 data_loader 统一为 'student_id'。

    Args:
        data_dir: 数据目录路径

    Returns:
        pd.DataFrame: 合并后的完整数据集
    """
    config = get_config()
    mapping = config.get("id_column_mapping", {})

    # 文件加载顺序：先加载群体聚类作为基底
    base_file = "群体画像最终聚类结果.csv"
    file_order = [base_file] + [f for f in mapping.keys() if f != base_file]

    # 加载基底数据
    merged = load_portrait_file(base_file, data_dir)
    print(f"基底数据: {base_file} ({len(merged)} 条)")

    # 依次合并其他画像
    for filename in file_order[1:]:
        try:
            df = load_portrait_file(filename, data_dir)

            # 为列名添加前缀以避免冲突（保留student_id不加前缀）
            portrait_name = _get_prefix(filename)
            rename_cols = {}
            for col in df.columns:
                if col != "student_id" and col in merged.columns:
                    rename_cols[col] = f"{portrait_name}_{col}"

            if rename_cols:
                df = df.rename(columns=rename_cols)

            # 去重（以student_id去重，保留第一条）
            df = df.drop_duplicates(subset="student_id", keep="first")

            # 左连接，保留基底数据中的所有学生
            merged = merged.merge(df, on="student_id", how="left", suffixes=("", f"_{portrait_name}"))
            print(f"合并: {filename} ({len(df)} 条) -> 总计 {len(merged)} 条")

        except Exception as e:
            print(f"合并跳过: {filename} - {e}")

    print(f"\n最终合并数据: {len(merged)} 名学生, {len(merged.columns)} 个字段")
    return merged


def _get_prefix(filename):
    """根据文件名生成列名前缀"""
    prefixes = {
        "画像1_课堂参与度与专注力评级结果.csv": "p1",
        "画像2_学习投入度与自律性.csv": "p2",
        "画像3_挂科退学高危预警.csv": "p3",
        "画像4_学习轨迹退化预警画像.csv": "p4",
        "画像6_规律性生活.csv": "p6",
        "画像7_身体素质与意志力.csv": "p7",
        "画像8_网络依赖.csv": "p8",
        "画像9_综合竞争力.csv": "p9",
        "画像10_就业竞争力与匹配.csv": "p10",
    }
    for key, prefix in prefixes.items():
        key_clean = key.replace('"', '').replace('"', '')
        fname_clean = filename.replace('"', '').replace('"', '')
        if key_clean in fname_clean or fname_clean in key_clean:
            return prefix
    return "unknown"


def get_numeric_features(merged_df):
    """从合并后的数据中提取数值型特征列

    Returns:
        list of str: 数值型列名列表
    """
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
    # 排除PCA分量和聚类ID
    exclude = {"Cluster", "Cluster_ID", "PCA_Component_1", "PCA_Component_2"}
    return [c for c in numeric_cols if c not in exclude]


def get_students_by_risk(merged_df, risk_level="high"):
    """按风险等级筛选学生

    Args:
        merged_df: 合并后的DataFrame
        risk_level: 'low', 'medium', 'high', 'severe'

    Returns:
        pd.DataFrame: 符合条件的学生子集
    """
    config = get_config()
    thresholds = config.get("risk_thresholds", {})

    fail_col = "预测概率"
    if fail_col not in merged_df.columns:
        return pd.DataFrame()

    prob = merged_df[fail_col].fillna(0)
    ft = thresholds.get("fail_probability", {})

    if risk_level == "severe":
        mask = prob >= ft.get("severe", 0.9)
    elif risk_level == "high":
        mask = (prob >= ft.get("high", 0.5)) & (prob < ft.get("severe", 0.9))
    elif risk_level == "medium":
        mask = (prob >= ft.get("medium", 0.2)) & (prob < ft.get("high", 0.5))
    else:  # low
        mask = prob < ft.get("medium", 0.2)

    return merged_df[mask]
