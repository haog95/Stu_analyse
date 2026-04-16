"""
SHAP 可解释性分析运行脚本
"""
import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_merger import merge_all_portraits
from src.explanation.surrogate_model import SurrogateModel
from src.explanation.shap_analyzer import SHAPAnalyzer
from src.explanation.risk_attribution import RiskAttribution


def main():
    parser = argparse.ArgumentParser(description="SHAP可解释性分析")
    parser.add_argument("--output", type=str, default="output/shap", help="输出目录")
    parser.add_argument("--student-id", type=str, default=None, help="查看单个学生的风险归因")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K特征数")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 合并数据
    print("=== 步骤1: 合并画像数据 ===")
    merged_df = merge_all_portraits()
    print()

    # 2. 训练代理模型
    print("=== 步骤2: 训练代理模型 ===")
    surrogate = SurrogateModel()
    scores = surrogate.train(merged_df)
    print()

    # 3. 计算 SHAP 值
    print("=== 步骤3: 计算SHAP值 ===")
    from src.explanation.surrogate_model import FEATURE_COLUMNS
    available_cols = [c for c in FEATURE_COLUMNS if c in merged_df.columns]
    X = merged_df[available_cols].fillna(0).replace([float('inf'), float('-inf')], 0)

    analyzer = SHAPAnalyzer(surrogate)
    analyzer.compute_shap_values(X)

    # 缓存SHAP值
    analyzer.save_shap_cache(output_dir)
    print()

    # 4. 生成风险归因
    print("=== 步骤4: 生成风险归因 ===")
    attribution = RiskAttribution(analyzer, merged_df)

    if args.student_id:
        attr = attribution.get_attribution(args.student_id, args.top_k)
        print(json.dumps(attr, ensure_ascii=False, indent=2))
    else:
        attribution.save_attributions(output_dir / "risk_attributions.json")

    print("\nSHAP分析完成!")


if __name__ == "__main__":
    main()
