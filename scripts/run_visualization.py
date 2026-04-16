"""
可视化生成脚本 - 读取学生画像 JSON 文件，生成配套可视化图表

用法:
    # 单个学生（通过画像 JSON 文件）
    python scripts/run_visualization.py --profile reports/pjxyqwbj001_20260415_100115_profile.json

    # 单个学生（通过学生ID，自动加载数据）
    python scripts/run_visualization.py --student-id pjxyqwbj001

    # 批量生成（所有 JSON 文件）
    python scripts/run_visualization.py --batch-dir reports/

    # 指定输出目录
    python scripts/run_visualization.py --student-id pjxyqwbj001 --output output/viz
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config
from src.portraits.registry import PortraitRegistry
from src.core.data_merger import merge_all_portraits
from src.explanation.surrogate_model import SurrogateModel
from src.explanation.shap_analyzer import SHAPAnalyzer
from src.explanation.risk_attribution import RiskAttribution


def generate_all_charts(profile_json: dict, merged_df, registry,
                        student_id: str, output_dir: Path):
    """为单个学生生成所有可视化图表

    Args:
        profile_json: 画像 JSON 数据（来自 context_builder.build()）
        merged_df: 合并后的 DataFrame
        registry: 画像注册表
        student_id: 学生ID
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. 雷达图
    try:
        from src.visualization.student_radar import plot_student_radar
        profile = registry.get_all_for_student(student_id)
        path = str(output_dir / f"{student_id}_radar.png")
        plot_student_radar(profile, save_path=path)
        generated.append(("雷达图", path))
        print(f"  [OK] 雷达图: {path}")
    except Exception as e:
        print(f"  [WARN] 雷达图生成失败: {e}")

    # 2. SHAP 瀑布图
    try:
        from src.visualization.shap_waterfall import plot_shap_waterfall
        shap_data = {
            "shap_top3": profile_json.get("shap_risk_factors", []),
            "shap_top3_protective": profile_json.get("shap_protective_factors", []),
        }
        if shap_data["shap_top3"] or shap_data["shap_top3_protective"]:
            path = str(output_dir / f"{student_id}_shap_waterfall.png")
            plot_shap_waterfall(shap_data, student_id, top_k=10, save_path=path)
            generated.append(("SHAP瀑布图", path))
            print(f"  [OK] SHAP瀑布图: {path}")
    except Exception as e:
        print(f"  [WARN] SHAP瀑布图生成失败: {e}")

    # 3. 群体对比图
    try:
        from src.visualization.group_comparison import (
            plot_group_comparison,
            plot_group_distribution,
        )
        path = str(output_dir / f"{student_id}_group_comparison.png")
        plot_group_comparison(merged_df, student_id, save_path=path)
        generated.append(("群体对比图", path))
        print(f"  [OK] 群体对比图: {path}")

        # 关键指标分布位置图
        key_metrics = ["final_score", "沉迷指数", "ZYNJ_Score"]
        for metric in key_metrics:
            if metric in merged_df.columns:
                mpath = str(output_dir / f"{student_id}_dist_{metric}.png")
                plot_group_distribution(merged_df, student_id, metric, save_path=mpath)
                generated.append((f"分布图({metric})", mpath))
                print(f"  [OK] 分布图({metric}): {mpath}")
    except Exception as e:
        print(f"  [WARN] 群体对比图生成失败: {e}")

    # 4. 行为轨迹对比图
    try:
        from src.visualization.trajectory_plot import (
            plot_behavior_trajectory,
            plot_group_trajectory_comparison,
        )
        path = str(output_dir / f"{student_id}_trajectory.png")
        plot_behavior_trajectory(merged_df, student_id, save_path=path)
        generated.append(("行为轨迹图", path))
        print(f"  [OK] 行为轨迹图: {path}")

        path2 = str(output_dir / f"{student_id}_group_trajectory.png")
        plot_group_trajectory_comparison(merged_df, student_id, save_path=path2)
        generated.append(("群体轨迹对比图", path2))
        print(f"  [OK] 群体轨迹对比图: {path2}")
    except Exception as e:
        print(f"  [WARN] 行为轨迹图生成失败: {e}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="学生画像可视化图表生成")
    parser.add_argument("--student-id", type=str, help="学生ID（自动加载数据）")
    parser.add_argument("--profile", type=str, help="画像 JSON 文件路径")
    parser.add_argument("--batch-dir", type=str, help="批量处理目录（扫描 *_profile.json）")
    parser.add_argument("--output", type=str, default="output/viz", help="输出目录")
    parser.add_argument("--portraits", type=str, default="data/", help="画像数据目录")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # 加载数据
    print("=== 加载数据 ===")
    registry = PortraitRegistry()
    registry.initialize(args.portraits)
    merged_df = merge_all_portraits(args.portraits)
    print()

    if args.profile:
        # 通过 JSON 文件
        json_path = Path(args.profile)
        if not json_path.exists():
            print(f"文件不存在: {json_path}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            profile_json = json.load(f)

        student_id = profile_json.get("student_id", "")
        if not student_id:
            # 从文件名推断
            student_id = json_path.stem.split("_")[0]

        print(f"=== 生成可视化: {student_id} ===")
        charts = generate_all_charts(
            profile_json, merged_df, registry, student_id, output_dir
        )
        print(f"\n完成! 共生成 {len(charts)} 个图表")

    elif args.student_id:
        # 通过学生ID，自动构建画像
        print(f"=== 构建画像: {args.student_id} ===")

        # 训练代理模型获取 SHAP 数据
        from src.reporting.context_builder import ContextBuilder
        surrogate = SurrogateModel()
        surrogate.train(merged_df)
        X, _ = surrogate.prepare_features(merged_df)
        analyzer = SHAPAnalyzer(surrogate)
        analyzer.compute_shap_values(X)
        risk_attribution = RiskAttribution(analyzer, merged_df)

        context_builder = ContextBuilder(
            risk_attribution, surrogate.get_metrics(), merged_df=merged_df
        )
        profile = registry.get_all_for_student(args.student_id)
        profile_json = context_builder.build(profile)

        print(f"\n=== 生成可视化: {args.student_id} ===")
        charts = generate_all_charts(
            profile_json, merged_df, registry, args.student_id, output_dir
        )
        print(f"\n完成! 共生成 {len(charts)} 个图表")

    elif args.batch_dir:
        # 批量处理
        batch_dir = Path(args.batch_dir)
        json_files = list(batch_dir.glob("*_profile.json"))

        if not json_files:
            print(f"在 {batch_dir} 中未找到 *_profile.json 文件")
            return

        print(f"=== 批量生成可视化 ({len(json_files)} 个文件) ===\n")

        total_charts = 0
        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                profile_json = json.load(f)

            student_id = profile_json.get("student_id", json_path.stem.split("_")[0])
            print(f"[{student_id}]")
            charts = generate_all_charts(
                profile_json, merged_df, registry, student_id, output_dir
            )
            total_charts += len(charts)
            print()

        print(f"批量完成! 共处理 {len(json_files)} 名学生，生成 {total_charts} 个图表")

    else:
        print("请指定 --student-id、--profile 或 --batch-dir 参数")


if __name__ == "__main__":
    main()
