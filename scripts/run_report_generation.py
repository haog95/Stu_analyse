"""
报告生成运行脚本
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config
from src.portraits.registry import PortraitRegistry
from src.core.data_merger import merge_all_portraits
from src.explanation.surrogate_model import SurrogateModel
from src.explanation.shap_analyzer import SHAPAnalyzer
from src.explanation.risk_attribution import RiskAttribution
from src.reporting.context_builder import ContextBuilder
from src.reporting.knowledge_retriever import KnowledgeRetriever
from src.reporting.prompt_assembler import PromptAssembler
from src.reporting.llm_client import LLMClient
from src.reporting.report_generator import ReportGenerator
from src.reporting.batch_processor import BatchProcessor


def main():
    parser = argparse.ArgumentParser(description="KERAG 报告生成")
    parser.add_argument("--student-id", type=str, help="指定学生ID")
    parser.add_argument("--batch", action="store_true", help="批量生成")
    parser.add_argument("--output", type=str, default="reports", help="输出目录")
    parser.add_argument("--portraits", type=str, default="data/", help="画像数据目录")
    parser.add_argument("--knowledge-base", type=str, default="configs/knowledge/", help="知识库目录")
    parser.add_argument("--max-workers", type=int, default=3, help="最大并行数")
    parser.add_argument("--filter", type=str, default=None, help="筛选条件（如 risk_level=high）")
    parser.add_argument("--dry-run", action="store_true", help="只输出提示词，不调用LLM")
    args = parser.parse_args()

    config = load_config()

    # 1. 初始化画像注册表
    print("=== 加载画像数据 ===")
    registry = PortraitRegistry()
    registry.initialize(args.portraits)
    print()

    # 2. SHAP 分析
    print("=== 训练代理模型 & SHAP 分析 ===")
    merged_df = merge_all_portraits(args.portraits)
    surrogate = SurrogateModel()
    surrogate.train(merged_df)

    X, _ = surrogate.prepare_features(merged_df)
    analyzer = SHAPAnalyzer(surrogate)
    analyzer.compute_shap_values(X)

    risk_attribution = RiskAttribution(analyzer, merged_df)
    print()

    # 3. 初始化 KERAG 组件（增强版：传入 merged_df 以启用群体统计和数据质量分析）
    model_metrics = surrogate.get_metrics()
    context_builder = ContextBuilder(
        risk_attribution, model_metrics, merged_df=merged_df
    )
    knowledge_retriever = KnowledgeRetriever(args.knowledge_base)
    prompt_assembler = PromptAssembler()
    llm_client = LLMClient(config.get("llm", {}))

    generator = ReportGenerator(
        context_builder, prompt_assembler, llm_client,
        knowledge_retriever=knowledge_retriever,
    )

    if args.dry_run:
        # 只输出提示词
        if args.student_id:
            profile = registry.get_all_for_student(args.student_id)
            context = context_builder.build_json_string(profile)
            keywords = profile.get_risk_keywords()
            chunks = knowledge_retriever.retrieve(keywords)
            from src.reporting.report_classifier import classify_report_types
            types = classify_report_types(profile)
            if types:
                sys_prompt, user_prompt = prompt_assembler.assemble(context, chunks, types[0])
                print("=== System Prompt ===")
                print(sys_prompt[:500] + "...")
                print("\n=== User Prompt (first 1000 chars) ===")
                print(user_prompt[:1000] + "...")

                # 展示上下文结构
                ctx = context_builder.build(profile)
                print("\n=== Context Structure ===")
                print(f"Keys: {list(ctx.keys())}")
                if "group_statistics" in ctx:
                    print(f"Group stats: {len(ctx['group_statistics'].get('comparisons', []))} metrics compared")
                if "data_quality" in ctx:
                    print(f"Data quality: {ctx['data_quality'].get('quality_level', 'unknown')}")
                if "discovered_patterns" in ctx:
                    print(f"Patterns discovered: {len(ctx['discovered_patterns'])}")
                if "analysis_results_catalog" in ctx:
                    print(f"Analysis results: {len(ctx['analysis_results_catalog'])} items")
        return

    if args.student_id:
        # 单个学生
        print(f"=== 生成报告: {args.student_id} ===")
        profile = registry.get_all_for_student(args.student_id)
        result = generator.generate(profile)

        # 确保输出目录存在
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 文件名含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 转换为 Markdown 并保存
        md_content = ReportGenerator.to_markdown(result)
        md_path = output_dir / f"{args.student_id}_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n报告已保存到 {md_path}")

        # 保存学生画像 JSON（用于可视化脚本）
        json_path = output_dir / f"{args.student_id}_{timestamp}_profile.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.get("context", {}), f, ensure_ascii=False, indent=2)
        print(f"画像数据已保存到 {json_path}")

    elif args.batch:
        # 批量生成
        print("=== 批量生成报告 ===")
        student_ids = list(registry.get_common_student_ids())

        # 筛选
        if args.filter and "risk_level=high" in args.filter:
            high_risk_ids = []
            for sid in student_ids:
                profile = registry.get_all_for_student(sid)
                if profile.overall_risk.value in ("高风险", "严重风险"):
                    high_risk_ids.append(sid)
            student_ids = high_risk_ids
            print(f"筛选后: {len(student_ids)} 名高风险学生")

        profiles = [registry.get_all_for_student(sid) for sid in student_ids[:10]]  # 先处理前10名

        processor = BatchProcessor(generator)
        results = processor.process_students(profiles, args.max_workers, args.output)
        print(f"\n完成! 共生成 {len(results)} 份报告")

    else:
        print("请指定 --student-id 或 --batch 参数")


if __name__ == "__main__":
    main()
