"""
报告生成器 - KERAG 管道主编排器
"""
import json
from pathlib import Path
from typing import Optional

from src.core.student_profile import StudentProfile
from .report_classifier import classify_report_types, get_report_info
from .context_builder import ContextBuilder
from .prompt_assembler import PromptAssembler
from .llm_client import LLMClient
from .report_formatter import ReportFormatter


class ReportGenerator:
    """报告生成器

    编排 KERAG 四层管道：分类 → 上下文构建 → 知识检索 → 提示组装 → LLM 调用
    """

    def __init__(self, context_builder: ContextBuilder,
                 prompt_assembler: PromptAssembler,
                 llm_client: LLMClient,
                 knowledge_retriever=None):
        self.context_builder = context_builder
        self.prompt_assembler = prompt_assembler
        self.llm_client = llm_client
        self.knowledge_retriever = knowledge_retriever
        self.formatter = ReportFormatter()

    def generate(self, profile: StudentProfile,
                 report_type: str = None) -> dict:
        """为学生生成报告

        Args:
            profile: 学生画像
            report_type: 指定报告类型，None则自动分类

        Returns:
            {"student_id": str, "reports": [{"type": str, "title": str, "content": str}],
             "context": dict}
        """
        # 1. 确定报告类型
        if report_type:
            report_types = [report_type]
        else:
            report_types = classify_report_types(profile)

        # 2. 构建上下文
        context = self.context_builder.build(profile)
        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        # 3. 知识检索
        knowledge_chunks = []
        if self.knowledge_retriever:
            keywords = profile.get_risk_keywords()
            knowledge_chunks = self.knowledge_retriever.retrieve(keywords)

        # 4. 为每种报告类型生成内容
        reports = []
        for rtype in report_types:
            info = get_report_info(rtype)

            # 组装提示
            system_prompt, user_prompt = self.prompt_assembler.assemble(
                context_json, knowledge_chunks, rtype
            )

            # 调用 LLM
            try:
                raw_content = self.llm_client.generate(system_prompt, user_prompt)

                # 二次防护：检查格式化前的内容
                if not raw_content or not raw_content.strip():
                    print(f"[报告生成警告] 学生 {profile.student_id} 的 "
                          f"{rtype} 报告 LLM 返回空内容")
                    reports.append({
                        "type": rtype,
                        "title": info["name"],
                        "content": f"[报告生成失败: LLM 返回空内容，"
                                   f"请检查 API 配置和模型可用性]",
                        "error": "empty_llm_response",
                    })
                    continue

                formatted = self.formatter.format(raw_content, rtype)

                reports.append({
                    "type": rtype,
                    "title": info["name"],
                    "content": formatted,
                })
            except Exception as e:
                reports.append({
                    "type": rtype,
                    "title": info["name"],
                    "content": f"[报告生成失败: {e}]",
                    "error": str(e),
                })

        return {
            "student_id": profile.student_id,
            "overall_risk": profile.overall_risk.value,
            "reports": reports,
            "context": context,
        }

    @staticmethod
    def to_markdown(result: dict) -> str:
        """将报告结果转换为 Markdown 格式

        Args:
            result: generate() 返回的字典

        Returns:
            完整的 Markdown 字符串
        """
        student_id = result.get("student_id", "")
        overall_risk = result.get("overall_risk", "")
        reports = result.get("reports", [])

        lines = [
            f"# 综合预警干预报告",
            "",
            f"**学生编号**: {student_id}  ",
            f"**整体风险等级**: {overall_risk}",
            "",
            "---",
            "",
        ]

        for report in reports:
            title = report.get("title", "报告")
            content = report.get("content", "")
            lines.append(f"## {title}")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def generate_single(self, profile: StudentProfile,
                        report_type: str) -> str:
        """生成单份报告，返回纯文本"""
        result = self.generate(profile, report_type)
        if result["reports"]:
            return result["reports"][0]["content"]
        return ""
