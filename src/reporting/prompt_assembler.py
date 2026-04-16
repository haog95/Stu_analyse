"""
提示词组装器 - KERAG 四层提示工程的核心实现
"""
from pathlib import Path

from src.core.config import PROJECT_ROOT


class PromptAssembler:
    """KERAG 四层提示组装器

    将四个层次的提示组件组装为完整的 LLM 消息：
    1. System Role & Guardrails（系统角色与约束）
    2. Context & Data Grounding（学生画像结构化数据）
    3. Knowledge Retrieval Injection（外部知识检索）
    4. Task Execution & Formatting（任务模板与范式约束）
    """

    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = PROJECT_ROOT / "configs" / "prompt_templates"
        self.template_dir = Path(template_dir)

    def assemble(self, context_json: str, knowledge_chunks: list[str],
                 report_type: str) -> tuple[str, str]:
        """组装完整的 KERAG 提示

        Args:
            context_json: 第二层的结构化画像 JSON 字符串
            knowledge_chunks: 第三层的知识检索片段列表
            report_type: 报告类型（motivation/addiction/success/career/comprehensive）

        Returns:
            (system_prompt, user_prompt) 元组
        """
        # 第一层：系统角色与约束
        system_prompt = self._load_template("system_role.txt")

        # 第四层：报告范式约束 + 报告类型模板
        report_paradigm = self._load_template("report_paradigm.txt")
        report_template = self._load_report_template(report_type)

        # 组装用户提示
        user_parts = []

        # 第二层：数据注入
        user_parts.append("## 学生画像数据\n")
        user_parts.append("以下是该学生的结构化画像数据：\n")
        user_parts.append("```json")
        user_parts.append(context_json)
        user_parts.append("```\n")

        # 第三层：知识注入
        if knowledge_chunks:
            user_parts.append("## 参考知识\n")
            user_parts.append("以下是相关的专业知识和干预经验，请在撰写报告时参考：\n")
            for i, chunk in enumerate(knowledge_chunks, 1):
                user_parts.append(f"### 参考资料 {i}")
                user_parts.append(chunk)
                user_parts.append("")

        # 第四层：任务执行
        user_parts.append("## 报告撰写要求\n")
        user_parts.append(report_paradigm)
        user_parts.append("\n")
        user_parts.append(report_template)

        user_prompt = "\n".join(user_parts)

        return system_prompt, user_prompt

    def _load_template(self, filename: str) -> str:
        """加载提示词模板文件"""
        filepath = self.template_dir / filename
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return ""

    def _load_report_template(self, report_type: str) -> str:
        """加载报告类型对应的模板"""
        template_map = {
            "motivation": "report_type_motivation.txt",
            "addiction": "report_type_addiction.txt",
            "success": "report_type_success.txt",
            "career": "report_type_career.txt",
            "comprehensive": "report_type_comprehensive.txt",
        }
        filename = template_map.get(report_type, "report_type_comprehensive.txt")
        return self._load_template(filename)
