"""
报告格式化器 - 后处理 LLM 输出
"""
from datetime import datetime


class ReportFormatter:
    """报告格式化与后处理"""

    def format(self, raw_content: str, report_type: str) -> str:
        """格式化报告内容

        Args:
            raw_content: LLM 原始输出
            report_type: 报告类型

        Returns:
            格式化后的报告文本
        """
        content = raw_content.strip()

        # 添加报告头部
        header = self._build_header(report_type)
        return f"{header}\n\n{content}"

    def _build_header(self, report_type: str) -> str:
        """构建报告头部"""
        date_str = datetime.now().strftime("%Y年%m月%d日")
        return f"生成日期：{date_str}"

    def to_markdown(self, title: str, content: str, student_id: str) -> str:
        """转为 Markdown 格式"""
        return f"# {title}\n\n**学生编号**: {student_id}\n\n{content}"
