"""
知识库检索器 - KERAG 第三层：基于关键词的知识检索（含引用追踪）
"""
from pathlib import Path
from typing import Optional

from src.core.config import PROJECT_ROOT


# 关键词到知识文件的映射
KEYWORD_FILE_MAP = {
    "网络依赖": "intervention_manual.txt",
    "沉迷": "intervention_manual.txt",
    "网瘾": "intervention_manual.txt",
    "拖延": "experience_knowledge.txt",
    "学习被动": "experience_knowledge.txt",
    "缺乏动力": "experience_knowledge.txt",
    "退化": "experience_knowledge.txt",
    "挂科预警": "intervention_manual.txt",
    "高风险": "intervention_manual.txt",
    "作息紊乱": "psychology_frameworks.txt",
    "熬夜": "psychology_frameworks.txt",
    "生活不规律": "psychology_frameworks.txt",
    "职业": "career_guidance.txt",
    "就业": "career_guidance.txt",
    "考研": "career_guidance.txt",
    "心理": "psychology_frameworks.txt",
    "干预": "intervention_manual.txt",
}

# 知识文件的理论框架标注（用于报告引用）
KNOWLEDGE_SOURCE_INFO = {
    "intervention_manual.txt": {
        "title": "认知行为干预手册",
        "frameworks": ["CBT (认知行为疗法)", "渐进式网络断戒策略", "学业预警分级干预"],
    },
    "experience_knowledge.txt": {
        "title": "学习经验知识库",
        "frameworks": ["番茄工作法", "艾宾浩斯遗忘曲线", "SQ3R阅读法"],
    },
    "psychology_frameworks.txt": {
        "title": "心理学理论框架",
        "frameworks": [
            "TTM 跨理论模型（阶段变化理论）",
            "自我决定理论 (SDT)",
            "归因理论 (Weiner)",
            "自我效能感理论 (Bandura)",
            "习惯形成理论 (Lally et al., 2010)",
            "拖延的认知-情感模型 (Sirois & Pychyl)",
        ],
    },
    "career_guidance.txt": {
        "title": "职业发展指导",
        "frameworks": ["Holland 职业兴趣理论", "舒伯生涯发展理论", "IKIGAI 模型"],
    },
}


class KnowledgeRetriever:
    """知识检索器

    基于关键词匹配从知识库文件中检索相关知识片段。
    支持后续升级为向量数据库检索。
    """

    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = PROJECT_ROOT / "configs" / "knowledge"
        self.knowledge_dir = Path(knowledge_dir)
        self._cache = {}

    def retrieve(self, keywords: list[str], top_k: int = 3) -> list[str]:
        """根据关键词检索知识片段

        Args:
            keywords: 风险关键词列表
            top_k: 返回的最大片段数

        Returns:
            知识片段列表
        """
        results = self.retrieve_with_sources(keywords, top_k)
        return [chunk for chunk, _ in results]

    def retrieve_with_sources(self, keywords: list[str],
                               top_k: int = 3) -> list[tuple[str, str]]:
        """根据关键词检索知识片段（含来源追踪）

        Args:
            keywords: 风险关键词列表
            top_k: 返回的最大片段数

        Returns:
            [(知识片段, 来源文件名), ...] 列表
        """
        # 确定需要检索的文件
        target_files = set()
        for kw in keywords:
            for key, filename in KEYWORD_FILE_MAP.items():
                if key in kw or kw in key:
                    target_files.add(filename)

        # 如果没有匹配，返回所有知识文件的部分内容
        if not target_files:
            target_files = {"intervention_manual.txt", "psychology_frameworks.txt"}

        # 加载和分段知识文件
        chunks = []
        for filename in target_files:
            content = self._load_knowledge(filename)
            if content:
                # 将知识文本按段落分割
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for p in paragraphs:
                    # 检查段落是否与关键词相关
                    relevance = sum(1 for kw in keywords if kw in p)
                    if relevance > 0:
                        chunks.append((relevance, p, filename))

        # 按相关性排序
        chunks.sort(key=lambda x: x[0], reverse=True)

        return [(chunk, source) for _, chunk, source in chunks[:top_k]]

    def get_source_info(self, filename: str) -> dict:
        """获取知识来源的详细信息（用于报告引用）

        Args:
            filename: 知识文件名

        Returns:
            包含 title 和 frameworks 的字典
        """
        return KNOWLEDGE_SOURCE_INFO.get(filename, {
            "title": filename,
            "frameworks": [],
        })

    def get_all_source_info(self) -> dict:
        """获取所有知识来源信息"""
        return KNOWLEDGE_SOURCE_INFO.copy()

    def _load_knowledge(self, filename: str) -> str:
        """加载知识文件（带缓存）"""
        if filename not in self._cache:
            filepath = self.knowledge_dir / filename
            if filepath.exists():
                self._cache[filename] = filepath.read_text(encoding="utf-8")
            else:
                self._cache[filename] = ""
        return self._cache[filename]
