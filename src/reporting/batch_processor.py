"""
批量报告处理器 - 并行生成多个学生的报告
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .report_generator import ReportGenerator


class BatchProcessor:
    """批量报告处理器"""

    def __init__(self, report_generator: ReportGenerator):
        self.generator = report_generator

    def process_students(self, profiles, max_workers: int = 3,
                         output_dir: str = "reports") -> list[dict]:
        """批量生成报告

        Args:
            profiles: StudentProfile 列表
            max_workers: 最大并行数
            output_dir: 输出目录

        Returns:
            生成结果列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._generate_one, profile, output_dir): profile
                for profile in profiles
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="生成报告"):
                profile = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "student_id": profile.student_id,
                        "error": str(e),
                    })

        # 保存索引
        self._save_index(results, output_dir)
        return results

    def _generate_one(self, profile, output_dir: Path) -> dict:
        """生成单个学生的报告"""
        result = self.generator.generate(profile)

        # 保存为 Markdown 文件（文件名含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_content = self.generator.to_markdown(result)
        filepath = output_dir / f"{profile.student_id}_{timestamp}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 保存画像 JSON
        json_path = output_dir / f"{profile.student_id}_{timestamp}_profile.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.get("context", {}), f, ensure_ascii=False, indent=2)

        return result

    def _save_index(self, results: list, output_dir: Path):
        """保存报告索引"""
        index = []
        for r in results:
            entry = {
                "student_id": r.get("student_id", ""),
                "overall_risk": r.get("overall_risk", ""),
                "report_count": len(r.get("reports", [])),
            }
            if "error" in r:
                entry["error"] = r["error"]
            index.append(entry)

        index_path = output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
