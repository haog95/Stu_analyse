# 学生多维画像分析系统（开发者版）

本文件面向开发、部署与二次扩展。

## 1. 环境准备

推荐环境：Python 3.10

```bash
conda create -n stu_analyse python=3.10 -y
conda activate stu_analyse
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 配置说明

1) 复制密钥模板

```bash
cp .env.example .env
```

2) 在 .env 中配置对应供应商密钥（如 OPENAI_API_KEY、DEEPSEEK_API_KEY 等）

3) 在 configs/settings.yaml 中选择提供商

```yaml
llm:
  provider: "openai"
```

当前默认模型配置：
- provider: openai
- model: gpt-5.4

## 3. 常用脚本

```bash
# 系统完整性验证
python scripts/run_validation.py

# SHAP 解释计算
python scripts/run_shap_explanation.py --output output/shap

# 单学生报告（先看提示词）
python scripts/run_report_generation.py --student-id pjxyqwbj001 --dry-run

# 单学生报告（实际调用 LLM）
python scripts/run_report_generation.py --student-id pjxyqwbj001 --output reports/

# 批量报告（当前脚本默认取筛选后前10名）
python scripts/run_report_generation.py --batch --filter "risk_level=high" --output reports/

# 启动可视化仪表盘
python scripts/run_dashboard.py

# 单学生图表生成
python scripts/run_visualization.py --student-id pjxyqwbj001 --output output/viz

# 基于 reports 目录的 profile.json 批量图表生成
python scripts/run_visualization.py --batch-dir reports/ --output output/viz

# 运行测试
pytest tests/ -v
```

## 4. 代码结构

- src/core: 配置加载、数据读取、ID 映射、数据合并
- src/portraits: 10个核心画像维度 + 群体聚类画像注册
- src/explanation: 代理模型训练、SHAP 分析、风险归因
- src/reporting: 报告分类、上下文构建、知识检索、提示组装、LLM 调用
- src/visualization: 雷达图、风险热力图、SHAP 瀑布图、群体对比、轨迹图、Dashboard

## 5. 当前实现边界

- 画像5（考证/升学意图）暂未接入主流水线
- 批量报告默认处理筛选后前10个学生（见 scripts/run_report_generation.py）
- OpenAI provider 的默认 base_url 以 configs/settings.yaml 为准

## 6. 排障建议

- 报告为空或调用失败：优先检查 .env 密钥、provider 配置、网络可达性
- 部分图表未生成：检查对应字段是否存在于 merged_df
- 训练失败：检查 data 目录是否完整且含10份画像 CSV + 群体聚类结果

## 7. 开发规范建议

- 配置与密钥分离：密钥只放 .env，不放 settings.yaml
- 新增画像维度时，同步更新 portraits 注册表与数据合并逻辑
- 修改提示词模板时，优先保持 KERAG 四层结构稳定

## 8. 文档入口

- 对外介绍：README_PUBLIC.md
- 综合说明：README.md
