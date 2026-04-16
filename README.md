# 学生多维画像分析与个性化干预报告生成系统

> 基于多源校园行为数据，构建 10 个核心画像维度 + 1 个群体聚类画像，结合 XGBoost 风险预测与 SHAP 可解释性分析，通过 KERAG（Knowledge-Enhanced Retrieval-Augmented Generation）知识增强提示架构驱动大语言模型，全自动生成"千人千面"的个性化评价与干预报告。

## 文档导航

- 对外发布版（适合汇报、介绍、立项材料）：[README_PUBLIC.md](README_PUBLIC.md)
- 开发者版（适合部署、运行、二次开发）：[README_DEV.md](README_DEV.md)
- 当前文档（完整技术说明）：[README.md](README.md)

---

## 项目背景

高校学生管理面临"数据丰富、洞察匮乏"的困境——校园一卡通、教务系统、图书馆闸机、线上学习平台、体测系统等多源异构数据虽已沉淀，但缺乏统一的整合分析与智能解读手段。辅导员面对数百名学生，难以逐一基于数据做出精准的学业预警与干预决策。

本项目突破性地引入大语言模型（LLM），通过**结构化画像标签 + SHAP 风险归因 + 外部知识库检索**的四层知识增强提示架构（KERAG），实现从"数据"到"洞察"再到"行动"的闭环转化，为每位学生自动生成兼具心理学深度与同理心的个性化干预报告。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        多源校园数据层                                │
│  教务考勤 │ 图书馆闸机 │ 线上学习 │ 门禁 │ 体测 │ 上网 │ 就业去向    │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     画像构建与建模层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ 无监督聚类    │  │ 有监督学习    │  │ 异常/时序检测 │              │
│  │ 画像1,2,6    │  │ 画像3,10    │  │ 画像4,8     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────┐                                                   │
│  │ 规则/统计模型 │                                                   │
│  │ 画像7,9      │                                                   │
│  └──────────────┘                                                   │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SHAP 可解释性层                                    │
│  基于 XGBoost 代理模型的 pred_contribs → 逐实例特征贡献归因             │
│  （缺勤频率 × 深夜在线时长 × 图书馆借阅量 → 量化风险因子排序）          │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│              KERAG 知识增强提示工程层（核心创新）                      │
│                                                                     │
│  ① System Role & Guardrails                                        │
│     └─ 角色对齐：资深辅导员 + 职业规划导师 + 心理学专家               │
│                                                                     │
│  ② Context & Data Grounding                                        │
│     └─ 学生画像标签 JSON + SHAP 风险归因 → 结构化事实锚定            │
│                                                                     │
│  ③ Knowledge Retrieval Injection                                    │
│     └─ 关键词匹配检索 → 干预手册 / 经验知识库 / 行业白皮书            │
│                                                                     │
│  ④ Task Execution & Formatting                                      │
│     └─ 报告范式约束：数据叙事 → 心理归因 → 干预建议                  │
└──────────────────────┬──────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM 报告生成层                                     │
│  支持: DeepSeek / 智谱GLM / Moonshot / OpenAI / Claude / Ollama     │
│  输出格式: Markdown（2000+ 字深度报告）                               │
│  《深度学业与行为发展动态归因与全景干预报告》                           │
│  《学习动机归因深度分析报告》                                         │
│  《网络依赖对学业风险的深度归因诊断报告》                               │
│  《面向学困生的成功经验参考与逆袭路径蓝图》                             │
│  《个性化职业发展路径与跨区域人才流动规划建议》                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 画像体系（10维 + 群体聚类）

### 一、学业成就与认知画像

| 编号 | 画像名称 | 建模方法 | 核心输入字段 | 输出标签 |
|:---:|---------|---------|------------|---------|
| 1 | 课堂参与度与专注力 | 无监督聚类 | 抬头率、低头率、前排率、签到率、讨论数 | 听课质量评级（沉浸/一般/游离）、隐性逃课判定 |
| 2 | 学习投入度与自律性 | 无监督聚类 | 任务完成率、图书馆打卡、专题阅读时长、作业时间 | 主动学习型/普通型/被动应付型、拖延指数诊断 |
| 3 | 挂科/退学高危预警 | 有监督(XGBoost) | 课程成绩、考勤状态、学籍异动、补考重修、综合表现 | 不及格/退学概率预测、SHAP高危因子解释 |
| 4 | 学习轨迹"退化"预警 | 异常/时序检测 | 综合表现时序、上网时长趋势、考勤周频次 | 退化拐点识别、事前预警干预建议 |
> 注：画像5（考证/升学意图）当前版本未纳入主流水线。

### 二、生活习惯与综合素质

| 编号 | 画像名称 | 建模方法 | 核心输入字段 | 输出标签 |
|:---:|---------|---------|------------|---------|
| 6 | 规律性生活 | 无监督聚类 | 门禁刷卡时间、跑步打卡、图书馆进出、签到时间 | 作息分类（早起型/熬夜型/规律型）、心理健康风险提示 |
| 7 | 身体素质与意志力 | 规则/统计 | 体测总分、BMI、50m/800m/1000m、跑步打卡周次 | 体质与意志力综合分群、个性化运动处方 |
| 8 | 网络依赖与消遣 | 异常检测 | 上网累计时长、统计月度、学校均值、课堂低头率 | 网络依赖风险指数、网瘾归因分析 |
| 9 | 综合竞争力 | 规则/统计 | 专业年级排名、奖学金级别、竞赛获奖、成绩/素质/能力排名 | 综合梯队定位、成功经验模板参考 |

### 三、毕业去向与发展画像

| 编号 | 画像名称 | 建模方法 | 核心输入字段 | 输出标签 |
|:---:|---------|---------|------------|---------|
| 10 | 就业竞争力与匹配 | 有监督 | 毕业去向、单位性质/行业、绩点、竞赛/奖学金 | 就业去向模拟预测、路径参考与职业建议 |

### 四、群体聚类画像

| 编号 | 画像名称 | 建模方法 | 核心输入字段 | 输出标签 |
|:---:|---------|---------|------------|---------|
| G1 | 群体画像最终聚类 | 聚类结果映射 | Group_Profile、Cluster 等群体标签字段 | 群体类型归属、群体对比分析锚点 |

---

## 技术实现

### 项目结构

```
Students_ana/
├── src/
│   ├── core/                           # 核心数据层
│   │   ├── config.py                   # 全局配置加载
│   │   ├── id_mapper.py                # 学生ID列名标准化（处理4种不同列名）
│   │   ├── data_loader.py              # CSV统一加载器（编码/表头自动检测）
│   │   ├── data_merger.py              # 多文件外连接合并
│   │   └── student_profile.py          # StudentProfile 数据类 + 风险计算
│   ├── portraits/                      # 画像模块（10维 + 群体聚类）
│   │   ├── base.py                     # 抽象基类 PortraitDimension
│   │   ├── classroom.py                # 画像1：课堂参与度
│   │   ├── study_engagement.py         # 画像2：学习投入度
│   │   ├── risk_prediction.py          # 画像3：挂科退学预警
│   │   ├── degradation.py              # 画像4：学习轨迹退化
│   │   ├── lifestyle.py                # 画像6：规律性生活
│   │   ├── physical.py                 # 画像7：身体素质与意志力
│   │   ├── network.py                  # 画像8：网络依赖
│   │   ├── competitiveness.py          # 画像9：综合竞争力
│   │   ├── career.py                   # 画像10：就业竞争力
│   │   ├── group_clustering.py         # 群体聚类结果
│   │   └── registry.py                 # PortraitRegistry 统一注册表
│   ├── explanation/                    # SHAP 可解释性
│   │   ├── feature_catalog.py          # 特征名→中文描述映射表
│   │   ├── surrogate_model.py          # XGBoost 代理模型（4分类，AUC ≥ 99%，五折CV）
│   │   ├── shap_analyzer.py            # pred_contribs 特征贡献计算
│   │   └── risk_attribution.py         # 风险/保护因子结构化输出
│   ├── reporting/                      # KERAG 报告生成
│   │   ├── report_classifier.py        # 画像→报告类型自动匹配
│   │   ├── context_builder.py          # 第二层：画像JSON + 模型指标 + 聚类谱系
│   │   ├── knowledge_retriever.py      # 第三层：关键词知识检索
│   │   ├── prompt_assembler.py         # 四层提示组装器
│   │   ├── llm_client.py               # LLM 统一客户端（6个提供商，max_tokens=8000）
│   │   ├── report_generator.py         # 主编排器（支持 Markdown 输出）
│   │   ├── report_formatter.py         # 报告后处理
│   │   └── batch_processor.py          # 批量并行生成
│   └── visualization/                  # 可视化
│       ├── student_radar.py            # 学生画像雷达图
│       ├── risk_heatmap.py             # 风险等级热力图
│       ├── shap_waterfall.py           # SHAP 归因瀑布图
│       ├── group_comparison.py         # 个体-群体指标对比图
│       ├── group_scatter.py            # 群体聚类 PCA 散点图
│       ├── trajectory_plot.py          # 行为轨迹对比图
│       ├── font_config.py              # 中文字体自动配置
│       └── dashboard.py                # Streamlit 交互仪表盘
├── configs/
│   ├── settings.yaml                   # 全局配置（不含 API Key）
│   ├── prompt_templates/               # KERAG 提示词模板（7个文件）
│   └── knowledge/                      # 知识库（4个文件）
├── scripts/                            # 运行脚本
├── tests/                              # 测试（27个用例）
├── data/                               # 画像结果数据（10个CSV）
├── .env.example                        # API Key 配置模板
├── .gitignore                          # Git 排除规则（含 .env）
└── requirements.txt                    # Python 依赖
```

### 核心模块说明

| 模块 | 说明 |
|------|------|
| `src/core/id_mapper.py` | 解决 10 个 CSV 文件中学生 ID 列名不一致问题（student_id / XH / SID / 学号） |
| `src/core/data_merger.py` | 以群体聚类文件为基底，外连接合并 10 个画像 CSV，输出统一分析数据集 |
| `src/explanation/surrogate_model.py` | 在 21 个数值特征上训练 XGBoost 代理模型（4分类），含 AUC(OvR) 和五折交叉验证指标计算 |
| `src/explanation/shap_analyzer.py` | 使用 XGBoost 的 `pred_contribs` 计算逐实例特征贡献，输出 (n_samples, 21, 4) 的 SHAP 值矩阵 |
| `src/reporting/context_builder.py` | 组装画像数据 + SHAP 归因（Top K）+ 模型性能指标 + 群体聚类全谱系，注入 LLM 上下文 |
| `src/reporting/prompt_assembler.py` | KERAG 核心实现：将系统角色 + 画像JSON + 知识检索 + 报告模板组装为完整的 LLM 消息 |
| `src/reporting/llm_client.py` | 统一客户端，支持 OpenAI 兼容接口 + Claude SDK + 本地 Ollama |

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n stu_analyse python=3.10 -y
conda activate stu_analyse
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 API Key

```bash
# 复制模板并填写你的 Key
cp .env.example .env
```

编辑 `.env` 文件，填写你使用的提供商的 API Key：

```env
# 例如使用 DeepSeek（推荐，国内直连，性价比高）
DEEPSEEK_API_KEY=sk-your-key-here
```

`.env` 文件已被 `.gitignore` 排除，**不会被提交到 Git**。

### 3. 选择 LLM 提供商

编辑 `configs/settings.yaml`，修改 `llm.provider` 字段：

```yaml
llm:
  provider: "openai"    # 改为你要用的提供商
```

| 提供商 | provider 值 | 环境变量 | 默认 base_url（以 settings.yaml 为准） |
|--------|------------|---------|----------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `https://openai.com/api/` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| 智谱 GLM | `zhipu` | `ZHIPU_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` |
| Moonshot | `moonshot` | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` |
| Claude | `claude` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` |
| 本地 Ollama | `local` | 无需 Key | `http://localhost:11434` |

当前默认模型配置：`openai.model = gpt-5.4`。

---

## 快速开始

```bash
conda activate stu_analyse

# 系统完整性验证
python scripts/run_validation.py

# SHAP 特征贡献分析
python scripts/run_shap_explanation.py --output output/shap

# 查看某个学生的提示词（不调用LLM）
python scripts/run_report_generation.py --student-id pjxyqwbj001 --dry-run

# 为单个学生生成报告（需配置好 LLM API Key，输出为 Markdown）
python scripts/run_report_generation.py --student-id pjxyqwbj001 --output reports/

# 批量生成（前10名高风险学生）
python scripts/run_report_generation.py --batch --filter "risk_level=high" --output reports/

# 启动可视化仪表盘
python scripts/run_dashboard.py

# 为单个学生生成配套图表（雷达图/SHAP瀑布图/群体对比/轨迹图）
python scripts/run_visualization.py --student-id pjxyqwbj001 --output output/viz

# 基于已生成的 profile.json 批量可视化
python scripts/run_visualization.py --batch-dir reports/ --output output/viz

# 运行全部测试
pytest tests/ -v
```

---

## 数据概况

`data/` 目录下已产出以下画像结果（覆盖约 2500 名学生）：

| 文件 | 学生数 | 核心字段 |
|------|:------:|---------|
| 画像1_课堂参与度与专注力评级结果.csv | 2500 | 听课质量评级、是否隐性逃课 |
| 画像2_学习投入度与自律性.csv | 2500 | 18 维特征 + 诊断报告文本 |
| 画像3_挂科退学高危预警.csv | 2501 | XGBoost 预测概率 |
| 画像4_学习轨迹“退化”预警画像.csv | 2499 | 行为偏移分值、画像描述、干预建议权重 |
| 画像6_规律性生活.csv | 2489 | 作息类型聚类标签 |
| 画像7_身体素质与意志力.csv | 2434 | 锻炼习惯、体型、意志力梯队 |
| 画像8_网络依赖.csv | 2433 | 沉迷指数、沉迷等级、SVM 标签 |
| 画像9_综合竞争力.csv | 2499 | 13 维竞争力评分 + 预警标签 |
| 画像10_就业竞争力与匹配.csv | 2501 | 推荐去向1、推荐去向2 |
| 群体画像最终聚类结果.csv | 2433 | 5 群体聚类 + 16 维综合画像 |

### 群体聚类分布

| 群体 | 人数 | 占比 | 特征画像 |
|------|:----:|:----:|---------|
| 稳健应付与中庸型 (Passive Casuals) | 1051 | 43.2% | 排名中游，风险低，听课一般 |
| 积极主导与规律成就型 (Active Achievers) | 1000 | 41.1% | 排名靠前，低风险，自律性高 |
| 行为轨迹时序退化型 (Degrading At-risk) | 191 | 7.8% | 学习退化趋势，听课游离 |
| 隐性逃课与全面游离型 (Disconnected Riskers) | 167 | 6.9% | 挂科概率极高(>99%)，排名垫底 |
| 极端异常离群群体 (Outliers) | 24 | 1.0% | 行为极端异常 |

---

## KERAG 提示工程架构

本项目的核心创新——通过四层模块化提示工程，将结构化画像数据转化为 LLM 可理解、可推理的上下文，确保生成报告的精准性、专业性与个性化。

### 第一层：结构化指令与角色对齐（System Role & Guardrails）

融合"**数据科学家 + 高校资深辅导员 + 心理危机干预专家 + 职业发展导师**"四重身份，具备对 SHAP、TOPSIS、LSTM-Autoencoder、DBSCAN 等算法输出的专业解读能力，同时深谙认知行为疗法（CBT）、动机访谈（MI）等循证干预框架。

7 条报告撰写铁律：数据锚定、叙事弧结构、心理深度、干预颗粒度、正向锚定、同理心语调、禁止事项。

### 第二层：微观数据与 SHAP 推演成果注入（Context & Data Grounding）

将学生的核心画像标签、XGBoost 预测结果、SHAP 归因、模型评估指标、群体聚类全谱系转化为结构化 JSON：

```json
{
  "student_id": "pjxyqwbk401",
  "group": "隐性逃课与全面游离型",
  "overall_risk": "严重风险",
  "fail_probability": 0.999,
  "high_risk_dimensions": ["网络依赖", "学习投入度"],
  "portraits": {
    "课堂参与度": {"label": "游离", "risk_level": "高风险"},
    "学习投入度": {"label": "被动应付型", "procrastination_index": 0.73},
    "挂科退学预警": {"fail_probability": 0.999},
    "网络依赖": {"沉迷指数": 0.83, "依赖等级": "重度沉迷"},
    "综合竞争力": {"综合排名百分位": 92.5}
  },
  "shap_risk_factors": [
    {"factor": "沉迷指数偏高(0.83)", "contribution": 0.32},
    {"factor": "深夜活跃型作息模式", "contribution": 0.28},
    {"factor": "图书馆到访次数为零", "contribution": 0.19}
  ],
  "shap_protective_factors": [...],
  "model_performance": {
    "train_accuracy": 0.9988,
    "test_accuracy": 0.9815,
    "auc_ovr": 0.9956,
    "cv_5fold_mean": 0.9612,
    "n_features": 21,
    "n_classes": 4
  },
  "cluster_taxonomy": {
    "-1": "极端异常离群群体",
    "0": "积极主导与规律成就型",
    "1": "稳健应付与中庸型",
    "2": "行为轨迹时序退化型",
    "3": "隐性逃课与全面游离型"
  }
}
```

### 第三层：动态外部领域知识挂载（Knowledge Retrieval Injection）

基于关键词匹配，自动检索 `configs/knowledge/` 中的专业文献：

| 风险关键字 | 检索知识库 | 检索内容 |
|-----------|----------|---------|
| 网络依赖、沉迷 | 《大学生心理健康危机干预指导手册》 | 认知行为干预策略、渐进式断网方案 |
| 拖延指数、学业退化 | 《学长学姐考研保研成功经验知识库》 | 时间管理模板、番茄工作法实践案例 |
| 就业去向、职业选择 | 《区域人才流动趋势白皮书》 | 行业趋势、区域人才需求、薪资对比 |
| 作息紊乱、熬夜 | 《心理学干预框架与行为改变理论》 | 行为改变跨理论模型、自我决定理论 |

### 第四层：任务导向的逻辑输出约束（Task Execution & Formatting）

采用"**数据叙事弧**"（Data Narrative Arc）四幕结构，每份报告严格遵循：

```
第一幕 · 情境引入（Setup）
  → 全谱系5类群体对比叙事 + 多维雷达映射 + 潜能锚定

第二幕 · 冲突揭示（Conflict）
  → 附模型性能指标的风险概率预警 + 行为退化拐点时序重现

第三幕 · 归因探究（Resolution）
  → 维度一：SHAP 个体归因  |  维度二：时序动态分析  |  维度三：深层心理演化

第四幕 · 行动号召（Call-to-Action）
  → 阻断期（0-7天）+ 重构期（8-21天）+ 发展期（长期）微观干预矩阵
```

---

## 报告生成类型

基于画像组合与风险等级，系统自动匹配不同的报告模板。所有报告均遵循"数据叙事弧"四幕结构，输出为 Markdown 格式。

| 报告类型 | 触发画像 | 适用群体 | 核心内容 |
|---------|---------|---------|---------|
| 学习动机归因深度分析报告 | 画像2 + 画像4 | 被动应付型/学业退化 | 投入度退化时序、拖延心理归因、动机重建方案 |
| 网络依赖对学业风险的深度归因诊断 | 画像8 + 画像3 | 重度沉迷/高危挂科 | 依赖度量化评估、行为挤出效应、CBT干预处方 |
| 面向学困生的成功经验参考与逆袭路径蓝图 | 画像9 + 画像1 | 综合竞争力低 | 同辈行为模式提炼、个性化适配、逆袭行动计划 |
| 个性化职业发展路径与跨区域人才流动规划 | 画像10 + 群体聚类 | 就业选择期 | 多路径对比分析、竞争力缺口诊断、职业行动蓝图 |
| 深度学业与行为发展动态归因与全景干预报告 | 全部画像 | 高危/退化群体 | 全景生态位扫描、SHAP多维归因、三阶段干预矩阵 |

### 报告核心结构（数据叙事弧）

```
一、发展基线与全景生态位扫描（Setup）
  ├── 全谱系5类群体对比叙事
   ├── 多维能力雷达映射
   └── 潜能与优势锚定

二、核心冲突与时序轨迹突变（Conflict）
   ├── 风险阈值突破预警（附 AUC/交叉验证 模型性能指标）
   └── 行为退化拐点时序重现

三、动力学多维归因剖析（Resolution）
   ├── 维度一：全局静态风险因子解构（SHAP 个体归因）
   ├── 维度二：时序动态轨迹与行为突变分析
   └── 维度三：深层特征交互与心理演化机制

四、知识增强赋能与微观干预矩阵（Call-to-Action）
   ├── 阻断期（0-7天，物理与环境干预）
   ├── 重构期（8-21天，认知与习惯干预）
   ├── 发展期（长期，价值与路径干预）
   └── 推荐干预资源清单
```

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 数据处理 | Python 3.10, Pandas, NumPy |
| 建模 | XGBoost, Scikit-learn |
| 可解释性 | XGBoost pred_contribs（等价 SHAP TreeExplainer），含 AUC 和五折交叉验证 |
| LLM | DeepSeek / 智谱GLM / Moonshot / OpenAI / Claude / Ollama |
| 报告引擎 | KERAG 四层提示工程（数据叙事弧结构，2000+字深度报告，Markdown输出） |
| 可视化 | Matplotlib, Seaborn, Streamlit |
| 测试 | Pytest（27 个用例） |

---

## 已有数据成果

详细的原始数据字段定义参见 `字段汇总.docx`，画像维度的输入/输出字段定义参见 `画像字段输入输出.docx`。

---

## 安全说明

- 所有 API Key 通过 `.env` 文件管理，该文件已被 `.gitignore` 排除
- `configs/settings.yaml` 不包含任何密钥，仅包含 base_url 和模型名称
- `.env.example` 是公开的配置模板，不含实际密钥
- 上传到 GitHub 前请确认 `.env` 文件不会被意外提交
