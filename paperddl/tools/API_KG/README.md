# arXiv HTML Academic KG Extractor

从 arXiv HTML 论文页面抽取科研导向的 claim-evidence 知识图谱。

核心目标不是抽关键词，而是抽取科研论证结构：

- 论文解决的问题是什么
- 本文提出了什么 claim / method / contribution
- claim 被哪些 evidence / result / experiment 支撑
- 方法由哪些 component 组成
- 实验使用了哪些 benchmark / metric / baseline
- 论文有哪些 limitation / future work

## 目录结构

```text
academic_kg_extractor.py   # LLM 抽取器、严格 PE、dry-run 规则抽取
arxiv_html_processor.py    # arXiv HTML 抓取、缓存、元数据和章节解析
paper_assets.py            # 提取并下载论文相关图片，默认取最相关的第一张图
kg_schema.py               # 节点/边 schema、校验、去重、质量报告
kg_cache.py                # Neo4j 优先缓存，本地 JSON 兜底
main_arxiv_html.py         # 单篇论文入口
batch_eval_arxiv_html.py   # 批量评估入口
config.py                  # 环境变量配置
requirements.txt           # 精简依赖
data/                      # 运行时输出目录
```

## 安装

Windows PowerShell:

```powershell
conda activate kg
python -m pip install -r requirements.txt
```

## 最低成本测试

dry-run 不调用大模型 API，适合测试解析、缓存、schema 和输出格式。

```powershell
conda activate kg
python .\main_arxiv_html.py --url https://arxiv.org/html/2602.22727v1 --dry-run --max-sections 5
```

输出文件会写到：

```text
data/<arxiv_id>_kg.json
```

## 真实 LLM 抽取

```powershell
conda activate kg
$env:LLM_API_KEY="your_api_key"
$env:LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
$env:LLM_MODEL="glm-5.1"
python .\main_arxiv_html.py --url https://arxiv.org/html/2602.22727v1 --max-sections 5
```

成本控制逻辑：

1. 解析 arXiv HTML。
2. 提取并下载论文相关图片，默认保存第一张高相关图片。
3. 按章节选择高价值 section。
4. 计算 section 文本 hash。
5. 先查 Neo4j 或本地 JSON cache。
6. 只有缓存缺失时才调用 API。
7. API 结果通过 schema 校验、压缩、去重后入库。

## 多模态图片资产

系统会从 arXiv HTML 中寻找论文相关图片，优先选择正文 figure、overview、framework、architecture、pipeline、method 类图片。默认下载 1 张，用作多模态知识图谱的图片资产。

保存路径：

```text
data/images/<arxiv_id>/paper_image_1.<ext>
```

输出 JSON 中会写入：

```json
{
  "paper": {
    "images": [
      {
        "url": "remote image url",
        "local_path": "data/images/<arxiv_id>/paper_image_1.png",
        "caption": "figure caption",
        "alt": "image alt text",
        "content_type": "image/png",
        "bytes": 12345
      }
    ]
  },
  "assets": {
    "images": []
  }
}
```

如果启用 Neo4j，`Paper` 节点会保存首图的 `image_url` 和 `image_path`。

## Neo4j 缓存与入库

不配置 Neo4j 时，会自动使用本地缓存：

```text
data/extraction_cache.json
```

启用 Neo4j:

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="your_password"
$env:NEO4J_DATABASE="neo4j"
```

## 批量评估

```powershell
conda activate kg
$env:ARXIV_HTML_CACHE_FIRST="1"
python .\batch_eval_arxiv_html.py --dry-run --max-sections 5
```

也可以传入自己的 URL 文件：

```powershell
python .\batch_eval_arxiv_html.py --url-file .\urls.example.txt --dry-run --max-sections 5
```

汇总结果：

```text
data/batch_eval_summary.csv
```

## 输出格式

顶层 JSON：

```json
{
  "paper": {},
  "assets": {
    "images": []
  },
  "kg": {
    "nodes": [],
    "edges": []
  },
  "quality_report": {},
  "stats": {}
}
```

LLM 只允许输出内部 KG：

```json
{
  "nodes": [
    {
      "id": "claim_1",
      "type": "Claim",
      "name": "short canonical name",
      "aliases": [],
      "summary": "one short sentence",
      "evidence": {
        "section": "Abstract",
        "text": "short copied evidence span"
      },
      "confidence": 0.8,
      "research_value_score": 0.9
    }
  ],
  "edges": [
    {
      "source": "claim_1",
      "target": "evidence_1",
      "type": "supported_by",
      "evidence": {
        "section": "Abstract",
        "text": "short copied evidence span"
      },
      "confidence": 0.8
    }
  ]
}
```

## 主要节点类型

```text
Claim, EvidenceSpan, Experiment, PriorLimitation,
ResearchProblem, Contribution, Method, MethodComponent,
Dataset, Benchmark, Metric, Model, Baseline,
ExperimentSetting, Result, Finding, AblationFactor,
TheoreticalClaim, TheoreticalGuarantee, Limitation,
FutureWork, CodeResource, Funding, Table, Figure
```

## 主要关系类型

```text
supported_by, evaluated_by, reports_result, supports,
addresses, proposes, has_component, uses_dataset,
evaluates_on, uses_metric, uses_model, compares_with,
improves_over, has_limitation, suggests_future_work,
funded_by, has_resource, mentions
```

## 质量报告

每次输出都会附带：

```text
node_count
edge_count
nodes_by_type
claim_count
evidence_count
result_count
isolated_node_count
low_confidence_nodes
missing_evidence_nodes
```

科研上建议优先关注：

- `claim_count > 0`
- `evidence_count > 0`
- `isolated_node_count` 尽量低
- `missing_evidence_nodes` 必须为 0
