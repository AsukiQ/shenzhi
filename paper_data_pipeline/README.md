# 深知论文数据准备工具

本工具完成第一版检索数据准备：

1. 读取 `深知PAPER_DATA_20260528.json`；
2. 使用其中的 `arxiv_id`（实际为来源记录 ID）合并重复记录；
3. 与 Neo4j ZIP 的 `papers.csv` 对齐并沿用已有 canonical `paper_id`；
4. 仅根据来源 ID 为 JSON 中新增论文生成稳定的哈希 `paper_id`；
5. 输出干净论文元数据、检索文档、ID 映射和质量报告。

原始 JSON 和 ZIP 均保持只读。输出目录如果已经存在，程序会拒绝覆盖。

## 2026 canonical 新图：rich schema 重建（当前方案）

`canonical_*_2026.zip` 以及 ICML/VLDB/S&P 的 `*_graph.zip` 使用统一的
`{graph:{nodes,edges},audit}` 格式。当前新图方案不再把 Institution、Method、Funding、
Conference 等放在 staging，而是按来源 canonical ID 合并八类节点和九类关系：

- 节点：Paper、Author、Institution、Method、Funding、Topic、Conference、Venue；
- 关系：AUTHORED_BY、AFFILIATED_WITH、PROPOSES、USES_AS_BASELINE、FUNDED_BY、
  HAS_TOPIC、PUBLISHED_IN、PART_OF、CITES。

构建全部新图候选：

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
PYTHONPATH=paper_data_pipeline/src python3 -m shenzhi_paper_data.build_rich_graph \
  --zip canonical_ACL_2026.zip \
  --zip canonical_CVPR_2026.zip \
  --zip canonical_FSE_2026.zip \
  --zip canonical_IJCAI_2026.zip \
  --zip canonical_INFOCOM_2026.zip \
  --zip canonical_POPL_2026.zip \
  --zip canonical_SIGKDD_2026.zip \
  --zip canonical_SIGMOD_2026.zip \
  --zip ICML2026_graph.zip \
  --zip VLDB2026_graph.zip \
  --zip 'S&P2026_graph.zip' \
  --output-dir derived/neo4j_rich_candidate_20260821_all_2026_v2

PYTHONPATH=paper_data_pipeline/src python3 -m shenzhi_paper_data.validate_rich_graph \
  derived/neo4j_rich_candidate_20260821_all_2026_v2 \
  --output derived/neo4j_rich_candidate_20260821_all_2026_v2/validation_report.json
```

输出包括 `paper_kg/*.csv`、`retrieval_documents.jsonl`、`constraints.cypher`、
`import_neo4j_admin.sh`、输入/输出 SHA-256 manifest 和完整质量报告。主论文缺少的作者边会从
有序 `Paper.authors` 属性确定性补齐，并标记 `derived=true`；不会推断不存在的机构关系。
Dataset/Code 在当前 canonical 输入中是 Paper 属性，因此保留为 JSON 属性，不虚构独立节点。

旧 `paper_kg_neo4j_import_ready.zip` 是另一套扁平 CSV schema，只作为历史基线保留，不能与
上述新图 CSV 直接混用。下文的 A 层/staging/旧图 additive merge 是 2026-08-17 的过渡流程，
不再是新 canonical 数据的默认接入路径。

## 运行

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/paper_data_pipeline

PYTHONPATH=src python3 -m shenzhi_paper_data.prepare \
  --json ../深知PAPER_DATA_20260528.json \
  --zip ../paper_kg_neo4j_import_ready.zip \
  --output-dir ../derived/paper_data_v1
```

也可以安装为可编辑包：

```bash
python3 -m pip install -e .
shenzhi-prepare-papers --help
```

## 输出

| 文件 | 用途 |
|---|---|
| `papers_clean.jsonl` | 全部去重论文元数据，包括占位摘要论文 |
| `retrieval_documents.jsonl` | 只含正常摘要，可直接送入 BM25/向量索引 |
| `paper_id_map.csv` | `source_id -> canonical paper_id` 映射 |
| `quality_report.json` | 机器可读的数据质量、输入哈希和输出哈希 |
| `quality_report.md` | 人工阅读版摘要 |

## 当前边界

- 同标题但来源 ID 不同的记录不会自动合并，以免误合并不同论文。
- 作者/编辑列表若出现完整序列重复，会折叠为一份，同时保留原始顺序。
- `subjects` 当前是会议栏目/轨道，不是结构化 Method。
- 本工具不会修改 ZIP 中的 Author/Keyword/Citation 等关系表。
- Institution、Method、Funding 等节点需要独立的数据补全流程。

## 新图谱 JSON 的只读 dry-run

`ICML2026_graph.zip`、`VLDB2026_graph.zip` 和 `S&P2026_graph.zip` 不是旧的 CSV import ZIP，而是
“每篇论文一个 `graph.nodes/graph.edges` JSON”。使用独立适配器可以把三个包一起
规范化，原始 ZIP 不会被修改，输出目录已存在时也不会覆盖：

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/paper_data_pipeline
PYTHONPATH=src python3 -m shenzhi_paper_data.prepare_graph_json \
  --zip ../ICML2026_graph.zip \
  --zip ../VLDB2026_graph.zip \
  --zip ../S\&P2026_graph.zip \
  --output-dir ../derived/graph_json_ingest_dryrun
```

输出分为三层：

- `retrieval_documents.jsonl`：可直接建立 BM25/向量索引的主论文文档；
- `graph_import/paper_kg/`：只含 A 层 Paper、从 `Paper.authors` 重建的完整 Author、Topic→Keyword、Venue、Year 及对应边；
- `staging/`：完整保留 Method、Institution、Funding 和全部 CITES，等待质量门与数据挖掘负责人确认后再接入。

`quality_report.json` 会记录输入哈希、论文/作者覆盖、异常 arXiv ID、引用分级和接入边界。该目录是 dry-run 产物，不能直接替换线上 Neo4j 或启动最终训练。

## 生产合并前的 canonical 对齐

不同图谱导出的 `Author/Keyword/Venue` ID 不能直接当作同一实体。正式合并前先执行：

```bash
PYTHONPATH=src python3 -m shenzhi_paper_data.align_neo4j_addition \
  --base-zip ../paper_kg_neo4j_import_ready.zip \
  --addition-dir ../derived/graph_json_ingest_dryrun_20260817_three_graphs/graph_import \
  --output-dir ../derived/neo4j_addition_aligned

PYTHONPATH=src python3 -m shenzhi_paper_data.merge_neo4j_import \
  --base-zip ../paper_kg_neo4j_import_ready.zip \
  --addition-dir ../derived/neo4j_addition_aligned \
  --output-dir ../derived/neo4j_merge_candidate \
  --output-zip ../derived/neo4j_merge_candidate.zip
```

对齐器只自动接受 arXiv/DOI 精确 Paper 身份和规范化实体名称；标题-only、异常 arXiv ID、同标题冲突会进入 `alignment_report.json` 隔离。合并器遇到新增冲突或悬空关系会直接失败。候选包通过备份图导入和服务 smoke 后才能切换线上。

## staging 清洗与引用候选

在 A 层 canonical 对齐完成后，再对三张图谱的 Method、Institution、Funding 和 CITES 做保守清洗。这个步骤不改写原始 `staging/`，每条 clean 记录都保留 `raw` 字段，无法可靠对齐的记录进入 quarantine：

```bash
PYTHONPATH=src python3 -m shenzhi_paper_data.clean_staging \
  --staging-dir ../derived/graph_json_ingest_dryrun_20260817_three_graphs/staging \
  --merged-papers-csv ../derived/neo4j_merge_candidate_20260817_three_graphs_aligned_v2/root/paper_kg_neo4j_import_merged/paper_kg/papers.csv \
  --alignment-report ../derived/neo4j_addition_aligned_20260817_three_graphs_v2/alignment_report.json \
  --output-dir ../derived/graph_json_staging_clean_20260817
```

输出目录包含：

- `methods_clean.jsonl`、`institutions_clean.jsonl`、`funding_clean.jsonl`：规范化后的实体候选；
- `citations_clean.jsonl` 和 `cites_import.csv`：source/target 均已对齐且置信度达标的引用候选；
- `citations_quarantine.jsonl`、`rejected.jsonl`：标题-only、低置信度、抽取残片等待复核记录；
- `quality_report.json`：数量、质量门和下一版图契约。

这里默认三张图谱的事实来源可信，清洗阶段不做事实真伪审核，只处理明显抽取噪声、格式规范化和 Paper ID 对齐。当前有 18 条 CITES 通过自动质量门；Method、Institution、Funding 以及这 18 条 CITES 已标记为 `ready_for_incremental_import`。上线时仍应先在备份库执行增量导入和服务 smoke，再切换线上实例，这是发布流程而不是事实审核。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
