# 深知 KG 论文检索后端

这是可运行的论文检索后端，直接使用已经清洗好的
`derived/paper_data_v1/retrieval_documents.jsonl`。索引是派生文件，可以随时
删除后重建；`paper_id` 始终沿用 canonical ID，不把 SQLite 行号当业务 ID。

当前主链是 `BM25 + canonical Stage0 → (按需图过滤/扩展) → RRF →
可选 SkillRouter Reranker → 返回`。模型与图组件均可选；没有 GPU 或 Neo4j 时可降级为
BM25/SQLite 关系回退，且不会改变 canonical `paper_id` 和返回协议。

中文查询支持

论文正文和当前静态索引主要是英文，因此中文请求统一先经过
`data/domain_glossary.v1.json` 和可选的 `QUERY_TRANSLATOR_URL` 改写为英文，随后只走
一条英文 BM25/Stage0/reranker 链路。原始中文 query 始终保留用于审计；翻译服务超时
或失败时记录失败状态并不伪造召回结果，不再维护中文 alias BM25 或中英三路融合。
`/query-rewrite` 可单独检查改写结果，`/search` 返回 `query_rewrite`，便于线上审计。

在改写前，HTTP `/search` 和 `/multistep-search` 会先运行
`query_parser.py` 的确定性解析器：只识别带明确标记或白名单的年份、会议、作者、
关键词/主题条件，并识别“同作者”“同主题”等关系意图。解析出的条件进入硬过滤，
关系意图进入图扩展；未达到高置信度的文本保留在 `semantic_query`，继续交给翻译服务，
不会被猜成作者或会议。响应中的 `query_parse` 字段包含原 query、语义文本、关系文本、
最终过滤条件和提取记录，方便前端和日志审计。只有过滤条件而没有可检索主题时，系统
使用显式的 `BROWSE_PAPER_FILTERED` 浏览路径，不把空 query 静默替换成“最新论文”。
搜索响应中的 `query_rewrite.request_query` 保留完整用户输入；改写服务本身处理的是
解析后的 `semantic_query`。

英文查询仍保留受控的拼写纠错和前缀召回通道（`FUZZY_WEIGHT`，默认 `0.35`）；该通道
只作为低权重补充，不会重新引入中文多路召回。

这套改写支持不等于已经完成跨语言训练。Stage0/Stage2 不在当前中文工程改造中重训；
等待新的工程数据后，再构造中文查询到英文论文的标注数据并重新训练和评测。

## 运行

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
bash retrieval_backend/build_paper_index.sh

python3 retrieval_backend/paper_search.py search \
  --db retrieval_backend/papers_fts.db --top-k 10 \
  --year-gte 2023 --conference AAAI \
  'knowledge graph completion with graph neural networks'
```

需要前端联调时，可以直接启动标准库 HTTP 服务（无需 FastAPI）：

```bash
python3 retrieval_backend/http_server.py --db retrieval_backend/papers_fts.db --port 8080
curl --get 'http://127.0.0.1:8080/search' \
  --data-urlencode 'q=knowledge graph completion' \
  --data-urlencode 'year_gte=2023' --data-urlencode 'top_k=5'

# 中文条件会先解析年份/会议，再把剩余主题交给改写服务
curl --get 'http://127.0.0.1:8080/search' \
  --data-urlencode 'q=2020年以后在AAAI发表的图神经网络论文' \
  --data-urlencode 'top_k=5'
```

健康检查：

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

输出包含论文元数据、canonical `paper_id`、各召回/精排分数和可序列化的
`state`。`state` 记录 BM25、Stage0、Reranker、Neo4j 等已执行操作，主要用于
审计；普通一次搜索不会因此变成多步 Stage2 任务。

## 按需图增强

单步 `/search` 不会对所有查询调用图。只有显式传入年份、会议、作者、关键词或
subject 条件，或者查询中出现关系意图（例如“同作者的其他论文”“same author”“同领域”）
时，才会启动图过滤/扩展。图扩展从文本召回的前 `GRAPH_SEED_K` 篇论文开始，最多加入
`GRAPH_EXPAND_K` 个候选，并以 RRF 与 BM25/Stage0 合并；图的邻居计数不会直接和 BM25
原始分数相加。

支持的关系扩展为 `authors`、`keywords`、`subjects` 和 `venue_year`。未配置 Neo4j
时，关系意图会使用 SQLite 派生关系表作为确定性回退；Neo4j 短暂失败时同样回退，
并在 `state.failed_operations` 中记录错误。没有文本种子时不会凭空从图中搜索。
固定的数据/运行契约位于 `data/graph_retrieval_contract.v1.json`；当前源图虽然存有
`CITES`，但引用方向和用户意图尚未形成独立契约，因此不把它冒充“相关论文”扩展。

可调参数及默认值：

```text
GRAPH_WEIGHT=0.8
GRAPH_SEED_K=12
GRAPH_EXPAND_K=100
GRAPH_VALUE_LIMIT=16
GRAPH_INTENT_ENABLED=1
```

紧急回退或 A/B 基线可以设置 `GRAPH_INTENT_ENABLED=0`，此时关系语言不会触发扩展；
显式 HTTP 元数据条件仍由 SQLite 过滤。是否连接 Neo4j 仍由 `NEO4J_HTTP_URI` 单独控制。

正式索引已由当前数据生成：`retrieval_backend/papers_fts.db`（119,514 条，
约 931 MiB）。它是本地派生缓存，换机器或数据版本变化时重新执行 `build` 即可。

`/health` 是进程存活检查；`/ready` 会确认已配置的 Stage0 已完成加载。hybrid/full
启动脚本默认先执行一条预热查询，模型加载失败时服务不会假装就绪。HTTP 层默认限制
1 MiB 请求体、16 个并发请求和 30 秒连接超时；正式环境还应通过
`AUTH_TOKEN_ENV` 指定保存 Bearer token 的环境变量，并继续由反向代理负责 TLS、来源
控制和全局限流。

## 当前检索流程

1. 标题、摘要、关键词和 subjects 写入 SQLite FTS5；当前静态索引按英文词切分，
   中文请求先经中译英改写后进入同一套索引。
2. canonical vNext Stage0 可对 119,514 篇论文做 dense 召回。
3. BM25 与 dense 使用 Reciprocal Rank Fusion，不直接比较原始分数。
4. SkillRouter-Reranker-0.6B 对有界 Top-N 候选精排。
5. 对结构化条件，SQLite 始终执行元数据过滤；显式启用 Neo4j 后，相同条件还会经过
   参数化 Cypher 图校验。关系型查询再从 Top-N 文本种子沿图扩展，并通过 RRF 合并。
   图中不存在的新增论文不会被旧图无条件删除。

多步接口采用混合执行策略：初始搜索先应用硬条件，关系扩展按查询中识别到的作者、
关键词、主题或会议/年份顺序执行；每轮扩展后再次应用相同硬条件，最后返回融合排序结果。
未配置 Stage2 模型时由 `RuleActionPolicy` 确定性选择关系动作，模型不是运行必需项。

## canonical Stage0 与 Stage2 边界

最新源码位于：
`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source`。
每篇论文已经适配为 canonical vNext Stage0 候选，`paper_id` 同时作为
`skill_id`。Stage0 负责一次请求里的静态 dense 召回；普通“输入一句话，返回论文”
不需要 Stage2。

只有拿到真实跨节点查询、每一步执行结果和下一步选择标签后才训练 vNext Stage2。
当前不会把一次检索人为拆成多步，也不会伪造 Paper→Author→Institution 轨迹。

### 当前 Stage2 bootstrap（工程联调）

项目已经额外生成一版可执行的弱监督 Stage2 数据：
`../derived/action_vnext_stage2_v1/`。它来自本地 SQLite 论文索引的真实回放，
包含 `SEARCH_PAPERS → 扩展动作 → 扩展动作 → STOP` 轨迹（train 96 条、dev 48
条）及 40/24 组 causal branch pair。数据契约和动作 inventory 审计均通过，
但这不是用户日志，也没有人工相关性标签；因此只能证明训练链路、动作合法性和
多步执行器可运行，不能宣称真实用户泛化能力。拿到真实多步查询后应替换这些文件
并重新训练、评测。

Stage2 全量训练当前选择 step 800：
`outputs/action_vnext_stage2_skillrouter_full/checkpoints/clstr_vnext_stage2-step800.pt`。
它通过 causal、safety、coarse-recall 和 gradient quality gates。由于现有数据没有
独立 held-out open-pool 分层，open-pool release selector 会 fail-closed；该模型可
用于项目内 GPU 联调，但暂不作为真实泛化发布模型。

HTTP 多步接口默认继续使用规则策略；配置 Stage2 后按请求创建独立 recurrent session：

```bash
python3 retrieval_backend/http_server.py \
  --db retrieval_backend/papers_fts.db \
  --stage2-checkpoint outputs/action_vnext_stage2_skillrouter_full/checkpoints/clstr_vnext_stage2-step800.pt \
  --stage2-skills outputs/action_vnext_stage0_skillrouter_full/selected_skills.jsonl \
  --stage2-clstr-source /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source \
  --stage2-device cuda
```

模型只从执行器提供的合法 skill 中选动作；作者、关键词、subject、venue/year
等具体参数仍由当前候选论文确定性绑定。模型加载或运行失败时自动回退规则策略，
不会返回非法动作或破坏原 `/search` 接口。

启用 Neo4j 时，扩展动作会沿 `Paper→Author/Topic/Conference` 真实关系
执行参数化 Cypher；没有图服务时才使用 SQLite 派生关系表作为可复现回退。

GPU 在线 smoke 可通过 `sbatch retrieval_backend/sbatch_stage2_online_smoke.sh` 提交；
它检查真实 checkpoint、515 个动作、合法动作约束、执行结果反馈和 STOP。当前报告
`outputs/stage2_online_smoke.json` 已通过并被 release manifest 哈希绑定。

冻结 dev 多查询评测（48 条弱监督可执行轨迹）结果：teacher-forced Top-1
`0.7604`、Top-5 `0.9531`；closed-loop 停止率和非空结果率均为 `1.0`，非法动作
与执行失败均为 `0`，平均 `3.69` 步、约 `948 ms/query`。报告位于
`outputs/stage2_policy_eval.json`。这些是生成回放数据上的工程指标，不能替代真实
用户查询和人工相关性评测。

## Stage0 训练数据

`../derived/paper_stage0_v1/` 是早期 `retrieval_warmup/unified_v2` bootstrap：

- `skill_pool.jsonl`：119,514 篇论文，每篇以 `paper_id` 作为 `skill_id`；
- `retrieval.jsonl`：239,028 条弱监督查询（标题模板和摘要首句各一条）；
- `splits.json`：按论文 ID 哈希切分，train/dev/test = 190,930/24,014/24,084；
- 每条查询有 8 个同会议/年份优先的确定性 hard negative；
- `manifest.json` 明确标注 `label_quality=weak_bootstrap`。

它不能直接传给当前 canonical vNext Stage0。现在已通过
`build_vnext_stage0_paper_data.py` 生成 `../derived/paper_vnext_stage0_v1/`：

- `skills.jsonl`：119,514 个论文候选；
- `retrieval_train.jsonl` / `retrieval_dev.jsonl`：190,930 / 24,014 条；
- `static_route_train.jsonl` / `static_route_dev.jsonl`：与论文检索共享查询和正例，
  用于满足 Stage0 的统一静态检索/路由双目标；
- `inventory_catalogs.jsonl`：绑定全部合法论文候选的哈希目录；
- `data_contract.json`：绑定以上输入文件路径和 SHA-256；
- 24,084 条 test query 不进入任何 trainer 输入。

所有样本都显式记录为零历史单步查询，未伪造论文多步轨迹。当前 vNext Stage0
会在完整合法候选池内在线挖 hard negative，所以早期数据中的 8 个负例只保留为
bootstrap 生成记录，不直接塞入新 trainer。

重新生成命令：

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
python3 retrieval_backend/build_vnext_stage0_paper_data.py \
  --skills derived/paper_stage0_v1/skill_pool.jsonl \
  --retrieval derived/paper_stage0_v1/retrieval.jsonl \
  --output-dir derived/paper_vnext_stage0_v1
```

训练入口是当前代码的 `scripts/run_clstr_vnext_stage0_train.py`。20-step A800
smoke 已完成并通过质量门；全量训练使用：

```bash
sbatch retrieval_backend/sbatch_stage0_full.sh
```

这些数据适合验证 SkillRouter 初始化和 Stage0 管线，不是最终人工相关性评测集。
正式训练前仍应补充真实用户查询、人工正例和更可靠的 hard negative。

当前可用的本地模型：

- Embedding：`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Embedding-0.6B`
- Reranker：`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Reranker-0.6B`

Embedding 用于 Stage0 粗召回；Reranker 用于召回 Top-N 后的二阶段重排。Reranker
不是 Stage0 的训练目标，也不是 CLSTR vNext Stage2 的替代品。

真实 GPU smoke 已确认 BM25、canonical Stage0 dense 和本地 0.6B Reranker 在同一
请求中成功执行，报告在 `outputs/gpu_inference_smoke.json`。

全量 Stage0 已完成 1,000 steps，严格 artifact 验证通过。最终 checkpoint 为
`outputs/paper_vnext_stage0_skillrouter_full/checkpoints/clstr_vnext_stage0-step1000.pt`，
SHA-256 是 `e109bbab0a93764639197c6406f9b7b314220bb3711142e093f8aea1d074f525`；
selection score 从 0.000245 提升到 0.721950。

## Neo4j

Neo4j 使用 Apptainer 和 Neo4j 5.26 Community 运行，不依赖主机 Java 8。原始 ZIP
保持只读；`prepare_neo4j_import.py` 会验证节点、去掉 BOM、按
`(start_id, end_id, type)` 去重关系，并生成带源 ZIP SHA-256 的 manifest。

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
python3 retrieval_backend/prepare_neo4j_import.py \
  --source-zip paper_kg_neo4j_import_ready.zip \
  --output-dir runtime/neo4j/import_ready
bash retrieval_backend/import_neo4j_apptainer.sh
NEO4J_AUTH=none bash retrieval_backend/run_neo4j_apptainer.sh console
```

当前 final 实例已验证 661,378 个节点、347,148 个 Paper（其中正式 Paper 113,653）、
8 个唯一约束和 20 个索引。线上 SQLite 的 110,598 个 Paper ID 全部能在图中精确
命中。生产环境不要使用 `NEO4J_AUTH=none`，密码应通过环境变量提供。

## 一键启动

配置示例见 `retrieval_backend/retrieval.env.example`。正式服务可按需导出其中变量；
密码不要提交到仓库。全量服务启动方式：

```bash
export RETRIEVAL_MODE=full
export NEO4J_HTTP_URI=http://127.0.0.1:17474
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='replace-me'
bash retrieval_backend/run_retrieval_service.sh
```

`RETRIEVAL_MODE` 可选 `bm25`、`hybrid`、`full`。Neo4j 只在设置
`NEO4J_HTTP_URI` 后启用。`http_smoke_test.py` 可验证健康检查、模型组件和图过滤。
没有 GPU 或模型时使用 `RETRIEVAL_MODE=bm25`；没有 Neo4j 时不要设置
`NEO4J_HTTP_URI`，HTTP 协议和 canonical `paper_id` 不变。

启用图增强的服务配置示例：

```bash
export NEO4J_HTTP_URI=http://127.0.0.1:17474
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='replace-me'
export GRAPH_WEIGHT=0.8
export GRAPH_SEED_K=12
export GRAPH_EXPAND_K=100
export GRAPH_VALUE_LIMIT=16
bash retrieval_backend/run_retrieval_service.sh
```

`output_graphs.zip` 不能单独作为 Neo4j 图源：它没有边且大多数 Paper ID 为空。当前
生产使用的是合并并清洗后的 final 图，不应再切回旧 `paper_kg_neo4j_import_ready.zip`。
后续增量应继续使用带 canonical ID、来源和关系证据的数据合并到 final 图。

## 评测

`evaluate_retrieval.py` 支持 BM25、dense、hybrid 和可选 Reranker；报告 MRR、
graded NDCG、hit@K、多正例 Recall@K 和时延。冻结弱监督 test 有 24,084 条，
只用于工程回归，因为查询来自标题或摘要首句。

`evaluate_graph_ablation.py` 对同一批查询比较文本基线与图增强路径，报告图操作次数、
SQLite 回退次数、失败类型、返回数量和延迟；输入包含人工 qrels 时才额外输出
MRR/Recall，不会自动制造相关性标签。示例：

```bash
export NEO4J_PASSWORD='replace-me'
python3 retrieval_backend/evaluate_graph_ablation.py \
  --db retrieval_backend/papers_fts.db \
  --queries evaluation/human_queries_seed_v1.jsonl \
  --neo4j-http-uri http://127.0.0.1:17474 \
  --neo4j-user neo4j \
  --top-k 10 \
  --output outputs/graph_ablation.json
```

`evaluation/` 另含 24 条中英双语真实风格查询和待标 qrels；
`QRELS_ANNOTATION.md` 定义 0-3 级相关性、双人标注和复核流程。`null` 相关性不会
被评测脚本当成正例。

当前可复现结果：

- 24,084 条冻结弱监督 test 的 BM25：MRR 0.981842、NDCG@100 0.985975、
  Recall@100 0.999294，12-worker wall time 658.0 秒；
- 固定 200 条 GPU 回归集：dense MRR 0.687730 / Recall@100 0.945，
  hybrid MRR 0.869019 / Recall@100 1.0，hybrid+reranker Top-20 MRR 1.0；
- full HTTP smoke 已真实执行 BM25、Stage0、Neo4j 和 SkillRouter Reranker；
- 新图增强 HTTP smoke 已真实执行 `FILTER_PAPER_NEO4J` 和
  `EXPAND_PAPER_AUTHORS_NEO4J`；8 条未标注操作查询中图路径执行 8 次图操作、改变
  4 条 Top-10 并新增 8 个结果，0 次执行失败。该结果只证明图被正确使用，不代表相关性提升；
- BM25-only HTTP 回退也已独立 smoke，通过时仅执行 `RECALL_PAPER_BM25`；
- 24 条人工查询对应 466 个候选仍全部待人标，不能把上述弱监督指标当作真实用户
  泛化结论。

最终发布检查：

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/retrieval_backend
bash verify_release.sh
```

脚本会逐个检查 shell 语法、自动发现并运行检索后端与数据清洗管线的全部测试、严格
验证 Stage0 与 qrels 状态，最后生成 fail-closed 的 `outputs/release_manifest.json`。
当前检索后端 39/39、数据管线 7/7 测试均通过，release manifest 状态为 `ok`。

## 数据边界

`Institution`、`Method`、`Funding` 等尚未出现在当前清洗输出的节点，不在本
原型中臆造。它们需要后续从论文全文、DBLP/Crossref 或人工标注补充，并在 Neo4j
中建立关系后才能作为硬过滤条件。
# Qwen Embedding + Zilliz（准备阶段）

`vector_ingest.py` 从 `papers_fts.db` 流式读取论文，调用 OpenAI-compatible 的 Qwen3-Embedding-8B API，再批量 upsert 到 Zilliz。它支持 `--dry-run`、`--limit` 和 `--resume`，默认批量大小为 8，适合当前低内存服务器。`vector_retriever.py` 提供与现有 RRF 主链兼容的查询适配器，但在完成平台 smoke 前不会自动接入线上服务。

先复制 `vector_ingest.env.example` 到未跟踪的环境文件并填写 Qwen API 与 Zilliz 参数，然后执行：

```bash
python3 vector_ingest.py --dry-run --limit 10
python3 vector_ingest.py --limit 10
python3 vector_ingest.py --resume
```

需要启用 Zilliz SDK 时，在隔离的向量任务环境安装 `requirements-vector.txt`；不要为当前 BM25 服务强行安装或加载它。

当前 API 默认返回 768 维；实测设置 `QWEN_EMBEDDING_DIMENSIONS=1024` 可返回 1024 维。不同维度必须使用不同的 Zilliz Collection，并重新生成向量。

第一次真实运行只建议 10 篇 smoke；确认向量维度、集合索引和中文/英文查询后，再逐步增加批量。Qwen 或 Zilliz 不可用时，线上检索仍使用现有 BM25 + Neo4j 回退。
