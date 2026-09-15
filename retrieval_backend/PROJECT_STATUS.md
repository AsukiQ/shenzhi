# 深知 KG 论文检索项目进度

更新时间：2026-08-24

## 一、已经完成

### 1. 数据准备

- 原始 JSON 137,301 条记录完成去重和字段合并；
- 得到 122,561 篇唯一论文；
- 119,514 篇论文有可用摘要并进入检索文档；
- 早期 107,200 篇沿用旧 Neo4j ZIP 的 canonical `paper_id`；当前线上 SQLite 已更新为
  110,598 篇，全部能按 `Paper.paper_id` 在 final 图精确命中；
- final 图现有 661,378 个节点，其中 Paper 347,148、正式 Paper 113,653；
- 原始 JSON 和 ZIP 保持只读。

### 2. 基础后端检索

- SQLite FTS5 + BM25 关键词检索；
- 中文请求统一中译英后进入英文 BM25 索引；不维护独立中文 alias/BM25 分支；
- 标题/摘要/关键词/subjects 检索；
- 年份、会议、作者、关键词和 subject 过滤；
- 按查询意图触发的图过滤/关系扩展；
- HTTP 单步和多步接口已接入确定性中文条件解析：年份、会议白名单、显式作者、已验证关键词/主题，以及同作者/同主题等关系意图；
- 纯结构化条件使用明确的 `BROWSE_PAPER_FILTERED` 候选浏览路径，不把空语义查询静默替换为最新论文；
- JSON CLI 和 `/search` HTTP 接口；
- 已生成 `retrieval_backend/papers_fts.db`。

### 3. Stage0 数据

- 早期 `unified_v2` 数据仍保留在 `derived/paper_stage0_v1/`；
- 已新增当前 canonical vNext Stage0 数据
  `derived/paper_vnext_stage0_v1/`；
- 119,514 个论文候选；
- train/dev/test query 数量为 190,930 / 24,014 / 24,084，其中 test 完全排除在
  trainer 输入之外；
- 已生成 retrieval/static-route train/dev、全局 inventory catalog 和绑定 SHA-256
  的 `data_contract.json`；
- 所有样本均为显式零历史单步查询，没有伪造多步论文轨迹；
- 该数据标注为 `weak_bootstrap`，不是最终人工 qrels。

## 二、最新源码的实际 canonical 主线

你提供的最新分支 `clstr-qwen06-native-sync-smoke-source` 当前代码已经转向
vNext 两阶段主线：

```text
Stage0 → Stage2
```

- `Stage0`：统一静态检索基础，训练 query/document 路由和静态候选召回；
- `Stage2`：从 Stage0 checkpoint 出发，训练带 causal memory 的统一路由。

代码证据是：

- `scripts/run_clstr_vnext_stage0_train.py` → `clstr.vnext_stage0_train.train_vnext_stage0`；
- `scripts/run_clstr_vnext_stage2_train.py` → `clstr.vnext_stage2_train.train_vnext_stage2`；
- `scripts/finalize_clstr_vnext_full_chain.py` 最终链只校验 `Stage0`、可选
  `candidate_compressor` 和 `Stage2`。

旧的 `run_clstr_stage1_heads_init.py`、`run_clstr_stage2_full_base_train.py`、
`run_clstr_stage4_act_train.py` 仍保留在仓库中，属于旧版/兼容实验链，不能拿来
代表当前 vNext canonical 主线。`candidate_compressor` 是可选的静态路由适配器，
不是一个编号 stage；`precompute` 只是数据预处理，也不是训练 stage。

`SkillRouter-Reranker` 是外部可选重排模型，同样不属于 CLSTR 的 Stage0/Stage2。

深知 KG 的第一版实际链路应当是：

```text
BM25/规则基线
    ↓
Stage0：SkillRouter Embedding 初始化的论文静态检索（可选）
    ↓
按查询意图：Neo4j 过滤或从 Top-N 种子沿 Author/Topic/Conference 扩展
    ↓
RRF 融合文本与图候选
    ↓
Reranker：对 Top-N 候选重排（可选）
    ↓
返回论文
```

旧 Stage1/Stage2/Stage4 链不属于当前主线。当前 vNext Stage2 是后面的多步检索
扩展，不是单步论文检索的必然后续。对普通“输入一句话，返回论文”的请求，BM25，
或者 Stage0 + Reranker，就已经是完整的一次检索链路。

如果以后要做“先找方法，再沿 Paper→Author→Institution 扩展，再按会议过滤”这类
多步查询，论文版可以参考当前 vNext 的 Stage2；不需要把旧版 Stage1/Stage2/Stage4
链路原样搬过来。

## 三、各阶段当前状态

下表描述本地完整研究链。当前 paperddl 远端由于只有 2 核 CPU、3.5 GiB 内存且无
GPU，Python 主接口实际运行的是 BM25；Stage0 文件已经部署但只供升级硬件后启用，
Reranker 和 Stage2 没有放入当前远端运行链。Neo4j final 图已供 Node `/api/kg/*`
使用；Python 图适配代码和低并发 canary 已验证，但生产 `8080` 尚未重启切换。

| 阶段 | 论文任务状态 | 说明 |
|---|---|---|
| BM25 | 已完成 | 历史训练/评测 119,514 篇；当前线上 SQLite 110,598 篇 |
| Stage0 | 已完成 | 全量 119,514 候选、1000-step checkpoint、严格质量门和 GPU 接入均通过 |
| Reranker | 已完成 | SkillRouter 0.6B Top-N 精排、GPU 矩阵和 HTTP smoke 均通过 |
| vNext Stage2 | 弱监督 bootstrap 已训练 | 96/48 条可执行轨迹、40/24 组 causal pair；step 800 checkpoint 通过工程质量门，但没有真实用户多步泛化证据 |
| Neo4j 过滤/扩展 | canary 已完成 | final 图 661,378 节点；Python schema、回退和 2 并发验证通过，待维护窗口切换 8080 |
| Method/Institution/Funding | 图中已有、检索未开放 | 需先审计覆盖率和用户需求，再增加过滤/扩展动作 |
| 评测 | 工程评测已完成 | 人工 qrels 雏形已生成，仍需团队真实标注后才能做用户泛化结论 |
| 发布 | 部分上线 | Node final 图和 Python BM25 已上线；Python 图增强已过 canary，待维护窗口切换 8080 |

## 四、最终验证结果

1. Stage0 最终选择 step 1000，selection score 0.000245 → 0.721950，checkpoint
   SHA-256 为 `e109bbab0a93764639197c6406f9b7b314220bb3711142e093f8aea1d074f525`；
2. 24,084 条冻结弱监督 test 的 BM25：MRR 0.981842、NDCG@100 0.985975、
   Recall@100 0.999294；
3. 固定 200 条 GPU 回归集：dense MRR 0.687730 / Recall@100 0.945，hybrid
   MRR 0.869019 / Recall@100 1.0，hybrid+reranker Top-20 MRR 1.0；
4. full HTTP smoke 实际执行 `RECALL_PAPER_BM25`、`RECALL_PAPER_STAGE0`、
   `FILTER_PAPER_NEO4J`、`RERANK_PAPER_SKILLROUTER`；
5. BM25-only 回退 HTTP smoke 在未启用 GPU 模型和 Neo4j 时独立通过；
6. final Neo4j 验证 347,148 Paper、8 constraints、20 indexes；SQLite 的 110,598
   个 ID 全部对齐；
7. 历史版本发布校验曾通过检索后端 39/39、数据管线 7/7；final 图切换后需要重跑并
   生成新的 `outputs/release_manifest.json`，不能沿用旧报告。
8. 新增真实 Neo4j 图增强 smoke：单步 `/search` 已执行过滤和 Author 扩展；8 条
   未标注操作查询执行 8 次图操作、改变 4 条 Top-10、加入 8 个新结果且 0 失败。
   这些数字只证明运行路径和候选变化，不能替代人工相关性结论。

上述冻结 query 来自论文标题/摘要，只能用于工程回归，不能单独证明真实用户查询的
泛化质量。

## 五、外部验收与后续条件

工程实现已经可以交付。仍需由团队完成的外部工作有两类：

1. 对 `evaluation/human_qrels_annotation_v1.jsonl` 的 466 个候选做双人相关性标注，
   当前全部为 `null`，状态明确为 `awaiting_human_labels`；
2. 如果以后确认需要跨多类节点的连续决策，再采集真实多步 query、每一步动作、执行
   结果和下一步标签后训练 Stage2。在这些数据出现之前不伪造轨迹。

## 六、Stage2 当前工程状态

Stage2 数据位于 `derived/action_vnext_stage2_v1/`，来源标记为
`weak_bootstrap_executable_replay`。它是由真实论文索引执行器回放构造的训练雏形，
不是用户行为日志；因此当前结论限于“数据契约、训练流程、checkpoint 加载和多步
执行闭环可运行”。全量训练选择：

`outputs/action_vnext_stage2_skillrouter_full/checkpoints/clstr_vnext_stage2-step800.pt`

该 checkpoint 通过 causal route、causal/ordinary safety、coarse recall 和 gradient
health gates。独立 open-pool held-out release 仍 fail-closed，因为现有弱监督数据没
有独立 open-pool 分层；在真实数据到来前只能用于内部 GPU 联调。`/multistep-search`
已支持可选 Stage2 policy，按请求创建独立 session，模型异常自动回退到规则策略。

当启用 Neo4j 时，单步关系型查询以及 `EXPAND_AUTHORS`、`EXPAND_KEYWORDS`、
`EXPAND_SUBJECTS` 和 `EXPAND_VENUE_YEAR` 会优先通过参数化 Cypher 沿真实图关系扩展；
未启用图服务或 Neo4j 短暂失败时回退到 SQLite 派生关系表/文本候选。图过滤和图扩展
均保持候选论文 ID 的确定性绑定，普通主题查询不会强制调用图。

GPU 在线 smoke 脚本为 `retrieval_backend/stage2_online_smoke.py`，提交脚本为
`retrieval_backend/sbatch_stage2_online_smoke.sh`；作业 136960/136962 已完成，报告
状态为 `ok`。另对冻结 48 条
Stage2 dev 轨迹完成独立 GPU 评测：192 个 teacher-forced decision 的 Top-1/Top-5
为 0.7604/0.9531；closed-loop 48/48 正常停止且有结果，非法动作与执行失败均为 0，
平均 3.69 步、948 ms/query。该评测明确限定为弱监督工程回归，不代表真实用户泛化。

`Institution`、`Method`、`Funding` 已存在于 final 图，但当前 Python API 还没有对应
的结构化过滤字段。正式开放前仍需审计覆盖率、实体质量并补充接口与评测，不能仅因
节点存在就直接加入默认排序。
