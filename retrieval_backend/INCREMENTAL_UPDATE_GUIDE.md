# 知识底座增量更新操作说明

更新时间：2026-10-14。

入口：`/home/user2/jiansuo/retrieval_backend/import_new_batch.py`。

支持按 `paper_id` 新增、修改论文，同步 Neo4j 节点/关系及 SQLite 元数据、FTS 和筛选关联表。已有参考文献节点可以升级为主论文。同一批文件重复执行会复用批次记录；失败后可按批次号继续。线上数据库原地按事务更新，不需要重建整库。

## 1. 准备输入

支持 canonical ZIP、JSON、JSONL 和 RAR。RAR 需要 `unrar.cffi`；建议交付 ZIP。JSONL 每行必须是一个完整论文图，而不是普通的论文元数据行。

每个图包含 `graph.nodes` 和 `graph.edges`（或 `relationships`）。一个文件只描述一篇主论文及其关联实体、参考文献。推荐明确提供顶层 `main_paper_id`；没有该字段时，脚本根据 `external=false`、论文的作者/主题/发表关系识别主论文，识别不唯一则停止。

```json
{
  "main_paper_id": "paper:example-2026-001",
  "graph": {
    "nodes": [
      {"id": "paper:example-2026-001", "label": "Paper", "properties": {
        "title": "Incremental Knowledge Graph Retrieval",
        "abstract": "An example research abstract.",
        "year": 2026,
        "keywords": ["retrieval"]
      }},
      {"id": "topic:knowledge_graph", "label": "Topic", "properties": {"name": "knowledge graph"}}
    ],
    "edges": [
      {"source": "paper:example-2026-001", "target": "topic:knowledge_graph", "type": "HAS_TOPIC"}
    ]
  }
}
```

修改旧论文时必须沿用原 `paper_id`，不能因为标题修改重新生成 ID。同名论文使用了不同 ID 时，脚本会报告冲突，需先确认数据身份。同一批内同一 ID 的冲突快照也会停止。现存历史包就存在“标题 ID”和 arXiv ID 对应同名论文的情况，所以不要把历史 `new/` 目录直接全部重灌。

支持 Paper、Author、Topic、Method、Institution、Venue、Conference、Funding。支持的关系为 AUTHORED_BY、HAS_TOPIC、PUBLISHED_IN、CITES、USES_AS_BASELINE、PROPOSES、FUNDED_BY、HAS_AFFILIATION、AFFILIATED_WITH、PART_OF。输入出现未知标签、关系或缺失端点会在正式写入前停止，避免悄悄丢数据。

## 2. 先预览

```bash
cd /home/user2/jiansuo/retrieval_backend
python3 import_new_batch.py /path/to/canonical_new_batch.zip --dry-run
```

输出 `new_to_sqlite` 和 `existing_papers_to_update`。这是根据 SQLite 的 ID 判断新增/已有，不等于 Neo4j 主论文数量的变化。预览不写数据库，只检查输入结构、元数据及 ID/标题冲突；Neo4j 连通性、唯一性约束和作者 ID 解析会在正式导入时检查。

## 3. 正式导入

```bash
python3 import_new_batch.py /path/to/canonical_new_batch.zip --apply
```

默认读取 `deployment/paperddl.env.bm25` 中的 Neo4j 连接信息，环境变量可覆盖。密码不放在命令行。可以用 `--env-file`、`--db` 和 `--state-dir` 指定其他环境。

默认 `--relation-mode replace`：输入视为该论文**完整的关系快照**。输入包含的作者、主题、引用、基金等关系新增或保留，旧的、输入未包含的主论文出边会删除。论文节点和关联实体不自动删除，其他论文指向该论文的引用会保留。

如果仅交付补充关系，使用：

```bash
python3 import_new_batch.py /path/to/partial_graph.zip --relation-mode merge --apply
```

`merge` 只增加/更新关系，保留已有出边。缺失的标题以外标量字段保留现有 SQLite 值；明确传空摘要或 null 年份则清空该字段。作者、主题、基金从对应图关系同步，关键词由论文的 `keywords` 字段更新。`replace` 模式的空关系表示确实清空相应关系，不能拿不完整抽取结果当完整快照。

论文属性里的 `authors` 数组不替代 AUTHORED_BY，`subjects` 数组不替代 HAS_TOPIC；需要图筛选与检索筛选一致时，应交付相应节点和关系。

### 可选图片

```bash
python3 import_new_batch.py /path/to/canonical_new_batch.zip \
  --image-archive /path/to/new_primary_images.zip --apply
```

图片按原有命名方式匹配：`canonical_CONF/name_canonical.json` 对应 `CONF/name_minimal.png`，也支持原有日期后缀。每篇论文只接收一张匹配的主图，多图冲突则停止。可重复传 `--image-archive`。图片先保存到批次目录，校验 SHA-256 后发布到 `images/incremental/` 并原子更新图片 manifest。

如果只想更新图片，也可以携带同一篇论文图形成新的批次。预览不校验图片匹配；正式导入在任何目标数据写入前校验并暂存图片。

## 4. 失败后继续

开始时会输出批次号，例如：

```json
{"batch_id":"0123456789abcdef0123","status":"started"}
```

失败输出包含恢复参数。执行：

```bash
python3 import_new_batch.py --apply --resume 0123456789abcdef0123
```

批次目录已保存解析后的输入、固定的作者映射、更新计划及图片，因此恢复不依赖原始压缩包。自定义 `--db`、`--env-file`、`--state-dir` 时，恢复也要指定相同目标配置。

不要在未完成批次之后另开新的更新批次。脚本用数据库级文件锁阻止两个导入进程同时写入，并检查默认状态目录中的未完成批次。不要通过更换 `--state-dir` 绕开检查，也不要在批次恢复期间用旧导入/修补脚本修改同一批数据。

## 5. 日志、备份和一致性

默认目录：`data/incremental_imports/<batch_id>/`。

- `manifest.json`：输入文件校验和、目标库、关系更新模式、创建时间。
- `journal.sqlite3`：每篇论文更新计划、pending/graph_pending/graph_done/completed/failed 状态及事件日志。
- `before_<序号>.json`：该论文修改前的 SQLite/FTS/筛选表记录，以及此次涉及的 Neo4j 节点、出边快照。
- `assets/`：可选图片，恢复时无需原始图片压缩包。

查看状态：

```bash
python3 - /path/to/batch/journal.sqlite3 <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
for row in conn.execute('SELECT status,count(*) FROM jobs GROUP BY status'):
    print(row)
for row in conn.execute("SELECT pid,error FROM jobs WHERE status='failed'"):
    print(row)
PY
```

每篇论文先在一个 Neo4j HTTP 事务内写入节点和关系，再在一个 SQLite 事务内同步元数据、FTS 和关联表。出现超时或提交后进程退出时，重试会幂等地重放该论文，不会因为 SQLite 已有论文而跳过 Neo4j。FTS 按 rowid 替换，清理该论文遗留重复行；每次启动只扫描一次 FTS 建立临时映射，避免每更新一篇论文就扫描整库。

两个数据库之间没有分布式事务，因此导入过程中及失败到恢复之间可能短暂不一致。一批导入也不是全有或全无：前面完成的论文已提交。应以 `completed` 为完成标志；出错及时恢复。快照用于审计和人工恢复，当前不提供一键整批回滚或自动删除论文功能。

## 6. 生效与边界

本次修复后，检索服务使用支持 WAL 的 SQLite 只读连接，已有连接能看到提交后的更新；图片 manifest 按变化重新读取。以后增量更新通常无需重启 API。Neo4j 学者/机构/基金列表有约 30 秒缓存，健康统计有约 15 秒缓存。

当前只同步 **Neo4j + SQLite/BM25 + 可选图片**。不自动更新 MySQL `/api/papers` 数据集、YAML 会议日历或 Zilliz 向量索引。当前线上 embedding 关闭；以后开启向量检索前，要先同步相应向量数据。BM25 在线更新不会删除向量检索代码。

## 7. 验证

```bash
python3 -m unittest -v test_incremental_import test_paper_search test_index_roundtrip
```

真实 Neo4j 集成测试：

```bash
LIVE_INCREMENTAL_TEST=1 python3 -m unittest -v test_incremental_import.LiveGraphTest
```

集成测试用随机前缀创建 Paper/Topic 测试节点，结束后清理，只触碰该测试前缀。覆盖参考文献升级、完整主论文保护、关系替换、保留入边、重复执行和事务回滚。
