# API 接口检查与性能更新报告（2026-10-16）

## 结论

- 检查入口：`http://158.247.231.197:18088`（Nginx 外部入口）。
- `apiguide.md` 列出的接口及其参数错误、未登录保护契约均通过：**38/38 通过，0 失败**。
- 主服务、检索服务、Neo4j 和 Nginx 当前均为 active；`/api/retrieval/ready` 返回 `ready=true`。
- 线上检索组件是 **SQLite FTS5 BM25 + Neo4j filter**；embedding/Zilliz 当前关闭，健康检查 `dense=null`。本次没有删除向量检索代码，只保持默认关闭。

## 本次修复与更新

### 1. 修复 BM25 冷查询慢

原查询使用 `bm25(...) AS bm` 再 `ORDER BY bm`，SQLite FTS5 会对匹配结果建立临时排序；服务重启后首次查询可能达到数秒甚至十几秒。

现在改用 FTS5 的 `rank MATCH 'bm25(5.0,1.0,3.0,2.0)'` 和 `ORDER BY rank`，保留原来的标题、摘要、关键词、主题权重，同时让 FTS5 使用内部 top-k 排序路径。

同时为 `paper_funding(value_key, paper_id)` 补充索引，并用 `CROSS JOIN` 固定关系表优先过滤，基金检索从约 0.8–2.8 秒降到本次复测的约 3–7 ms（热缓存）。

### 2. 启动预热

检索服务启动时执行一次 BM25 warmup，避免第一位用户承担磁盘冷缓存成本。Node 服务启动前预热会议、分类和论文 venue 聚合；会议 YAML 和 SQLite 聚合结果分别缓存 60 秒/5 分钟，并在新增会议时失效。

### 3. 文档同步

已同步修正：

- 外部服务地址为 `158.247.231.197:18088`。
- 默认检索模式为 BM25 + Neo4j，`dense` 为空。
- `/api/kg/graph` 默认排除 `reference_stub`，可用 `includeReferenceStubs=true` 显式包含。
- `/api/papers`、venues、tracks 支持 `record_type=canonical|external_full|all`。
- `/api/categories` 是会议 YAML 的分类筛选项，不等同于论文 Topic。

## 接口检查结果

### 功能接口

| 检查项 | 方法与路径 | HTTP | 响应时间 | JSON 结构 |
|---|---|---:|---:|---|
| retrieval search POST | `POST /api/retrieval/search` | 200 | 201.8 ms | object: results, state, query_parse, query_rewrite |
| retrieval search GET | `GET /api/retrieval/search?q=graph%20neural%20network&top_k=3` | 200 | 210.5 ms | object: results, total, total_exact, state, query_parse, query_rewrite |
| retrieval multistep search | `POST /api/retrieval/multistep-search` | 200 | 215.8 ms | object: results, state, query_parse, query_rewrite |
| knowledge graph title search | `GET /api/kg/search?q=graph&limit=3` | 200 | 72.4 ms | array(3) |
| scholar search | `GET /api/retrieval/scholars/search?q=Marcel%20Binz&limit=3&offset=0` | 200 | 19.5 ms | object: results, query |
| subject paper search | `GET /api/retrieval/search/by-subject?subject=reinforcement&offset=0&limit=2` | 200 | 16.9 ms | object: results, total, state |
| funding paper search | `GET /api/retrieval/search/by-funding?funding=NSF&top_k=2` | 200 | 6.6 ms | object: results, total, state |
| funding candidate search | `GET /api/retrieval/fundings/search?q=NSF&limit=2&offset=0` | 200 | 23.2 ms | object: results, query |
| institution paper search | `GET /api/retrieval/search/by-institution?institution=Stanford&top_k=2` | 200 | 68.5 ms | object: results, total, state |
| institution candidate search | `GET /api/retrieval/institutions/search?q=Stanford&limit=2&offset=0` | 200 | 18.1 ms | object: results, query |
| paper summary | `GET /api/knowledge/papers/summary` | 200 | 65.9 ms | object: paper_count |
| research asset summary | `GET /api/knowledge/research-assets/summary` | 200 | 3.2 ms | object: research_asset_count |
| paper detail | `GET /api/kg/paper?paperId=paper%3A17203_aaai%3A911ff38f19e8` | 200 | 20.7 ms | object: paper_id, title, authors, year, venue, abstract, doi, pdf_url, record_type |
| paper graph default | `GET /api/kg/graph?paperId=paper%3A17203_aaai%3A911ff38f19e8&depth=1` | 200 | 134.2 ms | object: rootId, nodes, lines |
| paper graph including reference stubs | `GET /api/kg/graph?paperId=paper%3A17203_aaai%3A911ff38f19e8&depth=1&includeReferenceStubs=true` | 200 | 95.2 ms | object: rootId, nodes, lines |
| scholar detail | `GET /api/retrieval/scholars/author%3Alegacy%3Af1ce255aceabf63b1c578466` | 200 | 18.2 ms | object: scholar_id, name, institutions, paper_count, years, conferences, topics, funding, coauthors, papers |
| paper list | `GET /api/papers?limit=2&offset=0` | 200 | 103.4 ms | object: items, total, limit, offset |
| paper venues | `GET /api/papers/venues` | 200 | 3.6 ms | array(121) |
| paper tracks | `GET /api/papers/tracks?conference=AAAI` | 200 | 7.8 ms | array(4181) |
| categories | `GET /api/categories` | 200 | 4.3 ms | array(10) |
| conferences | `GET /api/conferences` | 200 | 17.4 ms | array(354) |
| main health | `GET /api/health` | 200 | 50.9 ms | object: ok, time, papers |
| retrieval health | `GET /api/retrieval/health` | 200 | 54.4 ms | object: status, components, lexical, graph, dense |
| retrieval readiness | `GET /api/retrieval/ready` | 200 | 77.8 ms | object: status, components, lexical, graph, dense, ready |
| logout idempotent | `POST /api/auth/logout` | 200 | 3.7 ms | object: ok |

### 参数校验与认证保护

| 检查项 | 方法与路径 | HTTP | 响应时间 | JSON 结构 |
|---|---|---:|---:|---|
| auth me unauthenticated contract | `GET /api/auth/me` | 401 | 3.7 ms | object: message |
| favorites list unauthenticated contract | `GET /api/favorites` | 401 | 3.2 ms | object: message |
| search empty query structured contract | `POST /api/retrieval/search` | 200 | 3.3 ms | object: results, state, query_parse, query_rewrite |
| multistep missing query | `POST /api/retrieval/multistep-search` | 400 | 2.5 ms | object: error |
| subject missing parameter | `GET /api/retrieval/search/by-subject` | 400 | 1.9 ms | object: error |
| funding missing parameter | `GET /api/retrieval/search/by-funding` | 400 | 2.0 ms | object: error |
| institution missing parameter | `GET /api/retrieval/search/by-institution` | 400 | 1.6 ms | object: error |
| paper detail missing parameter | `GET /api/kg/paper` | 400 | 3.0 ms | object: message |
| graph missing parameter | `GET /api/kg/graph` | 400 | 2.3 ms | object: message |
| register missing fields | `POST /api/auth/register` | 400 | 18.2 ms | object: message |
| login missing fields | `POST /api/auth/login` | 400 | 4.3 ms | object: message |
| favorite write unauthenticated contract | `POST /api/favorites` | 401 | 3.3 ms | object: message |
| favorite delete unauthenticated contract | `DELETE /api/favorites/audit-nonexistent` | 401 | 2.6 ms | object: message |

注册和登录的成功路径没有在生产环境创建临时账号；已验证缺少字段返回 400、未登录收藏/用户接口返回 401、退出登录幂等返回 200。这样不会修改真实用户数据。

## 当前数据与服务状态

| 项目 | 当前值 |
|---|---:|
| SQLite 总记录 | 540,176 |
| SQLite canonical 主论文 | 121,769 |
| SQLite external_full | 190,899 |
| SQLite reference_stub | 227,508 |
| Neo4j Paper 节点 | 711,360 |
| Neo4j funding 节点 | 12,978 |
| 论文总览接口 | 121,769 |
| 研究资产总览接口 | 12,978 |

监听端口：Nginx `18088`、Node `5000`、检索 `8080`、Neo4j HTTP `17474`、Bolt `17687`。服务由 user-level systemd 管理并设置为 enabled。

## 性能复测

本次通过外部入口串行复测，服务已预热：

- `/api/retrieval/search` POST：约 202 ms；GET：约 211 ms。
- `/api/retrieval/multistep-search`：约 216 ms。
- `/api/retrieval/search/by-funding`：约 7 ms。
- `/api/kg/search`：约 72 ms（标题图谱查询）。
- `/api/kg/paper`：约 19 ms。
- `/api/kg/graph`：约 95–135 ms（缓存命中更低）。
- `/api/papers/venues`、`/api/papers/tracks`、`/api/conferences`：缓存命中约 4–17 ms。
- 四个并发检索请求均返回 200，耗时约 0.49–0.65 秒。

单次冷启动 warmup 属于服务启动阶段，不再由首个用户请求承担；服务就绪后再接受外部请求。

## 复现方式

```bash
python3 /tmp/final_api_audit.py
cat /home/user2/jiansuo/api_interface_audit_20261016.json
```

服务检查：

```bash
systemctl --user status paperddl-node.service
systemctl --user status paperddl-retrieval.service
systemctl --user status paperddl-neo4j.service
systemctl --user status paperddl-nginx.service
```

## 迁移说明

本报告、修复后的 Node/检索代码、API 使用手册和最新 SQLite 快照已写入 `/home/user2/shenzhi-dev-migration`。迁移目录的清单和 SHA256 会在数据库快照复制完成后重新生成；原有完整包保留，另生成本次更新包，避免删除其中唯一的论文图片归档。
