# 知识底座现状与上线质量汇报

> 更新时间：2026 年 10 月 13 日
> 当前线上入口：`http://158.247.231.197:18088`

## 一、结论摘要

当前知识底座已经具备论文检索、论文详情、引用图谱、作者/机构/基金查询和多步关系检索能力，可以供小前端直接接入。

当前线上检索链路为：

```text
BM25（SQLite FTS5） + Neo4j 图过滤/关系扩展
```

embedding 向量检索代码仍然保留，但当前运行配置关闭了 dense 召回，健康检查中显示 `dense: null`。这样做是为了避免 embedding 上游服务波动影响核心检索链路。

截至本次检查，以下服务均正常运行并设置了开机自启：

- Neo4j 图数据库
- Python 检索服务
- Node 业务 API
- Nginx 反向代理

真实线上 smoke 已通过，普通检索、Neo4j 条件过滤和作者关系扩展均可正常执行。

## 二、当前系统架构

```text
小前端 / 浏览器
        |
        v
Nginx :18088
        |
        +-- /api/retrieval/*  --> Python 检索服务 :8080
        |                         |
        |                         +-- SQLite FTS5 / BM25
        |                         +-- Neo4j HTTP :17474
        |                         +-- 可选 embedding / Zilliz
        |
        +-- /api/*             --> Node API :5000
                                  |
                                  +-- MySQL 业务数据
                                  +-- Neo4j Bolt :17687
```

### 组件职责

| 组件 | 作用 | 当前状态 |
| --- | --- | --- |
| SQLite FTS5 | 论文标题、摘要和元数据的 BM25 召回 | 已启用 |
| Neo4j | 论文、作者、机构、基金、主题、引用关系和图过滤 | 已启用 |
| Python 检索服务 | 统一检索、查询改写、多步关系检索 | 已启用 |
| Node API | 论文列表、会议、登录、收藏、KG 详情接口 | 已启用 |
| Nginx | 对外统一入口、路由、CORS、超时和日志 | 已启用 |
| embedding / Zilliz | 向量召回 | 代码保留，当前关闭 |

### 当前数据规模

健康接口返回的当前规模为：

- SQLite 论文索引：约 540,176 篇
- Neo4j 论文节点：约 711,360 个
- Neo4j 基金节点：约 12,978 个

## 三、服务和部署状态

当前 systemd 用户服务：

```text
paperddl-neo4j.service
paperddl-retrieval.service
paperddl-node.service
paperddl-nginx.service
```

四个服务均设置为 `enabled`，并配置了失败自动重启。user2 已启用 linger，因此服务器重启后不需要先登录用户会话，服务也可以自动启动。

常用运维命令：

```bash
systemctl --user status \
  paperddl-neo4j.service \
  paperddl-retrieval.service \
  paperddl-node.service \
  paperddl-nginx.service

systemctl --user restart paperddl-neo4j.service
systemctl --user restart paperddl-retrieval.service
systemctl --user restart paperddl-node.service
systemctl --user reload paperddl-nginx.service
```

健康检查：

```bash
curl http://158.247.231.197:18088/api/health
curl http://158.247.231.197:18088/api/retrieval/health
curl http://158.247.231.197:18088/api/retrieval/ready
```

内部端口只供本机服务使用：

```text
5000   Node API
8080   Python 检索服务
17474  Neo4j HTTP
17687  Neo4j Bolt
```

对外应该只使用 Nginx 的 `18088` 入口。

## 四、对外接口

### 4.1 论文检索

推荐接口：

```http
POST /api/retrieval/search
```

请求示例：

```json
{
  "query": "2022年以后关于图神经网络的论文",
  "top_k": 10,
  "year_gte": 2022,
  "conference": ["AAAI"]
}
```

支持的主要条件：

- `year_gte`
- `year_lte`
- `conference`
- `author`
- `keyword`
- `subject`
- `institution`
- `funding`

简单查询也可以使用：

```http
GET /api/retrieval/search?q=graph%20neural%20network&top_k=10
```

### 4.2 多步关系检索

```http
POST /api/retrieval/multistep-search
```

适合“同一作者的其他论文”“相关主题论文”“同会议论文”等关系型查询。

```json
{
  "query": "查找同一作者在相同主题下的其他论文",
  "top_k": 10,
  "max_steps": 6
}
```

### 4.3 查询改写

```http
GET /api/retrieval/query-rewrite?q=知识图谱补全
```

中文查询会优先使用领域术语表改写为英文检索词。例如：

```text
知识图谱补全 -> knowledge graph completion
检索增强生成 -> retrieval augmented generation
```

### 4.4 图谱和论文详情

```http
GET /api/kg/paper?paperId={paper_id}
GET /api/kg/graph?paperId={paper_id}&depth=1
GET /api/kg/search?q=graph&limit=20
```

`paperId` 应从检索结果中的 `paper_id` 获取，并进行 URL 编码。

`depth` 建议前端默认使用 `1`。深度越大，Neo4j 查询和返回数据量越大。

### 4.5 学者、机构和基金

```http
GET /api/retrieval/scholars/search?q=graph&limit=20
GET /api/retrieval/institutions/search?q=university&limit=20
GET /api/retrieval/fundings/search?q=AI&limit=20
```

### 4.6 论文列表和业务接口

```http
GET /api/papers?q=graph&conference=AAAI&year=2024&limit=20&offset=0
GET /api/papers/venues
GET /api/papers/tracks?conference=AAAI
GET /api/categories
GET /api/conferences
```

业务侧还提供：

```text
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/auth/me
GET    /api/favorites
POST   /api/favorites
DELETE /api/favorites/{conferenceId}
```

## 五、小前端接入方式

如果小前端和知识底座部署在同一个 Nginx 入口下，推荐使用相对路径：

```js
const result = await fetch("/api/retrieval/search", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: "图神经网络",
    top_k: 10,
    year_gte: 2020
  })
}).then((response) => response.json());
```

详情和图谱页面可以这样串联：

```js
const search = await fetch("/api/retrieval/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "graph neural network", top_k: 10 })
}).then((response) => response.json());

const paperId = search.results[0].paper_id;

const detail = await fetch(
  `/api/kg/paper?paperId=${encodeURIComponent(paperId)}`
).then((response) => response.json());

const graph = await fetch(
  `/api/kg/graph?paperId=${encodeURIComponent(paperId)}&depth=1`
).then((response) => response.json());
```

前端展示时建议：

1. 使用 `rank` 和返回顺序展示相关性，不要把 `score` 当作百分比。
2. 论文列表先请求 `top_k: 10` 或 `20`，不要一次请求几百条。
3. 图谱默认 `depth=1`，用户主动展开时再请求更深层级。
4. 对重复搜索和详情请求做前端缓存或去重。
5. 使用 `/api/retrieval/ready` 判断检索是否已经完成启动，而不是只判断进程端口是否存在。
6. 前端不直接访问 `8080`、`5000`、`17474`、`17687`。

## 六、当前检索逻辑

普通英文或中文术语查询主要走 BM25：

```text
查询解析
  -> 中文术语改写
  -> SQLite BM25 召回
  -> 可选 Neo4j 元数据过滤
  -> 结果读取和排序
```

带明确年份、会议、作者、主题、机构或基金条件的请求会触发 Neo4j 过滤。

包含“同作者”“相关主题”“同会议”等关系表达时，会触发 Neo4j 关系扩展。

普通主题查询不会无条件调用 Neo4j，这样可以避免所有请求都承担图数据库耗时。

embedding 仍然保留在代码和配置中，恢复时可以切换到 hybrid 模式。但恢复前需要重新验证：

- embedding 上游延迟和可用性
- 向量维度是否为 1024
- Zilliz collection 是否匹配
- dense 召回与 BM25 的融合权重
- 中文、英文和混合查询质量

## 七、上线质量审核流程

### 7.1 代码和接口检查

```bash
cd /home/user2/MKC_Paperddl
npm run server:smoke

cd /home/user2/jiansuo/retrieval_backend
python3 -m unittest discover -q
node --check /home/user2/MKC_Paperddl/server/src/index.js
python3 -m py_compile retrieval_pipeline.py http_server.py
```

### 7.2 真实 HTTP smoke

```bash
cd /home/user2/jiansuo/retrieval_backend

python3 http_smoke_test.py \
  --base-uri http://127.0.0.1:18088/api/retrieval \
  --expect-component neo4j_filter \
  --output /tmp/paperddl-http-smoke.json
```

该检查应覆盖：

- `/health`
- `/ready`
- 普通检索
- 带年份或会议条件的 Neo4j 过滤
- 同作者关系扩展
- 结果不为空
- `executed_operations` 与预期一致

### 7.3 数据质量检查

每次导入或更新数据后，至少审核：

- `paper_id` 唯一性
- 标题、年份、会议空值率
- SQLite 与 Neo4j 论文 ID 对齐率
- 作者、机构、基金孤儿节点
- 引用边两端是否存在
- 索引是否在线
- 数据量是否和导入报告一致
- 中英文、年份、会议、作者、主题抽样结果

### 7.4 检索质量检查

准备一套固定 query 和人工相关性标注，记录：

- Recall@10
- Recall@20
- MRR
- nDCG@10
- 空结果率
- 错误结果率
- 中文和英文查询的差异
- 带过滤条件和关系查询的差异

接口返回 200 只能说明服务成功响应，不能证明检索结果质量正确。

### 7.5 性能和稳定性检查

建议采用以下上线门槛：

| 接口类型 | 建议 p95 门槛 |
| --- | ---: |
| 健康检查 | < 100ms |
| 普通论文检索 | < 500ms |
| 带元数据过滤检索 | < 800ms |
| 论文详情 | < 300ms |
| depth=1 图谱 | < 1000ms |
| 5xx/502 错误率 | 0 |

需要同时观察 Nginx 日志和应用日志：

```text
/home/user2/web/nginx/logs/access.log
/home/user2/web/nginx/logs/error.log
/home/user2/MKC_Paperddl/server/runtime.log
/home/user2/jiansuo/retrieval_backend/retrieval.log
```

Nginx 日志会记录：

- 总请求耗时
- upstream 连接耗时
- upstream 首字节耗时
- upstream 响应耗时
- HTTP 状态码

## 八、推上线步骤

### 8.1 前端构建

```bash
cd /home/user2/MKC_Paperddl
npm run web:build
rsync -a --delete web/dist/ /home/user2/web/web-deployed-dist/
```

### 8.2 上线前审核

先执行代码测试、数据检查和 HTTP smoke，确认结果通过后再重启服务。

### 8.3 重启服务

```bash
systemctl --user restart paperddl-neo4j.service
systemctl --user restart paperddl-retrieval.service
systemctl --user restart paperddl-node.service
systemctl --user reload paperddl-nginx.service
```

### 8.4 上线后检查

```bash
systemctl --user is-active \
  paperddl-neo4j.service \
  paperddl-retrieval.service \
  paperddl-node.service \
  paperddl-nginx.service

curl http://158.247.231.197:18088/api/health
curl http://158.247.231.197:18088/api/retrieval/health
curl http://158.247.231.197:18088/api/retrieval/ready
```

### 8.5 回滚

前端回滚到上一份 `web-deployed-dist` 备份，后端回滚代码后执行：

```bash
systemctl --user restart paperddl-retrieval.service
systemctl --user restart paperddl-node.service
systemctl --user reload paperddl-nginx.service
```

## 九、正式上线前的剩余事项

当前系统已经可以供小前端接入，但正式对外使用前建议完成以下工作：

1. 为检索 API 启用 token 或网关认证。
2. 把 Nginx 的 `Access-Control-Allow-Origin: *` 限制为实际前端域名。
3. 配置正式域名和 HTTPS，当前入口仍是 HTTP 的 `18088` 端口。
4. 确认公网只开放 Nginx 入口，内部端口不暴露。
5. 把 API 文档中的旧地址统一更新为正式域名。
6. 固化 query 质量集和性能基线，每次数据或代码更新都自动执行。
7. 继续保持 embedding 灰度开关，等上游稳定后再做小流量对比，不直接切换全量流量。

## 十、当前验收结论

本次线上检查已确认：

- Node API 正常
- Python 检索服务正常
- Neo4j HTTP 和 Bolt 正常
- Nginx 反向代理正常
- BM25 检索正常
- Neo4j 元数据过滤正常
- Neo4j 作者关系扩展正常
- Node server smoke 通过
- Python HTTP smoke 通过
- 检索服务当前未启用 embedding dense 召回

当前适合进入“小前端联调和灰度上线”阶段。正式全量上线前，应优先完成认证、HTTPS、CORS 收紧和固定质量基线。

## 十一、Neo4j 当前节点和边统计

> 统计时间：2026 年 10 月 14 日
> 统计数据库：当前线上 Neo4j `neo4j` 数据库
> 统计方式：Cypher `MATCH (n)` 和 `MATCH ()-[r]->()`

### 11.1 总体规模

| 指标 | 数量 |
| --- | ---: |
| 节点总数 | 1,093,826 |
| 边总数 | 2,869,943 |
| 平均每个节点关联边数 | 约 2.62 |
| Paper 节点 | 711,360 |
| Author 节点 | 185,340 |
| Topic 节点 | 141,224 |

### 11.2 各类节点数量

| 节点类型 | 数量 | 主要用途 |
| --- | ---: | --- |
| `Paper` | 711,360 | 论文 |
| `Author` | 185,340 | 作者 |
| `Topic` | 141,224 | 主题 |
| `Method` | 35,737 | 方法 |
| `Funding` | 12,978 | 基金 |
| `Institution` | 6,917 | 机构 |
| `Conference` | 173 | 会议 |
| `FundingRaw` | 56 | 原始基金名称 |
| `Venue` | 41 | 期刊或出版 venue |

各类节点数量合计为 1,093,826。当前统计中节点按上述标签分类，没有发现需要单独拆分统计的多标签重复口径。

### 11.3 各类边数量

| 关系类型 | 数量 | 含义 |
| --- | ---: | --- |
| `HAS_TOPIC` | 1,240,286 | 论文属于某个主题 |
| `CITES` | 843,992 | 论文引用另一篇论文 |
| `AUTHORED_BY` | 583,308 | 论文由作者撰写 |
| `PUBLISHED_IN` | 121,534 | 论文发表在会议、期刊或 venue |
| `USES_AS_BASELINE` | 36,867 | 论文使用某方法作为 baseline |
| `FUNDED_BY` | 17,179 | 论文得到某基金支持 |
| `PROPOSES` | 13,572 | 论文提出某个方法 |
| `AFFILIATED_WITH` | 12,854 | 作者属于某个机构 |
| `PART_OF` | 173 | 节点属于某个更高层级节点 |
| `HAS_AFFILIATION` | 90 | 关联机构信息 |
| `NORMALIZED_FROM` | 88 | 实体规范化来源关系 |

各类关系数量合计为 2,869,943。

### 11.4 论文节点字段覆盖

| 指标 | 数量 |
| --- | ---: |
| 有 `paper_id` 的论文 | 711,360 |
| 有标题的论文 | 710,495 |
| 有年份的论文 | 541,068 |
| `external=true` 的论文 | 588,726 |

`paper_id` 基本完整；标题缺失约 865 篇；年份字段覆盖约 54.1 万篇。`external=true` 的节点主要用于补充引用关系和外部论文，不一定都有完整摘要和全文检索字段。

### 11.5 论文关系覆盖

| 关系 | 关联论文数 | 关系边数 | 平均值或说明 |
| --- | ---: | ---: | --- |
| `HAS_TOPIC` | 121,477 | 1,240,286 | 平均每篇 Paper 约 1.74 个主题边 |
| `AUTHORED_BY` | 121,425 | 583,308 | 仅按全部 Paper 计算约 0.82 条作者边 |
| `PUBLISHED_IN` | 121,476 | 121,534 | 有 venue 关系的论文 |
| `FUNDED_BY` | 8,186 | 17,179 | 有基金关系的论文 |
| `PROPOSES` | 13,572 | 13,572 | 有方法提出关系的论文 |
| 对外 `CITES` | 31,047 | 843,992 | 有引用出边的论文 |
| 被引用 | 602,911 | — | 至少被一篇论文引用 |

当前只有约 12.1 万篇论文有 Topic 关系，约 12.1 万篇有作者关系，说明图谱关系覆盖不是所有 Paper 节点都一致完整。

`CITES` 的方向为：

```text
(source: Paper)-[:CITES]->(target: Paper)
```

表示 `source` 论文引用了 `target` 论文。图中有大量被引用的外部论文节点，因此“有引用出边的论文数”和“被引用论文数”差异很大。

### 11.6 Neo4j 与 SQLite 检索库的口径差异

当前 SQLite BM25 索引约有 540,176 篇论文，Neo4j 有 711,360 个 `Paper` 节点。

| 存储 | 规模 | 主要用途 |
| --- | ---: | --- |
| SQLite FTS5 | 540,176 篇 | 标题、摘要和元数据全文检索 |
| Neo4j | 711,360 个 Paper 节点 | 论文关系、引用、作者、主题、机构和基金 |

两者数量不一致是预期现象：Neo4j 包含了更多用于引用关系补全的外部论文节点，SQLite 主要保存当前可以直接进行 BM25 全文检索的论文集合。

因此可能出现一篇论文能在 Neo4j 引用图里看到，但在 SQLite 全文检索中没有完整摘要或无法直接检索到的情况。

### 11.7 对当前图谱质量的判断

当前图谱已经具备以下关系框架：

```text
论文
 ├── 作者
 ├── 主题
 ├── 机构
 ├── 基金
 ├── 方法
 ├── 会议/venue
 └── 引用论文
```

当前适合支持：

- 论文详情
- 引用关系展示
- 作者关系扩展
- 主题关系扩展
- 条件过滤
- 多步关系检索

继续完善知识底座时，应重点审核：

1. SQLite、Neo4j、MySQL 的 `paper_id` 对齐率。
2. 没有作者、主题、venue 关系的 Paper 节点比例。
3. 外部 Paper 节点的标题、年份和来源字段完整性。
4. 引用边两端是否都能对应有效论文。
5. Topic、Author、Institution 的规范化和重复实体情况。
