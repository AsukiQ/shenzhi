# Python 主检索接入 final Neo4j：Canary 报告

日期：2026-08-24

## 结论

Python `/api/retrieval/search` 与 final Neo4j 的 ID、schema 和运行性能兼容。适配代码已
同步到 `/root/jiansuo/retrieval_backend`，但线上 `8080` 进程未重启，所以当前生产接口
仍为 BM25-only，健康检查中的 `graph` 仍是 `null`。临时 canary 已关闭。

## 兼容性

- final 图：661,378 个节点，其中 Paper 347,148、正式 Paper 113,653；
- SQLite：110,598 篇，全部能按 `Paper.paper_id` 在 final 图精确命中；
- Conference：`Paper-[:PUBLISHED_IN]->Conference-[:PART_OF]->Venue`；
- Keyword/Subject 检索槽统一映射为 `Paper-[:HAS_TOPIC]->Topic`；
- Author 保持 `Paper-[:AUTHORED_BY]->Author`；年份直接使用 `Paper.year`。

## Canary 结果

canary 监听 `127.0.0.1:18081`，最多 2 个工作线程，只连接现有 Neo4j，不修改图数据。

| 用例 | 结果 | P50 | P95 |
|---|---:|---:|---:|
| 普通 BM25 | 8/8，结果与线上一致 | 66ms | 78ms |
| Conference 过滤 | 8/8 | 94ms | 133ms |
| Topic 过滤 | 8/8 | 100ms | 109ms |
| 作者关系扩展 | 8/8 | 72ms | 245ms |
| 中文年份+会议条件 | 8/8 | 97ms | 101ms |
| 2 并发混合请求 | 8/8 HTTP 200 | - | 150ms |

同作者、同主题、同会议扩展均实际改变 Top-10 并加入新候选，所有图操作失败数为 0。
canary 常驻 RSS 约 64MB。没有人工 qrels，因此这里只能证明运行正确和候选发生变化，
不能据此声称相关性一定提升。

## 已加入的保护

- Neo4j 查询独立使用 3 秒默认超时，可由 `NEO4J_TIMEOUT` 调整；
- 图运行中不可用时，搜索回退到 SQLite/文本链路；
- `/health` 和 `/ready` 将图故障标记为 `degraded`，BM25 仍保持 ready；
- 并发槽增加 100ms 有界等待，消除响应完成与工作线程释放之间的瞬时 503；
- 参数化 Cypher、候选上限和图扩展上限保持不变。

## 生产切换前置条件

canary 使用 Neo4j 容器 IP `172.21.0.2`，该地址在容器重建后可能变化，不能作为长期
生产配置。推荐在维护窗口给 Neo4j HTTP 端口增加仅绑定 loopback 的映射：

```text
127.0.0.1:17474 -> neo4j:7474
```

然后为 Python 服务设置 `NEO4J_HTTP_URI=http://127.0.0.1:17474`、认证信息和
`NEO4J_TIMEOUT=2`，受控重启 `8080`，重复本报告的 smoke 后再放量。不要把 Neo4j
端口绑定到 `0.0.0.0`。切换前版本保存在：

```text
/root/jiansuo/backups/graph_final_20260824_before_canary
```
