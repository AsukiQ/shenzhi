# paperddl 部署包

目标目录：`/root/jiansuo`。

当前服务器只有 2 核 CPU、3.5 GiB 内存且没有 GPU，因此默认先运行 BM25：

```bash
bash /root/jiansuo/retrieval_backend/deployment/start_bm25.sh
```

健康检查：

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

`paperddl.env.hybrid` 已绑定训练后的 Stage0 checkpoint、论文 skills、CLSTR 源码和
SkillRouter Embedding。它用于升级后的 GPU/大内存服务器；当前主机不建议直接加载。

final Neo4j 已在当前服务器供 Node `/api/kg/*` 使用；Python 图增强代码和低并发
canary 已通过，但生产 `8080` 仍未启用图。启用前需要提供稳定的 loopback HTTP 映射，
再设置 `NEO4J_HTTP_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`，并按需调整
`GRAPH_WEIGHT`、`GRAPH_SEED_K`、`GRAPH_EXPAND_K` 和 `GRAPH_VALUE_LIMIT`。普通主题查询
不会调用图；只有结构化条件或关系型查询才会触发图过滤/扩展。Neo4j 故障时服务回退到
SQLite 关系表或文本结果。需要紧急关闭关系扩展时设置 `GRAPH_INTENT_ENABLED=0`。

本部署包不包含 SkillRouter Reranker 或当前弱监督 Stage2 的运行实例；Neo4j 由同机
Docker Compose 独立维护。`output_graphs.zip` 没有边，不能直接导入作为图源。Python
接入 final 图的验证记录见 `FINAL_GRAPH_CANARY_20260824.md`。

部署包现在自带一份可审核的领域术语表。中文查询会先做术语约束改写，再走英文
BM25；设置 `QUERY_TRANSLATOR_URL` 后，可让一个独立的小翻译模型改写未覆盖的句式。
翻译服务不可用时会回退到术语表；如果术语表也无法形成英文 query，服务会记录改写
失败并返回空结果，避免把无关论文伪装成中文搜索结果。

服务已经限制请求体、并发数和连接超时。需要让其他机器访问时仍应保持服务监听
`127.0.0.1`，通过 Nginx 提供 TLS 和限流。若需接口认证，在受保护的 systemd
EnvironmentFile 中设置 `AUTH_TOKEN_ENV=RETRIEVAL_AUTH_TOKEN` 和实际 token；不要把
token 写进本部署目录或仓库。
