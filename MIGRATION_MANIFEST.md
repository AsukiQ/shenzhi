# Shenzhi migration manifest

代码仓库：`https://github.com/AsukiQ/shenzhi.git`

## 服务

| 服务 | 当前地址 | 迁移内容 |
|---|---|---|
| Production retrieval | `127.0.0.1:8080` | `retrieval_backend/`, production SQLite, env template |
| AAAI demo retrieval | `127.0.0.1:18080` | `demo_aaai2026/`, demo SQLite, systemd unit |
| KG API | Docker `server:5000` | KG/Web source and Docker configuration |
| Neo4j | Docker `7473/7687` | Neo4j snapshot or import CSV, credentials separately |
| Nginx | port 80 | deployment Nginx configuration |

## Data packages

Do not commit these files to GitHub. Upload them separately through SwissTransfer or an equivalent authenticated channel.

| Package | Source | Approx. size |
|---|---|---:|
| production SQLite | `retrieval_backend/papers_fts.db` | 1.9 GB |
| Neo4j preservation snapshot | `neo4j_store_preserved_20260924.tar.gz` | 301 MB |
| graph CSV | `graph_data/` | 452 MB |
| images/new data | `new/` | 599 MB |
| AAAI demo | `demo_aaai2026/` | 69 MB |

## Delivery rules

1. Stop writes before taking a Neo4j or SQLite snapshot.
2. Generate `SHA256SUMS` after packaging.
3. Transfer each package independently; do not create a second full-size copy on this host.
4. The receiver must verify SHA256 before extraction.
5. Credentials are exchanged separately and never committed to GitHub.
