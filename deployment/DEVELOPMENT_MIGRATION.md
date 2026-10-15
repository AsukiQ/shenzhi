# 开发环境迁移

本仓库是代码；数据库和图片通过独立迁移包传输。建议 Linux、Docker Engine + Compose，至少 4 GB 内存，推荐 8 GB；磁盘预留约 20 GB 供包解压、数据库导入及镜像使用。

## 1. 获取代码和备份

```bash
git clone https://github.com/AsukiQ/shenzhi.git
cd shenzhi
mkdir -p migration-data runtime-data
```

把 `shenzhi-databases-20261015.tar.zst`、`shenzhi-images-20261015.tar.zst` 和 `SHA256SUMS` 下载到本地，先在包所在目录执行 `sha256sum -c SHA256SUMS`。校验文件如包含代码包，须把代码包一起下载。

```bash
tar --zstd -xf /path/to/shenzhi-databases-20261015.tar.zst -C migration-data
tar --zstd -xf /path/to/shenzhi-images-20261015.tar.zst -C runtime-data
cp migration-data/papers_fts.db runtime-data/
cp migration-data/paper_images.json runtime-data/
# Neo4j 容器用户需要能读取 dump，MySQL SQL 仍可保持 600。
chmod 755 migration-data
chmod 644 migration-data/neo4j.dump
```

MySQL SQL 包含真实用户资料及密码哈希，开发环境应限制访问并按需脱敏。不要将任何迁移数据目录放进 Git。

## 2. 配置开发环境凭据

```bash
cp deployment/.env.example deployment/.env
```

修改里面的 Neo4j、MySQL、JWT 密钥。Compose 默认只暴露 Web `18088`；Neo4j 管理端口绑定本机 `17474`、`17687`。SQLite 使用可写目录挂载，WAL 文件与数据库一起持久化，不能只挂载单个 db 文件。

后续命令都从仓库根目录执行。自定义包目录时可在 `deployment/.env` 配置绝对路径 `MIGRATION_BACKUP_DIR`、`MIGRATION_DATA_DIR`。

## 3. 导入 Neo4j

首次导入必须在目标 Neo4j 未运行时进行：

```bash
docker compose -f deployment/compose.yml run --rm --no-deps neo4j \
  neo4j-admin database load neo4j --from-path=/backups
docker compose -f deployment/compose.yml up -d neo4j paper-mysql
```

包使用 Neo4j Community **5.26.27** 导出，目标固定同版本，避免存储格式不兼容。这里仅导入业务数据库 `neo4j`，不导入生产 `system` 认证库；新库使用 `deployment/.env` 里的开发密码。

如果卷里已存在数据，load 会拒绝覆盖；请先确认目标卷是否确实为开发环境，勿直接覆写生产库。

## 4. 配置两个 MySQL

服务现在明确分为两个数据库：

- `USER_MYSQL_*`：已有用户数据库，保存 `users`、`favorites`；可以继续使用原服务器，不需要把它搬到开发机。
- `PAPER_MYSQL_*`：开发机上的 `paper-mysql`，只保存论文和全文索引。Compose 首次创建卷时会自动执行 `deployment/paper-mysql-init/001_papers.sql`。

把 `deployment/.env.example` 复制为 `deployment/.env`，填写现有用户库连接和新的论文库密码。不要把包含用户密码哈希的 `mysql.sql` 直接导入论文库；它是旧的混合快照，且论文数量未完成对齐。只有明确要迁移用户数据时，才将其中 `users`、`favorites` 导入用户库。

导入 Neo4j 主论文到新的论文 MySQL：

```bash
python3 sync_mysql_from_primary.py \
  --input primary_papers.jsonl \
  --env /path/to/paperddl/server/.env \
  --apply \
  --report mysql_paper_sync_report.json
```

`.env` 中必须设置 `PAPER_MYSQL_HOST`、`PAPER_MYSQL_DATABASE`、`PAPER_MYSQL_USER`、`PAPER_MYSQL_PASSWORD`。后续只同步新增论文时加 `--incremental`。

## 5. 启动接口与页面

```bash
docker compose -f deployment/compose.yml up -d --build
curl http://127.0.0.1:18088/api/health
curl http://127.0.0.1:18088/api/retrieval/health
curl -H 'Content-Type: application/json' \
  -d '{"query":"knowledge graph completion","top_k":5}' \
  http://127.0.0.1:18088/api/retrieval/search
```

若 Neo4j 还在启动，图接口可能暂时失败，等服务就绪再验收。检索 `/ready`、`/health`、检索结果及 Node `/api/health` 应分别验证。`/api/categories` 是会议领域分类；它不是论文 Topic。

如果需要与原服务器相同的前端静态页面，可另加 Compose override，把 `migration-data/frontend-deployed` 只读挂到 web 的 `/usr/share/nginx/html`。默认镜像构建 `paperddl/web` 中的源码。

Nginx 记录总耗时和上游耗时；Node、Python 日志可通过 `docker compose ... logs -f node retrieval web` 查看。配置使用 `restart: unless-stopped`；还需操作系统启用 Docker 服务，才能随系统启动恢复。

## 6. 开发与增量更新

不使用容器的源码开发：Python 3.10+ 标准库足以跑 BM25 + Neo4j HTTP，Node 建议 22 LTS。`cd paperddl && npm ci` 后，设置 Node 的 `server/.env`（参考下方），分别执行 `npm run server:dev` 和 `npm run web:dev`。Vite 已将 `/api/retrieval`、`/api/knowledge` 转给 Python，其余 `/api` 转给 Node。

Node `.env` 字段：`PORT`、`USER_MYSQL_HOST`、`USER_MYSQL_PORT`、`USER_MYSQL_DATABASE`、`USER_MYSQL_USER`、`USER_MYSQL_PASSWORD`、`PAPER_MYSQL_HOST`、`PAPER_MYSQL_PORT`、`PAPER_MYSQL_DATABASE`、`PAPER_MYSQL_USER`、`PAPER_MYSQL_PASSWORD`、`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`JWT_SECRET`。旧代码模式仍兼容 `MYSQL_*`，但新部署不要用一套变量指向两个数据库。本机图端口为 `bolt://127.0.0.1:17687`，Python 图地址为 `http://127.0.0.1:17474`。

Python 可复制 `retrieval_backend/deployment/paperddl.env.example` 后设置自己的路径和图连接信息，再 `set -a; source /path/to/local.env; set +a`，执行 `bash retrieval_backend/run_retrieval_service.sh`。默认已改为便携路径和 bm25，无需 GPU。

容器增量更新示例：把一个 canonical ZIP 放进 `runtime-data`，然后：

```bash
docker compose -f deployment/compose.yml exec retrieval \
  python3 retrieval_backend/import_new_batch.py /data/new_batch.zip --db /data/papers_fts.db --dry-run
docker compose -f deployment/compose.yml exec retrieval \
  python3 retrieval_backend/import_new_batch.py /data/new_batch.zip --db /data/papers_fts.db --apply
```

数据库默认路径与源码路径不同，必须传 `--db /data/papers_fts.db`。批次状态在持久化的 retrieval-imports 卷里。完整规则见 [增量更新说明](../retrieval_backend/INCREMENTAL_UPDATE_GUIDE.md)。

## 7. 可选能力与边界

- 当前迁移包支持 BM25 + Neo4j；远程 Zilliz 数据不在包里。若开发环境继续使用原 Zilliz，需单独配置访问权限；若换集群，需另导入向量。embedding 代码和开关仍保留。
- Stage0/Stage2/精排模型代码保留在 `clstr_source`，模型权重另行准备。默认 Docker 镜像未安装这些可选依赖。
- `/api/kg/enrich` 需要 `paperddl/tools/API_KG` 的 Python 依赖、抽取模型/API 配置；默认 Node 镜像未安装它们，因此该按需抽取接口不会仅凭基础栈就自动可用。
- YAML 会议数据随代码提供。线上 Node 论文列表使用独立的论文 MySQL；SQLite 检索库仍由 Python BM25 服务使用，二者可以按批次分别更新。新增论文不自动同步到论文 MySQL，需运行增量同步脚本。
- 数据包为各库独立一致性快照，不是跨库同一时间点的分布式快照；导入前后应核对 `mysql.sql.manifest.json` 和数据库包中的统计清单。

## 8. 交付验证范围

本机可执行 Python 测试、Node 语法检查、Vue 生产构建、Compose 配置校验、SQLite 备份校验和 Neo4j dump 信息校验。当前账号没有 Docker daemon 访问权限，未宣称已在容器中完成全栈启动或 MySQL 回灌验收；到开发机按第 3–5 步验收。
