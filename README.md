# Shenzhi 知识底座

论文检索、Neo4j 知识图谱、会议日历、用户与收藏接口。代码同步自 2026-10-15 的运行版本。

- `retrieval_backend/`：BM25/SQLite、查询解析与改写、图过滤/扩展、可选 embedding/Zilliz、HTTP API、可恢复增量更新。
- `paperddl/server/`：Node 用户、收藏、论文列表、会议与图接口。
- `paperddl/web/`：Vue 前端源码；当前线上静态页面另随迁移包备份，尚未确认与此源码是同一构建。
- `paperddl/conference/`：会议日历及其研究领域 YAML。
- `deployment/`：开发环境 Docker Compose、Nginx、备份工具及迁移文档。
- `paper_data_pipeline/`、`shenzhi_paper_data/`：保留的数据准备、构图和校验工具。
- `clstr_source/`：保留的可选 Stage0/精排模型及研究依赖，不是默认运行所需。

开发迁移入口：[deployment/DEVELOPMENT_MIGRATION.md](deployment/DEVELOPMENT_MIGRATION.md)。

默认运行 **BM25 + Neo4j**，embedding 关闭但代码保留。通用中文翻译需要另配翻译服务，目前本地词表只有少量术语。MySQL、SQLite、Neo4j 各自承担不同接口，不会自动相互同步。

日常更新：[增量更新说明](retrieval_backend/INCREMENTAL_UPDATE_GUIDE.md)。

数据库、图片、模型、token 和真实 `.env` 均不提交到 GitHub。部署前请创建开发环境自己的密码和 JWT secret。


## 双 MySQL 部署

正式服务使用两个逻辑数据库：`USER_MYSQL_*` 连接已有用户库，`PAPER_MYSQL_*` 连接本机论文库。论文查询和导入走 `paperDb.js`，登录、用户和收藏继续走 `db.js`。不要把旧的混合 `mysql.sql` 直接导入论文库。详细步骤见 `deployment/DEVELOPMENT_MIGRATION.md`。
