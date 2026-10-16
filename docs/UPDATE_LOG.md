# Shenzhi 更新维护记录

> 每次代码、接口、数据或部署变更只在本文件新增一条记录。详细接口契约以 `API_USAGE_GUIDE.md` 和产品要求文档为准。

## 2026-10-16

- **检索数据职责**：论文、论文元数据、FTS5/BM25 索引、主题和基金映射使用 SQLite；Neo4j 保留图谱关系、过滤和关系扩展；MySQL 用于用户、登录和收藏业务。
- **基金接口**：`/api/retrieval/fundings/search` 与 `/api/retrieval/search/by-funding` 统一 SQLite 匹配、主论文范围和去重规则。
- **产品接口**：新增 Topic 与 Scholar 产品接口，字段按产品要求返回；论文列表复用现有论文库。
- **学者搜索**：Neo4j 全文索引不可用时自动降级为姓名包含匹配。
- **增量更新**：保留幂等重放、断点恢复、SQLite/FTS 更新、Neo4j 关系替换和图片 manifest 更新能力。
- **运行状态**：Node、Retrieval、Neo4j、Nginx 用户服务均启用并完成健康检查。
- **数据状态**：Topic/Scholar 元数据表已建立，待产品侧提供真实目录、画像、机构、头像和引用数据后导入。
- **验证**：18 项自动测试通过；2 项真实 Neo4j 集成测试因未设置 `LIVE_INCREMENTAL_TEST=1` 跳过。
- **代码版本**：GitHub `main` 提交 `e156b47`。

## 迁移包内容

完整迁移包必须包含：`repository/`、`database-backup/`、`primary_papers.jsonl`、`shenzhi-images-20261015.tar.zst`、迁移说明和校验文件。图片归档是完整项目迁移的一部分，不可省略。
