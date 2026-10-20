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
- **学者库首批数据**：已导入 OpenAlex 学者 `openalex:A5000000004`（Şaziye Melike IŞIK），包含机构、总引用 46、2022—2026 年度引用和 OpenAlex 来源信息；当前尚未与 Neo4j 作者节点建立论文关联，因此 `paper_count=0`，论文列表为空。
- **接口文档**：补充 Scholar 接口的数据口径：画像可先于 Neo4j 作者关联返回；`paper_count` 仅统计平台论文库中已关联且去重的论文；OpenAlex 等外部作品未匹配平台论文 ID 时不计入论文列表。

## 2026-10-20

- 导入两份 OpenAlex 学者 ZIP，各 1,000 条；合计 2,000 位独立学者，新增 1,999 位、更新原有 1 位。
- 规范化 DOI 后，与平台主论文建立 15 条关系，覆盖 4 位学者、15 篇论文；关系保存在现有 SQLite 的 scholar_paper_links 中，不复制论文、不按姓名合并 Neo4j 作者。
- 学者详情、论文列表和画像搜索读取新增论文关系；索引重建保留关联表。
- 修复重复机构记录导致导入失败的问题；重复导入不增加学者数。
- 来源保留 OpenAlex，源文件更新时间不冒充引用统计截至日期，citation_meta.as_of 未知时返回 null。
- 校验：18 项测试执行，16 项通过、2 项 Neo4j 集成测试跳过；通过 Nginx 统一入口验证学者详情、论文列表、搜索及健康接口。
- 原始 ZIP、转换数据和导入报告保留在 /home/user2/jiansuo/，报告目录 scholar-import-20261020。
