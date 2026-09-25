# 索引构建

统一入口为 `build_paper_index.sh`，从当前 SQLite 元数据快照建立新索引：

```bash
PAPER_INDEX_TARGET=/path/with/enough/space/papers.new.db bash build_paper_index.sh
```

目标必须不存在。脚本保留 paper_id、source_id、标题、摘要、会议、年份、authors、keywords、subjects、funding、image_id，重建关联表及 FTS，完成 SQLite/FTS 完整性校验后才生成目标文件。构建不会替换线上库或重启服务。空间须容纳新库和临时 JSONL；当前服务器不建议直接做全量重建。

图片从现有 manifest 补入 image_id。主题与基金名称共同进入现有 FTS subjects 列，返回的 subjects 仍仅含主题；作者通过关联表筛选。评分、排名、retrieval_mode 等为请求时计算，无需保存。

新 JSONL 数据必须提供同样的字段，交给 `PaperSearchIndex.build`。该构建器已纳入 funding 和 image_id。不要再使用历史逐字段修补脚本重建线上库；不要关闭 SQLite 日志或直接覆盖运行中的数据库。

此入口保证重建不丢已有字段，不自动补齐缺失论文，也不自动同步 Neo4j/Zilliz。切换仍需单独维护：停止服务、保留旧库、切换经验证的新库、启动并验证接口。

运行 `python audit_database_coverage.py` 可只读对比三库 ID。报告位于 `data/database_coverage_report.json`。Zilliz 按 paper_id 去重，不能把分块数当论文数；本报告衡量覆盖范围，不证明每篇内容或向量文本完全一致。
