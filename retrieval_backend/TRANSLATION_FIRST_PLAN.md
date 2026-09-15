# 中文静态检索收敛方案

## 已确定的唯一链路

中文请求不单独建立中文 BM25，也不做中文 BM25、英文 BM25、中文/英文 dense 三路融合。
统一流程为：

```text
原始中文 query
  -> 解析结构化条件和关系意图
  -> 领域术语表/翻译服务中译英
  -> 英文 BM25 +（已配置时）Stage0 dense
  -> RRF
  -> Neo4j 硬过滤和按需图扩展
  -> 英文 reranker
  -> 返回结果、原 query、translated_query 和状态
```

原始 query 只用于审计、条件解析和失败诊断；英文改写结果是静态检索和精排的统一输入。

解析器在翻译前执行高置信度结构化抽取：年份范围/上下界、会议白名单、显式作者标记、
带“关键词/主题”等字段标记的已验证术语，以及关系短语。显式 HTTP 参数优先于自然语言
抽取结果。抽取后的 `semantic_query` 用于翻译，`graph_query` 保留关系语句给图规划器；
两者和 `query_parse` 一起写入接口响应。仅有结构化条件时走 `BROWSE_PAPER_FILTERED`
候选浏览，不生成任意文本召回。

## 已删除的分支

- 中文 alias FTS5 表和 `recall_chinese` 召回接口；
- `bm25_zh` 与 `bm25_en` 并行排名源；
- 中英三路 RRF 融合；
- 中文 query 直接送入英文 Stage0/reranker 的隐式路径；
- 无查询词时按年份返回最新论文的静默回退。

翻译失败时只记录 `QUERY_REWRITE_ZH_EN:unavailable`，不伪造中文召回结果。

## 当前实现要求

1. `QueryRewriteService` 使用已验证领域术语表，并可选调用 `QUERY_TRANSLATOR_URL`；
2. `HybridPaperSearch` 只创建一个 `bm25` 排名源；
3. Stage0 和 reranker 使用同一个 `translated_query`；
4. HTTP 响应保留 `original_query`、`translated_query`、改写来源和失败状态；
5. 翻译服务不可用时，服务保持可用，但中文请求应明确返回无改写状态，而不是返回最新论文作为假结果。

## 等新数据到位后的工作

- 构造中英文 query-document 对并重新训练 Stage0/Stage2；
- 评测术语表覆盖率、翻译成功率、中文 Recall@K、MRR/NDCG 和 P95 延迟；
- 根据评测结果决定是否需要增加中文标题/关键词字段；在评测证明必要前，不重新引入中文 alias 召回分支。
