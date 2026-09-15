# Shenzhi

论文知识图谱与混合检索服务。

## 目录

- `retrieval_backend/`: BM25、Qwen/Zilliz dense、RRF、Neo4j 过滤和 HTTP API
- `paper_data_pipeline/`: 论文数据准备与图数据流水线
- `shenzhi_paper_data/`: rich graph 构建、校验和合并工具
- `clstr_source/`: 可选的 CLSTR 研究与训练代码

运行数据、模型、数据库、图谱归档和凭据不在仓库中。部署时请根据
`retrieval_backend/deployment/paperddl.env.example` 创建本地环境文件，并单独准备
论文 SQLite 数据库、Neo4j、模型和 Zilliz/Qwen 凭据。

线上检索接口的使用说明见 `API_USAGE_GUIDE.md`。
