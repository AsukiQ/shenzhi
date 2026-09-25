# Progress

- Latest proxy check: clash-for-linux inactive, no mihomo/clash processes, no 7890/7891/9090 listeners. SSH reverse-forward listener remains on 127.0.0.1:12345; listener alone does not establish upstream proxy usability. Did not restart proxy or resume paused rebuild.

- API latency diagnostic (no changes): local Nginx tests returned HTTP 200 for summary 86ms, assets 10ms, paper detail 5ms, scholars 226ms, institutions 303ms, funding candidates 21ms, subject papers 13ms, funding papers 3.912s, general search 4.234s. General search explicitly reports RECALL_PAPER_DENSE:RuntimeError and bm25+fuzzy fallback; health dense:ready does not prove provider query success. Server load low, available RAM ~1.4 GiB. External-client latency not established. Rebuild remains paused per user; this diagnostic does not authorize restarting Clash or ingestion.

- Verified `/api/knowledge/papers/summary` locally and through public IP: HTTP 200, `{"paper_count":536241}`. Route reads lexical SQLite health count, not Neo4j count; difference from graph after partial canary is expected unsynchronized state. No changes required or performed. User's explicit graph-rebuild pause remains in effect; do not resume ingestion merely due to completion hooks.

- User requested pause and Clash shutdown. Stopped `clash-for-linux.service`; proxy ports closed. Verified Neo4j container online, retrieval service active, local and public `/api/retrieval/health` HTTP 200. SQLite reports 536,241 papers; Neo4j reports 538,558 after the retained partial canary writes; dense component ready. Rebuild/export is paused; no further graph writes performed.

- Installed APOC 5.26.27 via Clash after confirming 5.26.0/5.26.2 dependency incompatibility; Neo4j restarted and `apoc.version()` verified. Enabled file export in `apoc.conf`. APOC full graph export completed in one batch: 889,854 nodes, 2,591,112 relationships, 15,501,026 properties; output `/root/paper_kg_neo4j_import_ready/paper_kg/full_graph_export_20260924.csv` is ~1.4 GiB. Production Neo4j and retrieval health remain online. This is a complete APOC export, but its single-file APOC format still needs conversion into neo4j-admin per-label/per-relationship CSV before rebuild; free disk ~3.6 GiB.

- Clarified rebuild options: a complete CSV can technically be exported from the recovered Neo4j, but current ~3.7 GiB free is insufficient for safe simultaneous export/merge/import artifacts. The running batch importer remains the practical path; it is alive at checkpoint 32 with disk ~3.7 GiB and available memory ~1.9 GiB. Zilliz is independent: existing vectors are unaffected; only new/changed papers require embedding upsert after graph/SQLite validation. The six old archives/quarantine paths were not deleted because the confirmation command was interrupted; they total only ~12 MiB and are not material to capacity.

- User authorized cleanup. Read-only inventory identified `/root/jiansuo/merged_rich_candidate_20260924` as an obsolete, incomplete derived graph (802 MiB) whose Paper baseline was only 240,300 and which cannot safely rebuild production. Removed it with interactive exact-path confirmation. Raw `/root/jiansuo/new`, build/merge scripts, production Neo4j, SQLite, and 301 MiB preservation archive retained. Free disk increased to ~3.7 GiB; retrieval health remains OK (SQLite 536,241; Neo4j 537,924 after partial canary). Unused python image (~125 MiB) remains available for importer.

- 2026-09-24: Confirmed no importer process persists between turns; the slow per-record canary was stopped, and no background job is currently running. Production retrieval/Neo4j remain healthy, with the canary's idempotent partial writes retained. Before any restart, replace it with UNWIND batch transactions and an explicit checkpoint file; do not relaunch the old script.

- User requested no interim reports until the graph is fully rebuilt and verified. Subsequent work should remain silent between tool actions; do not claim completion until Neo4j, SQLite, Zilliz, API, and consistency checks all pass.

- Canary importer ran ~15 minutes with per-node/per-edge autocommit and was interrupted after becoming effectively blocked/too slow. It had already added data idempotently: retrieval health now reports Neo4j 537,924 Paper nodes versus SQLite 536,241 (net +1,683 Paper nodes); graph health remains OK, services remain up, disk ~3.0 GiB free, memory available ~1.9 GiB. No rollback was attempted because writes use MERGE and may include valid related nodes/edges. Per-record importer is unsuitable; replace with UNWIND batch transactions and checkpointing before continuing. Full graph rebuild has not completed.

- Started guarded 1,000-file canary importer in disposable Python/Neo4j-driver container. First attempt stopped on a nested-map property rejected by Neo4j; updated adapter serializes nested maps to JSON strings and reran from the beginning (MERGE is idempotent). Current canary process is still running with no new error output; retrieval service was not stopped. Do not start the full import until this process completes and counts/relationship checks pass.

- Prepared disposable importer runtime: pulled `python:3.11-slim`, installed Neo4j Python driver only inside the ephemeral container, mounted `/root/jiansuo/new` read-only, and verified Bolt connectivity to the running Neo4j (`RETURN 1` -> `1`). No writes were performed. This removes the host/container driver blocker; next step is implementing and running the guarded 1,000-paper canary.

- User reported VS Code Server was reopened. Confirmed `/root/.vscode-server` occupied ~1.7 GiB and multiple server processes were running. Terminated those processes and removed the exact VS Code Server directory with interactive confirmation. Disk returned from ~1.4 GiB to ~3.1 GiB free; retrieval service remains active and Neo4j container remains up. No production database or source data touched.

- User authorized batched import. Preflight found the host Python environment lacks the Neo4j driver and the Neo4j container has no Python, so no importer was run or written blindly. Existing tools only create full CSV imports and cannot perform guarded online upserts. Production remains unchanged; next implementation must use the container's cypher-shell/HTTP endpoint or add a temporary pinned driver environment, with a 1,000-paper canary and checkpoint before full batches.

- 2026-09-24 read-only new-data audit: scanned all readable non-image ZIPs in `/root/jiansuo/new`. Found 250,048 Paper node records and 195,971 unique Paper IDs; compared with production SQLite (536,241 IDs), 11,933 already exist and 184,038 are new. No database or source archive changed. Full candidate CSV remains unsuitable as an import baseline; next implementation is a streamed, checkpointed new-only upsert with guarded property preservation.

- 2026-09-24: User explicitly authorized removal of databases/neo4j_incomplete_20260924. Initial force-removal command was rejected before execution; used interactive recursive removal with confirmation after exact-path inspection. Candidate removed, freeing ~1.2 GiB; root available space now ~3.1 GiB. Production health via port 80 remains OK, Neo4j and SQLite both report 536241 papers. Production preservation archive (301 MiB) retained. Deleted failed candidate has no binary backup; its merged CSV remains available for reconstruction. No new-data import performed.

- Post-recovery merge audit: existing merge_neo4j_import.py builds CSV/archive artifacts with temporary SQLite bookkeeping; it is not an online incremental writer. The current merged candidate includes obsolete baseline records, so replaying it with unrestricted SET would risk undoing production metadata fixes. No production ingestion performed. Failed candidate databases/neo4j_incomplete_20260924 occupies ~1.2 GiB and is separate from restored production; removal is proposed, not executed this turn. New-data-only importer and disk headroom remain prerequisites.

- User approved missing-log recovery. Preserved mismatched logs by directory rename, temporarily enabled db.recovery.fail_on_missing_files=false, and recovered production successfully. Counts: 536241 Paper, 887046 nodes, 616046 CITES. Removed override, cleanly stopped database, ran Neo4j 5.26.27 offline consistency check with 512m heap/384m off-heap/one thread/1400m container limit and property-owner checks enabled: exit 0. Normal restart succeeded. Retrieval health reports graph OK and matching SQLite count; real search returns BM25/fuzzy results but dense recall has RuntimeError despite ready health. Frontend proxy restarted and /api/retrieval/health verified via port 80. Other application containers remain stopped. Free disk ~1.9 GiB; available memory ~2.1 GiB. New-data merge is NOT completed, candidate must not replace production.

- Preservation completed: /root/jiansuo/neo4j_store_preserved_20260924.tar.gz is approximately 301 MiB. tar comparison against stopped production directory emitted no differences/errors. Disk remains approximately 2.4 GiB free. External backup space is no longer a prerequisite for a local recovery attempt. Missing-log recovery remains pending explicit risk approval; Neo4j is stopped and the production store is unchanged.

- Follow-up read-only audit: production store is approximately 1.9 GiB, not the earlier 4.2 GiB estimate. Debug log records a successful production shutdown checkpoint at 2026-09-24 06:18 UTC, txId 40. Started bounded gzip/tar preservation with a 1.5 GiB output limit; no forced recovery authorized or executed.

- 2026-09-24 recovery audit: stopped Neo4j. Current transaction logs are byte-identical to neo4j_import_mismatch_20260924; the other backup has a different header but also a September store creation ID, not the August production ID. No matching production transaction logs found under Docker volumes or project. Actual databases directory contains neo4j, neo4j_incomplete_20260924, and system; neo4j_previous_20260924 is NOT present. Production store must be preserved. Free disk approximately 2.7 GiB, available RAM 2.7 GiB. Missing-log forced recovery is explicitly integrity-risking per Neo4j logs and has NOT been attempted this turn. Requires a preserved copy/external backup and explicit risk decision before proceeding; merge remains incomplete.

- 2026-09-24: 磁盘只读盘点：剩余约 1.8G；可候选清理项包括损坏 SQLite 备份 `papers_fts.damaged_preserved_20260918.db`（约 807MB）、旧 `merge_handoff/legacy_graph.zip`（约 90MB）、ECCV 旧 RAR（约 7.2MB）及重复 canonical 压缩包；当前生产 `papers_fts.db`、Neo4j/rich 图 CSV、新 canonical 数据均标记为保留。本轮未删除任何文件。

- 2026-09-24: 开始制定新数据增量接入方案：现有 `build_rich_graph.py` 可将 canonical JSON 转成 rich CSV，但完整临时构建会复制约 388MB 现有 `papers/cites` CSV；当前磁盘仅约 1.8G，暂不直接重建/切换。已确认 ACL/ECCV 样例均为 `graph.nodes/edges` canonical 结构，下一步应流式解析、去重并生成最小 addition bundle，再调用既有对齐/合并工具。
- 2026-09-24: 停止检索、Neo4j 和相关 Docker 服务后，使用 `build_rich_graph.py` 生成独立候选目录 `incremental_rich_candidate_20260924`。候选包含 4,147 个 JSON，生成 237,850 个去重节点、303,989 条去重关系；但质量门槛未通过：仅 ECCV 的 895 篇被标记为 `external=false` 主论文，其余数据缺少主论文标记；21 个 JSON 因 `HAS_AFFILIATION` 与当前 `AFFILIATED_WITH` 契约不一致而被拒绝。未修改生产图、SQLite、Zilliz 或原始输入，不能直接合并，需先适配数据契约和主论文判定。
- 2026-09-24: 完成新数据适配器：对缺失 `external` 的 Paper，使用“有标题+非空摘要+作者列表”作为保守主论文判定；将旧 `HAS_AFFILIATION` 映射为 `AFFILIATED_WITH`，对不合法的 Paper→Institution 边仅跳过该边并保留其余记录。重新生成候选图后，4,147 个 JSON 全部解析，识别 3,991 篇主论文、3,989 篇可检索文档、249,008 个去重节点、321,452 条去重关系；质量门槛 `neo4j_import_candidate_ready=true`，原始数据和生产库未修改。
- 2026-09-24: 以现有 rich 图为基线生成独立合并目录 `merged_rich_candidate_20260924`，按节点 ID 与关系端点去重追加。新增 Paper 186,140、Author 15,342、Institution 1,767、Method 11,073、Funding 4,726、Topic 11,427；新增 CITES 245,896 条及其他关系，全部关系端点校验无悬空。合并目录约 802MB，生产 Neo4j/SQLite/Zilliz 尚未切换，当前磁盘剩余约 4.5GB。
- 2026-09-24: 对合并目录完成导入前校验，17 个 CSV 共约 1,329,721 行，关系端点无悬空。检查发现现有 Neo4j 数据目录约 4.2GB，而系统仅剩约 4.4GB；全量 `neo4j-admin import` 还需新库文件和回退空间，当前直接导入有磁盘耗尽和数据库损坏风险，因此暂未执行导入。
- 2026-09-24: 删除旧 `neo4j_legacy_backup_20260826` 后释放约 1.7GB，执行了 Neo4j 5.26 离线导入预演。节点阶段和关系阶段写入新库目录约 1.2GB，但 Community 版不支持通过 Cypher 创建新数据库，直接改目录名后默认库状态为 unavailable。已恢复原 `neo4j` 目录并验证旧库可用（887,046 节点）；不完整的新库保留在 `databases/neo4j_imported_20260924`，尚未切换生产服务。
- 2026-09-24: 按 Neo4j Community 正确离线流程完成新图导入：停容器、移走旧 `neo4j` 目录作为 `neo4j_previous_20260924`，用临时 `neo4j-admin` 容器执行全量导入（`--overwrite-destination=true`），再启动原容器。新库 smoke 通过：542,910 节点、763,132 条关系、10,441 个 `external=false` 主论文、576,881 条 `CITES`；当前 Neo4j 已运行，旧库目录仍保留可回退。
- 2026-09-24: 复核发现新图的 Paper 数量低于生产 SQLite/旧 Neo4j，因为 rich CSV 基线仅含 240,300 篇 Paper，而 SQLite 有 536,241 篇。新图不能替代生产图。尝试恢复旧 Neo4j 目录后，Neo4j 系统元数据将默认库标记为 unavailable，当前需进一步修复数据库注册/事务目录；未启动检索服务，未删除旧库目录。

- 2026-09-24: 新数据复查完成：新增 `canonical_ACL_2026` 1779、`canonical_CHI_2026` 550、`canonical_EUROCRYPT_2026` 141、`canonical_PLDI_2026` 107、`ECCV2026.zip` 895 篇 JSON；旧包仍有重复 `(1)` 版本。当前选取的 canonical/新包合计 4147 个 JSON 文件（未按 paper ID 去重），磁盘剩余约 1.8G。全部样例均为 graph JSON，但来源 ID 同时包含 arXiv、DOI 和 title，导入前必须按现有 canonical 解析器做 ID 去重与字段审计，尚未导入线上。

- 2026-09-24: 检查 `/root/jiansuo/new/ECCV2026.zip`：文件名虽为 `.zip`，实际魔数为 `RAR!`，`file` 判定为 RAR v5，`unzip` 无法读取；大小仍约 7.2MB。它不是可直接解压的 ZIP，需改名为 `.rar` 或使用 RAR 工具解压后再核验内容。

- 2026-09-24: 用户确认 ECCV2026 RAR 仅约 7.2MB；已核实压缩包体积不会显著增加磁盘风险，但服务器没有 unrar/7z/rarfile，暂不能读取其内部内容。ECCV 仍待解压核验，其他 ZIP 数据审计结论不变。

- 2026-09-24: 审计 `/root/jiansuo/new` 新数据：发现 canonical ASPLOS/EuroSys/HPCA/HPDC/ISCA/PPoPP 压缩包，去重后约 675 篇论文候选；带 `(1)` 文件为重复版本，ACL/PPoPP/images 压缩包主要是图片，ECCV2026 为 RAR 尚未解压核验。当前磁盘约剩 2.2G，未执行解压、Neo4j/SQLite/Zilliz 导入，避免空间和服务风险。

- 2026-09-22: 按既有表格格式更新 `/root/jiansuo/223131.md`，新增 `/api/knowledge/papers/summary` 与 `/api/knowledge/research-assets/summary` 的参数、示例和统一 Funding 口径说明；其他文档内容未改，`git diff --check` 通过。

- 2026-09-22: 已实现知识库总览统计接口：`GET /api/knowledge/papers/summary` 返回 `paper_count=536241`；`GET /api/knowledge/research-assets/summary` 按统一 Funding 口径返回 `research_asset_count=7353`。Neo4j 健康统计改为分开计数，避免 Paper×Funding 笛卡尔积导致启动超时；前端 Nginx 已增加 `/api/knowledge/` 代理并 reload，公网入口验证通过。

- 2026-09-22: 最终确认：按统一 Funding 口径，`1.zip` 中的论文总数、研究资产总数和跨实体检索需求均可实现；项目/专利/基金不独立分类，推荐返回 `paper_count` 与 `research_asset_count`，统一实体类型为 `funding`。

- 2026-09-22: 用户确认不区分项目、专利和基金类型，统一按 Funding 处理。后续总览接口可返回统一 `research_asset_count`；若前端固定要求 `project_count`/`patent_count`/`funding_count`，三者只能映射同一 Funding 总数，并需在文档中明确不代表独立分类。跨库检索统一使用 `entity_type=funding`。

- 2026-09-22: 根据用户澄清修正 `1.zip` 需求评估：项目、专利、基金在当前模型中统一归入 Funding；此前“项目/专利实体完全缺失”的表述不准确。接口可基于 Funding 实现，但需先审计名称、来源或属性并建立 project/patent/grant 分类规则；若无可靠类型信息，只能准确返回统一 funding_count，不能伪造三类细分数量。

- 2026-09-22: 评估 `/root/jiansuo/1.zip` 总览页需求：文档要求跨库返回论文/学者/主题/项目/基金，以及论文、项目、专利、基金数量摘要。当前系统已有论文、学者、主题、基金数据和对应部分接口，但无 Project/Patent 节点或可靠字段；通用 `/api/retrieval/search` 目前只返回论文，尚无 scope/entity_type 跨库适配。因此接口外壳可实现，真实完整需求需先明确并接入项目/专利数据，不能直接用 0 或伪造数量冒充完成。

- 2026-09-22: 通用 `/api/retrieval/search` 已支持 `offset`/`limit` 分页，保留旧 `top_k`；分页请求返回 `results`、`total`、`total_exact`。由于混合 BM25/模糊/Zilliz 召回没有全库精确计数路径，`total` 是最多 1000 条召回窗口的数量，并用 `total_exact` 明确是否触达窗口上限。服务已重启，`q=reinforcement&offset=2&limit=2` 验证返回 2 条、total=4。

- 2026-09-22: 按既有接口手册格式更新 `/root/jiansuo/aoi222.md`：补充主题检索的 `offset`/`limit`/`total` 真实分页用法、页码计算、兼容 `top_k` 及空页行为；未改动其他接口内容，`git diff --check` 通过。

- 2026-09-22: 已实现主题检索真实分页：`/api/retrieval/search/by-subject` 支持 `offset`/`limit`/`total`，通过 SQLite `paper_subjects` 索引执行 COUNT 与分页，保留旧 `top_k` 兼容路径。已通过语法检查、原有 4 项论文检索测试、临时库分页测试；重启 `jiansuo-retrieval.service` 后公网前本地接口验证 `reinforcement` 返回 total=1824、分页 2 条。

- 2026-09-22: 评估 `xuqiu1.md` 主题论文分页需求：确认当前 `/api/retrieval/search/by-subject` 仅支持 `top_k`，尚未处理 `offset`、`limit` 或 `total`。需求可实现，建议利用 SQLite `paper_subjects` 索引执行独立 COUNT 与分页查询，限制 limit 并保持稳定排序；无需重建 Neo4j/论文库。当前仅完成评估，未修改代码或重启服务。

- 2026-09-21: 修正 newapi.md 排版：基金候选作为接口总览独立业务行放在基金论文检索之后；独立章节移除，参数、返回、清洗边界说明合入该行，其他接口行保持不变。

- 2026-09-21: newapi.md 文档收尾完成：已核对基金候选说明、保留其他接口内容并确认无 patent 条目；本次文档请求无剩余工作。

- 2026-09-21: 按原有表格与简洁风格更新 newapi.md 基金候选说明：补参数标题、编码/分页/400/空结果行为，明确基金候选与论文检索区别及 funding_id 不用于精确论文过滤。其他接口条目未改；专利接口说明已不存在。

- 2026-09-21: 收尾复核：本次 Funding 清洗及 patent 路由删除已完成并验证；原始数据可恢复。没有本次任务剩余实施步骤，历史 ROR 阶段仍按用户转向保持延期，不自动恢复。

- 2026-09-21: Funding 清洗完成：14 个确认污染节点隔离，42 个结构化节点规范化，349 篇论文 funding_json/paper_funding/FTS 同步。原节点改为 FundingRaw，原关系保留，新节点以 NORMALIZED_FROM 追溯；受影响数据备份位于 retrieval_backend/data/funding_repair_20260921T093007667111。统一建索引入口增加同样清洗规则，防止重建重新引入污染。9 项测试通过，SQLite quick_check=ok，重复预检变更为 0，污染 funding 和 FTS 命中均为 0。服务健康，基金公网接口正常，专利路由/文档已删除，公网返回 404。未清除无法确认的自由文本，未声称资助关系全部真实。

- 2026-09-21: 更正 Funding 专利审计结论：未发现显式专利标记不等于没有被误分类的专利。用户指出可能无特殊词条；已确认需追溯 sources_json、关联论文和原始抽取规则才能判定。本轮为结论澄清，尚未开展来源级审计，不将其记录为已排除专利。

- 2026-09-21: 针对“专利是否在 Funding”完成只读核查：扫描全部 7,356 个 Funding 的 ID、name、properties_json、sources_json，未发现 patent/专利标记；properties_json 解析后的键仅 name。编号样例为科研资助编号，未发现可识别的专利记录；未修改图谱。

- 2026-09-21: 基金候选接口已上线，公网关键词/分页/空结果/非法参数验证通过；专利数据缺失明确返回 503 PATENT_DATA_UNAVAILABLE。机构接口和健康检查正常，newapi.md 已追加独立说明。实测基金源数据含疑似模板（NSF grant 12345）及字典字符串名称，未伪装成已清洗标准基金库；专利实际检索仍需数据。

- 2026-09-21: 开始落实基金/专利候选接口。复核 Neo4j 标签仍无 Patent，基金存在 funding_id/name。新增基金实体查询及专利未接入的明确 503 响应，待部署验证；ROR/向量化不在本次实施范围。

- 2026-09-21: 用户将 ROR 接入讨论转向机构向量检索。已说明独立 Zilliz collection、1024 维多语言 Embedding、名称匹配与向量融合、源数据清洗及不能补齐论文机构关系的边界。ROR 仅完成访问/样例核查，未上线；机构向量化仍是方案，未建 collection、未调用 Embedding、未改线上逻辑。

- 2026-09-21: 已解释 ROR 的机构标准 ID、名称/别名数据及适用边界；它不能直接补齐论文机构关系，也不保证中文覆盖。仅回答概念问题，未执行数据接入或部署。

- 2026-09-21: 回答机构别名维护成本问题，建议权威目录补充、中文查询改写及缓存、歧义候选选择；这些是建议，未实施别名写入、翻译调用或新部署。当前说明请求已完成。

- 2026-09-21: 已解释机构候选接口中文查询边界：目前只匹配已有机构名称/ID，不做自动翻译；中英文别名匹配是建议方案，尚未实现或导入别名数据。本次为说明请求，无新增部署任务。

- 2026-09-21: 收尾复核：机构候选检索 Phase 13 已完成，无本任务剩余阶段。反复出现的 0/0 提示源于规划检查上下文/格式不一致：会话 cwd 为 /root，项目计划在 /root/jiansuo；检查器识别 `### Phase` 与 `**Status:**`，项目使用 `## Phase` 与 `Status:`。未修改历史任务状态或检查器来隐藏提示。

- 2026-09-21: 完成机构候选检索部署。公网 Stanford 返回 institution:stanford_university（116 篇关联论文）；分页、空关键词浏览、无匹配、非法 limit=400、health 均验证。newapi.md 改用实测示例。发现源数据有名称变体及模板占位机构，已注明，未擅自清洗图谱。此前“无工具”回复不准确，代码实际上已写入，本轮完成重启验证。

- 2026-09-21: 按用户要求将机构论文检索接口及当前机构数据覆盖率说明，以现有表格风格追加到 `/root/jiansuo/newapi.md`；未修改其他接口内容，`git diff --check` 通过。

- 2026-09-21: 机构覆盖率进一步核实：30,423 篇有 Paper→Author→Institution 路径，全部属于 117,850 篇主论文；现有数据不足以安全推断其余论文机构，需 affiliation 原始字段或外部元数据源补充。

- 2026-09-21: 规划文件收尾检查完成；将历史阶段状态统一为 `complete`，保留其延期说明在正文中，避免计划检查误判未完成。

- 2026-09-21: 核查机构覆盖率：Neo4j 共有 536,241 篇 Paper，约 30,423 篇可通过 Paper→Author→Institution 关联机构（约 5.7%，并非全部论文）。同时修正机构过滤 Cypher 的实际关系路径，重启后 Stanford 查询验证通过。

- 2026-09-21: 新增机构论文检索接口 `/api/retrieval/search/by-institution`；扩展 SearchFilters、Neo4j Institution 关系硬过滤、HTTP 路由及 API.md。服务重启后公网验证通过，执行了 `FILTER_PAPER_NEO4J`。

- 2026-09-20: 按用户要求在 `API.md` 原有接口总览格式中补充主题论文检索与基金论文检索说明，明确两者均返回论文结果（不返回学者），并说明 `top_k` 返回前 N 篇相关论文；原有内容未改写。

- 2026-09-20: 按用户要求将学者检索说明加入 `/root/jiansuo/api.md`，包含搜索、画像详情、参数、返回示例及同名作者 ID 注意事项；保留文档原有简洁格式。
- 2026-09-20: 整理 task_plan.md 完成状态，将历史阻塞阶段标记为已完成/延期跟进，并添加学者检索验收清单。
- 2026-09-20: 根据用户要求改写 `/root/jiansuo/API.md`，仅按原有接口总览表格格式新增学者检索两行，未改动原有章节和内容。
- 2026-09-20: 新增独立论文主题/基金检索入口：`/api/retrieval/search/by-subject` 与 `/api/retrieval/search/by-funding`，返回与普通论文检索相同的论文结果结构；已验证公网基金检索返回论文并更新 API.md。

- Scholar portrait discussion: current `author` is only a paper filter, not a scholar entity/aggregation API. Recommended additive `/api/scholars/search` and `/api/scholars/{id}` backed by Neo4j Author relationships, preserving existing paper search contract and requiring author disambiguation.

- Confirmed complete-field API example is documented and based on a live result; no runtime changes made.

- API_USAGE_GUIDE.md response example replaced with a real result containing populated authors, keywords, subjects, funding, image_id, BM25/dense scores, and retrieval_mode (`paper:2303.08250v5`). `git diff --check` passed.

- Explained `source_scores`: per-retriever raw score/rank and RRF contribution for BM25, fuzzy, and dense channels; raw values are not cross-channel probabilities, and clients should use result order/rank for display.

- Final verification for unified index workflow: git diff --check and shell syntax check passed; retrieval service active. Requested rebuild-code unification and ID coverage audit complete; production full rebuild and missing-record ingestion not performed.

- Unified rebuild entry now preserves authors/subjects/funding/image_id, reconstructs relations and FTS, validates a new staged database without touching production. Five tests pass including funding-only recall plus author/subject filters across a rebuild. Coverage audit: SQLite=Neo4j=536241 IDs, main=117850; Zilliz=114763 unique papers/114764 chunks, all present in SQLite; 3087 main papers lack vectors. New canonical batch: 624 IDs, only 5 in SQLite/Neo4j, zero in Zilliz. Live databases unchanged. Details in INDEX_BUILD_GUIDE.md and data/database_coverage_report.json.

- Explained query-only usage: explicit author/subject filters are optional; natural-language condition extraction is not guaranteed. Topic words can participate in relevance retrieval without becoming mandatory subject filters. No code or service changes requested for this explanation.

- Clarified author/subject request semantics from paper_search.py: optional result filters; AND across fields, OR within each list, local matching uses case-insensitive substrings. subject targets subjects labels and should not be assumed automatically translated. Explanation request complete; no service changes.

- API_USAGE_GUIDE.md example updated to real funded paper `paper:2605.24041v2`, showing nonempty `funding` while retaining concise, abbreviated response fields. `git diff --check` remains clean.

- API_USAGE_GUIDE.md updated to current response contract: authors/subjects/funding/image_id, scoring diagnostics, funding query usage without a dedicated funding filter, and author-order caveat. Kept existing concise structure; git diff --check passed. Documentation request complete.

- 2026-09-20: 当前结果格式已确认：`authors` 已回填，`subjects` 来自 Topic，`funding` 为基金名称列表，`image_id` 为文件名或 null；检索诊断字段和 RRF 分数仍按原协议返回。

- Author backfill API verification passed: searching the cognitive-model paper returns authors [Eric Schulz, Marcel Binz]. Current author-backfill request complete.

- Author backfill completed: filled 106,636 external=false main papers from Neo4j AUTHORED_BY, updating authors_json and paper_authors in a durable transaction without rebuilding FTS. Existing nonempty authors preserved; 324 main papers remain without names. Names alphabetically ordered because publication order is not established by this export.

- Author coverage audit: matched all 117,850 Neo4j external=false papers to SQLite. 106,960 have empty authors_json; 106,636 have nonempty Author names via AUTHORED_BY and can be backfilled. Remaining 324 have no usable author names through that relationship. Read-only audit; no backfill performed.

- 2026-09-20: Confirmed example paper has authors_json=[] in Neo4j but AUTHORED_BY points to Marcel Binz and Eric Schulz; SQLite reconstruction omitted relationship-derived authors. Added funding list to SearchResult hydration with compatibility for databases lacking funding_json. Four paper-search tests passed; restarted service. Author relationship backfill remains outstanding.

- 2026-09-20: 实测搜索接口正常。结果已返回 `subjects`（来自 Neo4j Topic）、`image_id`、评分与检索模式；基金已进入 SQLite/FTS 用于召回，但当前 API 尚未返回独立 `funding` 字段。

- 2026-09-18: 基金同步已完成并验证：5,123 篇论文、9,392 条关系写入 SQLite，基金名称参与 FTS 检索，服务正常。

- 2026-09-18: 基金同步完成：从 Neo4j 导入 5,123 篇有基金的论文、9,392 条基金关系；SQLite 新增 `funding_json` 与 `paper_funding`，并将基金名称并入 FTS 主题检索字段。服务恢复，`DARPA AIQ` 查询已验证可返回结果，数据库完整性检查通过。

- 2026-09-18: 磁盘清理完成并记录：释放约 2.2GB，当前剩余约 3.7GB；有效检索库和 Neo4j/Zilliz 运行所需文件均保留。

- 2026-09-18: 按授权清理磁盘：删除重复的损坏 SQLite 副本和已用于重建、可由 Neo4j 重新导出的 400MB JSONL 中间文件；保留当前有效 `papers_fts.db`、一份损坏库备份、Neo4j 数据和 Zilliz 相关文件。可用空间由约 1.5GB 增至约 3.7GB（87% 使用率）。

- 2026-09-18: 核查 Neo4j 基金数据：7,356 个 `Funding` 节点，9,392 条 `FUNDED_BY` 关系。可同步到 SQLite 为 `funding_json`/`paper_funding`，并加入 FTS；因磁盘仅剩约 1.5GB，需先清理空间。

- 2026-09-18: Topic 同步完成：117,542 篇论文成功匹配，写入 1,220,960 条主题关系，`subjects_json` 与 FTS 已更新；服务恢复并验证主题查询正常。磁盘剩余约 1.5GB（95% 使用率），后续操作需先清理空间。

- 2026-09-18: 恢复完成：从 Neo4j 重建 536,241 篇论文 SQLite 检索库并恢复服务；原损坏文件已保留，主题 Topic 尚未同步到新库。

- 2026-09-18: 原 `papers_fts.db` 只读扫描在第 56 条记录报损坏，已保留损坏副本；从 Neo4j 分页导出 536,241 个 Paper 节点并重建 `papers_fts.db`（约 1.4GB），检索服务恢复，`machine learning` 查询已返回结果。新库暂未同步 Topic，待后续单独批量处理。

- 2026-09-18: 评估 SQLite 损坏后的恢复路径：项目未丢失；Neo4j 仍保有约 536,241 个 Paper 节点及 Topic/引用关系，canonical ZIP 和 Zilliz 也仍在。需先保护损坏 DB，尝试 `.recover`/页恢复，再必要时从 Neo4j 重建论文元数据索引。

- 2026-09-18: 批量 Topic 同步前发现 `papers_fts.db` 已损坏（`PRAGMA integrity_check` 报 freelist/page 错误）；服务可启动但查询返回 `DatabaseError`。已停止继续写入，主题同步未落盘。需要先从可靠备份或原始数据重建 SQLite，再恢复检索功能。

- 2026-09-18: 诊断主题同步失败原因：逐篇 SQLite/FTS 写入造成磁盘 I/O 阻塞；改进方案为临时表 + 内存 ID 映射 + 单事务批量更新/重建 FTS，并在完整性检查后恢复服务。

- 2026-09-18: 按授权暂停并尝试批量同步 Topic；Neo4j 批量查询成功返回约 117k 条，但 SQLite 更新/FTS 重建因磁盘 I/O 长时间阻塞，已终止进程并确认未完成写入。已重新启动 `jiansuo-retrieval.service`，BM25、Zilliz dense、Neo4j filter 恢复正常。

- 2026-09-18: 确定加速方案：维护期间暂停检索服务但保持 Neo4j 运行；采用批量导出 Topic、单事务更新 SQLite，最后一次性重建 FTS，避免逐篇更新造成的性能瓶颈。

- 2026-09-18: 尝试将 Neo4j Topic 同步到 SQLite `subjects_json`；两种分页方式均因 Neo4j 查询/逐条 FTS 更新过慢而中止，核对确认当前实际写入为 0 条，线上检索库未改变。后续需改为批量导出后本地批量事务更新。

- 2026-09-18: 核对无 Topic 节点组成：418,699 篇无 Topic 论文中，418,383 篇 `external=true`（主要是引用/关联节点），316 篇 `external=false`（主图论文但无 Topic）。因此“无 Topic=引用”只能近似成立，不能绝对化。

- 2026-09-18: 统计 Neo4j Topic 覆盖率：536,241 篇 Paper 中 117,542 篇有 `HAS_TOPIC`，418,699 篇没有 Topic。因此不能为所有论文填充 subjects；同步时应只写入有 Topic 的论文，其余保持空值。

- 2026-09-18: 结论：`subject` 与 Neo4j `Topic` 语义可统一，但不能简单改名；需要把 Topic 同步到检索库并在 API 中以 `topics` 为主、保留 `subjects` 兼容别名。

- 2026-09-18: 复核 Neo4j：不存在 `Subject` 标签；主题信息使用 `Topic` 节点和 `HAS_TOPIC` 关系存储，共 130,678 个 Topic 节点、1,220,960 条 Paper-TOPIC 关系。SQLite 的 `subjects` 为空与 Neo4j 图谱主题数据并不矛盾。

- 2026-09-18: 核查 `subjects`：当前 SQLite 检索库 115,386 篇论文中非空 subjects 为 0；五个新增 canonical 压缩包的主论文 subjects 也全部为空。当前检索不依赖该字段。

- 2026-09-18: 评估将新增数据接入 Neo4j/Zilliz：Zilliz 可对 623 篇新增主论文做增量 Embedding；Neo4j 仍需先将 canonical JSON 转换为去重后的节点/关系 CSV，不能直接原样导入。当前未修改线上 Neo4j/Zilliz，避免半成品切换。

- 2026-09-18: 核对 `/root/jiansuo/new/canonical_*.zip` 导入结果：共 624 个主论文记录，其中 623 篇为新增、1 篇为已有记录更新；SQLite 检索库由 114,763 增至 115,386 篇，净增 623 篇。压缩包内所有 Paper 节点记录约 42,426 个（去重约 37,847 个），其余关联/引用节点尚未导入检索库或 Neo4j。

- Inspected `/root/jiansuo/new`: five canonical ZIPs plus images ZIP. Existing prepare_graph_json.py rejects every canonical archive because Paper nodes omit external=false root markers, although records contain title/abstract/authors/keywords/primary_image. No production databases changed; import is blocked pending a dedicated adapter and root/duplicate policy.

- Simplified image API at user request: top-level image_id is basename or null; removed images response array. Verified all archive basenames unique and no multiple images per matched paper. Importer rejects future collisions. Updated API guide, passed four tests and local contract assertions, restarted service and verified public search plus image endpoint for present/absent image cases.

- Image integration: read ZIP metadata without extraction; 4074 PNG/JPG/JPEG images, 3842 exact matches to searchable paper IDs, 232 unmatched in paper_images.report.json. Added reproducible importer and paper_images.json manifest. Search and image endpoint now return image metadata or null. Four paper-search tests pass. First post-restart request got transient 502 during startup; subsequent public verification succeeded for search, encoded paper image endpoint, and null case.

- Follow-up cleanup authorized: deleted historical outputs/paper_vnext_stage0_skillrouter_full and duplicate merge_handoff/new_rich_graph/paper_kg. Compared both CSV directories byte-for-byte before deletion; preserved graph_data original CSV and active Neo4j volume/import mount. Confirmed configured RETRIEVAL_MODE=bm25 does not require Stage0 artifacts.

- Disk cleanup completed with user authorization: removed historical VS Code servers, npm cache, old SQLite backup, old graph transfer archive, FlClash download and unused pre-rich Docker image. Initial force-removal command was rejected; interactive removal succeeded. Root now 84% used with 4.5 GiB available. Retrieval health OK: BM25, dense ready, Neo4j OK. Training outputs and graph CSV retained; identified historical 1000-step SkillRouter/CLSTR Stage0 training artifacts from GPU cluster.

- User requested removal of the newly added search arxiv_id field. Removed field, normalization helper, hydration assignment and guide additions only; existing graph properties untouched. Four tests passed, service restarted, public search verified field absent and bm25+zilliz_dense working. Tracked source diff is empty.

- Inspected result hydration, service launch and API guide. No preexisting project-root plan files.
- Added nullable normalized arxiv_id to shared search result hydration and documented the field.
- Four existing paper search tests passed; checked modern/versioned/URL/legacy identifiers and invalid identifiers, plus real SQLite example.
- Restarted jiansuo-retrieval.service; active. Public search returned paper:2305.14299v3 with arxiv_id 2305.14299 and bm25+zilliz_dense. Diff whitespace check passed.
- 2026-09-20: 核查项目与专利数据：Neo4j 当前仅有 Paper、Author、Topic、Method、Funding、Institution、Conference、Venue 节点，没有 Project 或 Patent 节点，也未发现对应属性；当前检索不支持独立项目/专利查询。
- 2026-09-20: 新增学者检索接口：`GET /api/retrieval/scholars/search?q=...` 支持姓名/author_id 模糊搜索，`GET /api/retrieval/scholars/{scholar_id}` 返回论文数、年份、会议、主题、基金、机构、合作者和论文摘要列表；使用 Neo4j 稳定 author_id，避免同名混淆。服务已重启，公网搜索和详情接口验证通过。
- 2026-09-24: AAAI 2026 demo scope prepared: 5,123 papers in independent SQLite, BM25 service on 18080, health/search/summary/image routes verified. Added systemd unit `jiansuo-retrieval-demo.service`; production 8080 remains unchanged. Graph-backed scholar/institution/funding routes intentionally return unavailable until a scoped AAAI graph is built.
- 2026-09-24: AAAI demo graph feasibility audit: available disk ~2.5 GB; full graph CSV ~2.2 GB. Only 272 of 5,123 demo paper IDs directly matched current `papers.csv`, so a direct scoped graph build would be incomplete. Need source/arXiv ID mapping audit before constructing graph.
- 2026-09-24: Built scoped AAAI 2026 graph export without copying or modifying production Neo4j: 5,123 Paper, 19,290 Author, 13,223 Topic, 889 Institution, 1 Conference nodes and 85,290 relationships. Export size ~30 MB under `/root/jiansuo/demo_aaai2026/graph`; counts and CSV readability verified. No CITES/FUNDED_BY relationships exist inside current AAAI scope.
- 2026-09-24: Clarified demo routing: port 18080 exposes direct paths such as `/search`, `/health`, and `/knowledge/papers/summary`; it does not currently expose an `/api` prefix. Existing `/api/...` routes remain production/full-library endpoints.
- 2026-09-24: Documented AAAI demo endpoint boundary: `/health`, `/ready`, `/search`, `/multistep-search`, summary, subject/funding/institution paper routes, rewrite and image routes are exposed directly on port 18080; graph-backed scholar/entity endpoints and `/api` prefix are not configured. Production `/api/...` remains unchanged.
- 2026-09-24: Reviewed `/root/jiansuo/121.md` for demo adaptation. Data/query/graph routes can be scoped to the 5,123 AAAI papers; auth and favorites require a separate user store. Recommended adding `/api/...` compatibility aliases on port 18080 while leaving production routes unchanged.
- 2026-09-24: Explained empty AAAI demo CITES/FUNDED_BY relations: scoped export reflects current Neo4j source, where no internal AAAI citation or funding edges match; this is source coverage/linkage absence, not API filtering.
- 2026-09-24: Rechecked `/root/jiansuo/AAAI-2026.zip`: 1,366 canonical records contain 1,670 Funding nodes. Corrected earlier conclusion: funding exists in raw AAAI data but was not imported/linked into current Neo4j scoped graph. The demo graph export must be rebuilt from the raw archive for Funding edges.
- 2026-09-24: Imported 1,670 raw AAAI `FUNDED_BY` edges into production Neo4j with idempotent MERGE; total graph funding edges now 10,810 across 8,649 Funding nodes. Updated production and AAAI demo SQLite funding tables (326 and 189 matched papers respectively) and rebuilt scoped graph funding files (1,209 Funding nodes, 1,404 scoped edges). Retrieval services restarted; FTS rebuild was initiated for updated SQLite indexes.
- 2026-09-24: Added `/api/retrieval/*`, `/api/knowledge/*`, and `/api/kg/search` compatibility routing to the demo HTTP service. Restarted demo and verified `/api/retrieval/health`, `/api/retrieval/ready`, `/api/retrieval/search`, and `/api/knowledge/papers/summary`; production service unchanged.
- 2026-09-25: Recommended separating migration delivery: source code via GitHub with secrets excluded; database/images/vector manifests via SwissTransfer with consistent snapshots, checksums, versioned filenames, and a small deployment bundle rather than one monolithic archive.
