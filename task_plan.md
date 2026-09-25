# Search arXiv ID

Overall status: current funding cleanup complete; ROR deferred

### Phase 17: Evaluate subject pagination requirement
**Status:** complete

Reviewed `xuqiu1.md` against the current subject-search route and retrieval/index implementation. Confirmed feasibility and identified the required offset/limit/count changes; no production code or service state was changed.

### Phase 18: Document subject pagination in aoi222
**Status:** complete

Updated `/root/jiansuo/aoi222.md` in the existing concise table format with offset/limit/total usage, page calculation, compatibility with top_k, and empty-page behavior. No unrelated documentation content was changed.

### Phase 19: Add pagination to general paper search
**Status:** complete

Added `offset`/`limit` handling to `/api/retrieval/search`, preserving legacy `top_k`. Paginated responses include `total` and `total_exact`; the latter documents the bounded mixed-recall window rather than claiming an exact full-database count. Restarted and verified the retrieval service, then updated `aoi222.md`.

### Phase 20: Assess knowledge-base overview API requirements
**Status:** complete

Inspected `/root/jiansuo/1.zip`. Confirmed paper, scholar, topic, and funding capabilities exist, while project and patent entities/data are absent. Determined that endpoint shells and paper/funding summaries are feasible, but complete cross-library results and real project/patent counts require source data and an entity adapter layer.

### Phase 21: Unify research assets as Funding
**Status:** complete

User clarified that project, patent, and funding need not be distinguished. Updated the assessment to use one Funding asset count and `entity_type=funding`; fixed legacy wording that treated project/patent as entirely absent.

### Phase 22: Document knowledge overview endpoints
**Status:** complete

Updated `/root/jiansuo/223131.md` with the paper summary and unified Funding research-asset summary endpoints, preserving the existing table format and documenting the counting scope.

### Phase 23: Audit new paper data drop
**Status:** complete

Inspected `/root/jiansuo/new` without mutating production. Identified duplicate canonical archives, approximately 675 deduplicated paper candidates, separate image-only archives, and an uninspected ECCV RAR. Deferred import because only about 2.2G disk space remains and full extraction/indexing could endanger services.

### Phase 24: Assess ECCV archive size and extraction blocker
**Status:** complete

Confirmed `ECCV2026.rar` is approximately 7.2MB, so the archive itself is not a disk concern. The host lacks unrar/7z/rarfile support, so its contents remain unverified and are deferred until an extraction tool or locally extracted files are available.

### Phase 25: Re-audit expanded new data drop
**Status:** complete

Confirmed `ECCV2026.zip` is now a readable ZIP and counted the expanded canonical set: ACL 1779, CHI 550, EUROCRYPT 141, PLDI 107, ECCV 895, plus prior venues, totaling 4147 JSON files before ID deduplication. Disk space is about 1.8G. No production import was performed; next step is archive deduplication and canonical field/ID audit.

### Phase 28: Build AAAI 2026 demo service
Status: complete

Created a 5,123-paper AAAI 2026 SQLite scope and isolated BM25 service on port 18080. Health, search, summary, pagination and image routes are verified. Built a separate ~30 MB scoped graph export with 5,123 papers, 19,290 authors, 13,223 topics, 889 institutions and 85,290 relationships. Production Neo4j remains untouched; importing the CSV into a separate Neo4j instance is a deployment follow-up.

### Phase 29: Import AAAI funding data
Status: complete

Validated `AAAI-2026.zip`, imported its 1,670 `FUNDED_BY` edges idempotently into production Neo4j, updated production and demo SQLite funding tables, and regenerated scoped demo funding CSVs. Retrieval services were restarted after the updates.

### Phase 30: Add demo API compatibility paths
Status: complete

Demo service accepts `/api/retrieval/*`, `/api/knowledge/*`, and `/api/kg/search` aliases on port 18080. Health, readiness, search, and paper summary were verified through compatibility paths; production routing was unchanged.

### Phase 26: Plan low-space incremental import
**Status:** in_progress

Recovery is complete but new-data ingestion remains pending. Existing merge tools emit CSV/archives, not a bounded online Neo4j upsert. The merged CSV includes an obsolete baseline and must not be used to overwrite recovered production properties. Build a new-data-only, bounded, idempotent importer with ID/relationship validation and property-preservation tests before ingestion. Disk has ~1.9 GiB free; failed candidate database occupies ~1.2 GiB. Keep the recovered production store, its compressed preservation backup, SQLite, and raw inputs. Further cleanup scope and import headroom must be resolved before production writes.

Read-only source audit: all new canonical archives contain 195,971 unique Paper IDs (250,048 Paper node records); 11,933 IDs already exist in production SQLite and 184,038 are absent. This is a large addition, so do not stage a second full graph. Stream batches from canonical archives, maintain a durable checkpoint, and use Neo4j MERGE/ON CREATE plus guarded property updates. SQLite and Zilliz updates follow graph validation.

### Phase 27: Audit removable disk artifacts
**Status:** complete

Performed a read-only disk inventory. Identified the damaged SQLite preservation copy, old graph handoff archive, obsolete ECCV RAR, and duplicate canonical archives as candidates; explicitly retained production SQLite, Neo4j/rich CSV data, and new canonical inputs. No deletion was performed pending user authorization.

### Phase 16: Repair funding data and remove patent route
**Status:** complete

Audit extraction contamination, retain raw records while normalizing reliable lists, synchronize SQLite funding and FTS, remove patent route/documentation, verify service.

Quarantined 14 contaminated records, normalized 42 structured records, synced 349 papers and their FTS funding text. Raw graph nodes/edges retained as FundingRaw and backed up before mutation. Nine tests, SQLite quick_check, idempotence dry run and public routes passed. Patent endpoint now 404; source extraction truth beyond confirmed artifacts remains unverified.

### Phase 15: Funding candidates and patent availability
**Status:** complete

Implement and verify funding entity keyword search, document unavailable patent data without fabricated search results.

Public search, pagination, empty results, invalid limits, patent 503, existing institution search and health verified. Patent retrieval requires a real dataset and schema; only the unavailable response is implemented.

### Phase 14: ROR institution lookup integration
**Status:** deferred

Verify v2 API and Chinese name coverage; implement bounded cached name expansion without graph merges; test fallback and pagination; deploy and document limitations.

User redirected discussion to institution embeddings before implementation. Only ROR API accessibility and sample coverage were inspected. ROR deployment remains deferred; institution-vector design has been explained but not implemented. Do not resume ROR integration automatically.

## Phase 13: Deploy institution candidate search
Status: complete

Verified existing implementation, restarted retrieval, checked public keyword search, pagination, empty/no-match queries, invalid parameters and health. Updated newapi.md with a real response and source-data limitations. Entity cleanup remains outside this endpoint task.

## Phase 12: Add institution paper search
Status: complete

Added the independent paper-returning institution endpoint, wired Institution graph filtering through the retrieval pipeline, updated API documentation, restarted the service, and verified the public route.

Coverage audit: 30,423 of 536,241 Paper nodes currently have an Author-to-Institution path (~5.7%); the filter uses the corrected Paper→Author→Institution path.
Additional audit: all 30,423 institution-linked papers are among 117,850 main papers; no reliable source is currently available to infer institutions for the remaining papers.

## Phase 11: Add subject/funding paper-search documentation
Status: complete

Added concise rows for `/api/retrieval/search/by-subject` and `/api/retrieval/search/by-funding` to `API.md`, preserving the existing table style and clarifying that both endpoints return papers only.

## Phase 8: Unify rebuild entry and audit database ID coverage
Status: complete

Built safe staged index workflow, validated metadata preservation and funding recall; completed read-only SQLite/Neo4j/Zilliz ID comparison. Missing-data ingestion is separate follow-up work.

## Phase 7: Assess new canonical data compatibility
Status: complete

## Phase 6: Simplify image response to filename image_id or null
Status: complete

## Phase 5: Integrate uploaded image manifest into search and image endpoint
Status: complete

## Phase 6: Remove historical training outputs and duplicate CSV
Status: complete

## Phase 5: Authorized disk cleanup and training artifact explanation
Status: complete

## Phase 4: Remove added API field at user request and verify deployment
Status: complete

## Phase 1: Inspect identifier sources
Status: complete

## Phase 2: Add nullable normalized arxiv_id and document it
Status: complete

## Phase 3: Verify and restart retrieval service
Status: complete

## Phase 9: Audit project and patent data availability
Status: complete

Checked Neo4j labels, Paper properties, and local index tables. No Project/Patent entities or fields are currently loaded; no production data was changed.

## Phase 10: Implement scholar search
Status: complete

Added Neo4j-backed scholar search/detail endpoints and concise API documentation. Verified local and public routes after service restart.

## Completion Checklist
### Recovery after failed replacement
Status: complete

Preserved and compared a 301 MiB compressed production backup. User approved missing-log recovery. Recovery succeeded, temporary override removed, offline full consistency check passed (exit 0, including property ownership), normal restart verified. Production: 536241 Paper, 887046 nodes, 616046 CITES. Retrieval and port-80 frontend proxy restarted. Dense query runtime failure remains; BM25 fallback works. Phase 26 remains incomplete: do not promote the incomplete candidate graph; use recovered production as the complete baseline.

- [x] Project/patent data availability audited
- [x] Scholar search endpoint implemented
- [x] Scholar detail endpoint implemented
- [x] Public endpoint verified
- [x] API documentation updated
- [x] Subject and funding paper-search documentation updated
- [x] Institution paper-search endpoint implemented and verified
