# BGE-M3 Backbone Study Tables

This file records only the BGE-M3 backbone round from
`.planning/2026-07-08-bge-m3-backbone-study`.

`pending` means the job has not finished yet. `blocked` requires an explicit blocker
report path. Do not mix these rows into `finalwork/table.md` until the BGE round has a
clear include/appendix/abandon decision.

User-confirmed pool protocol: ToolBench-G3 and ALFWorld use the large/global skill pool; tau2 and
ToolSandbox use benchmark-local small skill pools. Rows with different pool protocols
must stay labeled and should not be compared as if they used the same candidate space.

## Model Scope

| Component | Local path | Parameter scale | Role |
|---|---|---:|---|
| BGE-M3 | `models/BAAI/bge-m3` | 0.568B | embedding / retriever backbone |
| BGE-reranker-v2-M3 | `models/BAAI/bge-reranker-v2-m3` | 0.568B | sequence-classification reranker |

## Table 1. BGE Retrieval Sanity

Primary metric is MRR for route-style benchmarks. Rows must use strict denominators
where a benchmark can drop or skip examples.

| Benchmark | Method | Params | R@1 | R@5 | MRR | Rows | Candidate scope | Report path / status |
|---|---|---:|---:|---:|---:|---:|---|---|
| ToolBench-G3 | BGE-M3 retrieval-only | 0.6B | 0.010279 | 0.044053 | 0.034188 | 1362/1362 strict | full pool -> top100 | `outputs/bge_route_eval/toolbench_g3_retrieval_full_rawstate_20260709_092204/route_eval_report.json`; BGE raw-state query mode; gold@top100 0.286344; old instruct-query MRR 0.008512 |
| ToolBench-G3 | BGE-M3 + BGE-reranker-v2-M3 | 1.2B | 0.033774 | 0.107195 | 0.070718 | 1362/1362 strict | full pool -> top100 -> rerank | `outputs/bge_route_eval/toolbench_g3_rerank_full_rawstate_20260709_093535/route_eval_report.json`; BGE raw-state query mode; gold@top100 0.286344; old instruct-query MRR 0.010200 |
| ToolSandbox | BGE-M3 retrieval-only | 0.6B | 0.304348 | 0.930435 | 0.540518 | 115/115 strict | row candidates | `outputs/bge_route_eval/toolsandbox_retrieval_smoke_20260709_093535/route_eval_report.json`; benchmark-local small pool |
| ToolSandbox | BGE-M3 + BGE-reranker-v2-M3 | 1.2B | 0.286957 | 0.973913 | 0.543271 | 115/115 strict | row candidates -> rerank | `outputs/bge_route_eval/toolsandbox_rerank_full_20260709_095005/route_eval_report.json`; benchmark-local small pool |
| tau2 | BGE-M3 retrieval-only | 0.6B | 0.117567 | 0.331056 | 0.231744 | 13907/13907 strict | row candidates | `outputs/bge_route_eval/tau2_retrieval_full_rawstate_20260709_095005/route_eval_report.json`; benchmark-local small pool; larger row count than legacy 1217-row table |
| tau2 | BGE-M3 + BGE-reranker-v2-M3 | 1.2B | 0.079888 | 0.247501 | 0.210041 | 13907/13907 strict | row candidates -> rerank | `outputs/bge_route_eval/tau2_rerank_full_rawstate_20260709_095005/route_eval_report.json`; benchmark-local small pool; reranker hurts vs retrieval-only |

## Table 2. BGE SkillRouter-Style Reimplementation

These rows are paper-style reimplementations on our data, not official SkillRouter
training code.

| Benchmark | Method | Params | R@1 | R@5 | MRR | Rows | Report path / status |
|---|---|---:|---:|---:|---:|---:|---|
| ToolBench-G3 | BGE-SR-Emb | 0.6B | pending | pending | pending | pending | pending paper-style full BGE-M3 finetune |
| ToolBench-G3 | BGE-SR-Emb + BGE-SR-Rank | 1.2B | 0.234949 | 0.506608 | 0.360227 | 1362/1362 strict | `outputs/bge_route_eval/toolbench_g3_bge_sr_emb_rank_full_bgeclf_20260710_000429/route_eval_report.json`; full pool -> top100 |
| ToolSandbox | BGE-SR-Emb | 0.6B | 0.408696 | 0.973913 | 0.647184 | 115/115 strict | `outputs/bge_route_eval/toolsandbox_bge_sr_emb_full_20260710_0034/route_eval_report.json`; row candidates |
| ToolSandbox | BGE-SR-Emb + BGE-SR-Rank | 1.2B | 0.373913 | 0.939130 | 0.580952 | 115/115 strict | `outputs/bge_route_eval/toolsandbox_bge_sr_emb_rank_full_20260710_0034/route_eval_report.json`; row candidates; hurts vs BGE-SR-Emb |
| tau2 | BGE-SR-Emb | 0.6B | 0.075861 | 0.264040 | 0.231911 | 13907/13907 strict | `outputs/bge_route_eval/tau2_bge_sr_emb_full_20260710_0034/route_eval_report.json`; row candidates |
| tau2 | BGE-SR-Emb + BGE-SR-Rank | 1.2B | pending | pending | pending | pending | pending Phase 2 |

## Table 3. BGE ToolRex-Style Reimplementation

Full ToolRex-style means document-expanded Tool-Embed plus Tool-Rank. Embedding-only
rows must not be labeled full ToolRex.

| Benchmark | Method | Params | R@1 | R@5 | MRR | Rows | Report path / status |
|---|---|---:|---:|---:|---:|---:|---|
| ToolBench-G3 | BGE-Tool-Embed retrieval-only | 0.6B | 0.102056 | 0.297357 | 0.202891 | 1362/1362 strict | `outputs/bge_route_eval/toolbench_g3_bge_toolrex_embed_full_20260710_0018/route_eval_report.json`; full pool -> top100 |
| ToolBench-G3 | BGE-Tool-Embed + BGE-Tool-Rank | 1.2B | 0.062408 | 0.213656 | 0.147703 | 1362/1362 strict | `outputs/bge_route_eval/toolbench_g3_bge_toolrex_embed_rank_full_20260710_0018/route_eval_report.json`; full pool -> top100; hurts vs Tool-Embed retrieval-only |
| ToolSandbox | BGE-Tool-Embed retrieval-only | 0.6B | pending | pending | pending | pending | pending Phase 3 |
| ToolSandbox | BGE-Tool-Embed + BGE-Tool-Rank | 1.2B | pending | pending | pending | pending | pending Phase 3 |
| tau2 | BGE-Tool-Embed retrieval-only | 0.6B | pending | pending | pending | pending | pending Phase 3 |
| tau2 | BGE-Tool-Embed + BGE-Tool-Rank | 1.2B | pending | pending | pending | pending | pending Phase 3 |

## Table 4. BGE-CLSTR

These rows require fresh BGE-compatible Stage0, Stage1/2, and Stage4 checkpoints.

| Benchmark | Method | Params | Primary metric | Secondary | Rows / episodes | Report path / status |
|---|---|---:|---:|---:|---:|---|
| ToolBench-G3 | BGE-CLSTR full | 0.6B + heads | pending | pending | pending | pending Phase 4/5 |
| ToolSandbox | BGE-CLSTR full | 0.6B + heads | pending | pending | pending | pending Phase 4/5 |
| tau2 | BGE-CLSTR full | 0.6B + heads | pending | pending | pending | pending Phase 4/5 |
| ALFWorld | BGE-CLSTR full + Qwen3-14B executor | 0.6B + heads | pending | pending | pending | pending Phase 4/5; closed-loop only |

## Run Log

| Time | Item | Status | Evidence |
|---|---|---|---|
| 2026-07-08 | Phase 0 BGE backend smoke | completed | Slurm job `108673`; `outputs/bge_backend_smoke/full_20260709_003147/metrics.json` |
| 2026-07-09 | Phase 1 ToolBench-G3 BGE-M3 retrieval-only sanity | submitted, waiting on smoke | Slurm job `108694`, dependency `afterok:108673`; supersedes cancelled unstarted jobs `108688`, `108690` |
| 2026-07-09 | Phase 1 tau2 BGE-M3 retrieval-only sanity | rerun submitted | Slurm job `108701` with corrected `TAU_DATA_ROOT`; earlier job `108695` failed with `missing_tasks_json=3` |
| 2026-07-09 | Phase 1 ToolBench-G3 BGE-M3 retrieval-only full | submitted | Slurm job `108703`; `MAX_EVAL_ROWS=ALL`, no reranker |
| 2026-07-09 | Phase 1 tau2 BGE-M3 retrieval-only full | completed | Slurm job `108713`; `outputs/bge_route_eval/tau2_retrieval_full_20260709_010720/route_eval_report.json` |
| 2026-07-09 | Phase 1 ToolBench-G3 BGE-M3 + BGE-reranker smoke | completed | Slurm job `108714`; `outputs/bge_route_eval/toolbench_g3_rerank_smoke_20260709_010720/route_eval_report.json` |
| 2026-07-09 | Phase 1 ToolBench-G3 BGE-M3 + BGE-reranker full | completed | Slurm job `108718`; `outputs/bge_route_eval/toolbench_g3_rerank_full_20260709_011551/route_eval_report.json` |
| 2026-07-09 | Phase 1 ToolBench-G3 BGE-M3 + BGE-reranker full raw-state | completed | Slurm job `108771`; `outputs/bge_route_eval/toolbench_g3_rerank_full_rawstate_20260709_093535/route_eval_report.json` |
| 2026-07-09 | Phase 1 ToolSandbox BGE-M3 retrieval-only | completed | Slurm job `108772`; `outputs/bge_route_eval/toolsandbox_retrieval_smoke_20260709_093535/route_eval_report.json` |
| 2026-07-09 | Phase 1 tau2 BGE-M3 retrieval-only raw-state | completed | Slurm job `108783`; `outputs/bge_route_eval/tau2_retrieval_full_rawstate_20260709_095005/route_eval_report.json` |
| 2026-07-09 | Phase 1 tau2 BGE-M3 + BGE-reranker raw-state | completed | Slurm job `108784`; `outputs/bge_route_eval/tau2_rerank_full_rawstate_20260709_095005/route_eval_report.json` |
| 2026-07-09 | Phase 1 ToolSandbox BGE-M3 + BGE-reranker | completed | Slurm job `108785`; `outputs/bge_route_eval/toolsandbox_rerank_full_20260709_095005/route_eval_report.json` |
