# CLSTR Main Table Design

Date: 2026-07-01

## Purpose

This document fixes the experiment table format before running or rerunning more jobs. The goal is to avoid changing the table shape after seeing metrics.

The main paper table should answer one question:

> Does complete CLSTR improve over the closest small-router baseline, finetuned SkillRouter, on trajectory/state-aware skill routing benchmarks?

A second same-scale retrieval/rerank baseline can be included in the main table if it is runnable across the same route rows with a documented small model size.

## Main Table

Use this as the primary paper table. Do not include goal-only/static ablations here.

| Benchmark | Evaluation scope | Samples | Primary metric | SkillRouter finetuned | Tool-DE 0.6B if runnable | Complete CLSTR | Delta vs SkillRouter | Secondary metrics | Status |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| ToolBench-G3 | trajectory next-skill routing, strict denominator | 1362 source rows | MRR | TBD | TBD | TBD | TBD | R@1, R@5, retained rows, avg candidates | ready |
| tau2 | domain-local next-tool routing | 1217 rows | MRR | TBD | TBD | TBD | TBD | R@1, R@5, avg candidates | ready |
| ToolSandbox | source-derived required-tool route diagnostic | 115 rows | MRR | TBD | TBD | TBD | TBD | R@1, R@5, usable scenarios | ready, diagnostic not official pass-rate |
| ALFWorld valid_seen | closed-loop Qwen3-14B executor success | 140 episodes | Success | TBD | TBD | TBD | TBD | avg steps, reward | ready |
| TrajectBench | trajectory next-skill / route diagnostic | TBD | MRR or Recall@K | TBD | TBD | TBD | TBD | R@1/R@5/MRR or Recall@20/50/100 | not yet clean enough |

### Main Table Rules

- Complete CLSTR means Stage4 checkpoint is loaded when the evaluator supports Stage4.
- For ToolBench-G3, use the complete Stage4 + online memory + inventory64 run, not Stage2-memory diagnostics.
- For ALFWorld, use the Qwen3-14B executor-gated comparison with the same denominator for SkillRouter and CLSTR.
- For tau2 and ToolSandbox, label the scope as route diagnostics, not official interactive task success.
- TrajectBench should not enter the main table until complete CLSTR and SkillRouter finetuned are evaluated under the same route scope. Current Stage0-only retrieval numbers are not enough.
- Goal-only/static SkillRouter is an ablation, not a main-table baseline.
- Tool-DE 0.6B can enter the main table if its 0.6B retriever/reranker checkpoints are available and runnable under the same route-row protocol. It should be marked as a retrieval/rerank baseline, not a trajectory-transition model.

## Current Main Table Fill Plan

| Benchmark | CLSTR artifact | SkillRouter artifact | Action needed |
|---|---|---|---|
| ToolBench-G3 | `outputs/toolbench_g3_official_skillrouter_comparison/clstr_full_route_function_aug_v2_stage4_inventory64_full_20260629_132413/full_clstr_route_eval_report.json` | `outputs/toolbench_g3_official_skillrouter_comparison/skillrouter_finetune_function_aug_v2_full_20260628_230542/train_eval_report.json` | Fill metrics from existing reports |
| tau2 | `outputs/tau2_full_clstr_route_eval/function_aug_v2_stage4_full_20260628_203245/tau2_full_clstr_route_eval_report.json` | `outputs/tau2_skillrouter_finetuned_eval/full_20260628_222647/tau2_skillrouter_finetuned_eval_report.json` | Fill metrics from existing reports |
| ToolSandbox | `outputs/toolsandbox_full_clstr_route_eval/function_aug_v2_stage4_full_20260629_205005/toolsandbox_full_clstr_route_eval_report.json` | `outputs/toolsandbox_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260629_205005/toolsandbox_skillrouter_finetuned_eval_report.json` | Fill metrics from existing reports |
| ALFWorld valid_seen | `outputs/alfworld_eval/qwen14b_clstr_function_aug_v2_stage4_global_w025_loopguard_valid_seen_full_20260629_110508/gate_summary.json` | `outputs/alfworld_eval/qwen14b_skillrouter_ft_loopguard_valid_seen_full_20260628_235102/gate_summary.json` | Fill metrics from existing reports |
| TrajectBench | current outputs are incomplete/mixed-scope | current outputs are frozen or retrieval-only | Decide whether to implement same-scope full route eval or move to appendix |

## Baseline Priority Matrix

This matrix preserves the baseline decisions and prevents static ablations from blocking the main table.

| Baseline | Parameter scale | Table placement | Status / action |
|---|---:|---|---|
| SkillRouter-Embedding-0.6B, goal-only | 0.6B | Ablation | Worth doing later as a static/no-history control. Not main table. |
| SkillRouter-Embedding-0.6B, history | 0.6B | Main baseline component | Already available through the ToolBench SkillRouter finetune report retrieval metrics; extend/fill for other benchmarks where needed. |
| SkillRouter-Embedding + SkillRouter-Reranker 0.6B, history | 1.2B | Main baseline | Already available for several route evals. This is the closest released SkillRouter-style comparison. |
| Qwen3-Embedding-0.6B, history | 0.6B | Secondary/main optional baseline | Worth doing after the first main table fill. Requires model download and unified route-eval adapter. |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 1.2B | Secondary/main optional baseline | Worth doing after the first main table fill. Requires model download and unified route-eval adapter. |
| Qwen3-Reranker-8B | 8B | Upper-bound appendix | Useful as a large-reranker upper bound only. Not a fair main-table baseline. |
| Tool-DE / Tool-Embed / Tool-Rank 0.6B | 0.6B/1.2B depending on retriever/reranker use | Main optional baseline if runnable | Upgraded to a main-table candidate because the paper reports 0.6B models. Need code/model availability check. |
| Re-Invoke | backend-dependent | Appendix zero-shot retrieval baseline | Not discarded. Use only after backend size and rewriting cost are fixed. |
| ToolRerank | backend/hierarchy-dependent | ToolBench-specific appendix | Not discarded. Best for ToolBench/StableToolBench where tool hierarchy exists; not universal across all selected benchmarks. |

## Secondary Baseline Table

Use this only after the main table is filled. These methods are useful, but they should not delay the main CLSTR vs SkillRouter table.

| Method | Paper | Method type | Comparable to CLSTR? | Suggested table placement | Notes |
|---|---|---|---|---|---|
| Tool-DE / Tool-Embed / Tool-Rank | `arXiv:2510.22670` | document-expanded tool retrieval; dense retriever plus reranker, with 0.6B/4B variants reported by the paper | Yes as a same-scale retrieval/rerank baseline. It is not a trajectory policy, but that is a useful contrast | Main table optional column if 0.6B checkpoints are runnable; otherwise appendix | Good candidate if code/models are available. Use state+history query for fairness, and report backend size. |
| Re-Invoke | `arXiv:2408.01875` | zero-shot tool invocation rewriting and multi-view similarity ranking | Partially. It is dynamic at query/intention rewriting time, but not learned trajectory transition | Appendix zero-shot retrieval table | Useful because it is training-free. Need to record the LLM/embedding backend; not a 0.6B fair baseline unless backend is small. |
| ToolRerank | `arXiv:2403.06551` | adaptive and hierarchy-aware reranking for tool retrieval | Partially. It reranks retrieved tools, but is mostly query/tool hierarchy driven | ToolBench-specific appendix table | Best fit for ToolBench/StableToolBench where tool hierarchy is explicit. Less suitable as a universal five-benchmark main baseline. |

### Secondary Table Format

| Benchmark | Scope | SkillRouter finetuned | Tool-DE / Tool-Rank | Re-Invoke | ToolRerank | CLSTR | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| ToolBench-G3 | same route rows | TBD | TBD | TBD | TBD | TBD | ToolRerank only if hierarchy is available |
| tau2 | domain-local route rows | TBD | TBD | TBD | N/A | TBD | ToolRerank likely not applicable |
| ToolSandbox | required-tool route rows | TBD | TBD | TBD | N/A | TBD | Re-Invoke/Tool-DE possible as retrieval baselines |
| TrajectBench | route/retrieval rows | TBD | TBD | TBD | N/A | TBD | Strong candidate for retrieval-focused comparison |
| ALFWorld | closed-loop or route proxy | TBD | TBD | TBD | N/A | TBD | Do not mix route proxy with executor success in the same row |

## 0.6B Fairness Policy

- The main baseline remains finetuned SkillRouter because it is the closest released skill-router style method.
- Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B can be added later as generic same-scale retriever/reranker baselines.
- Qwen3-Reranker-8B is an upper bound, not a fair main-table baseline.
- Tool-DE, Re-Invoke, and ToolRerank must report their backend size. If they rely on large LLM expansion/reranking, they go to appendix or an upper-bound table.

## Final Table Ordering

1. Main Table: CLSTR vs finetuned SkillRouter on the selected benchmarks.
2. Ablation Table: no memory / no Stage4 / goal-only SkillRouter / history SkillRouter.
3. Retrieval Baseline Appendix: Tool-DE, Re-Invoke, ToolRerank, Qwen3-Embedding/Reranker.
4. Limitation Table: benchmarks where CLSTR underperforms or scope is diagnostic only.
