# CLSTR Main Comparison Tables

This file fixes the table layout and records metrics already available on disk.
`N/A` means the metric scope is not cleanly applicable for that benchmark/method combination.

Status note, 2026-07-04: CLSTR rows are **legacy pre-belief-fix diagnostics** unless explicitly marked
as repaired. A code review found that the old CLSTR path used inconsistent train/eval `m_t` semantics
(`retrieval_logits` memory during Stage1/2 training versus `belief_logits` memory during rollout/eval), and the clean
Stage0 checkpoint leaves `logit_scale_belief=log(0.2)` with zero belief bias. These rows can remain as historical
baselines, but they are not final evidence for recurrent belief-state routing until the repaired Stage1/2/4 chain is
rerun and strict evals are regenerated.

Audit note: all filled values below should be traceable to the listed JSON report. Candidate columns are benchmark-local:
`full pool -> topK` means a retriever first keeps topK candidates from a larger skill pool, while `avg row candidates`
means the benchmark supplies a dynamic per-row candidate set.

Leakage note: legacy pre-clean CLSTR ToolBench-G3 rows are excluded from main conclusions because the old CLSTR
training source `data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2` contains all 1362 rows from the
post-hoc ToolBench eval split. The repaired ToolBench-G3 CLSTR row below uses the clean filtered data root
`data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean`.

Metric note: main route tables must use strict denominators when a candidate handoff can drop rows. Retained-only
metrics are diagnostic-only. TRAJECT-Bench route evaluation is teacher-forced next-skill/tool routing, not closed-loop
trajectory success.

## Table 1. Main Result Summary

Primary metrics only. Detailed R@K/MRR/success breakdowns are in the per-benchmark tables below.

| Method | Params | ToolBench-G3 | tau2 | BFCL | ALFWorld | ToolSandbox | TrajectBench | Notes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 1.2B | 0.161285 | 0.284626 | N/A | N/A | 0.602319 | 0.201480 | Same-scale generic retriever/reranker; ALFWorld is closed-loop executor success and is not a route-rerank metric |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B, adapter-tuned | 1.2B | 0.175192 | 0.290776 | N/A | N/A | 0.602754 | 0.201987 | Same architecture with unified adapter tuning; ALFWorld route-rerank is not directly applicable |
| SkillRouter finetuned baseline | 0.6B-1.2B | 0.131589 | 0.250241 | N/A | 0.042857 | 0.539720 | 0.145842 | Closest SkillRouter-style baseline available per benchmark: ToolBench uses official embedding+reranker pipeline; tau2/ToolSandbox/TrajectBench use finetuned bi-encoder adapter; ALFWorld uses action adapter |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 16B | 0.129204 | 0.302919 | N/A | N/A | 0.634203 | 0.193108 | Large retriever/reranker reference; ALFWorld route-rerank is not directly applicable |
| Tool-REX / Tool-Embed-0.6B retrieval-only | 0.6B | 0.074397 | 0.264559 | N/A | N/A | 0.561988 | 0.086518 | Official same-scale Tool-DE/Tool-REX retriever; ALFWorld uses the adapter-prior row below |
| Tool-REX / Tool-Embed-0.6B, adapter-tuned | 0.6B | 0.128288 | 0.243730 | N/A | 0.200000 | 0.517464 | 0.103645 | Same Tool-Embed backbone with unified adapter tuning; embedding-only top100 for route evals, Qwen3-14B executor on ALFWorld |
| SkillRouter finetuned clean67k/global-pool | 0.6B | 0.140362 | 0.193566 | 0.210809 | N/A | 0.039174 | N/A | Clean 67k unified adapter evaluated over the same full/global pool protocol as repaired CLSTR; ToolBench appends 8 missing skills, tau2 appends 45 unseen skills, ToolSandbox appends 31 unseen skills; BFCL uses frozen 6614-row global-pool protocol |
| CLSTR unified-memory current full3000, protocol-specific | ~0.6B + lightweight heads | 0.434817 | 0.302094 | 0.257426 | 0.164286 | 0.744203 | 0.102661 | Current Stage0/Stage1-2/Stage4 unified-memory chain; ToolBench and BFCL use clean67k/global-pool strict, tau2/ToolSandbox use benchmark-local frozen pools pending final protocol choice; ALFWorld is Qwen3-14B official-env closed-loop success with `clstr_prior_mode=unified_memory`; TrajectBench uses same-row replay-prefix full eval and shows positive dynamic-vs-static MRR delta |
| CLSTR vNext foundation-preserving route, final same-protocol | ~0.6B + lightweight heads | 0.311853 | 0.330649 | N/A | N/A | 0.605942 | N/A | Final schema-v12 route: open/global pools use adapted static + recurrent dynamic; fully enumerated legal pools fitting Top500 use the lineage-bound preserved foundation. All three primary MRRs exceed SR/ToolREx; tau2 uses 1208 routing rows, with equivalent aligned ToolREx target 0.266530. See `finalwork/clstr_vnext_foundation_preserving_results.md`. |
| CLSTR repaired clean67k/global-pool | ~0.6B + lightweight heads | 0.264709 | 0.086867 | 0.083740 | N/A | 0.001561 | N/A | Repaired belief/memory chain; ToolBench/BFCL are clean full-pool strict evals, tau2/ToolSandbox dynamically append unseen skills and are Stage0-limited |
| CLSTR legacy pre-belief-fix diagnostic | ~0.6B + lightweight heads | N/A* | 0.296849 | 0.210318 | 0.235714 | 0.634369 | 0.117278 | Old complete-chain diagnostic with Stage4 loaded where applicable; not final belief-state evidence until repaired Stage1/2/4 is rerun; `*` current ToolBench CLSTR value is contaminated by training-source overlap |

Primary metric by benchmark:

| Benchmark | Primary metric | Secondary metrics |
|---|---|---|
| ToolBench-G3 | MRR | R@1, R@5 or Recall@10, retained rows, candidate scope |
| tau2 | MRR | R@1, R@5, avg row candidates |
| BFCL | MRR | R@1, R@5, strict/global-pool row coverage |
| ALFWorld | Success rate | avg steps, reward, episodes |
| ToolSandbox | MRR | R@1, R@5, usable scenarios |
| TrajectBench | MRR | R@1/R@5/MRR; legacy complete-chain CLSTR, finetuned SkillRouter, and retrieval baselines use the same 4693 teacher-forced eval rows |

## Table 2. ToolBench-G3

Scope: trajectory next-skill routing. Use the strict denominator for CLSTR. If a baseline reports Recall@10 instead of R@5, mark it explicitly.
Legacy pre-clean CLSTR ToolBench rows are excluded because the post-hoc eval rows were present in the old unified
training source. The `CLSTR repaired clean67k full-pool` row below uses the clean filtered data root and strict metrics.

| Method | Params | R@1 / Hit@1 | R@5 / Recall@10 | MRR / MRR@10 | Rows | Candidate scope | Report path / status |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 1.2B | 0.088840 | 0.247430 | 0.161285 | 1362/1362 strict | full pool -> embedding top100; gold@100 0.465492 | `outputs/qwen3_embedding_reranker_route_eval/toolbench_g3_full_20260701_181231/qwen3_embedding_reranker_route_eval_report.json` |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B, adapter-tuned | 1.2B | 0.093245 | 0.271659 | 0.175192 | 1362/1362 strict | full pool -> embedding top100; gold@100 0.541116 | `outputs/qwen3_embedding_adapter_route_eval/toolbench_g3_full_20260702_001422/qwen3_embedding_reranker_route_eval_report.json` |
| SkillRouter-Embedding-0.6B + SkillRouter-Reranker-0.6B, finetuned | 1.2B | 0.043319 | 0.402349 | 0.131589 | 1362 | full pool 37,617; official pipeline reports Recall@10/MRR@10 | `outputs/toolbench_g3_official_skillrouter_comparison/skillrouter_finetune_function_aug_v2_full_20260628_230542/train_eval_report.json` |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 16B | 0.057269 | 0.207048 | 0.129204 | 1362/1362 strict | full pool -> embedding top100; gold@100 0.469897 | `outputs/qwen3_embedding8_reranker8_route_eval/toolbench_g3_full_20260702_014841/qwen3_embedding_reranker_route_eval_report.json` |
| Tool-REX / Tool-Embed-0.6B retrieval-only | 0.6B | 0.024229 | 0.115272 | 0.074397 | 1362/1362 strict | full pool -> embedding top100; gold@100 0.501468 | `outputs/toolrex_embedding_route_eval/toolbench_g3_full_20260701_200724/qwen3_embedding_reranker_route_eval_report.json`; retrieval-only |
| Tool-REX / Tool-Embed-0.6B, adapter-tuned | 0.6B | 0.065345 | 0.180617 | 0.128288 | 1362/1362 strict | full pool -> embedding top100; gold@100 0.640235 | `outputs/toolrex_embedding_adapter_route_eval/toolbench_g3_full_20260702_095435/qwen3_embedding_reranker_route_eval_report.json`; embedding-only adapter |
| SkillRouter-Embedding-0.6B, clean 67k finetuned global-pool | 0.6B | 0.073421 | 0.192364 | 0.140362 | 1362/1362 strict | clean67k+8 appended; full-pool rank | `outputs/toolbench_g3_global_pool_skillrouter_eval/global_pool_toolbench_g3_skillrouter_cleanft_20260706_193218/toolbench_g3_global_pool_skillrouter_eval_report.json`; R@20 0.393539 / R@50 0.544787 / R@100 0.632893 / R@500 0.804699 |
| CLSTR unified-memory clean67k full-pool, current full3000 | ~0.6B + heads | 0.327460 | 0.578561 | 0.434817 | 1021/1362 strict | clean 67k pool; Stage0 top500 -> 64 candidates | `outputs/toolbench_g3_full_clstr_route_eval/unified_memory_stage4_full3000_fastbuild_20260707_181548/full_clstr_route_eval_report.json`; retained dynamic MRR 0.580040 vs static 0.556113; Stage0 top500 next coverage 1021/1362; report strict prior/delta needs regeneration |
| CLSTR repaired clean67k full-pool | ~0.6B + heads | 0.176211 | 0.338473 | 0.264709 | 1158/1362 strict | clean 67k pool; Stage0 top500 -> 64 candidates | `outputs/toolbench_g3_official_skillrouter_comparison/toolbench_g3_repaired_clean67k_stage4_full_20260706_192633/full_clstr_route_eval_report.json`; strict prior MRR 0.229359; Stage4 delta +0.035350; Stage0 top500 next coverage 1158/1362 |

## Table 3. tau2

Scope: domain-local next-tool routing.

| Method | Params | R@1 | R@5 | MRR | Rows | Avg row candidates | Report path / status |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 1.2B | 0.115037 | 0.422350 | 0.284626 | 1217 | 14.907 | `outputs/qwen3_embedding_reranker_route_eval/tau2_full_20260701_181834/qwen3_embedding_reranker_route_eval_report.json`; row-candidate topK covers all domain candidates |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B, adapter-tuned | 1.2B | 0.128184 | 0.424815 | 0.290776 | 1217 | 14.907 | `outputs/qwen3_embedding_adapter_route_eval/tau2_full_20260702_001422/qwen3_embedding_reranker_route_eval_report.json` |
| SkillRouter-Embedding-0.6B, finetuned adapter | 0.6B | 0.068200 | 0.414133 | 0.250241 | 1217 | 14.907 | `outputs/tau2_skillrouter_finetuned_eval/full_20260628_222647/tau2_skillrouter_finetuned_eval_report.json` |
| SkillRouter-Embedding-0.6B, clean 67k finetuned global-pool transfer | 0.6B | 0.046015 | 0.314708 | 0.193566 | 1217/1217 strict | 67k+45 appended; full-pool rank | `outputs/tau2_global_pool_skillrouter_eval/global_pool_tau2_skillrouter_cleanft_20260706_164932/tau2_global_pool_skillrouter_eval_report.json`; clean 67k unified adapter; rank mean 18.386; R@20 0.812654 / R@50 0.936730 / R@500 1.000000 |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 16B | 0.092030 | 0.587510 | 0.302919 | 1217 | 14.907 | `outputs/qwen3_embedding8_reranker8_route_eval/tau2_full_20260702_004216/qwen3_embedding_reranker_route_eval_report.json` |
| Tool-REX / Tool-Embed-0.6B retrieval-only | 0.6B | 0.089565 | 0.459326 | 0.264559 | 1217 | 14.907 | `outputs/toolrex_embedding_route_eval/tau2_full_20260701_193244/qwen3_embedding_reranker_route_eval_report.json`; retrieval-only |
| Tool-REX / Tool-Embed-0.6B, adapter-tuned | 0.6B | 0.098603 | 0.326212 | 0.243730 | 1217 | 14.907 | `outputs/toolrex_embedding_adapter_route_eval/tau2_full_20260702_095435/qwen3_embedding_reranker_route_eval_report.json`; embedding-only adapter |
| CLSTR repaired global-pool transfer | ~0.6B + heads | 0.081348 | 0.086278 | 0.086867 | 216/1217 strict | 64.000 retained; 67k+45 appended | `outputs/tau2_global_pool_clstr_route_eval/global_pool_tau2_full_20260706_161628/tau2_global_pool_clstr_route_eval_report.json`; clean 67k pool plus unseen tau2 append; Stage0 top500 coverage 216/1217; prior strict MRR 0.014813 |
| CLSTR unified-memory benchmark-local pool, current full3000 | ~0.6B + heads | 0.128184 | 0.486442 | 0.302094 | 1217 | 14.907 | `outputs/tau2_full_clstr_route_eval/unified_memory_stage4_full3000_local_prebuilt1217_main_20260707_211700/tau2_full_clstr_route_eval_report.json`; frozen 1217-row source/45-skill local pool; local `skill_table.E` rebuilt after 67k checkpoint shape mismatch; dynamic-vs-static MRR delta +0.044121 |
| CLSTR legacy pre-belief-fix diagnostic | ~0.6B + heads | 0.139688 | 0.483155 | 0.296849 | 1217 | 14.907 | `outputs/tau2_full_clstr_route_eval/function_aug_v2_stage4_full_20260628_203245/tau2_full_clstr_route_eval_report.json` |

## Table 4. ALFWorld

Scope: closed-loop task success with Qwen3-14B executor. Seen and unseen rows are
explicitly separated; none of these task-success rows are route-proxy metrics.

| Method | Split | Router params | Executor | Success | Avg steps | Episodes | Report path / status |
|---|---|---:|---|---:|---:|---:|---|
| **CLSTR adaptive abstract guidance + literal legacy `exact_prior`** | **valid_seen** | **~0.6B + heads** | **Qwen3-14B** | **0.285714** | **41.142857** | **140** | Job `120677`; `alfworld_legacy_guided_exact_prior_8ef3629/adaptive_full/full_8ef3629/metrics.json`; all steps use guidance + exact prior; runtime exact-action expansion disclosed |
| CLSTR adaptive abstract guidance + literal legacy `exact_prior` | valid_unseen | ~0.6B + heads | Qwen3-14B | 0.126866 | 46.447761 | 134 | Job `121301`; `alfworld_legacy_guided_exact_prior_8ef3629/adaptive_unseen_full/full_8ef3629/metrics.json`; 17/134, complete; all 6,224 steps use guidance + exact prior; runtime exact-action expansion disclosed |
| Qwen3-14B executor only | valid_seen | - | Qwen3-14B | 0.185714 | 43.185714 | 140 | `outputs/alfworld_eval/qwen14b_only_loopguard_valid_seen_full_20260702_101337/gate_summary.json` |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | valid_seen | 1.2B | Qwen3-14B | N/A | N/A | N/A | route-rerank baseline; no clean closed-loop ALFWorld controller for this row |
| SkillRouter-Embedding-0.6B action adapter | valid_seen | 0.6B | Qwen3-14B | 0.042857 | 48.585714 | 140 | `outputs/alfworld_eval/qwen14b_skillrouter_ft_loopguard_valid_seen_full_20260628_235102/gate_summary.json` |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | valid_seen | 16B | Qwen3-14B | N/A | N/A | N/A | route-rerank baseline; not a closed-loop ALFWorld method |
| Tool-REX / Tool-Embed-0.6B adapter prior | valid_seen | 0.6B | Qwen3-14B | 0.200000 | 43.585714 | 140 | `outputs/alfworld_eval/qwen14b_toolrex_adapter_loopguard_valid_seen_full_20260702_101337/gate_summary.json` |
| **CLSTR clean step-500 `skill_prompt`, adaptive** | **valid_seen** | **~0.6B + heads** | **Qwen3-14B** | **0.200000** | **43.192857** | **140** | Job `119540`; `alfworld_current_release/full_119540/metrics.json`; hard query-level static/dynamic route, no score fusion |
| CLSTR clean step-500 `skill_prompt`, forced static | valid_seen | ~0.6B + heads | Qwen3-14B | 0.157143 | 43.978571 | 140 | Job `119724`; `alfworld_skill_prompt_static_seen/full_119724/metrics.json`; matched control differing only in route mode |
| **CLSTR clean step-500 `skill_prompt`, adaptive** | **valid_unseen** | **~0.6B + heads** | **Qwen3-14B** | **0.149254** | **45.798507** | **134** | Job `119541`; `alfworld_current_release/full_119541/metrics.json`; no matched forced-static unseen control |
| CLSTR clean step-500 `exact_prior` ablation | valid_seen | ~0.6B + heads | Qwen3-14B | 0.235714 | 42.478571 | 140 | Job `118720`; `alfworld_current_release/full_118720/metrics.json`; exact-action score-fusion interface, retained as a separate ablation |
| CLSTR clean step-500 `exact_prior` ablation | valid_unseen | ~0.6B + heads | Qwen3-14B | 0.149254 | 45.417910 | 134 | Job `119114`; `alfworld_current_release_unseen/full_119114/metrics.json`; exact-action score-fusion interface |
| CLSTR unified-memory current full3000 | valid_seen | ~0.6B + heads | Qwen3-14B | 0.164286 | 44.035714 | 140 | `outputs/alfworld_eval/qwen14b_clstr_unified_memory_mT_valid_seen_full_20260708_144500/qwen_plus_clstr/metrics.json`; `stage4_overlay_loaded=true`; `clstr_prior_mode=unified_memory`; recurrent `m_t` step rate 0.977291 |
| CLSTR legacy pre-belief-fix diagnostic | valid_seen | ~0.6B + heads | Qwen3-14B | 0.235714 | 41.900000 | 140 | `outputs/alfworld_eval/qwen14b_clstr_function_aug_v2_stage4_global_w025_loopguard_valid_seen_full_20260629_110508/gate_summary.json` |

Matched `skill_prompt` seen attribution: adaptive and static share 21 successes;
adaptive has 7 exclusive successes and static has 1. The exact paired two-sided
McNemar/binomial value is `p=0.0703125`. Adaptive therefore has a positive but
not conventionally significant `+6/140 = +0.042857` recurrent-route delta.

On `valid_unseen`, the literal composition is not promoted over the clean
method: it reaches `17/134 = 0.126866`, versus `20/134 = 0.149254` for both
clean adaptive `skill_prompt` and the separate `exact_prior` ablation. Paired
discordances are 10 combined-only versus 13 skill-prompt-only
(`p=0.677639`), and 5 combined-only versus 8 exact-prior-only (`p=0.581055`).

## Table 5. ToolSandbox

Scope: source-derived required-tool route diagnostic. This is not official ToolSandbox interactive pass-rate.

| Method | Params | R@1 | R@5 | MRR | Rows | Usable scenarios | Report path / status |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 1.2B | 0.347826 | 0.991304 | 0.602319 | 115 | 83 | `outputs/qwen3_embedding_reranker_route_eval/toolsandbox_full_20260701_181834/qwen3_embedding_reranker_route_eval_report.json`; row-candidate topK covers all allowed/source-derived candidates |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B, adapter-tuned | 1.2B | 0.347826 | 0.991304 | 0.602754 | 115 | 83 | `outputs/qwen3_embedding_adapter_route_eval/toolsandbox_full_20260702_001422/qwen3_embedding_reranker_route_eval_report.json` |
| SkillRouter-Embedding-0.6B, finetuned adapter | 0.6B | 0.286957 | 0.939130 | 0.539720 | 115 | 83 | `outputs/toolsandbox_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260629_205005/toolsandbox_skillrouter_finetuned_eval_report.json` |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 16B | 0.347826 | 1.000000 | 0.634203 | 115 | 83 | `outputs/qwen3_embedding8_reranker8_route_eval/toolsandbox_full_20260702_004216/qwen3_embedding_reranker_route_eval_report.json` |
| Tool-REX / Tool-Embed-0.6B retrieval-only | 0.6B | 0.330435 | 0.956522 | 0.561988 | 115 | 83 | `outputs/toolrex_embedding_route_eval/toolsandbox_full_20260701_193244/qwen3_embedding_reranker_route_eval_report.json`; retrieval-only |
| Tool-REX / Tool-Embed-0.6B, adapter-tuned | 0.6B | 0.269565 | 0.930435 | 0.517464 | 115 | 83 | `outputs/toolrex_embedding_adapter_route_eval/toolsandbox_full_20260702_095435/qwen3_embedding_reranker_route_eval_report.json`; embedding-only adapter |
| SkillRouter-Embedding-0.6B, clean 67k finetuned global-pool transfer | 0.6B | 0.017391 | 0.052174 | 0.039174 | 115/115 strict | 83; 67k+31 appended; full-pool rank | `outputs/toolsandbox_global_pool_skillrouter_eval/global_pool_toolsandbox_skillrouter_20260706_201223/toolsandbox_global_pool_skillrouter_eval_report.json`; clean 67k unified adapter; R@20 0.095652 / R@50 0.191304 / R@100 0.304348 / R@500 0.556522 |
| CLSTR repaired global-pool transfer | ~0.6B + heads | 0.000000 | 0.000000 | 0.001561 | 6/115 strict | 83; 67k+31 appended | `outputs/toolsandbox_global_pool_clstr_route_eval/global_pool_toolsandbox_full_20260706_162342/toolsandbox_global_pool_clstr_route_eval_report.json`; clean 67k pool plus unseen ToolSandbox append; Stage0 top500 coverage 6/115; prior strict MRR 0.001561 |
| CLSTR unified-memory benchmark-local pool, current full3000 | ~0.6B + heads | 0.539130 | 1.000000 | 0.744203 | 115 | 83 | `outputs/toolsandbox_full_clstr_route_eval/unified_memory_stage4_full3000_local_rebuild_fix_mt_diag_20260707_202500/toolsandbox_full_clstr_route_eval_report.json`; corrected local-pool load rebuilds skipped `skill_table.E`; pairwise m_t effect: mean max-logit diff 0.470222, 17/115 argmax changes; dynamic-vs-static MRR delta -0.027122 |
| CLSTR unified-memory benchmark-local pool, invalid zero-E diagnostic | ~0.6B + heads | 0.426087 | 0.965217 | 0.634369 | 115 | 83 | `outputs/toolsandbox_full_clstr_route_eval/unified_memory_stage4_full3000_local_20260707_190123/toolsandbox_full_clstr_route_eval_report.json`; invalidated for m_t effect analysis because local `skill_table.E` was not rebuilt after 67k checkpoint shape mismatch, making unified logits insensitive to m_t |
| CLSTR legacy pre-belief-fix diagnostic | ~0.6B + heads | 0.426087 | 0.965217 | 0.634369 | 115 | 83 | `outputs/toolsandbox_full_clstr_route_eval/function_aug_v2_stage4_full_20260629_205005/toolsandbox_full_clstr_route_eval_report.json` |

## Table 6. TrajectBench

Scope: teacher-forced next-skill/tool route eval vs finetuned SkillRouter. Do not fill this table with Stage0-only or frozen retrieval-only diagnostics.

| Method | Params | Metric scope | R@1 / Recall@20 | R@5 / Recall@50 | MRR / Recall@100 | Rows | Report path / status |
|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 1.2B | top100 embedding + reranker, same eval rows | 0.096314 | 0.291498 | 0.201480 | 4693 | `outputs/qwen3_embedding_reranker_route_eval/trajectbench_full_20260701_212930/qwen3_embedding_reranker_route_eval_report.json`; embedding gold@100 0.905391 |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B, adapter-tuned | 1.2B | top100 embedding + reranker, same eval rows | 0.095461 | 0.292137 | 0.201987 | 4693 | `outputs/qwen3_embedding_adapter_route_eval/trajectbench_full_20260702_001422/qwen3_embedding_reranker_route_eval_report.json`; embedding gold@100 0.924569 |
| SkillRouter-Embedding-0.6B, finetuned adapter | 0.6B | complete route, same Stage0 top350 rows | 0.051779 | 0.208609 | 0.145842 | 4693 | `outputs/trajectbench_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260701_130731/trajectbench_skillrouter_finetuned_eval_report.json` |
| Qwen3-Embedding-8B + Qwen3-Reranker-8B | 16B | top100 embedding + reranker, same eval rows | 0.069252 | 0.298956 | 0.193108 | 4693 | `outputs/qwen3_embedding8_reranker8_route_eval/trajectbench_full_20260702_014841/qwen3_embedding_reranker_route_eval_report.json`; embedding gold@100 0.933092 |
| Tool-REX / Tool-Embed-0.6B retrieval-only | 0.6B | top100 retrieval-only, same eval rows | 0.016834 | 0.114639 | 0.086518 | 4693 | `outputs/toolrex_embedding_route_eval/trajectbench_full_20260701_212930/qwen3_embedding_reranker_route_eval_report.json`; embedding gold@100 0.901342 |
| Tool-REX / Tool-Embed-0.6B, adapter-tuned | 0.6B | top100 embedding-only adapter, same eval rows | 0.028766 | 0.137013 | 0.103645 | 4693 | `outputs/toolrex_embedding_adapter_route_eval/trajectbench_full_20260702_095435/qwen3_embedding_reranker_route_eval_report.json`; embedding gold@100 0.919455 |
| CLSTR unified-memory current full3000 replay-prefix full | ~0.6B + heads | current full CLSTR, same 4693-row protocol as SkillRouter baseline; `route_scorer=unified_memory` | 0.045600 | 0.140635 | 0.102661 | 4693 | `outputs/trajectbench_full_clstr_eval/unified_memory_full3000_prebuilt4693_replay_full_b128_resume_20260708_132000/eval_stdout.json`; replay prefix attached for 4693/4693 rows; dynamic-vs-static retained MRR delta +0.049377 |
| CLSTR unified-memory current full3000 no-replay diagnostic | ~0.6B + heads | current clean67k Stage0 top350 rebuilt candidates; not comparable to old 4693-row protocol and not valid dynamic-memory evidence | 0.038360 | 0.192328 | 0.130456 | 3806 | `outputs/trajectbench_full_clstr_eval/unified_memory_full3000_current_clean67k_h100_20260707_230554/eval_stdout.json`; `route_scorer=unified_memory`; Stage0 top350 retained 15392/25827 eligible source rows; replay prefix was not actually attached to eval rows, so `m_t=m_0` and dynamic-vs-static delta is 0 |
| CLSTR legacy pre-belief-fix diagnostic | ~0.6B + heads | old complete-chain Stage4 + memory, same Stage0 top350 rows; teacher-forced route eval | 0.044108 | 0.159812 | 0.117278 | 4693 | `outputs/trajectbench_full_clstr_eval/function_aug_v2_complete_stage4_full_20260701_121058/eval_stdout.json` |

## Table 7. Already-Run Auxiliary / Blocked Records

These results are already available and should not be rerun unless the protocol changes.
Some rows, such as the current BFCL clean67k/global-pool result, are also surfaced in Table 1; older or scope-mismatched rows remain here as diagnostics.

| Benchmark | Scope | CLSTR | SkillRouter | Rows / queries | Status / report |
|---|---|---:|---:|---:|---|
| APIBank | current unified-memory CLSTR route diagnostic; benchmark-local APIBank pool from frozen 565-row protocol | R@1 0.368142; R@5 1.000000; MRR 0.638142 | R@1 0.460177; R@5 1.000000; MRR 0.698555 | 565 | CLSTR: `outputs/apibank_full_clstr_route_eval/unified_memory_full3000_current_b128_prebuilt_retry_20260708_163000/apibank_full_clstr_route_eval_report.json`; SkillRouter: `outputs/apibank_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260629_175751/apibank_skillrouter_frozen_eval_report.json`; `route_scorer=unified_memory`; dynamic-vs-static retained MRR delta -0.012596 |
| APIBank | legacy pre-belief-fix CLSTR Stage4 route diagnostic vs finetuned adapter | R@1 0.352212; R@5 1.000000; MRR 0.631003 | R@1 0.460177; R@5 1.000000; MRR 0.698555 | 565 | CLSTR: `outputs/apibank_full_clstr_route_eval/function_aug_v2_stage4_full_20260629_175751/apibank_full_clstr_route_eval_report.json`; SkillRouter: `outputs/apibank_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260629_175751/apibank_skillrouter_frozen_eval_report.json` |
| BFCL v3 8-category | current unified-memory CLSTR global-pool strict route eval over clean 67k pool; not official generation score | R@1 0.147717; R@5 0.355609; MRR 0.257426 | R@1 0.126852; R@5 0.295888; MRR 0.210809 | CLSTR 5708/6614 strict; SkillRouter 6614/6614 | CLSTR: `outputs/bfcl_global_pool_clstr_route_eval/unified_memory_full3000_prebuilt_bfcl_20260707_222610/bfcl_global_pool_clstr_route_eval_report.json`; SkillRouter: `outputs/bfcl_global_pool_skillrouter_eval/global_pool_bfcl_skillrouter_cleanft_prebuilt_20260708_085941/bfcl_global_pool_skillrouter_eval_report.json`; `route_scorer=unified_memory`; BFCL skills already in clean pool; dynamic-vs-static retained MRR delta -0.078501, so keep as diagnostic/appendix rather than main CLSTR evidence |
| BFCL v3 8-category | repaired pre-unified global-pool strict route eval over clean 67k pool; not official generation score | R@1 0.017236; R@5 0.089809; MRR 0.083740 | R@1 0.126852; R@5 0.295888; MRR 0.210809 | CLSTR 6449/6614 strict; SkillRouter 6614/6614 | CLSTR: `outputs/bfcl_global_pool_clstr_route_eval/global_pool_bfcl_full_fast_20260706_173541/bfcl_global_pool_clstr_route_eval_report.json`; SkillRouter: `outputs/bfcl_global_pool_skillrouter_eval/global_pool_bfcl_skillrouter_cleanft_prebuilt_20260708_085941/bfcl_global_pool_skillrouter_eval_report.json`; historical repaired global-pool row before current unified-memory full3000 chain |
| BFCL v3 8-category | legacy pre-belief-fix next-function routing diagnostic, not official generation score | R@1 0.107046; R@5 0.312821; MRR 0.210318 | R@1 0.342304; R@5 0.641064; MRR 0.480662 | 6614 | CLSTR: `outputs/bfcl_full_clstr_route_eval/function_aug_v2_stage4_full8_exportenv_20260629_101841/bfcl_full_clstr_route_eval_report.json`; SkillRouter: `outputs/bfcl_skillrouter_finetuned_eval/function_aug_v2_unified_adapter_full8_exportenv_20260629_101841/bfcl_skillrouter_finetuned_eval_report.json` |
| tau3 | legacy pre-belief-fix CLSTR Stage4 route eval vs finetuned adapter | R@1 0.045167; R@5 0.348996; MRR 0.203141 | R@1 0.091142; R@5 0.370231; MRR 0.250751 | 14834 | CLSTR: `outputs/tau3_full_clstr_route_eval/function_aug_v2_stage4_full_b64_20260629_212456/tau3_full_clstr_route_eval_report.json`; SkillRouter: `outputs/tau3_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260629_205005/tau3_skillrouter_finetuned_eval_report.json` |
| TrajectBench Stage0 retrieval | retrieval-only diagnostic, not complete CLSTR Stage4 reranking | Recall@1 0.069985; Recall@5 0.266735; Recall@100 0.954402; NDCG@100 0.331324 | Recall@20 0.588072; Recall@50 0.778732; Recall@100 0.884076; NDCG@100 0.319504 | 38094 queries | CLSTR Stage0: `outputs/trajectbench_clstr_retrieval_eval/function_aug_v2_stage0_full_20260629_185940/metrics.json`; frozen SkillRouter: `outputs/trajectbench_skillrouter_eval/frozen_full_20260629_185940/metrics.json` |
| StableToolBench | static routing diagnostic, retrieval-only; not official SoPR | Recall@1 0.167213; Recall@5 0.479508; Recall@100 0.931694 | Recall@1 0.181421; Recall@5 0.544536; Recall@100 0.960383 | 61 queries | CLSTR: `outputs/stabletoolbench_static_routing/clstr_stage0_function_aug_v2_20260629_192056/metrics.json`; SkillRouter: `outputs/stabletoolbench_static_routing/skillrouter_frozen_20260629_192056/metrics.json` |
| Tau2 | repaired-adapter official closed-loop environment reward/task success | **39/100 = 0.390000** (airline 5/20; retail 20/40; telecom 14/40) | N/A | 100 official test tasks | Jobs `120050`, `120051`, `120052`; raw-verified aggregate `official_adapter_fixed_037a0e1/tau2_full/tau2_full_aggregate_raw_verified.json` binds exact matched-union task identities and all rewards; no private/oracle task fields enter the agent |
| ToolSandbox | final dual-evidence adaptive official closed-loop held-out execution | **mean similarity 0.820408**; 0 exceptions | N/A | 21 grouped test scenarios | Job `120916`; matched static control is `0.825114`; adaptive uses dual evidence on all 118 history-bearing dynamic selections, preserves every legal schema, and performs no post-hoc substitution |
| WebShop | closed-loop eval blocked | no clean metric | no clean metric | - | `outputs/webshop_eval/function_aug_v2_stage4_smoke_20260629_191845/blocker_report.json`; `outputs/webshop_eval/skillrouter_adapter_smoke_20260629_191845/blocker_report.json` |
