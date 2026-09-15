# Phase 1.5 Redundancy Audit

Date: 2026-05-24

## Summary
- Total Python files in scope: **240**
  - `clstr/` (incl. `bridges/`, `envs/`): 81
  - `scripts/`: 86
  - `tests/`: 73
- main_line (AppWorld-only, AppWorld+SkillX, infra shared by them): **35** files
- historical_alfworld + full_base/aux/dagger/online_hrpo/qwen-frozen stack: **88** files (largest bloc)
- historical_skillsbench: **12** files
- historical_qwen_direct_baseline: **6** files
- historical_scienceworld_webshop_dbbench: **14** files
- historical_skillrouter_warmstart (skillret / native_rerank / clean_router): **26** files
- orphan_test: **0** strict orphans (every test file's target module is still present); however many tests are in the historical buckets and would be removed alongside their target
- unclear: **5** files (see "needs human review")
- Total LOC potentially deletable (sum of historical_*): **~33.4K LOC**
  - historical_skillsbench: ~1,204 LOC
  - historical_scienceworld_webshop_dbbench: ~2,061 LOC
  - historical_qwen_direct_baseline: ~1,384 LOC
  - historical_alfworld (+ full_base/aux/dagger/online_hrpo/qwen-frozen training stack): ~23,313 LOC
  - historical_skillrouter_warmstart: ~5,432 LOC

## Methodology
1. Built import-edge graph by greping `from clstr.X` / `import clstr.X` across `clstr/`, `scripts/`, `tests/` (`/tmp/clstr_imports.txt`, 398 raw edges). Resolved per-module importer counts in `/tmp/clstr_import_counts.txt`.
2. Cross-checked every `clstr/*.py` and `scripts/*.py` reference in `description.md` (5,308 lines) for the most recent section-header date in which the file is mentioned, and whether the prose says "still using", "blocker only", "reference baseline", "dropped", etc.
3. Confirmed sbatch entrypoints by extracting `scripts/<name>.py` references from `scripts/sbatch/*.sh` — only AppWorld-stack scripts are wired to live sbatch jobs.
4. Categorized as historical when (a) the file's domain is ALFWorld/ScienceWorld/WebShop/DBBench/SkillsBench/Qwen-direct or (b) the file is a SkillRouter-warmstart helper around the SkillRET corpus, and (c) it is not transitively required by an AppWorld main-line file.

## Detailed table

Legend: `imports` = number of distinct files that import this module (excluding the implicit `clstr` package import).

### main_line (35)
| file | category | imports | desc_md_date | last_action | recommendation |
|------|----------|---------|--------------|-------------|----------------|
| clstr/__init__.py | main_line | 202 (pkg) | always | package root | KEEP |
| clstr/model.py | main_line | 19 | 2026-05-23/24 | training core | KEEP |
| clstr/data.py | main_line | 22 | 2026-05-22+ | dataset core | KEEP |
| clstr/encoders.py | main_line | 6 | 2026-05-22+ | encoder core | KEEP |
| clstr/heads.py | main_line | 3 | 2026-05-24 phase1 fix | KEEP |
| clstr/belief.py | main_line | 9 | 2026-05-22+ | KEEP |
| clstr/losses.py | main_line | 2 | 2026-05-22+ | KEEP |
| clstr/skill_embedding.py | main_line (listed) | 2 | 2026-05-22 | KEEP |
| clstr/multistep_stop.py | main_line | 3 | 2026-05-24 phase1 bug 5 fix | KEEP |
| clstr/native_rerank.py | main_line (listed) | 7 | 2026-05-22+ | KEEP |
| clstr/closed_loop_controller.py | main_line | 6 | 2026-05-24 | KEEP (used by ALFWorld eval, also by structured_controller & test) — still listed authoritative |
| clstr/appworld_routing.py | main_line | 15 | 2026-05-22+ | KEEP |
| clstr/appworld_routing_diagnostics.py | main_line | 2 | 2026-05-23 | KEEP |
| clstr/appworld_skillrouter_base.py | main_line | 4 | 2026-05-23+ | KEEP |
| clstr/appworld_eval.py | main_line | 6 | 2026-05-22+ | KEEP |
| clstr/appworld_clstr_eval.py | main_line | 4 | 2026-05-22+ | KEEP |
| clstr/appworld_executor.py | main_line | 7 | 2026-05-23+ | KEEP |
| clstr/appworld_multistep.py | main_line | 8 | 2026-05-23/24 | KEEP |
| clstr/appworld_act_hrpo.py | main_line | 5 | 2026-05-23/24 | KEEP |
| clstr/appworld_mt_fusion_train.py | main_line | 2 | 2026-05-24 phase1 | KEEP |
| clstr/appworld_act_verified_pairs.py | main_line | 3 | 2026-05-23+ | KEEP |
| clstr/bridges/skillrouter/__init__.py | main_line (infra) | 14 | 2026-05-22+ | KEEP |
| clstr/bridges/skillrouter/checkpoints.py | main_line | 1 (train.py) | KEEP |
| clstr/bridges/skillrouter/datasets.py | main_line | 12 | KEEP |
| clstr/bridges/skillrouter/evaluation.py | main_line | 2 | KEEP |
| clstr/bridges/skillrouter/serialization.py | main_line | 4 | KEEP |
| clstr/bridges/skillx/__init__.py | main_line | 5 | KEEP |
| clstr/bridges/skillx/appworld_adapter.py | main_line | 6 | 2026-05-24 leakage audit | KEEP |
| clstr/envs/__init__.py | main_line (infra) | 22 | KEEP |
| clstr/envs/base.py | main_line (infra) | 11 | KEEP |
| clstr/envs/appworld_env.py | main_line | 6 | KEEP |
| scripts/run_appworld_clstr_eval.py | main_line | sbatch-wired | KEEP |
| scripts/run_appworld_clstr_hrpo_train.py | main_line | sbatch-wired | KEEP |
| scripts/run_appworld_multistep_executor_eval.py | main_line | sbatch-wired | KEEP |
| scripts/run_appworld_mt_fusion_train.py | main_line | sbatch-wired | KEEP |

Additional AppWorld scripts that are sbatch-wired (also main_line):
`scripts/run_appworld_adapter_smoke.py`, `scripts/run_appworld_qwen_executor_eval.py`, `scripts/run_appworld_routing_diagnostics.py`, `scripts/run_appworld_skillrouter_base_eval.py`, `scripts/run_appworld_skillrouter_baseline.py`, `scripts/run_appworld_skillrouter_base_train.py`, `scripts/run_appworld_skillrouter_embedding_baseline.py`, `scripts/run_appworld_smoke.py`, `scripts/audit_skillx_appworld.py`, `scripts/build_appworld_act_verified_pairs.py`, `scripts/build_appworld_executor_comparison.py`, `scripts/build_appworld_routing_comparison.py`, `scripts/build_appworld_routing_data.py`, `scripts/build_appworld_skill_embedding_inputs.py`, `scripts/run_appworld_skill_pool_smoke.py`, `scripts/run_appworld_oracle_solution_smoke.py`, `scripts/build_appworld_multistep_comparison.py`, `scripts/build_appworld_multistep_failure_diagnostics.py`, `scripts/build_appworld_multistep_train_trajectories.py`, `scripts/build_appworld_oracle_multistep_trajectories.py`.

Main_line tests (KEEP — protect-listed): `tests/test_appworld_*.py`, `tests/test_belief.py`, `tests/test_bridge_skillrouter.py`, `tests/test_closed_loop_controller.py`, `tests/test_data.py`, `tests/test_encoders.py`, `tests/test_external_data.py`, `tests/test_heads.py`, `tests/test_losses.py`, `tests/test_metrics.py`, `tests/test_multistep_stop.py`, `tests/test_package_smoke.py`, `tests/test_phase1_*.py`, `tests/test_sbatch_scripts.py`, `tests/test_skillx_appworld_audit.py`.

`clstr/external_data.py` (31 importers): low-level JSON I/O helper used by both main_line and historical paths — KEEP (it is infra). Same for `clstr/data.py`.

---

### historical_skillsbench (12 files, ~1,204 LOC)
| file | imports | desc_md_date | last_action | recommendation |
|------|---------|--------------|-------------|----------------|
| clstr/skillsbench_harness.py (clstr/skillsbench_harness.py:1) | 2 (script+test) | not referenced in description.md after 2026-05 framing | abandoned; ID list at lines 16-25 are SkillsBench tasks | DELETE |
| clstr/skillsbench_transfer.py (clstr/skillsbench_transfer.py:1) | 2 | not in description.md | DELETE |
| clstr/agent_api.py (clstr/agent_api.py:1) | 2 (smoke+harness) | not in description.md | only used by SkillsBench harness path | DELETE |
| clstr/agent_config.py (clstr/agent_config.py:1) | 5 (all SB-only) | not in description.md | DELETE |
| scripts/run_skillsbench_harness_task.py | n/a (entrypoint) | not in description.md / not in sbatch | DELETE |
| scripts/run_skillsbench_transfer_retrieval_eval.py | n/a | not referenced | DELETE |
| scripts/check_skillsbench_data.py | n/a | not referenced | DELETE |
| scripts/collect_skillsbench_harness_results.py | n/a | not referenced | DELETE |
| scripts/export_skillsbench_skill_pool.py | n/a | not referenced | DELETE |
| scripts/import_skillsbench_historical_trajectories.py | n/a | not referenced | DELETE |
| scripts/run_agent_api_smoke.py | n/a | not referenced | DELETE |
| tests/test_agent_harness.py | n/a | n/a | DELETE (target SB harness gone) |

### historical_scienceworld_webshop_dbbench (14 files, ~2,061 LOC)
| file | imports | desc_md_date | last_action | recommendation |
|------|---------|--------------|-------------|----------------|
| clstr/envs/scienceworld_adapter.py (clstr/envs/scienceworld_adapter.py:1) | 4 | 2026-05-23 only as future hypothetical | DELETE |
| clstr/envs/webshop_adapter.py | 2 | not in description.md | DELETE |
| clstr/envs/webshop_official_adapter.py | 3 | not in description.md | DELETE |
| clstr/harness_controller.py (clstr/harness_controller.py:1) | 2 (sci+webshop scripts) | only ScienceWorld/WebShop closed-loop drivers | DELETE |
| scripts/run_scienceworld_clstr_eval.py | not in sbatch | DELETE |
| scripts/run_webshop_clstr_eval.py | not in sbatch | DELETE |
| scripts/run_dbbench_clstr_eval.py | not in sbatch | DELETE |
| scripts/setup_scienceworld_harness.py | not in sbatch | DELETE |
| scripts/setup_webshop_harness.py | not in sbatch | DELETE |
| tests/test_scienceworld_adapter.py | DELETE (target in this batch) |
| tests/test_scienceworld_eval.py | DELETE |
| tests/test_webshop_eval.py | DELETE |
| tests/test_webshop_official_adapter.py | DELETE |
| tests/test_dbbench_eval.py | DELETE |

### historical_qwen_direct_baseline (6 files, ~1,384 LOC)
| file | imports | desc_md_date | last_action | recommendation |
|------|---------|--------------|-------------|----------------|
| clstr/qwen_direct_policy.py (clstr/qwen_direct_policy.py:1) | 5 | 2026-05-21 (Qwen direct likelihood baseline) | reference-only baseline | DELETE if removing ALFWorld batch (tightly coupled to alfworld_eval) |
| clstr/qwen_teacher_rollout.py | 2 | 2026-05-21 | DELETE |
| scripts/run_qwen3_alfworld_direct_eval.py | not in sbatch | DELETE |
| scripts/run_qwen3_alfworld_teacher_rollout.py | not in sbatch | DELETE |
| tests/test_qwen_direct_policy.py | DELETE |
| tests/test_qwen_teacher_rollout.py | DELETE |

### historical_alfworld (+ full_base / aux / dagger / online_hrpo / qwen-frozen training stack) (88 files, ~23,313 LOC)

Per description.md after 2026-05-22, the AppWorld main line uses `appworld_act_hrpo` / `appworld_multistep` / `appworld_mt_fusion_train` and does NOT route through `full_base_train`, `aux_pretrain`, `online_hrpo`, `alfworld_eval`, `dagger_*`, `qwen_full_base_train`. These remain only because old ALFWorld results are still cited. None are reached transitively from any AppWorld main_line script.

Clstr modules (32 files):
`clstr/alfworld_action_skills.py`, `clstr/alfworld_eval.py`, `clstr/alfworld_expert_labeler.py`, `clstr/alfworld_policy_train.py`, `clstr/alfworld_replay.py`, `clstr/agentgym_data.py`, `clstr/online_hrpo.py`, `clstr/envs/alfworld_adapter.py`, `clstr/aux_pretrain.py`, `clstr/aux_trajectories.py`, `clstr/skillnet_aux_rebuild.py`, `clstr/full_base_train.py`, `clstr/full_base_preprocess.py`, `clstr/full_base_data.py`, `clstr/dagger_preprocess.py`, `clstr/dagger_qsuccess_report.py`, `clstr/qwen_full_base_train.py`, `clstr/qwen_external_encoder.py`, `clstr/qwen_planner_intent.py`, `clstr/structured_controller.py`, `clstr/structured_train.py`, `clstr/loss_ablation.py`, `clstr/dataset_registry.py`, `clstr/action_adapter.py`, `clstr/policy_replay_verifier.py`, `clstr/historical_trajectories.py`, `clstr/validator.py`, `clstr/infer.py`, `clstr/metrics.py`, `clstr/train.py`, `clstr/rollout.py`, `clstr/leakage.py`.

Notes per file (load-bearing decisions):
- `clstr/train.py` is the SkillsBench/ALFWorld stage-0/1/2 trainer. Imported by `clstr/appworld_act_hrpo.py:1` and `clstr/appworld_clstr_eval.py:15` only for `resolve_device` helper. Direct removal would break those two imports — call out as a small refactor before deletion.
- `clstr/rollout.py` is only imported by `clstr/appworld_act_verified_pairs.py`, `clstr/infer.py`, and tests; on the AppWorld side only `verified_pairs` uses it, for ALFWorld-flavored rollout helpers. **Verify before deletion.**
- `clstr/closed_loop_controller.py` is in the protect list (`main_line`) and must NOT be deleted, but it is currently only used by `clstr/alfworld_eval.py:19` (ALFWorld path) plus `clstr/harness_controller.py` and ALFWorld-flavored tests. After ALFWorld removal, only `clstr/structured_controller.py` (also historical) and tests would use it — recommend re-evaluating after Batch 5/6 land.
- `clstr/action_adapter.py` is used by `alfworld_eval`, `alfworld_policy_train`, `aux_pretrain`, `full_base_train` only — all historical.
- `clstr/external_data.py` is shared with main_line (`appworld_routing` etc) — KEEP regardless.

Scripts (30 files): all `scripts/*.py` whose name starts with `run_alfworld_`, `build_alfworld_`, `run_clstr_full_base`, `run_clstr_loss_ablation`, `run_clstr_dagger_success_train`, `run_clstr_structured_train`, `run_clstr_qwen3_full_base_train`, `run_clstr_qwen3_online_hrpo`, `run_aux_trajectory_pretrain`, `rebuild_aux_skillnet_data`, `import_aux_trajectories`, `extract_alfworld_policy_replay`, `audit_agentgym_agenttraj`, `check_alfworld_data`, `build_aux_pretrain_gate_report`, `build_clstr_full_base_data`, `build_clstr_full_base_registry`, `build_clstr_dagger_qsuccess_reports`, `build_clstr_dagger_train_data`, `build_clstr_qwen3_reports`, `build_clstr_controller_reports`, `build_historical_verified_pairs`, `build_verified_pairs`, `verify_policy_replay_candidates`, `export_successful_trajectories`, `run_leakage_audit`, `prepare_qwen3_8b`, `make_tiny_hf_model`. None of these scripts appear in any `scripts/sbatch/*.sh`.

Tests (26 files): tests/test_agentgym_data.py, tests/test_alfworld_*.py (5), tests/test_aux_trajectories.py, tests/test_full_base_*.py (3), tests/test_dagger_*.py (2), tests/test_dataset_registry.py, tests/test_historical_trajectories.py, tests/test_leakage.py, tests/test_loss_ablation.py, tests/test_online_hrpo.py, tests/test_policy_replay_verifier.py, tests/test_qwen_external_encoder.py, tests/test_qwen_planner_intent.py, tests/test_skill_embedding.py (target is main_line — RECLASSIFY KEEP), tests/test_skillnet_aux_rebuild.py, tests/test_structured_controller.py, tests/test_structured_train.py, tests/test_train_smoke.py, tests/test_infer_pipeline.py, tests/test_validator.py, tests/test_model_pipeline.py, tests/test_rollout.py. Caveat: `tests/test_skill_embedding.py` exercises `clstr/skill_embedding.py` which is protect-listed — KEEP it and re-trim if `skill_embedding` later transitions to main_line-only.

### historical_skillrouter_warmstart (26 files, ~5,432 LOC)

Skillret / SkillRouter-style retrieval helpers — used early but description.md confirms the AppWorld line now warm-starts from SkillRouter-Embedding-0.6B directly (e.g., description.md:1591 `outputs/.../appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage0_warmstart-step120.pt`). The `clstr/skillret*.py` family backs SkillRET corpus retrieval pretrain only.

Clstr modules (7): `clstr/skillret.py`, `clstr/skillret_official.py`, `clstr/skillrouter_style.py`, `clstr/native_rerank.py` (CAUTION: listed in protect set — see "needs human review"), `clstr/retrieval_warmup.py`, `clstr/clstr_retrieval_adapters.py`, `clstr/skill_embedding.py` (CAUTION: protect-listed).

Scripts (17): `scripts/import_skillret.py`, `scripts/run_skillret_official_eval.py`, `scripts/run_skillret_retrieval_warmup.py`, `scripts/run_skillrouter_style_skillret_finetune.py`, `scripts/run_skillrouter_eval_only.py`, `scripts/build_skillret_comparison_table.py`, `scripts/build_clean_router_data.py`, `scripts/build_clstr_native_routing_init.py`, `scripts/build_clstr_enriched_skill_embeddings.py`, `scripts/export_clstr_native_rerank_skillret_run.py`, `scripts/export_clstr_qdoc_rerank_skillret_run.py`, `scripts/export_clstr_qdoc_skillret_run.py`, `scripts/export_clstr_skillret_run.py`, `scripts/export_skillrouter_style_skillret_run.py`, `scripts/run_clstr_native_rerank_skillret_warmup.py`, `scripts/run_clstr_qdoc_rerank_skillret_warmup.py`, `scripts/run_clstr_qdoc_skillret_warmup.py`.

Tests (2): `tests/test_skillret.py`, `tests/test_skillrouter_eval.py`.

---

## Proposed deletion batches (ordered safest-first)

### Batch 1: orphan_test files for already-removed code
Status: **0 strict orphan tests**. Every test file's target module still exists in-tree. (Note: tests/test_skill_embedding.py target is protect-listed; tests inside historical batches are removed alongside their target.)

Estimated risk: n/a.

### Batch 2: historical_skillsbench
Files (12) listed above. SkillsBench harness path no longer runs (no sbatch wiring, no description.md mention after 2026-05).
LOC: ~1,204.
Estimated risk: **LOW**.

### Batch 3: historical_scienceworld_webshop_dbbench
Files (14) listed above. ScienceWorld/WebShop/DBBench were never integrated into the AppWorld main table (description.md:4713 mentions WebShop/ScienceWorld only as future reuse, no concrete runs after 2026-05-22 pivot).
LOC: ~2,061.
Estimated risk: **LOW**.

### Batch 4: historical_qwen_direct_baseline
Files (6) listed above. description.md:368-412 documents `qwen_direct_policy` / chat-template baseline as reference-only ("Qwen direct 仍是 reference baseline，不是 CLSTR 结果", description.md:412).
LOC: ~1,384.
Estimated risk: **MEDIUM** — these are reference baselines explicitly cited in description.md and may need to be preserved (or relocated to an `archive/` tree) so paper claims can be reproduced. Coupling: `qwen_direct_policy` is imported by `alfworld_eval` (line 17 / src clstr/alfworld_eval.py:23) which sits in Batch 5.

### Batch 5: historical_alfworld + full_base/aux/dagger/online_hrpo/qwen-frozen training stack
Files (88) listed above.
LOC: ~23,313.
Estimated risk: **MEDIUM** — description.md keeps ALFWorld as a "case study only" (per task brief, description.md 2026-05-21), but doesn't actively delete. Also includes the entire CLSTR full_base / aux / dagger / online_hrpo / structured training stack which only ran on ALFWorld replays. Coupling to main_line: `clstr/train.py` is imported by `clstr/appworld_act_hrpo.py:1` and `clstr/appworld_clstr_eval.py:15` purely for `resolve_device`; before deletion, inline that helper or move it to a small utility module. `clstr/rollout.py` only used by main-line via `clstr/appworld_act_verified_pairs.py` — verify the usage is necessary before deletion.

### Batch 6: historical_skillrouter_warmstart
Files (26) listed above (minus the two protect-listed ones: `clstr/native_rerank.py`, `clstr/skill_embedding.py`).
LOC: ~5,432 (gross; subtract LOC for the two protect-listed files if they are kept: 524+202 = 726, net ~4,706).
Estimated risk: **MEDIUM-HIGH** — `clstr/native_rerank.py` is in the protect list AND it imports from `clstr/skillret.py` and `clstr/skillret_official.py`. Removing the SkillRET stack requires first slimming `native_rerank.py` so it no longer references SkillRET helpers. Same for `clstr/skill_embedding.py` which imports `clstr.skillnet_aux_rebuild`. This is the highest-touch batch.

---

## Files we cannot safely classify (NEEDS HUMAN REVIEW)

1. `clstr/appworld_smoke.py` (clstr/appworld_smoke.py:1) — 100 LOC; only used by `scripts/run_appworld_smoke.py` (which IS sbatch-wired via `scripts/sbatch/run_appworld_smoke.sh`) and `tests/test_appworld_smoke.py`. The name says "smoke", not in the explicit main_line list. **Recommend KEEP** until confirmed redundant with `scripts/run_appworld_adapter_smoke.py`.
2. `clstr/external_data.py` — 554 LOC; 31 importers spanning both main_line and historical. Pure I/O helper. **KEEP regardless** but flag for possible compaction once historical files are removed (some helpers may go unused).
3. `clstr/data.py` (protect-listed) — 22 importers, including historical-only ones. **KEEP** but expect dead code inside (e.g., ALFWorld-specific dataclasses) once Batch 5 lands.
4. `clstr/native_rerank.py` (protect-listed) — imports `clstr.skillret` and `clstr.skillret_official`. Cannot remove SkillRET (Batch 6) without first refactoring `native_rerank` so it depends only on bridges/skillrouter. **NEEDS A PRE-DELETE REFACTOR PLAN.**
5. `clstr/skill_embedding.py` (protect-listed) — imports `clstr.skillnet_aux_rebuild` (which itself is Batch 5/6-adjacent). Same pre-delete refactor concern.

Two more soft uncertainties:
- `clstr/closed_loop_controller.py` (protect-listed) — actually only consumed by ALFWorld/structured paths today. Will become dead after Batch 5 if no AppWorld code re-imports it. Re-evaluate after Batch 5.
- `clstr/dataset_registry.py` — used by `tests/test_dataset_registry.py` and `tests/test_full_base_preprocess.py`. If full_base goes, this is orphaned (currently classified historical_alfworld stack).

## Estimated lines removable

Counted via `wc -l` on the file lists above (whitespace-blind):

| Batch | Files | Approx LOC |
|-------|-------|-----------|
| Batch 1 (orphan tests) | 0 | 0 |
| Batch 2 (skillsbench) | 12 | 1,204 |
| Batch 3 (scienceworld/webshop/dbbench) | 14 | 2,061 |
| Batch 4 (qwen direct baseline) | 6 | 1,384 |
| Batch 5 (alfworld + full_base/aux/dagger/online_hrpo) | 88 | 23,313 |
| Batch 6 (skillrouter warmstart, net) | 24 | ~4,706 |
| **Total** | **~144** | **~32.7K LOC** |

Caveats:
- Pre-delete refactors required for `train.py::resolve_device`, `native_rerank.py` (drop SkillRET deps), `skill_embedding.py` (drop skillnet_aux_rebuild dep), and `appworld_act_verified_pairs.py` (verify rollout dep).
- The ~33K LOC figure includes the entire historical-but-still-imported-by-itself ALFWorld stack. If only "files reachable from no AppWorld script" are removed (i.e., we keep `train.py` purely for its `resolve_device` import), the safe set drops by `clstr/train.py` (575 LOC) + `clstr/rollout.py` (285 LOC) = 860 LOC.
