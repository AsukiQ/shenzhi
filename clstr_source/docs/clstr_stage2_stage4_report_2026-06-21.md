# CLSTR Stage2/Stage4 Report 2026-06-21

## Summary

This round fixed the Stage2 ToolBench negative-transfer issue and produced a usable Stage4 logged-online memory adaptation result.

The defensible Stage4 narrative is not gradient RL yet. It is online belief/memory adaptation: CLSTR keeps the frozen Stage0/Stage2 routing prior and uses same-trajectory logged feedback to add a bounded memory residual when calibration shows it helps.

## Stage2 Result

Checkpoint:

`outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt`

Main fix:

`final_score = stage0_rank_prior + lambda * transition_residual`

This prevents the learned transition head from overwriting the strong Stage0 retriever order.

ToolBench full diagnostic:

| Metric | Stage0 prior | Stage2 |
|---|---:|---:|
| recall@5 | 0.3222 | 0.3318 |
| recall@20 | 0.5800 | 0.5913 |
| mean positive rank | 45.90 | 44.10 |

The old failure mode, where Stage2 was worse than Stage0 on about 78% of ToolBench rows, is materially fixed.

## Stage4 Result

Best mixed full gate:

`outputs/logged_online_stage4_full/mixed_v4_2_rankprior_stage2full_trajprefix_balanced_memory_sourcegate_l0_eval512_20260621_b/train_stdout.json`

Configuration:

- Stage0 top-M: 350
- Stage2 checkpoint: rank-prior residual full checkpoint above
- Gradient updates: 0
- Online memory: latest exact same-trajectory transition
- Gate: source-benchmark calibration, no hard-coded benchmark rules
- Source caps: 6000 per benchmark
- Eval caps: 512 per benchmark, except ALFWorld had 374 eligible eval suffix rows

Mixed eval:

| Metric | Prior/global-gated baseline | Source-gated Stage4 |
|---|---:|---:|
| recall@1 | 0.0681 | 0.0785 |
| recall@5 | 0.6995 | 0.7152 |
| MRR | 0.2955 | 0.3078 |

Gate decisions:

| Source | Decision | delta MRR | delta recall@5 |
|---|---|---:|---:|
| ToolBench-G3 | enabled | +0.0529 | +0.0527 |
| WebShop | enabled | +0.0524 | 0.0000 |
| ALFWorld | disabled | +0.0099 | -0.0134 |
| TrajectBench | disabled | -0.0002 | 0.0000 |

ToolBench-only full gate:

`outputs/logged_online_stage4_full/toolbench_g3_v4_2_rankprior_stage2full_trajprefix_memory_autogate_l0_eval512_20260621_a/train_stdout.json`

| Metric | Prior | Stage4 memory |
|---|---:|---:|
| recall@1 | 0.0938 | 0.1328 |
| recall@5 | 0.2441 | 0.3027 |
| MRR | 0.1765 | 0.2225 |

## Implementation Notes

New logged-online Stage4 controls:

- `benchmark_caps`: balanced source row construction before `max_rows`.
- `eval_benchmark_caps`: balanced trajectory-prefix eval suffix selection.
- `online_memory_gate_scope=source_benchmark`: source-conditional calibration.

The gate remains generic: it uses source labels only as calibration groups and never encodes benchmark-specific actions, APIs, or thresholds.

## Verification

Commands run after the final fix:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_logged_online_stage4_train.py tests/test_sbatch_scripts.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage4_act_train.py tests/test_logged_online_stage4_train.py tests/test_sbatch_scripts.py tests/test_stage4_quality_gate.py tests/test_stage2_quality_gate.py tests/test_stage1_heads_quality_gate.py tests/test_clstr_topm_candidate_handoff.py tests/test_unified_training_readiness.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/logged_online_stage4_train.py scripts/run_logged_online_stage4_train.py
bash -n scripts/sbatch/run_logged_online_stage4_train.sh
```

Results:

- targeted tests: 85 passed
- broader gate/readiness tests: 176 passed
- Python compile: passed
- sbatch syntax check: passed

## Risks

- This is not executor-based online RL. It should be presented as logged-online routing adaptation.
- Final Stage4 reports currently expose gate by source and aggregate final eval; if we need publication-quality per-source final eval tables, add explicit final eval-by-source reporting before the next full run.
- Stage4 gains depend on repeated same-trajectory transition feedback. It is strongest on ToolBench/WebShop in this run and neutralized by gate on ALFWorld/TrajectBench.
