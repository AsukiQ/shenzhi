# CLSTR Detailed Metrics Ledger

Last updated: 2026-07-23

This file separates the current clean matched vNext release from historical
diagnostics. Route-ranking metrics are not task-completion accuracy and must
not be presented as official closed-loop benchmark scores.

## 1. Current clean matched vNext release

All four route evaluations below use the same release-selected Stage2
checkpoint:

- checkpoint step: `500`
- checkpoint SHA-256: `100479d031f9d69e1d575efae0a2cd1eea1a69d364a75bd14e262ed4ffda8fa5`
- training skill inventory: `67,411`
- selection evidence: train/dev only; no benchmark identity, test metric,
  source dispatch, GT candidate injection, or teacher retention

### Main route-ranking metrics

| Benchmark | Split / candidate scope | Rows | Trajectories | R@1 | R@5 | R@10 | R@100 | MRR | Candidate-union recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ToolBench-G3 | clean test; global 67,411-skill pool | 1,362 | 562 | 0.240088 | 0.468429 | 0.571953 | 0.806167 | 0.347813 | 0.913363 |
| ToolSandbox | grouped official-style test routes; complete legal pool | 55 | 27 | 0.418182 | 1.000000 | 1.000000 | 1.000000 | 0.696970 | 1.000000 |
| Tau2 | official test tasks; complete legal pool | 4,484 | 649 | 0.638046 | 0.909456 | 0.956735 | 1.000000 | 0.753512 | 1.000000 |
| TrajectBench | deterministic held-out test; 725 visible public skills plus appended unseen test skills | 3,623 | 569 | 0.200386 | 0.502346 | 0.622688 | 0.934585 | 0.340172 | 1.000000 |

### Memory and counterfactual diagnostics

`Factual` is the deployed recurrent route. `Static` disables recurrent history
under the same evaluation contract. `h-only` removes the trajectory-memory
contribution. Mismatch and shuffle metrics use only their eligible subsets, so
their denominators differ from the main row.

| Benchmark | Factual MRR | Static MRR | Factual - static | h-only MRR | Mismatch MRR (n) | Shuffled-history MRR (n) |
|---|---:|---:|---:|---:|---:|---:|
| ToolBench-G3 | 0.347813 | 0.303214 | +0.044599 | 0.061313 | 0.185684 (1,362) | 0.335856 (800) |
| ToolSandbox | 0.696970 | 0.600000 | +0.096970 | 0.659091 | 0.937500 (8) | 0.666667 (12) |
| Tau2-test | 0.753512 | 0.663249 | +0.090263 | 0.321010 | 0.720249 (4,182) | 0.731448 (3,208) |
| TrajectBench-test | 0.340172 | 0.300762 | +0.039410 | 0.243662 | 0.177310 (3,622) | 0.276992 (2,485) |

### Memory execution coverage

| Benchmark | Recurrent `m_t` rows | Total rows | Recurrent rate | Actual-result corrections | Notes |
|---|---:|---:|---:|---:|---|
| ToolBench-G3 | 1,362 | 1,362 | 1.000000 | 1,362 | Every route row has verified executed-result correction. |
| ToolSandbox | 28 | 55 | 0.509091 | 0 | Source scenarios do not expose executed tool results in the route artifact. |
| Tau2-test | 3,835 | 4,484 | 0.855263 | 0 | Agent-visible rollout history is used; current route artifact has no result-correction field. |
| TrajectBench-test | 3,054 | 3,623 | 0.842948 | 3,558 | 2,056 rows use multi-positive supervision for parallel valid next tools. |

### Authoritative report artifacts

| Benchmark | Report |
|---|---|
| ToolBench-G3 | `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage2_coverage_control_p00_v1/final_matched_eval_1c94bc2_v1/toolbench/toolbench_g3_vnext_eval_report.json` |
| ToolSandbox | `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage2_coverage_control_p00_v1/final_matched_eval_1c94bc2_v1/toolsandbox/toolsandbox_vnext_eval_report.json` |
| Tau2-test | `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage2_coverage_control_p00_v1/final_matched_eval_1c94bc2_v1/tau2/tau2_vnext_eval_report.json` |
| TrajectBench-test | `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage2_coverage_control_p00_v1/diagnostics/trajectbench_release_9669b9a_v1/full_test/trajectbench_vnext_eval_report.json` |

## 2. ALFWorld closed-loop results

The first three rows are the current clean matched step-500 `skill_prompt`
release and its forced-static seen control. The literal combined row adds
adaptive abstract guidance to the unchanged legacy exact-action prior; it is
the strongest current seen result but retains disclosed runtime exact-action
row expansion. The following `exact_prior` rows remain separate ablations. The
full3000 row is a historical, different-lineage diagnostic and is not merged
with the current release.

| Method / protocol | Split | Episodes | Successes | Success rate | Avg. steps | Recurrent-step rate | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| **Current step-500 adaptive abstract guidance + literal legacy `exact_prior` + Qwen3-14B** | **`valid_seen`** | **140** | **40** | **0.285714** | **41.142857** | **0.975694** | Slurm job `120677`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_legacy_guided_exact_prior_8ef3629/adaptive_full/full_8ef3629/metrics.json`; guidance and prior active on 5,760/5,760 steps, 314 fused/Qwen action changes, 1,557 disclosed runtime exact-action rows |
| Current step-500 adaptive abstract guidance + literal legacy `exact_prior` + Qwen3-14B | `valid_unseen` | 134 | 17 | 0.126866 | 46.447761 | 0.978470 | Slurm job `121301`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_legacy_guided_exact_prior_8ef3629/adaptive_unseen_full/full_8ef3629/metrics.json`; guidance and prior active on 6,224/6,224 steps, 235 fused/Qwen action changes, 1,084 disclosed runtime exact-action rows |
| **Current clean step-500 vNext `skill_prompt`, adaptive + Qwen3-14B** | **`valid_seen`** | **140** | **28** | **0.200000** | **43.192857** | **0.976848** | Slurm job `119540`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_current_release/full_119540/metrics.json` |
| Current clean step-500 vNext `skill_prompt`, forced static + Qwen3-14B | `valid_seen` | 140 | 22 | 0.157143 | 43.978571 | 0.977262 | Slurm job `119724`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_skill_prompt_static_seen/full_119724/metrics.json` |
| **Current clean step-500 vNext `skill_prompt`, adaptive + Qwen3-14B** | **`valid_unseen`** | **134** | **20** | **0.149254** | **45.798507** | **0.978165** | Slurm job `119541`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_current_release/full_119541/metrics.json` |
| Current clean step-500 vNext recurrent concrete-action `exact_prior` ablation + Qwen3-14B | `valid_seen` | 140 | 33 | 0.235714 | 42.478571 | 0.976459 | Slurm job `118720`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_current_release/full_118720/metrics.json` |
| Current clean step-500 vNext recurrent concrete-action `exact_prior` ablation + Qwen3-14B | `valid_unseen` | 134 | 20 | 0.149254 | 45.417910 | 0.977982 | Slurm job `119114`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_current_release_unseen/full_119114/metrics.json` |
| Current clean step-500 direct-memory concrete-action pilot, adaptive + Qwen3-14B | `valid_seen` first 16 | 16 | 1 | 0.062500 | 47.312500 | 0.978864 | Slurm job `120296`; `alfworld_direct_memory_79a3b02/adaptive_pilot/smoke_120296/metrics.json`; rejected before full scaling |
| Current clean step-500 direct-memory concrete-action pilot, static + Qwen3-14B | `valid_seen` first 16 | 16 | 1 | 0.062500 | 47.312500 | 0.978864 | Slurm job `120295`; `alfworld_direct_memory_79a3b02/static_pilot/smoke_120295/metrics.json`; matched control |
| CLSTR `unified_memory_concrete_action` + Qwen3-14B executor | `valid_seen` | 140 | 36 | 0.257143 | not retained in the current ledger | 0.975584 | Slurm job `108571`; original output `outputs/alfworld_eval/qwen14b_clstr_unified_memory_concrete_action_valid_seen_full_20260708_185631`; surviving log `.tmp/slurm/alf_qwen_gate-108571.out` |
| Qwen3-14B executor only | `valid_seen` | 140 | 26 | 0.185714 | 43.185714 | N/A | `outputs/alfworld_eval/qwen14b_only_loopguard_valid_seen_full_20260702_101337/gate_summary.json` |
| Tool-REX adapter + Qwen3-14B executor | `valid_seen` | 140 | 28 | 0.200000 | 43.585714 | N/A | `outputs/alfworld_eval/qwen14b_toolrex_adapter_loopguard_valid_seen_full_20260702_101337/gate_summary.json` |

The literal combined method has 20 successes not achieved by clean adaptive
`skill_prompt`, while skill-prompt has eight not achieved by the combined
method (exact paired two-sided `p=0.035698`). Against legacy exact-prior, the
corresponding counts are 17 versus 10 (`p=0.247789`). This supports the value
of combining adaptive abstract guidance and exact-action fusion, while the
increment over exact-prior alone is not conventionally significant.

The literal composition does not transfer its seen gain to `valid_unseen`.
It scores 17/134, versus 20/134 for clean adaptive `skill_prompt` and 20/134
for the separate exact-prior ablation. All episode identities align exactly.
Against skill-prompt, seven successes are shared, ten are combined-only, and
13 are prompt-only (`p=0.677639`); against exact-prior, 12 are shared, five are
combined-only, and eight are prior-only (`p=0.581055`). The largest category
loss is `look_at_obj_in_light` (2/18 combined versus 12/18 skill-prompt and
9/18 exact-prior), so retain clean skill-prompt as the unseen main result.

The matched `skill_prompt` seen comparison shares 21 successful episodes.
Adaptive succeeds on 7 additional episodes while forced-static succeeds on 1,
for a `+6/140 = +0.042857` adaptive delta and exact paired two-sided
`p=0.0703125`. The direction supports recurrent routing, but the denominator is
not sufficient for a conventional `p<0.05` claim. Adaptive selects the dynamic
expert on 5,558/6,047 steps (`0.919133`); its fallback rate is `0.125682`, versus
`0.112717` for static. The proposed static-anchor follow-up was rejected because
only one static-exclusive success is available to recover while seven
adaptive-exclusive successes are at risk.

The clean direct-memory pilot was also rejected before full scaling. Adaptive
and static completed the same successful episode and had identical total steps.
Static overrode Qwen on 181/757 steps (`0.239102`), whereas adaptive overrode it
on only 4/757 (`0.005284`), but neither changed task outcomes. This scorer is a
negative transfer diagnostic for the current checkpoint, not a replacement for
the matched `skill_prompt` main result.

## 3. Current official closed-loop results

Partial or infrastructure-invalid shards are excluded. All rows below are now
complete under the current clean matched step-500 release. Seen and unseen
ALFWorld results remain separate.

| Benchmark | Official split / denominator | Metric | Current credible state | Evidence |
|---|---|---|---|---|
| ALFWorld | `valid_seen`, 140 episodes | Task success | **40/140 = 0.285714**, complete | Adaptive abstract guidance + literal legacy exact-prior; full `120677`; `alfworld_legacy_guided_exact_prior_8ef3629/adaptive_full/full_8ef3629/metrics.json`; runtime exact-action expansion disclosed |
| ALFWorld | `valid_unseen`, 134 episodes | Literal-composition task success | **17/134 = 0.126866**, complete | Adaptive abstract guidance + literal legacy exact-prior; full `121301`; `alfworld_legacy_guided_exact_prior_8ef3629/adaptive_unseen_full/full_8ef3629/metrics.json`; runtime exact-action expansion disclosed; below clean unseen |
| ALFWorld | `valid_seen`, 140 episodes | Clean fixed-inventory task success | **28/140 = 0.200000**, complete | `skill_prompt` adaptive; smoke `119538`; full `119540`; `alfworld_current_release/full_119540/metrics.json` |
| ALFWorld | `valid_unseen`, 134 episodes | Task success | **20/134 = 0.149254**, complete | `skill_prompt` adaptive; smoke `119539`; full `119541`; `alfworld_current_release/full_119541/metrics.json` |
| StableToolBench | G3 solvable, 61 queries, 3 judge trials | SoPR | **0.568306**; trial scores 0.565574 / 0.573770 / 0.565574; 0 missing labels | Generation job `119050`; judge resume `119164`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-sources/4613cbf/outputs/official_closed_loop/toolbench_step500_full_4613cbf/clstr_task_accuracy.json` |
| ToolSandbox | Grouped held-out test, 21 scenarios | Official similarity / exact success | **adaptive mean similarity 0.820408**; matched static **0.825114**; exact **0/21 = 0.000000**; 0 exceptions | Final dual-evidence pair `120916`; `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/toolsandbox_dual_evidence_45f2b3a/full_pair_r1/adaptive/clstr_task_accuracy.json`; near parity, not a claimed memory gain |
| Tau2 | Official test, 100 tasks (airline 20 / retail 40 / telecom 40) | Environment reward / task success | **39/100 = 0.390000**; all 100 simulations have rewards and all task identities exactly match the manifest | Repaired-adapter jobs `120050`, `120051`, `120052`; raw-verified aggregate `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/official_adapter_fixed_037a0e1/tau2_full/tau2_full_aggregate_raw_verified.json` binds all three reports and official `results.json` digests |

Tau2 domain breakdown:

| Domain | Tasks / rewards | Successes | Mean reward / success rate |
|---|---:|---:|---:|
| Airline | 20 / 20 | 5 | 0.250000 |
| Retail | 40 / 40 | 20 | 0.500000 |
| Telecom | 40 / 40 | 14 | 0.350000 |
| **Aggregate** | **100 / 100** | **39** | **0.390000** |

Tau2 online-route audit: the three domains contain 1,338 selection steps and
566 post-tool memory updates. Every selection reduces the domain tool set
(13--16 candidates) to top-8, so routing is intervention-effective rather than
a no-op. Of 1,075 history-bearing selections, 1,074 use the dynamic expert;
the remaining conversational/no-tool-history selections use static routing.
This establishes that recurrent routing is active, but a matched static
closed-loop control would still be needed to quantify its causal success gain.

StableToolBench uses CLSTR for the initial API-set retrieval and then evaluates
the complete Qwen3-14B answer with the official three-trial SoPR judge; its QA
executor does not update recurrent memory inside a query. ToolSandbox and Tau2
execute the release-native recurrent selector online. In the repaired
ToolSandbox full run, all 136 selection steps retained the complete legal tool
set because no step exposed more tracked tools than `top_k=8`. This is not a
selector no-op: CLSTR changed the serialized tool-schema order on 78/136 steps.
Among 105 history-bearing dynamic selections, dynamic and static rankings
differed on 50 steps and their top-1 differed on 26. These diagnostics establish
an active recurrent ranking intervention without deleting tools; a matched
static/adaptive official run is still required to attribute any similarity
delta to memory. The intervention audit is verified by Slurm job `120283`.
All three reports bind
checkpoint SHA-256
`100479d031f9d69e1d575efae0a2cd1eea1a69d364a75bd14e262ed4ffda8fa5`.
