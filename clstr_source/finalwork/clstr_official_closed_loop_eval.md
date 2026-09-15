# CLSTR Official Closed-Loop Evaluation Handoff

Last updated: 2026-07-20

This handoff evaluates the same clean matched CLSTR vNext step-500 release used
by the route-metric ledger. Formal jobs have not been submitted.

## Bound release

| Artifact | Identity |
|---|---|
| Stage2 checkpoint | step 500; SHA-256 `100479d031f9d69e1d575efae0a2cd1eea1a69d364a75bd14e262ed4ffda8fa5` |
| Training skill inventory | 67,411 skills |
| Matched release selection | `stage2_coverage_control_p00_v1/diagnostics/matched_release_1c94bc2_v1/clstr_vnext_stage2_matched_release.json` |
| Online route | native factual Stage2; static Top-500 plus up to 64 dynamic extras; hard reliability fallback |

## Evaluation contracts

| Benchmark | Official denominator and metric | CLSTR integration |
|---|---|---|
| ToolSandbox | 21 grouped held-out scenarios; milestone similarity, minefield similarity, combined similarity, exact success | Selects tools every agent turn. Completed tool calls are aligned to real execution results and update episode-local recurrent memory. |
| Tau2 | Official test split: 100 tasks across airline, retail, telecom; one trial by default; reward and exact task success | Selects tools every agent turn. Each simulation owns an independent memory session; tool results correct memory before the next decision. No task description, known/unknown info, or reference action path is read by the agent. |
| StableToolBench | G3 instruction queries; official generated answers followed by conversion and `eval_pass_rate.py`/SoPR | CLSTR selects the initial Top-K API set. StableToolBench's upstream QA pipeline then executes with that fixed set, so this is CLSTR-assisted tool-set retrieval plus official execution, not recurrent per-step routing. |

## Readiness checks

The following commands have already passed with exit code 0 and do not load
Torch or call external models:

```bash
RUN_EVAL=0 bash scripts/sbatch/run_clstr_vnext_toolsandbox_success.sh
RUN_EVAL=0 bash scripts/sbatch/run_clstr_vnext_tau2_success.sh
RUN_EVAL=0 bash scripts/sbatch/run_clstr_vnext_toolbench_official.sh
```

The launchers also pass `bash -n`; all new Python entry points pass AST parsing;
`git diff --check` passes.

## Formal launch shape

Only set `RUN_EVAL=1` after executor/API settings are supplied and the user has
authorized submission.

ToolSandbox requires `EXECUTOR_MODEL` and `EXECUTOR_BASE_URL` and writes
`clstr_task_accuracy.json` plus the official `result_summary.json`.

Tau2 requires the same executor settings. Its default is the official test
split, one trial, all three domains, and concurrency 4. It writes per-domain
official results and a combined `clstr_task_accuracy.json`.

StableToolBench runs four stages in one launcher: CLSTR API-set preparation,
official raw generation, answer conversion, and official pass-rate evaluation.
The executor/evaluator model and API variables remain configurable through the
existing StableToolBench environment contract.
