# AppWorld SkillRouter-Base And Executor Plan

## Scope

This plan advances the AppWorld main path after the completed CLSTR-base routing run.
It keeps CLSTR as a routing layer and evaluates downstream AppWorld task success with
a fixed Qwen3-8B executor.

All files and outputs stay under:

`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`

Cluster jobs must run through `sbatch`; login-node work is limited to light tests,
inspection, and script preparation.

## SkillRouter-Base Definition

The upstream SkillRouter checkout currently provides usable embedding/eval utilities,
but no local full fine-tuning entry point. For a fair AppWorld routing comparison, this
project defines **SkillRouter-base** as:

- frozen `SkillRouter-Embedding-0.6B` text embeddings,
- a small trainable routing scorer over AppWorld train qrels,
- the same AppWorld SkillX skill pool and qrels as CLSTR-base,
- dev eval with the same `recall@K` and `mrr@K` metrics.

This is distinct from:

- `skillrouter_embedding_0.6b`: frozen embedding retrieval with no AppWorld training,
- `skillrouter_serialization_lexical`: lexical fallback,
- `clstr_checkpoint_skill_table_logits`: CLSTR-base checkpoint scoring.

Outputs:

- `outputs/appworld_skillrouter_base_train_v1/model.pt`
- `outputs/appworld_skillrouter_base_train_v1/report.json`
- `outputs/appworld_skillrouter_base_eval/dev_v1/predictions.jsonl`
- `outputs/appworld_skillrouter_base_eval/dev_v1/report.json`

Current result:

- Train job `75235` completed.
- Eval job `75237` completed.
- Dev routing metrics: `recall@1=0.982456`, `recall@5=1.0`,
  `recall@20=1.0`, `mrr@20=0.991228`.
- Comparison artifact:
  `outputs/appworld_routing_comparison_full_v1_with_skillrouter_base/comparison.json`.

## Executor Benchmark

The executor benchmark uses Qwen3-8B as the downstream code generator:

1. A skill provider returns top-k SkillX skills for a task:
   - `qwen_only`: no retrieved SkillX skills,
   - `skillrouter_embedding`: frozen SkillRouter embedding top-k,
   - `skillrouter_base`: trained SkillRouter-base top-k,
   - `clstr_base`: completed CLSTR-base checkpoint top-k,
   - later `clstr_act`.
2. The prompt builder combines:
   - task instruction,
   - top-k skill text,
   - compact AppWorld API docs filtered by required apps and API refs,
   - a strict instruction to return Python code only.
3. Qwen3-8B generates Python code from the local model at `models/Qwen3-8B`.
4. The executor runs the code with `AppWorld.execute(code)`.
5. The loop records `task_completed()`, `evaluate(suppress_errors=True)`, generated code,
   execution output, exceptions, selected skills, and prompt metadata.

Smoke outputs:

- `outputs/appworld_executor_smoke/<method>/runs.jsonl`
- `outputs/appworld_executor_smoke/<method>/report.json`

Current smoke status:

- `qwen_only` dev smoke submitted as job `75238`.
- Output directory: `outputs/appworld_executor_smoke/qwen_only_dev3`.
- `xzf` Python uses project-local AppWorld dependencies from `.deps/appworld_py310`.

Dev benchmark outputs:

- `outputs/appworld_executor_dev/<method>/runs.jsonl`
- `outputs/appworld_executor_dev/<method>/report.json`

## Execution Order

1. Add tests for SkillRouter-base scoring, serialization, report schema, and checkpoint
   round trip.
2. Implement SkillRouter-base train/eval and sbatch entry points.
3. Run a local lightweight test suite.
4. Submit SkillRouter-base train/eval via `sbatch`.
5. Add tests for executor prompt building, code parsing, fake AppWorld execution, and
   report aggregation.
6. Implement executor module, script, and sbatch helper.
7. Run smoke on 3-5 train/dev tasks for `qwen_only`, `skillrouter_embedding`, and
   `clstr_base`.
8. If smoke passes, run full dev task success benchmark.
9. Only if `clstr_base + qwen` improves dev success over baselines, build train-only
   ACT verified pairs from successful train trajectories or oracle compiled solutions,
   then train real AppWorld CLSTR-act.

## Non-Goals For This Step

- No full-parameter fine-tuning of `SkillRouter-Embedding-0.6B`.
- No dev/test data in training.
- No CLSTR-act training before executor dev evidence justifies ACT data construction.
- No changes outside `/data/home/scyb713/run/xzf/AAAI/autodl-tmp`.
