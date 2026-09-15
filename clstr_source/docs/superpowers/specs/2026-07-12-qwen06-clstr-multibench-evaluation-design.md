# Qwen 0.6B CLSTR Multi-Benchmark Evaluation Design

## Goal

After Stage0/1/2/4 training completes, evaluate one immutable final CLSTR checkpoint chain on ToolBench-G3, tau2, ToolSandbox, and ALFWorld. Preserve fair frozen-input comparisons with SR/ToolREx while separately reporting CLSTR-native causal or closed-loop behavior.

## Evaluation Contract

The primary comparable layer uses the exact read-only corpora already frozen for the Qwen SR/ToolREx evaluation:

- ToolBench-G3: 1,362 rows and 27,486 declared skills;
- tau2: 13,907 rows and 45 declared skills;
- ToolSandbox: 115 rows and 31 declared skills;
- ALFWorld offline: 5,858 rows and 2,155 declared action skills.

These runs report strict next-tool/action Recall@1, Recall@5, MRR, positive coverage, source/retained denominators, known-versus-appended skill counts, and complete per-row predictions. They are routing metrics, not task success.

The CLSTR-native layer uses existing repository evaluators:

- ToolBench-G3 full trajectory routing;
- tau2 causal action-sequence routing;
- ToolSandbox causal scenario routing;
- ALFWorld official valid-seen and valid-unseen closed-loop success.

Only ALFWorld closed-loop is labeled task success. Native routing results use their own exact denominators and are not silently merged with the frozen SR/ToolREx comparison table.

## Checkpoint Identity

Every job resolves the same run root and requires:

- release-safe Stage0 selection and lineage;
- Stage1 quality gate and lineage;
- Stage2 quality gate, checkpoint, and lineage;
- Stage4 quality gate, checkpoint, and derived lineage;
- frozen Qwen backbone in every checkpoint configuration.

The resolver writes a self-contained final-chain manifest with absolute paths, content digests, parent lineage digests, model path, training skill-pool digest, and a chain digest. Benchmark jobs reject any mismatch instead of choosing a checkpoint by filename alone.

## Frozen Routing Adapter

Frozen rows are adapted without changing their query or candidate semantics:

- `raw_state` becomes the checkpoint-prompted `state_text`;
- `positive_skill_id` becomes `next_skill_id`;
- row-local `candidate_skill_ids` become the legal candidate inventory;
- ToolBench rows without row-local candidates use the complete frozen 27,486-skill pool;
- `row_id`, benchmark fields, and source provenance are retained.

The training skill pool remains the prefix of the model skill table. Benchmark skills already present in that prefix are reused; unseen tau2, ToolSandbox, or ALFWorld skills are appended in frozen order and initialized through the existing skill-text encoder. Reports expose appended-skill counts explicitly.

Frozen rows contain no lossless recurrent observation/action prefixes. Their primary score is therefore the model's learned initial-belief route, with `memory_active_rows=0` and exact zero-history fallback recorded. No synthetic replay prefix is fabricated from text-only history.

## Native Causal Evaluation

Existing native evaluators retain their current causal semantics and official source builders. They receive the resolved Stage0/2/4 paths and Qwen training skill pool from the final-chain manifest. Their reports remain separate because source rows and candidate protocols differ from the frozen comparable layer.

ALFWorld closed-loop uses the official environment, valid-seen and valid-unseen splits, admissible actions, real observations, and the final Stage4 checkpoint. It emits resumable episode records and success metrics for every official episode.

## Orchestration

The submitter has smoke and full scopes. Smoke runs use bounded rows/episodes and must pass identity, denominator, finite-score, prediction-completeness, no-gold-injection, and output-schema gates. Full jobs are submitted only from the accepted smoke-gate digest.

All outputs live under one CLSTR evaluation root with separate directories for frozen routing, native routing, and ALFWorld closed-loop. A final aggregator records job IDs/states, checkpoint-chain digest, corpus manifest digests, metrics, and failures without converting failures into zero scores.

## Failure Handling

Jobs fail closed on checkpoint-chain drift, corpus-manifest drift, missing positives, unknown candidate IDs after expansion, incomplete predictions, non-finite scores, altered denominators, trainable Qwen backbone parameters, or accidental gold insertion. Existing outputs may be resumed only when their full identity manifest matches.

## Non-Goals

- Building new official closed-loop executors for ToolBench, tau2, or ToolSandbox;
- changing benchmark prompts, candidates, splits, or official ALFWorld scoring;
- adapting or fine-tuning CLSTR on benchmark validation/test rows;
- modifying the concurrently active SR/ToolREx worktree;
- presenting routing recall as end-to-end task success.
