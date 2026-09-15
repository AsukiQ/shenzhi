# API/Schema-Aware Candidate Path Design

## Goal

Fix the current AppWorld Stage4 routing diagnosis by aligning the evaluated skill pool with the positive label space, then add a benchmark-safe API/schema candidate path that improves candidate recall without turning AppWorld ground truth into executor prompts.

## Root Cause

The latest `6171bbc_3` Stage0 gate used `data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl`, but its positive labels are `skillx/appworld/...` rows. Those AppWorld SkillX rows are absent from that unified pool. A `0/6` top350 result is therefore not evidence that Stage0 cannot retrieve the skills; the target rows are not in the search space.

The existing dynamic AppWorld route already supports `base skill pool + AppWorld SkillX append`, but the available dynamic pool was built from the older v4.1b base. We need the same dynamic append path for the current v4.2 nowweak pool.

## Design

1. Pool membership must be explicit in retrieval audits.
   - `stage0_retrieval_coverage_audit` will accept an optional `skill_pool_path`.
   - Rows whose positive skill is absent from the pool are counted separately as `positive_absent_from_pool`.
   - Only in-pool positives that miss top-K are emitted as Stage0 correction rows by default.

2. Dynamic AppWorld skill append is the default AppWorld candidate-space fix.
   - Reuse `build_appworld_current_route_dataset(..., base_skill_pool_path=...)`.
   - Build `data/clstr_appworld_dynamic_v4_2_nowweak_append/` from the current v4.2 base pool and `data/appworld_skill_pool/skill_pool.jsonl`.
   - Keep base rows first so existing checkpoint skill table rows remain prefix-compatible.
   - Mark appended AppWorld rows as `is_appended_after_checkpoint=true` and low coverage, so CLSTR transition/belief heads do not over-trust unseen dynamic skills.

3. API/schema evidence is candidate support, not final ranking.
   - API refs may be used to audit whether an in-pool skill can satisfy a correction target.
   - For train-only correction data, solution API refs are allowed.
   - For dev/test executor runs, official solution refs must not enter prompts or candidate construction.

4. Routing-only validation precedes any executor or full training job.
   - First validate that target AppWorld SkillX positives exist in the dynamic pool.
   - Then run dynamic routing audit/export on a small sample.
   - Only if pool membership and routing-only metrics are meaningful should we consider Stage4 online executor runs.

## Non-Goals

- Do not re-run Stage0/Stage1/Stage2 full training in this step.
- Do not use AppWorld dev/test solution traces as training data.
- Do not make AppWorld-specific prompt hacks part of the core CLSTR method.
- Do not claim Stage4 RL works until routing-only label-space checks pass.

## Success Criteria

- The audit report distinguishes absent positives from top-K misses.
- `6171bbc_3` positive SkillX rows are present in the current dynamic v4.2 pool.
- Existing dynamic skill coverage weighting remains intact for newly appended rows.
- CPU tests cover the new audit behavior.
- No Slurm/GPU job is submitted until the pool-space audit is clean.
