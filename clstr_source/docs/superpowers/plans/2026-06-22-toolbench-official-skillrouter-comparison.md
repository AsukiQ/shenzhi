# ToolBench Official SkillRouter Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal fair ToolBench-G3 comparison: Full CLSTR vs SkillRouter based on the official SkillRouter repository.

**Architecture:** Export ToolBench-G3 into the official SkillRouter `eval_core` layout, run official SkillRouter inference/evaluation on the same held-out rows, then evaluate existing Full CLSTR on exactly the same rows. Do not add broad ablations in this first experiment.

**SkillRouter baseline boundary:** Frozen SkillRouter must call the official repository entrypoint `python -m src.run_open_model_eval`. The official public repo currently exposes evaluation/model-loading code but no training entrypoint; if we add a finetuned baseline, report it as **official-compatible SkillRouter finetune** rather than official author-provided training.

**Tech Stack:** Python, pytest, Slurm, official SkillRouter repo at `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter`, cached SkillRouter 0.6B embedding/reranker models, existing CLSTR Stage0/Stage2/Stage4 checkpoints.

---

### Task 1: Build Official SkillRouter-Compatible ToolBench Data

**Files:**
- Create: `clstr/toolbench_skillrouter_official.py`
- Create: `scripts/build_toolbench_g3_skillrouter_official_eval.py`
- Test: `tests/test_toolbench_skillrouter_official.py`

- [x] Write tests for deterministic split and official `eval_core` fields.
- [x] Implement exporter that reads `data/toolbench_g3_routing_eval/{queries,qrels}.jsonl` and `data/toolbench_g3/skills.jsonl`.
- [x] Write `tasks.jsonl`, `relevance.json`, `easy/skills.jsonl`, and `hard/skills.jsonl` for `train` and `eval`.
- [x] Verify with pytest and a tiny generated fixture.

### Task 2: Official SkillRouter Frozen Smoke

**Files:**
- Create: `scripts/sbatch/run_toolbench_g3_official_skillrouter_eval.sh`

- [x] Run official repo `src.run_open_model_eval` on a small exported eval sample.
- [ ] Use cached local models:
  - `.cache/hf_models/SkillRouter-Embedding-0.6B`
  - `.cache/hf_models/SkillRouter-Reranker-0.6B`
- [x] Confirm output has retrieval and pipeline metrics.

Smoke result:
- job `94199`
- output `outputs/toolbench_g3_official_skillrouter_comparison/official_skillrouter_frozen_smoke/summary.json`
- official pipeline ran successfully on 16 eval queries and 256 smoke skills.

### Task 3: Finetuned SkillRouter Baseline Gate

**Files:**
- Reuse or extend: `clstr/skillrouter_style.py`
- Create: `scripts/run_toolbench_g3_skillrouter_finetune.py`
- Create: `scripts/sbatch/run_toolbench_g3_skillrouter_finetune.sh`

- [ ] Keep official model/query/skill serialization.
- [ ] If official repo has no train script, report this explicitly and implement only an official-compatible finetune baseline.
- [ ] Smoke train on a small train split before any full job.
- [ ] Evaluate on the same exported eval split with official-style metrics.

### Task 4: Full CLSTR Same-Split Eval

**Files:**
- Create or reuse a thin eval wrapper for existing Full CLSTR Stage0+Stage2+Stage4 on exported ToolBench eval query ids.

- [x] Use existing full CLSTR checkpoints; do not retrain CLSTR.
- [x] Evaluate recall@1, recall@5, MRR on the same held-out trajectory next-skill rows.
- [x] Write one markdown table.

Full trajectory fair-comparison result:
- shared trajectory eval data: `outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data`
- official SkillRouter full-pool result: `outputs/toolbench_g3_official_skillrouter_comparison/official_skillrouter_trajectory_full_fullpool/summary.json`
- CLSTR result: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_trajectory_full_fullpool/real_topm_eval_report.json`
- report: `docs/toolbench_g3_official_skillrouter_comparison_2026-06-22.md`

Capped smoke result:
- shared trajectory eval data: `outputs/toolbench_g3_official_skillrouter_comparison/trajectory_capped256_fullpool_data`
- official SkillRouter full-pool capped result: `outputs/toolbench_g3_official_skillrouter_comparison/official_skillrouter_trajectory_capped256_fullpool/summary.json`
- CLSTR capped result: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_trajectory_capped256_fullpool/real_topm_eval_report.json`

### Task 5: Decision Gate

- [ ] If SkillRouter finetuned beats Full CLSTR, stop and report the failure honestly.
- [ ] If Full CLSTR beats SkillRouter finetuned, keep this as the main ToolBench result.
- [ ] Do not add ablations until the main two-method comparison is complete.
