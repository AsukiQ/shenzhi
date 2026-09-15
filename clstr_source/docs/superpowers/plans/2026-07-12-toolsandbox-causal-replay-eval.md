# ToolSandbox Causal Replay Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eval-only causally aligned ToolSandbox replay mode and compare it with static and legacy replay using the same checkpoints.

**Architecture:** A ToolSandbox-local pure helper converts ordered milestone rows into explicit replay-prefix events based on `next_skill_id`. The existing evaluator consumes those explicit prefixes unchanged; launchers expose the mode and reports persist its identity.

**Tech Stack:** Python, PyTorch, pytest through Slurm, Bash, JSON.

---

### Task 1: Causal replay adapter

**Files:**
- Modify: `clstr/toolsandbox_route_eval.py`
- Test: `tests/test_toolsandbox_route_eval.py`

- [x] Add a failing test proving a two-step trajectory replays the first row's `next_skill_id`, not its `skill_id` or `action_text`, and uses the second row state as weak post-action observation.
- [x] Run the focused test through Slurm and verify the expected RED failure.
- [x] Implement `_attach_toolsandbox_causal_replay_prefixes()` with strict trajectory adjacency, bounded prefix length, explicit provenance, and skill-index validation.
- [x] Run focused tests through Slurm and verify GREEN.

### Task 2: Evaluation mode and report identity

**Files:**
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `scripts/run_toolsandbox_full_clstr_route_eval.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_native_route_eval.sh`
- Test: `tests/test_toolsandbox_route_eval.py`

- [x] Add failing tests for `legacy_auto` and `corrected_causal` mode validation and report fields.
- [x] Thread `toolsandbox_replay_mode` through CLI and launcher; attach corrected prefixes only in corrected mode.
- [x] Run ToolSandbox and launcher regressions through Slurm.
- [x] Commit the eval-only implementation.

### Task 3: Same-checkpoint diagnostic

- [x] Submit a corrected-causal full ToolSandbox evaluation in a fresh output directory.
- [x] Compare static, legacy native, and corrected native metrics over exactly 115 rows and the same mean 3.617 candidate pool.
- [x] Record results and provenance in `finalwork/clstr_benchmark_results.md`.

## Result

- Static MRR: `0.627619`
- Legacy native MRR: `0.603271` (`-0.024348` vs static)
- Corrected-causal native MRR: `0.611677` (`-0.015942` vs static)
- Corrected causal replay recovers `+0.008406` MRR but does not eliminate the remaining memory harm.
