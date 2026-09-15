# CLSTR Phase 2 Full Training + Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to walk this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is intended for a single autonomous agent session, not subagent dispatch — most steps are blocking I/O (sbatch wait).

**Goal:** Beat SkillRouter-base on AppWorld dev57 (CLSTR success_count > 14) using the Phase 1 bug-fix code that is already merged. If dev57 fails, additionally evaluate test_normal (168 tasks) as a sanity check before stopping.

**Architecture:** Phase 1 fixes (advantage fallback, closed-loop gradient, wrong-completion penalty, prior_residual export default, STOP head first-class) are already merged through commit `02bcf36`. The remaining work is to (a) confirm the in-flight 50-update smoke (`76372`) validates the routing_prior shape fix in real training, (b) submit a single full-scale from-zero training with `UPDATES=200 GROUP_SIZE=8 ENABLE_CLOSED_LOOP_GRADIENT=1`, (c) run full dev57 eval once, then (d) run test_normal once regardless of dev57 outcome.

**Tech Stack:** Python 3.10 (xzf conda env), PyTorch + Qwen3-8B frozen executor, SkillRouter-Embedding-0.6B frozen CLSTR backbone, AppWorld official runtime, SLURM (gpu_h200 partition).

---

## Constraints (from goal)

- Working dir: `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`
- No dev/test split used for training or hyperparameter tuning. Single dev57 run, single test_normal run.
- GPU jobs go through `scripts/sbatch/`, never on login node.
- task-success metric = `execution_ok AND task_completed AND evaluation_success`.
- New output paths must not collide with existing `outputs/` directories.
- Material findings written to `description.md`.
- If unable to fix a problem confidently, STOP and write diagnostics to `description.md` for user review.

---

## File Structure

No new source files are created in this plan — Phase 1 already added them. Edits are limited to:

- Modify: `description.md` (append progress + final result)
- Create: `outputs/appworld_clstr_train_phase2_full_v1/` (new training artefacts)
- Create: `outputs/appworld_multistep_executor_benchmark/phase2_dev57_v1/` (dev57 eval)
- Create: `outputs/appworld_multistep_executor_benchmark/phase2_test_normal_v1/` (test_normal eval)
- Create: `outputs/appworld_multistep_executor_benchmark/phase2_final_comparison/` (comparison markdown)

All new paths use the `phase2_*_v1` prefix to avoid collision with existing Phase 1 outputs.

---

## Task 1: Confirm smoke 76372 passes

**Files:**
- Read: `outputs/appworld_clstr_train_phase1_smoke_50updates/train_report.json`
- Read: `outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/`
- Read: `slurm-76372.out`

- [ ] **Step 1: Check if 76372 is still running**

Run: `squeue -u scyb713 | grep -E "76372|76244"`
Expected: 76372 either still `R` or no longer in queue. Note elapsed time.

- [ ] **Step 2: If 76372 still running and < 90 min elapsed, wait**

Run: `sleep 600` then re-check `squeue`. Repeat up to 4 times (covers ~40 extra minutes; original ETA was ~50 min and job submitted ~23:18 the day before).

If after 4 waits 76372 still running, write blocker note to `description.md` under heading `## 2026-05-25 Phase 2 blocked on 76372` describing elapsed time, then proceed to inspect partial output. Do NOT cancel 76372.

- [ ] **Step 3: When 76372 finishes, read its train_report.json**

Run: `cat outputs/appworld_clstr_train_phase1_smoke_50updates/train_report.json | python3 -m json.tool | head -60`

Expected fields to check:
- `status == "ok"` (no Python exception)
- `updates_completed >= 50`
- `advantage_nonzero_count_total > 0` (proves HRPO actually got signal somewhere)
- `routing_prior_loss` present and finite (proves f258cbb shape fix works in real run)

If any of these are missing or wrong, STOP and write diagnostic to `description.md`. Do not proceed to Task 2.

- [ ] **Step 4: Confirm a usable smoke checkpoint exists**

Run: `ls -la outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/`

Expected: at least one `.pt` file (typically `appworld_clstr_multistep_hrpo-step50.pt`).

If no checkpoint, STOP and write diagnostic.

- [ ] **Step 5: Record smoke result in description.md**

Append a section under `## 2026-05-25 Phase 2 launch — smoke 76372 verdict` listing: job state, advantage_nonzero_count_total, mean_reward, success_rate, routing_prior_loss, checkpoint path, decision (proceed to full / blocked).

Commit:
```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
git add description.md
git commit -m "docs(phase2): record smoke 76372 verdict before launching full training"
```

---

## Task 2: Launch full from-zero training

**Files:**
- Modify: `description.md` (append launch record)
- Read: `scripts/sbatch/run_appworld_clstr_hrpo_train.sh` (no edits — drive via env vars)

- [ ] **Step 1: Confirm no existing phase2 output dir**

Run: `ls outputs/appworld_clstr_train_phase2_full_v1 2>&1`
Expected: `No such file or directory`. If it exists, append `_attempt2` to OUTPUT_DIR and continue.

- [ ] **Step 2: Confirm there are no other big training jobs in queue or running**

Run: `squeue -u scyb713`
Expected: only the smoke (now completed) and small eval jobs. If a training >5h is already queued / running, STOP and append a note to `description.md` — do not stack two big trainings.

- [ ] **Step 3: Submit full training sbatch**

Run:
```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
sbatch --gpus=1 -p gpu_h200 --time=08:00:00 --job-name=phase2_train_v1 \
  --export=ALL,\
MODEL_CONFIG=configs/model/appworld_skillrouter_init_prior_residual.yaml,\
OUTPUT_DIR=outputs/appworld_clstr_train_phase2_full_v1,\
WARMSTART_CLSTR_CKPT=outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/appworld_clstr_multistep_hrpo-step50.pt,\
MAX_TASKS=90,\
GROUP_SIZE=8,\
UPDATES=200,\
TASK_BATCH_SIZE=3,\
TOP_K=16,\
CONTEXT_TOP_K=5,\
MAX_STEPS=5,\
MULTI_STEP=1,\
SKILL_CONTEXT_MODE=safe_metadata,\
LEARNING_RATE=2.0e-5,\
BETA_KL=0.01,\
BETA_ROUTING_PRIOR=0.02,\
ROLLOUT_TEMPERATURE=1.0,\
POLICY_SAMPLING_ALPHA=1.0,\
MAX_INTERACTIONS=10,\
MAX_APIS_PER_APP=12,\
TIMEOUT_SECONDS=60,\
MAX_NEW_TOKENS=768,\
DETACH_POLICY_INPUTS=0,\
ENABLE_CLOSED_LOOP_GRADIENT=1 \
  scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

Expected output: `Submitted batch job <NNNNN>`. Record the job id.

Justification for hyperparameters (from `02bcf36` analysis):
- `MAX_TASKS=90` = full AppWorld train split (per goal)
- `GROUP_SIZE=8` + `UPDATES=200` = 8 × 90 ≈ 720 rollouts per epoch × ~2.2 epochs = ~1600 rollouts (description says 3000-5000 needed for RL convergence; 200 updates × 3 task batch × 8 group ≈ 4800 rollouts)
- `DETACH_POLICY_INPUTS=0` + `ENABLE_CLOSED_LOOP_GRADIENT=1` = let TransitionPredictor/BeliefGate/TransHead receive ACT reward gradient (Phase 1 Bug #2 fix actually activated)
- `BETA_ROUTING_PRIOR=0.02` + `POLICY_SAMPLING_ALPHA=1.0` = keep routing prior anchor present but allow policy to actually explore (residual head trained, not just blended with frozen prior)
- `MAX_STEPS=5` for multi-step trajectories (was 3 in dev10 gate; AppWorld average task is ~8 steps)
- WARMSTART = smoke 76372 step50 checkpoint = continues from a checkpoint that already has the routing_prior shape fix exercised

- [ ] **Step 4: Append launch record to description.md**

Section heading: `## 2026-05-25 Phase 2 full training launched`

Record: job id, sbatch hyperparameters above (literal block), WARMSTART path, OUTPUT_DIR, expected ETA (~4-6h based on 76372 50 updates ≈ 1h scaled).

Commit:
```bash
git add description.md
git commit -m "docs(phase2): launch full training job <NNNNN>, 200 updates × 8 group × 3 task batch"
```

- [ ] **Step 5: Poll training every 30 minutes**

Run on each poll: `squeue -u scyb713 ; tail -5 slurm-<NNNNN>.out`
- Sleep 1800 s between polls.
- Maximum 12 polls (covers 6h). If training still running after 12 polls, write status to description.md and continue polling at 60-minute interval until completion or 12h wallclock cap.
- If `slurm-<NNNNN>.out` shows `RuntimeError` or `OOM`, STOP and inspect.

- [ ] **Step 6: When training completes, verify train_report.json exists and is sane**

Run: `cat outputs/appworld_clstr_train_phase2_full_v1/train_report.json | python3 -m json.tool | head -100`

Mandatory fields:
- `status == "ok"`
- `updates_completed == 200` (or close — if early stop, note it)
- `advantage_nonzero_count_total > 0`
- At least one rollout with `success == True` somewhere in the training rollouts
- Checkpoint file at `outputs/appworld_clstr_train_phase2_full_v1/checkpoints/appworld_clstr_multistep_hrpo-step200.pt` (or `_final.pt` / equivalent latest step)

If any check fails, STOP and write diagnostic.

---

## Task 3: Run full dev57 evaluation

**Files:**
- Read: `scripts/sbatch/run_appworld_multistep_executor_eval.sh`
- Create: `outputs/appworld_multistep_executor_benchmark/phase2_dev57_v1/report.json`

- [ ] **Step 1: Confirm eval sbatch script exists and supports needed env vars**

Run:
```bash
grep -E "TASKS_PATH|MAX_TASKS|CLSTR_CHECKPOINT_PATH|RANKING_MODE|CANDIDATE_SOURCE|RECURRENT_BELIEF|USE_STOP_HEAD" scripts/sbatch/run_appworld_multistep_executor_eval.sh | head -20
```
Expected: all 7 env vars are referenced. If any are missing, STOP and inspect script.

- [ ] **Step 2: Submit dev57 eval**

Run:
```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
sbatch --gpus=1 -p gpu_h200 --time=03:00:00 --job-name=phase2_dev57_v1 \
  --export=ALL,\
METHOD=clstr_multistep,\
CLSTR_MODEL_CONFIG=configs/model/appworld_skillrouter_init_prior_residual.yaml,\
CLSTR_CHECKPOINT_PATH=outputs/appworld_clstr_train_phase2_full_v1/checkpoints/appworld_clstr_multistep_hrpo-step200.pt,\
OUTPUT_DIR=outputs/appworld_multistep_executor_benchmark/phase2_dev57_v1,\
TASKS_PATH=data/appworld_routing/dev_tasks.jsonl,\
MAX_TASKS=57,\
TOP_K=5,\
MAX_STEPS=5,\
RANKING_MODE=policy_blend,\
CANDIDATE_TOP_K=16,\
POLICY_BLEND_ALPHA=0.25,\
CANDIDATE_SOURCE=routing_belief_union_after_update,\
RECURRENT_BELIEF=1,\
USE_STOP_HEAD=1,\
SKILL_CONTEXT_MODE=safe_metadata,\
DEDUPE_CANONICAL_SKILLS=1,\
MAX_AUTH_LIKE_SKILLS=1,\
MAX_INTERACTIONS=10,\
MAX_APIS_PER_APP=12,\
TIMEOUT_SECONDS=60,\
MAX_NEW_TOKENS=768,\
TEMPERATURE=0.0 \
  scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

Record the eval job id.

Justification:
- `MAX_STEPS=5` matches training horizon (Phase 1 ckpt used 3, but full Phase 2 trained with 5)
- `RANKING_MODE=policy_blend` + `POLICY_BLEND_ALPHA=0.25` = use trained policy head but anchored to base routing prior (per Phase 1 Bug #4 default)
- `CANDIDATE_SOURCE=routing_belief_union_after_update` = let m_t belief contribute to candidate pool (Phase 1 closed-loop gradient actually trained belief now)
- `RECURRENT_BELIEF=1` + `USE_STOP_HEAD=1` = enable all Phase 1 features in eval
- `DEDUPE_CANONICAL_SKILLS=1` + `MAX_AUTH_LIKE_SKILLS=1` = packing fix from May-23 still on
- `TEMPERATURE=0.0` = deterministic Qwen output, no run-to-run noise

- [ ] **Step 3: Poll eval every 15 minutes**

Run on each poll: `squeue -u scyb713 ; tail -5 slurm-<NNNNN>.out`
Sleep 900 s. Maximum 12 polls (3h).

- [ ] **Step 4: When eval completes, read report**

Run:
```bash
cat outputs/appworld_multistep_executor_benchmark/phase2_dev57_v1/report.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"success_count={d['success_count']}/{d['task_count']}, success_rate={d['success_rate']:.4f}, exec_failures={d['execution_failures']}, eval_success_count={d['evaluate_success_count']}\")"
```

Record the four numbers in `description.md` under `## 2026-05-25 Phase 2 dev57 result`.

---

## Task 4: Run full test_normal evaluation (regardless of dev57 outcome)

**Files:**
- Create: `outputs/appworld_multistep_executor_benchmark/phase2_test_normal_v1/report.json`

- [ ] **Step 1: Confirm test_normal task file exists**

Run: `wc -l data/appworld_routing/test_normal_tasks.jsonl`
Expected: 168 lines.

- [ ] **Step 2: Submit test_normal eval (same config as dev57 except TASKS_PATH and MAX_TASKS)**

Run:
```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
sbatch --gpus=1 -p gpu_h200 --time=06:00:00 --job-name=phase2_test_normal_v1 \
  --export=ALL,\
METHOD=clstr_multistep,\
CLSTR_MODEL_CONFIG=configs/model/appworld_skillrouter_init_prior_residual.yaml,\
CLSTR_CHECKPOINT_PATH=outputs/appworld_clstr_train_phase2_full_v1/checkpoints/appworld_clstr_multistep_hrpo-step200.pt,\
OUTPUT_DIR=outputs/appworld_multistep_executor_benchmark/phase2_test_normal_v1,\
TASKS_PATH=data/appworld_routing/test_normal_tasks.jsonl,\
MAX_TASKS=168,\
TOP_K=5,\
MAX_STEPS=5,\
RANKING_MODE=policy_blend,\
CANDIDATE_TOP_K=16,\
POLICY_BLEND_ALPHA=0.25,\
CANDIDATE_SOURCE=routing_belief_union_after_update,\
RECURRENT_BELIEF=1,\
USE_STOP_HEAD=1,\
SKILL_CONTEXT_MODE=safe_metadata,\
DEDUPE_CANONICAL_SKILLS=1,\
MAX_AUTH_LIKE_SKILLS=1,\
MAX_INTERACTIONS=10,\
MAX_APIS_PER_APP=12,\
TIMEOUT_SECONDS=60,\
MAX_NEW_TOKENS=768,\
TEMPERATURE=0.0 \
  scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

Record the job id.

- [ ] **Step 3: Poll test_normal every 30 minutes (longer eval)**

Run on each poll: `squeue -u scyb713 ; tail -5 slurm-<NNNNN>.out`
Sleep 1800 s. Maximum 10 polls (5h).

- [ ] **Step 4: When eval completes, read report**

Run:
```bash
cat outputs/appworld_multistep_executor_benchmark/phase2_test_normal_v1/report.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"success_count={d['success_count']}/{d['task_count']}, success_rate={d['success_rate']:.4f}, exec_failures={d['execution_failures']}, eval_success_count={d['evaluate_success_count']}\")"
```

Record in `description.md` under `## 2026-05-25 Phase 2 test_normal result`.

---

## Task 5: Final verdict & comparison report

**Files:**
- Create: `outputs/appworld_multistep_executor_benchmark/phase2_final_comparison/comparison.md`
- Modify: `description.md`

- [ ] **Step 1: Identify SkillRouter-base baseline reports**

Run: `ls outputs/appworld_executor_benchmark/skillrouter_base_dev57_v1/report.json outputs/appworld_executor_benchmark/skillrouter_base_dev10_v2_schema_hints/report.json 2>&1`

Expected: at least the dev57 report exists. If test_normal SkillRouter result doesn't exist yet, only compare dev57 against SkillRouter.

- [ ] **Step 2: Write comparison.md**

Run:
```bash
mkdir -p outputs/appworld_multistep_executor_benchmark/phase2_final_comparison
cat > outputs/appworld_multistep_executor_benchmark/phase2_final_comparison/comparison.md <<'EOF'
# Phase 2 Final Comparison

## dev57

| method | success_count | success_rate |
|---|---:|---:|
| SkillRouter-base | 14/57 | 0.2456 |
| CLSTR Phase 2 (this run) | __DEV57_CLSTR__/57 | __DEV57_RATE__ |

## test_normal

| method | success_count | success_rate |
|---|---:|---:|
| SkillRouter-base (if available) | __SR_TEST__/168 | __SR_TEST_RATE__ |
| CLSTR Phase 2 (this run) | __TEST_CLSTR__/168 | __TEST_RATE__ |

## Verdict

__VERDICT__
EOF
```

Then fill in placeholders by reading the two report.json files (use python3 to substitute via sed or rewrite the file directly).

Decision logic for `__VERDICT__`:
- If dev57 CLSTR > 14: write "GOAL ACHIEVED: dev57 strict-beats SkillRouter. test_normal is reported for context."
- Elif dev57 CLSTR == 14 AND test_normal CLSTR > SkillRouter test_normal (if available): write "GOAL PARTIAL: dev57 tied SkillRouter but test_normal beats it — recommend test_normal as headline result."
- Elif dev57 CLSTR < 14: write "GOAL NOT MET: dev57 = X/57 < 14. test_normal = Y/168. Failure-analysis section below."

- [ ] **Step 3: Append final verdict and per-task hit/miss diff to description.md**

Section heading: `## 2026-05-25 Phase 2 final verdict`

Content:
- dev57 and test_normal numbers (success_count, success_rate, execution_failures, evaluate_success_count)
- Comparison table (re-paste from comparison.md)
- If failure: list top 3 most-common failure modes from `runs.jsonl` (parse for repeated `error` / `execute_output` substrings). Do not propose new fixes — that decision belongs to the user.
- Final decision line: `Verdict: <SUCCESS|FAILURE>. Awaiting user direction.`

- [ ] **Step 4: Commit final artefacts**

Run:
```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
git add description.md outputs/appworld_multistep_executor_benchmark/phase2_final_comparison/
git commit -m "docs(phase2): final dev57 + test_normal verdict"
```

- [ ] **Step 5: Stop and wait for user**

Do not propose ScienceWorld migration or further AppWorld tuning. The user explicitly said: "失败就停"。

---

## Self-Review Notes

1. **Spec coverage check:**
   - Goal "dev57 success_count > 14" → Task 3 measures this
   - Hard constraint "no dev/test for training" → Tasks 2-4 all use `train_tasks.jsonl` for training, `dev_tasks.jsonl` + `test_normal_tasks.jsonl` only for eval, no early-stopping signal from dev
   - Hard constraint "all writes inside autodl-tmp" → all OUTPUT_DIR paths are under the project root
   - Hard constraint "GPU via sbatch" → every step that needs GPU goes through `scripts/sbatch/`
   - Hard constraint "task-success = exec_ok AND task_completed AND eval_success" → uses `report.json::success_count` which already enforces this (per commit `75316` fix)
   - Hard constraint "no path collision with existing outputs" → all new paths use `phase2_*_v1` prefix
   - Hard constraint "material findings to description.md" → Tasks 1/2/3/4/5 all append
   - Hard constraint "if stuck, stop and write diagnostic" → Tasks 1.5, 2.6, 3.4 all include STOP branches
   - Hard constraint "no AppWorld-specific prompt hack" → no prompt edits in this plan
   - User additional rule "also test on test_normal" → Task 4 unconditionally runs test_normal

2. **Placeholder scan:** All sbatch commands are literal — no TBD. comparison.md template has placeholder tokens (`__DEV57_CLSTR__` etc) but Step 2 explicitly says to substitute them by reading report.json. Acceptable.

3. **Type consistency:** All `OUTPUT_DIR` / `CHECKPOINT_PATH` / `MODEL_CONFIG` names match between Task 2 (training output) → Task 3/4 (consume same paths).

4. **Known fragilities:**
   - Step 200 checkpoint file name depends on `clstr/appworld_act_hrpo.py` saving convention. If actual file is `appworld_clstr_multistep_hrpo-step200.pt` vs `-final.pt`, Task 3/4 Step 2 may need adjustment. If checkpoint isn't there, Task 2 Step 6 already catches it.
   - If smoke 76372 step50 checkpoint name differs, Task 2 Step 3 WARMSTART_CLSTR_CKPT may need correction. Task 1 Step 4 verifies it first.
   - If `routing_belief_union_after_update` candidate source isn't actually implemented in eval sbatch, Task 3 Step 1 catches it.
