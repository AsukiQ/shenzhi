# CLSTR SkillNet Registry + ALFWorld Pseudo-Skill Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the SkillNet skill registry as CLSTR's canonical ALFWorld skill space while preserving ALFWorld admissible commands as the only executable closed-loop action space.

**Architecture:** SkillNet skills provide stable skill ids, descriptions, and bodies for CLSTR skill embeddings. ALFWorld admissible commands are mapped to those SkillNet skill ids through a single shared adapter, then CLSTR ranks the original admissible commands with policy, Q_success, transition, belief, STOP confidence, planner match, and loop penalties. This keeps CLSTR method structure intact without pretending ALFWorld text commands are native SkillNet executable skills.

**Tech Stack:** Python, PyTorch, CLSTR model heads, ALFWorld official text env, SkillNet local checkout at `/tmp/SkillNet`, pytest.

---

## Current Diagnosis

ALFWorld does not ship a reusable skill registry. It exposes per-state `admissible_commands`, such as `go to sinkbasin 1`, `take apple 1 from table 1`, and `heat mug 1 with microwave 1`. CLSTR, however, is built around a stable skill space used by `skill_head`, `trans_head`, `transition`, `belief`, routing, and STOP policy.

The existing code partially bridges this gap with pseudo skills, but the bridge is weak:

- `data/clstr_dagger_expert_corrected_train/skills.jsonl` has 121 SkillNet-style ALFWorld skills, but currently lacks skill bodies.
- ALFWorld action-to-skill mapping is duplicated across preprocessing and eval paths.
- Controller transition/belief scores in eval are not using the same head logits trained by `L_trans_skill_ce`.
- Remaining ablations should be paused until these alignment issues are fixed, otherwise loss conclusions are not reliable.

## What "Pseudo Skill" Means

A pseudo skill is a stable CLSTR skill label assigned to an ALFWorld text action template. It is not an executable ALFWorld action by itself.

Examples:

| ALFWorld admissible command | Pseudo / SkillNet skill id |
|---|---|
| `go to fridge 1` | `alfworld/alfworld-location-navigator` |
| `open cabinet 2` | `alfworld/alfworld-receptacle-opener` |
| `take apple 1 from table 1` | `alfworld/alfworld-object-picker` |
| `put apple 1 in fridge 1` | `alfworld/alfworld-object-placer` |
| `heat mug 1 with microwave 1` | `alfworld/alfworld-heat-object-with-appliance` |

We need this adapter because CLSTR learns in skill space, while ALFWorld evaluates exact text actions. The final closed-loop action must always be the exact admissible command, not the pseudo skill id.

## Key Decisions

1. SkillNet registry is used as the canonical ALFWorld skill registry for embedding and skill-space supervision.
2. ALFWorld admissible commands remain the closed-loop action candidates and the only values sent to `env.step`.
3. Pseudo skills are adapter labels, not official ALFWorld expert actions and not paper main-table skill-registry evidence.
4. SkillNet bodies should be included in skill embeddings through `skill_text_format=clstr_enriched`.
5. Training and eval must use one shared action-to-skill mapper.
6. Transition and belief eval scores must use trained CLSTR heads, not goal-cosine proxy scores.

## Issue Mapping From `docs/clstr_alfworld_issues_and_fixes.md`

### P0: `L_policy` Is Supervised CE Instead Of Outcome HRPO

Status: correct for the original CLSTR policy-loss spec, but supervised CE remains valid as a DAgger warm-start objective.

Current full-base / DAgger training optimizes cross-entropy over admissible actions using official expert action labels. That is not the same as markdown §6.1 GRPO/HRPO-style outcome supervision. For the current goal, this is acceptable only as expert-corrected warm-start. It should not be reported as online RL, GRPO, HRPO, or final CLSTR policy optimization.

Countermeasure:

- Keep expert-corrected CE as `L_policy_supervised_warm_start`.
- Record `grpo_policy_loss_used=false` unless an actual rollout-based HRPO stage runs.
- Add a later optional stage named `online_hrpo_after_dagger`, using train split only, grouped rollout rewards, KL-to-reference, and action-level log probabilities over `admissible_commands`.
- Do not let HRPO replace the DAgger expert labels before P2/P4 are fixed; otherwise outcome learning will optimize a controller whose transition/belief signals are still miswired.
- In reports, distinguish:
  - `L_policy_supervised_ce`: expert-corrected DAgger imitation.
  - `L_policy_hrpo`: grouped outcome optimization from train-split rollouts.

Acceptance signal:

- `train_report.json` truthfully reports whether HRPO was used.
- If HRPO runs, report contains `grpo_policy_loss_used=true`, `on_policy_rollout_used=true`, `kl_to_reference`, grouped reward statistics, and train-only rollout provenance.
- If HRPO does not run, final report states that policy was DAgger CE warm-start, not RL fine-tuning.

### P1: CLSTR-act `L_act` Is Not In The ALFWorld Main Training Path

Status: correct and important for claiming CLSTR-act.

The current DAgger/Q_success path has `L_trans_skill_ce`, but that is not identical to markdown `action_loss(model, verified_pairs)`. `L_act` requires verified next-skill pairs, exact or replayed belief `m_t`, and next-candidate supervision. Therefore the current model can be described as DAgger + Q_success + component-complete controller, but not full CLSTR-act unless this branch is added.

Countermeasure:

- Build ALFWorld verified-pair rows from train split only:
  - `state_before`
  - `action_at_t` mapped through the canonical SkillNet pseudo skill mapper
  - `obs_at_t`
  - `candidates_next`
  - `a_next_plus`
  - `replay_prefix`
  - `was_in_raw_topk`
- Add an `L_act` branch after P2/P4:
  - use exact `m_t` snapshot when available;
  - otherwise use full replay prefix with current model;
  - do not use approximate `subspace_obs` for paper-claim runs.
- Keep `L_trans_skill_ce` as a supervised next-skill transition objective, but do not call it full `L_act`.

Acceptance signal:

- `train_report.json` includes `action_loss_used > 0`, `retrieval_recall_raw`, and `m_t_source_counts`.
- The report separates `L_trans_skill_ce` from `L_act`.
- Verified-pair data manifest states `split=train` and does not include ALFWorld valid/test.

### P2: Transition / Belief Scores Are Disconnected From Training

Status: correct and high priority.

Current eval computes transition and belief contribution as a cosine delta against goal text. That is not the same objective as `L_trans_skill_ce`, where `trans_head` learns to rank next skills. This means transition/belief can appear "trained" offline but still not participate meaningfully in action selection.

Countermeasure:

- For each admissible command, map it to a canonical SkillNet pseudo skill id.
- Use the mapped skill/action candidate embedding with trained `trans_head` / `skill_head` scoring.
- Keep trace fields for raw and calibrated scores so we can verify whether transition/belief changes top-1 decisions.

Acceptance signal:

- `controller_diagnostic.json` reports nontrivial transition/belief variance.
- `run.jsonl` contains steps where `policy top-1 != final top-1`.
- The diagnostic states `component_score_source=trained_clstr_heads`, not `goal_cosine_delta`.

### P3: Frozen Qwen Pooled Embedding May Lose Planning Signal

Status: plausible bottleneck, not a direct code bug.

The Qwen direct baseline sees a chat-formatted prompt with explicit available actions and can use token-level planning. The CLSTR-Qwen external encoder path pools hidden states into a fixed vector, then relies on small CLSTR heads. That can discard action-list reasoning that Qwen direct uses.

Countermeasure:

- Do not treat this as the first root-cause fix.
- First repair P2/P4 so the CLSTR heads are evaluated correctly.
- Keep Qwen as frozen planner/teacher or intent encoder where useful, but do not let Qwen direct generation replace CLSTR final action selection.

Acceptance signal:

- If P2/P4 fixes still leave CLSTR-Qwen below Qwen direct, report the remaining gap as an encoder/planner compression bottleneck.
- Do not claim Qwen direct as CLSTR.

### P4: Training Vocabulary And Eval Vocabulary Are Not Fully Aligned

Status: correct and directly addressed by this SkillNet pseudo-skill plan.

Training losses such as `L_trans_skill_ce`, routing, and future `L_act` operate in a stable skill-id space. ALFWorld eval chooses among dynamic text commands. Without a single adapter between those spaces, offline skill metrics can look good while closed-loop action selection remains weak.

Countermeasure:

- Create one canonical `clstr/alfworld_action_skills.py`.
- Make preprocessing, DAgger conversion, full-base training data construction, controller eval, diagnostics, and leakage audit import the same `alfworld_action_to_skill_id` function.
- Treat SkillNet ALFWorld skills as the canonical skill registry.
- Keep final action selection over original `admissible_commands`; the pseudo skill id is only an internal adapter label.
- Add a vocabulary-alignment diagnostic comparing:
  - train `skill_id` distribution;
  - eval mapped candidate skill distribution;
  - unmapped candidate rate;
  - KL divergence or another explicit distribution gap metric.

Acceptance signal:

- A unit test proves preprocessing and eval use the same mapper object.
- `controller_diagnostic.json` reports `action_skill_mapper=canonical_skillnet_pseudoskill`.
- `unmapped_admissible_action_rate` is reported; if high, output a blocker instead of claiming transition/belief failure.

### P5: STOP Is Not A First-Class ALFWorld Candidate

Status: partially correct, but this plan must respect the project constraint that STOP cannot fake environment success or done.

The markdown method treats STOP as a first-class `[K+1]` policy action. The current ALFWorld controller mostly uses STOP as a per-candidate score/penalty. That can make STOP weak or hard to interpret. However, for ALFWorld eval, STOP must not be reported as env completion, because the official environment decides success/done.

Countermeasure:

- Keep official `env.done` / `won` as the only closed-loop success source.
- Add a STOP confidence diagnostic branch:
  - compute one state-level `stop_probability`;
  - record it in `run.jsonl`;
  - record whether STOP would have ranked above the best admissible action under a diagnostic `[K+1]` policy.
- For evaluation safety, default behavior is:
  - do not convert STOP into success;
  - if early termination is enabled for a diagnostic run, mark it as `agent_stop_terminated=true` and `env_done=false` unless env also returned done.
- Keep STOP BCE training from legal `done` labels only.

Acceptance signal:

- `run.jsonl` contains `stop_probability`, `stop_rank_vs_actions`, and `agent_stop_terminated`.
- Metrics separate `env_done`, `env_success`, and `agent_stop`.
- Reports do not claim STOP-induced termination as ALFWorld success.

### P6: Hardcoded Penalties Can Dominate

Status: correct as a calibration risk.

The current controller uses fixed penalties for actions like `help`, `inventory`, repeated `look`, reversible actions, and loops. These help suppress obviously bad behavior, but if true CLSTR head scores are weak or disconnected, penalties can become the dominant policy and push the agent into `go` / `examine` loops.

Countermeasure:

- Do not tune penalties before P2/P4 are fixed.
- After trained head scores are aligned, run a penalty ablation:
  - normal penalties
  - half penalties
  - zero penalties
- Keep penalty contribution visible in every component trace.

Acceptance signal:

- With reduced penalties, success does not collapse purely into repeated info actions.
- If zero-penalty performance is still unusable, report that model head signal remains insufficient.

注意我们不再考虑原先的P7，alfworld现在alfworld应该当做重要的任务

## Implementation Tasks

### Task 1: Create Shared ALFWorld Action-to-Skill Mapper

**Files:**
- Create: `clstr/alfworld_action_skills.py`
- Modify: `clstr/full_base_preprocess.py`
- Modify: `clstr/alfworld_eval.py`
- Test: `tests/test_alfworld_action_skills.py`

- [ ] **Step 1: Write failing mapper tests**

```python
from clstr.alfworld_action_skills import alfworld_action_to_skill_id


def test_alfworld_action_to_skill_id_maps_common_commands():
    known = {
        "alfworld/alfworld-location-navigator",
        "alfworld/alfworld-object-picker",
        "alfworld/alfworld-object-placer",
        "alfworld/alfworld-receptacle-opener",
        "alfworld/alfworld-heat-object-with-appliance",
    }
    assert alfworld_action_to_skill_id("go to fridge 1", known) == "alfworld/alfworld-location-navigator"
    assert alfworld_action_to_skill_id("take apple 1 from table 1", known) == "alfworld/alfworld-object-picker"
    assert alfworld_action_to_skill_id("put apple 1 in fridge 1", known) == "alfworld/alfworld-object-placer"
    assert alfworld_action_to_skill_id("open cabinet 2", known) == "alfworld/alfworld-receptacle-opener"
    assert alfworld_action_to_skill_id("heat mug 1 with microwave 1", known) == "alfworld/alfworld-heat-object-with-appliance"


def test_preprocess_and_eval_import_same_mapper():
    import clstr.full_base_preprocess as preprocess
    import clstr.alfworld_eval as eval_mod

    assert preprocess.alfworld_action_to_skill_id is eval_mod.alfworld_action_to_skill_id
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest -q tests/test_alfworld_action_skills.py
```

Expected: fails because `clstr.alfworld_action_skills` does not exist or eval/preprocess still use duplicated local functions.

- [ ] **Step 3: Implement shared mapper**

Create `clstr/alfworld_action_skills.py` with one canonical function:

```python
from __future__ import annotations

from typing import Iterable


def alfworld_action_to_skill_id(action: str, known_skill_ids: Iterable[str] | None = None) -> str | None:
    text = str(action or "").strip().lower()
    known = set(str(item) for item in known_skill_ids or [])
    candidates: list[str] = []
    if text.startswith(("take ", "pick up ", "grab ")):
        candidates = ["alfworld/alfworld-object-picker", "alfworld/alfworld-object-retriever"]
    elif text.startswith(("go to ", "go ")):
        candidates = ["alfworld/alfworld-location-navigator", "alfworld/alfworld-receptacle-navigator"]
    elif text.startswith("open "):
        candidates = ["alfworld/alfworld-receptacle-opener", "alfworld/alfworld-open-receptacle"]
    elif text.startswith("close "):
        candidates = ["alfworld/alfworld-receptacle-closer"]
    elif text.startswith(("put ", "place ", "move ")):
        candidates = ["alfworld/alfworld-object-placer", "alfworld/alfworld-object-storer"]
    elif text.startswith(("clean ", "wash ")):
        candidates = ["alfworld/alfworld-clean-object"]
    elif text.startswith(("heat ", "warm ", "cook ")):
        candidates = ["alfworld/alfworld-heat-object-with-appliance", "alfworld/alfworld-object-heater"]
    elif text.startswith(("cool ", "chill ")):
        candidates = ["alfworld/alfworld-object-cooler"]
    elif text.startswith(("look", "examine", "inspect")):
        candidates = ["alfworld/alfworld-object-state-inspector", "alfworld/alfworld-environment-scanner"]
    elif text in {"inventory", "help"}:
        candidates = ["alfworld/alfworld-environment-scanner"]
    for candidate in candidates:
        if not known or candidate in known:
            return candidate
    return None
```

- [ ] **Step 4: Replace local mapper copies**

In `clstr/full_base_preprocess.py`, import `alfworld_action_to_skill_id` and call it inside `_infer_skill_id` for `benchmark == "alfworld"`.

In `clstr/alfworld_eval.py`, remove `_skill_id_for_alfworld_action` and import `alfworld_action_to_skill_id`.

- [ ] **Step 5: Run mapper tests**

Run:

```bash
pytest -q tests/test_alfworld_action_skills.py
```

Expected: pass.

### Task 2: Build Enriched SkillNet Skill Registry

**Files:**
- Modify: `clstr/skill_embedding.py`
- Modify: `scripts/build_clstr_enriched_skill_embeddings.py`
- Modify: `clstr/model.py`
- Test: `tests/test_skill_embedding.py`

- [ ] **Step 1: Verify body preservation tests exist**

Run:

```bash
pytest -q tests/test_skill_embedding.py
```

Expected before implementation in a fresh tree: fail if SkillNet body is not preserved or serializer excludes body/action templates.

- [ ] **Step 2: Generate enriched registry**

Run:

```bash
python scripts/build_clstr_enriched_skill_embeddings.py \
  --skills_path data/clstr_dagger_expert_corrected_train/skills.jsonl \
  --train_path data/clstr_dagger_expert_corrected_train/train.jsonl \
  --output_dir data/clstr_dagger_expert_corrected_train_enriched \
  --report_output_dir outputs/clstr_dagger_expert_corrected_train_enriched \
  --skillnet_root /tmp/SkillNet/experiments/src/skills
```

Expected outputs:

- `data/clstr_dagger_expert_corrected_train_enriched/skills.jsonl`
- `outputs/clstr_dagger_expert_corrected_train_enriched/skill_embedding_report.json`

- [ ] **Step 3: Inspect enrichment report**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("outputs/clstr_dagger_expert_corrected_train_enriched/skill_embedding_report.json").read_text())
print(report)
assert report["valid_or_test_used_for_skill_enrichment"] is False
assert report["skills_with_positive_action_examples"] > 0
PY
```

Expected: assertions pass. `skills_with_body` should be greater than zero when the local SkillNet checkout contains matching ALFWorld skill bodies.

### Task 3: Align Controller Transition / Belief Scoring With Trained Heads

**Files:**
- Modify: `clstr/alfworld_eval.py`
- Modify: `clstr/closed_loop_controller.py`
- Test: `tests/test_alfworld_eval.py`
- Test: `tests/test_closed_loop_controller.py`

- [ ] **Step 1: Write failing controller scorer test**

Add a test that builds a tiny model with instrumented `trans_head` and `skill_head`, calls `make_controller_component_scorer`, and asserts both heads are called for candidate scoring. The test should fail on the current goal-cosine implementation.

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
pytest -q tests/test_alfworld_eval.py::test_controller_component_scorer_uses_trained_trans_and_skill_heads
```

Expected: fail because scorer uses goal-cosine delta rather than candidate head logits.

- [ ] **Step 3: Implement trained-head component scoring**

Use this data flow:

1. Encode state text into `h`.
2. Build `m_obs = _skill_memory(model, h)`.
3. Encode admissible command texts as candidate action embeddings.
4. Map each admissible command to a SkillNet pseudo skill id with `alfworld_action_to_skill_id`.
5. Use mapped skill ids for transition update labels when available.
6. Score candidate actions with trained CLSTR heads:
   - `transition_scores`: `trans_head(pred, candidate_embs)` or a candidate-specific projection compatible with the model.
   - `belief_scores`: `skill_head(candidate_embs, belief)`.
7. Record `component_score_source="trained_clstr_heads"`.

- [ ] **Step 4: Preserve trace semantics**

Ensure `run.jsonl` still records:

- `policy_score`
- `q_success_score`
- `transition_score`
- `belief_score`
- `stop_logit`
- `stop_probability`
- `planner_score`
- `loop_penalty`
- `action_prior_penalty`
- `final_score`
- `chosen_reason`

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest -q tests/test_alfworld_eval.py tests/test_closed_loop_controller.py
```

Expected: pass.

### Task 4: Re-Train One Enriched DAgger + Q_success Gate Model

**Files:**
- Modify: `scripts/run_clstr_dagger_success_train.py`
- Output: `outputs/clstr_dagger_success_train_enriched/`

- [ ] **Step 1: Train with enriched skills**

Run:

```bash
python scripts/run_clstr_dagger_success_train.py \
  --train_path data/clstr_dagger_expert_corrected_train/train.jsonl \
  --skills_path data/clstr_dagger_expert_corrected_train_enriched/skills.jsonl \
  --output_dir outputs/clstr_dagger_success_train_enriched \
  --max_steps 1000 \
  --batch_size 4 \
  --skill_text_format clstr_enriched
```

Expected:

- `outputs/clstr_dagger_success_train_enriched/checkpoints/clstr_full_base-step1000.pt`
- `outputs/clstr_dagger_success_train_enriched/train_report.json`
- report states `skill_text_format=clstr_enriched`.

### Task 5: Run Valid-Seen Gate Eval Before Resuming Ablation

**Files:**
- Output: `outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen/`

- [ ] **Step 1: Run ALFWorld valid_seen gate eval**

Run:

```bash
python scripts/run_alfworld_structured_controller_eval.py \
  --checkpoint outputs/clstr_dagger_success_train_enriched/checkpoints/clstr_full_base-step1000.pt \
  --output_dir outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen \
  --split valid_seen \
  --max_episodes 20 \
  --max_steps 50 \
  --controller_mode policy_plus_transition_belief_stop_loop_penalty
```

Expected:

- `metrics.json`
- `run.jsonl`
- `controller_diagnostic.json`

- [ ] **Step 2: Inspect diagnostic for component use**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
metrics = json.loads(Path("outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen/metrics.json").read_text())
diag = json.loads(Path("outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen/controller_diagnostic.json").read_text())
print(metrics)
print(diag)
PY
```

Expected: diagnostic identifies trained-head component scoring. If success remains zero or below previous best, write a blocker report rather than claiming improvement.

### Task 6: Resume Loss Ablation Only After Alignment Fixes

**Files:**
- Modify: `scripts/run_clstr_loss_ablation.py`
- Modify: `clstr/loss_ablation.py`
- Output: `outputs/clstr_loss_ablation_enriched/`

- [ ] **Step 1: Add resume/skip behavior**

If an ablation already has `train_report.json`, `offline_report.json`, and valid_seen `metrics.json`, skip it. If train exists but eval is missing, only run eval.

- [ ] **Step 2: Run enriched ablation in a new output directory**

Run:

```bash
python scripts/run_clstr_loss_ablation.py \
  --train_path data/clstr_dagger_expert_corrected_train/train.jsonl \
  --skills_path data/clstr_dagger_expert_corrected_train_enriched/skills.jsonl \
  --output_dir outputs/clstr_loss_ablation_enriched \
  --skill_text_format clstr_enriched
```

Expected:

- `outputs/clstr_loss_ablation_enriched/ablation_summary.json`
- `outputs/clstr_loss_ablation_enriched/ablation_table.md`
- `outputs/clstr_loss_ablation_enriched/recommended_slim_loss_config.json`

### Task 7: Update Reports and Description

**Files:**
- Modify: `description.md`
- Create or update: `outputs/clstr_dagger_qsuccess_comparison_table.md`
- Create or update: `outputs/clstr_dagger_qsuccess_comparison_summary.json`
- Create or update: `outputs/clstr_dagger_qsuccess_paper_report.md`
- Create or update: `outputs/clstr_dagger_qsuccess_paper_report.json`

- [ ] **Step 1: Record architectural clarification**

Add to `description.md`:

- SkillNet registry is used for skill-space representation.
- ALFWorld admissible commands remain executable actions.
- Pseudo skills are adapter labels.
- P2/P4 fixed before final loss ablation.
- P3 remains an encoder compression risk if Qwen direct outperforms CLSTR-Qwen.
- P6 penalties are calibration, not model signal.

- [ ] **Step 2: Generate final comparison**

Include:

- routing init baseline
- 0.6B CLSTR best
- Qwen3-8B direct reference
- previous structured CLSTR-Qwen
- DAgger + Q_success enriched model
- best ablation model
- slim loss model

### Task 8: Final Verification

**Files:**
- Tests under `tests/`
- Git status for protected repos

- [ ] **Step 1: Run required targeted tests**

Run:

```bash
pytest -q tests/test_alfworld_expert_labeler.py tests/test_qwen_teacher_rollout.py tests/test_dagger_preprocess.py
pytest -q tests/test_success_value.py tests/test_closed_loop_controller.py tests/test_structured_controller.py
pytest -q tests/test_loss_ablation.py tests/test_structured_train.py tests/test_alfworld_eval.py
pytest -q tests/test_alfworld_action_skills.py tests/test_skill_embedding.py
```

Expected: all pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all pass or failures are documented as unrelated hard blockers with exact stack traces.

- [ ] **Step 3: Confirm protected repos unchanged**

Run:

```bash
git -C /root/autodl-tmp/skillrouter status --short
git -C /root/autodl-tmp/alfworld_repo status --short
```

Expected: no modifications caused by this work.

## Reporting Rules

Do not claim improvement unless `metrics.json` shows closed-loop improvement. Do not call offline CE, recall, MRR, Q_success AUC, or controller diagnostic a closed-loop success metric. If ALFWorld remains weak after this plan, report whether the bottleneck is skill grounding, Qwen pooled representation, controller calibration, loss design, or ALFWorld long-horizon planning.
