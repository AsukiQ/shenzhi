# CLSTR SkillNet Registry Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the SkillNet skill registry as CLSTR's canonical ALFWorld skill space while preserving official ALFWorld admissible commands as the only executable closed-loop action set.

**Architecture:** SkillNet `SKILL.md` files provide stable skill identities and rich skill text. ALFWorld admissible commands are mapped into those skill identities for CLSTR transition/routing/belief supervision, but final environment actions are always selected by reranking `admissible_commands`. This fixes the current mismatch where training operates in pseudo-skill id space while evaluation scores dynamic action strings with partly duplicated mapping code.

**Tech Stack:** Python, PyTorch, existing CLSTR model/training/eval modules, local `/tmp/SkillNet/experiments/src/skills`, pytest.

---

## Scope And Constraints

- Only modify `/root/autodl-tmp/clstr`.
- Do not modify `/root/autodl-tmp/skillrouter`, `/root/autodl-tmp/alfworld_repo`, or `/tmp/SkillNet`.
- Do not use ALFWorld valid/test for training, rollout, skill enrichment, or mapping calibration.
- Do not replace ALFWorld admissible commands with SkillNet skills at eval time. SkillNet skills are latent/controller labels; ALFWorld commands remain executable actions.
- Do not claim offline recall, CE, MRR, or Q-success AUC as closed-loop success.

## Terminology

**SkillNet registry skill:** A stable skill document from SkillNet, for example `alfworld/alfworld-object-picker`, with name, description, body, workflow, and environment metadata.

**ALFWorld admissible command:** A legal text action returned by the official environment at one step, for example `take apple 1 from diningtable 1`.

**Pseudo skill:** A CLSTR adapter label that maps a dynamic text command or action template to a stable skill-space id. In the preferred implementation, pseudo skills should resolve to SkillNet registry ids whenever possible. Example: `take apple 1 from diningtable 1` maps to `alfworld/alfworld-object-picker`.

Pseudo skills are needed because ALFWorld does not provide a reusable skill registry. CLSTR, however, trains `skill_head`, `trans_head`, routing, belief, and STOP around a finite skill/action space. The pseudo-skill adapter is the bridge between dynamic ALFWorld commands and CLSTR's latent skill-space modules.

## Design Decision

Use SkillNet as the canonical ALFWorld skill registry, not as an executable policy. The controller remains:

```text
goal / observation / history / admissible_commands
    -> CLSTR state + belief
    -> map each admissible command to SkillNet skill id
    -> compute policy, Q_success, transition, belief, STOP, planner, loop scores
    -> choose one exact admissible command
    -> env.step(chosen_command)
```

This preserves the CLSTR method structure and avoids turning Qwen or SkillNet into a direct action generator.

## Current Evidence

- `clstr/skillnet_aux_rebuild.py` already loads SkillNet `SKILL.md` files and preserves `body`.
- `clstr/model.py` already supports `skill_text_format="clstr_enriched"`, including `body`, action templates, examples, preconditions, and effects.
- Current `data/clstr_dagger_expert_corrected_train/skills.jsonl` contains 121 ALFWorld-style SkillNet-derived skills but has `body_count=0`.
- Current action-to-skill mapping is duplicated between `clstr/full_base_preprocess.py` and `clstr/alfworld_eval.py`, which makes training/eval drift possible.
- Current controller transition/belief eval scores use goal-cosine deltas, not the trained `trans_head` / `skill_head` scoring functions.

## File Structure

- Create: `clstr/alfworld_action_skills.py`
  - Single canonical ALFWorld command -> SkillNet skill id mapper.
  - Owns normalization, preferred skill id order, confidence/reason metadata, and distribution audit helpers.
- Modify: `clstr/full_base_preprocess.py`
  - Replace local `_infer_skill_id(..., benchmark="alfworld")` logic with the canonical mapper.
- Modify: `clstr/dagger_preprocess.py`
  - Use the canonical mapper when converting expert-corrected rollout to DAgger rows.
- Modify: `clstr/alfworld_eval.py`
  - Remove local `_skill_id_for_alfworld_action`.
  - Use canonical mapper for candidate command metadata.
  - Score transition/belief with trained heads instead of goal-cosine deltas.
- Modify: `clstr/skill_embedding.py`
  - Make SkillNet body enrichment the default path when the local SkillNet root exists.
  - Keep train-only action examples and hard negatives from train rows only.
- Modify: `scripts/build_clstr_enriched_skill_embeddings.py`
  - Generate enriched `skills.jsonl` and report paths suitable for DAgger training.
- Modify: `scripts/run_clstr_dagger_success_train.py`
  - Accept enriched skills and `--skill_text_format clstr_enriched`.
- Modify: `scripts/run_clstr_loss_ablation.py`
  - Pass the enriched skill format into ablation training.
  - Resume completed ablations instead of rerunning finished configs.
- Tests:
  - `tests/test_alfworld_action_skills.py`
  - `tests/test_full_base_preprocess.py`
  - `tests/test_dagger_preprocess.py`
  - `tests/test_alfworld_eval.py`
  - `tests/test_skill_embedding.py`
  - `tests/test_loss_ablation.py`

---

### Task 1: Canonical ALFWorld Skill Mapping

**Files:**
- Create: `clstr/alfworld_action_skills.py`
- Test: `tests/test_alfworld_action_skills.py`

- [ ] **Step 1: Write failing mapping tests**

```python
from clstr.alfworld_action_skills import map_alfworld_action_to_skill_id


def test_maps_common_alfworld_actions_to_skillnet_ids():
    skill_ids = {
        "alfworld/alfworld-location-navigator",
        "alfworld/alfworld-object-picker",
        "alfworld/alfworld-object-placer",
        "alfworld/alfworld-receptacle-opener",
        "alfworld/alfworld-clean-object",
        "alfworld/alfworld-heat-object-with-appliance",
        "alfworld/alfworld-object-cooler",
        "alfworld/alfworld-object-state-inspector",
    }
    assert map_alfworld_action_to_skill_id("go to fridge 1", skill_ids).skill_id == "alfworld/alfworld-location-navigator"
    assert map_alfworld_action_to_skill_id("take apple 1 from table 1", skill_ids).skill_id == "alfworld/alfworld-object-picker"
    assert map_alfworld_action_to_skill_id("put apple 1 in fridge 1", skill_ids).skill_id == "alfworld/alfworld-object-placer"
    assert map_alfworld_action_to_skill_id("open cabinet 1", skill_ids).skill_id == "alfworld/alfworld-receptacle-opener"
    assert map_alfworld_action_to_skill_id("clean apple 1 with sinkbasin 1", skill_ids).skill_id == "alfworld/alfworld-clean-object"
    assert map_alfworld_action_to_skill_id("heat mug 1 with microwave 1", skill_ids).skill_id == "alfworld/alfworld-heat-object-with-appliance"
    assert map_alfworld_action_to_skill_id("cool tomato 1 with fridge 1", skill_ids).skill_id == "alfworld/alfworld-object-cooler"
    assert map_alfworld_action_to_skill_id("look", skill_ids).skill_id == "alfworld/alfworld-object-state-inspector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_alfworld_action_skills.py`

Expected: FAIL because `clstr.alfworld_action_skills` does not exist or the mapper is not implemented.

- [ ] **Step 3: Implement mapper**

Implement a frozen result dataclass with fields:

```python
skill_id: str | None
confidence: str
reason: str
normalized_action: str
```

Use deterministic verb/template rules and preferred SkillNet ids. If a preferred id is absent from `known_skill_ids`, fall back to the next preferred id. If none are present, return `skill_id=None`, `confidence="unmapped"`.

- [ ] **Step 4: Run mapping tests**

Run: `pytest -q tests/test_alfworld_action_skills.py`

Expected: PASS.

---

### Task 2: Replace Duplicate Training/Eval Mapping

**Files:**
- Modify: `clstr/full_base_preprocess.py`
- Modify: `clstr/dagger_preprocess.py`
- Modify: `clstr/alfworld_eval.py`
- Test: `tests/test_full_base_preprocess.py`
- Test: `tests/test_dagger_preprocess.py`
- Test: `tests/test_alfworld_eval.py`

- [ ] **Step 1: Write failing consistency tests**

Add tests asserting the same action and skill pool produce the same skill id in:

- full-base preprocessing
- DAgger preprocessing
- ALFWorld eval component scoring metadata

Use actions such as `go to fridge 1`, `take apple 1 from table 1`, `put apple 1 in fridge 1`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_full_base_preprocess.py tests/test_dagger_preprocess.py tests/test_alfworld_eval.py
```

Expected: FAIL until all paths import the canonical mapper.

- [ ] **Step 3: Replace local mapping logic**

Remove or deprecate:

- `clstr.full_base_preprocess._infer_skill_id` ALFWorld-specific branch
- `clstr.alfworld_eval._skill_id_for_alfworld_action`

All ALFWorld mapping calls should import `map_alfworld_action_to_skill_id`.

- [ ] **Step 4: Run consistency tests**

Run:

```bash
pytest -q tests/test_full_base_preprocess.py tests/test_dagger_preprocess.py tests/test_alfworld_eval.py
```

Expected: PASS.

---

### Task 3: Enriched SkillNet Skill Embeddings

**Files:**
- Modify: `clstr/skill_embedding.py`
- Modify: `scripts/build_clstr_enriched_skill_embeddings.py`
- Test: `tests/test_skill_embedding.py`

- [ ] **Step 1: Write failing tests**

Assert that enriched SkillNet rows include:

- `body`
- `positive_action_examples`
- `negative_action_examples`
- `action_templates`
- `preconditions`
- `effects`
- `valid_or_test_used_for_skill_enrichment=false`

- [ ] **Step 2: Run tests to verify they fail or expose missing coverage**

Run: `pytest -q tests/test_skill_embedding.py`

Expected: FAIL if enrichment does not preserve body or train-only grounding fields.

- [ ] **Step 3: Implement enrichment defaults**

Use `/tmp/SkillNet/experiments/src/skills` as a read-only source when present. If it is absent, keep the existing `skills.jsonl` rows and report `skills_with_body=0` rather than failing training.

- [ ] **Step 4: Generate enriched skill data**

Run:

```bash
python scripts/build_clstr_enriched_skill_embeddings.py \
  --skills_path data/clstr_dagger_expert_corrected_train/skills.jsonl \
  --train_path data/clstr_dagger_expert_corrected_train/train.jsonl \
  --output_dir data/clstr_dagger_expert_corrected_train_enriched \
  --report_output_dir outputs/clstr_dagger_expert_corrected_train_enriched \
  --skillnet_root /tmp/SkillNet/experiments/src/skills
```

Expected:

- `data/clstr_dagger_expert_corrected_train_enriched/skills.jsonl`
- `outputs/clstr_dagger_expert_corrected_train_enriched/skill_embedding_report.json`
- report shows no valid/test usage.

---

### Task 4: Controller Transition/Belief Scoring Alignment

**Files:**
- Modify: `clstr/alfworld_eval.py`
- Test: `tests/test_alfworld_eval.py`
- Test: `tests/test_closed_loop_controller.py`

- [ ] **Step 1: Write failing controller scorer test**

Create a fake model with instrumented `trans_head` and `skill_head`. The test must fail if `make_controller_component_scorer` computes transition/belief scores from goal-cosine deltas instead of calling those trained heads.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_alfworld_eval.py::test_controller_component_scorer_uses_trained_transition_and_belief_heads
```

Expected: FAIL until the scorer calls the trained heads.

- [ ] **Step 3: Implement scorer alignment**

For each admissible action:

1. Encode action text for policy/Q-success.
2. Map action text to canonical SkillNet skill id.
3. Build candidate skill embeddings from the model skill table where mapping exists.
4. Compute transition prior with the current transition module.
5. Compute `transition_scores` using `model.trans_head(pred, candidate_skill_or_action_embs)`.
6. Compute belief update using gate.
7. Compute `belief_scores` using `model.skill_head(candidate_action_embs, belief)`.
8. Keep `q_success_scores`, planner scores, loop penalties, and STOP diagnostics.

The scorer must record mapping metadata in trace diagnostics:

```json
{
  "candidate_action": "take apple 1 from table 1",
  "mapped_skill_id": "alfworld/alfworld-object-picker",
  "mapping_confidence": "high",
  "mapping_reason": "alfworld_pickup_command"
}
```

- [ ] **Step 4: Run scorer tests**

Run:

```bash
pytest -q tests/test_alfworld_eval.py tests/test_closed_loop_controller.py
```

Expected: PASS.

---

### Task 5: Resume-Safe Loss Ablation With Enriched Skills

**Files:**
- Modify: `scripts/run_clstr_loss_ablation.py`
- Test: `tests/test_loss_ablation.py`

- [ ] **Step 1: Write failing resume test**

Assert an ablation with existing checkpoint, train report, offline report, and eval metrics is loaded into summary without rerunning training or eval.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_loss_ablation.py::test_ablation_runner_skips_completed_eval_when_resuming`

Expected: FAIL until resume logic exists.

- [ ] **Step 3: Implement resume support**

Add:

- `--resume_existing`
- `--skill_text_format`
- existing artifact loader for `train_report.json`, `offline_report.json`, eval `metrics.json`

Pass `skill_text_format` to `run_clstr_full_base_train`.

- [ ] **Step 4: Run loss ablation tests**

Run: `pytest -q tests/test_loss_ablation.py`

Expected: PASS.

---

### Task 6: Re-train And Re-evaluate After Alignment

**Files:**
- Modify only scripts/reports as needed.
- Outputs under `outputs/` and `data/`.

- [ ] **Step 1: Train DAgger + Q_success with enriched skills**

Run:

```bash
python scripts/run_clstr_dagger_success_train.py \
  --train_path data/clstr_dagger_expert_corrected_train/train.jsonl \
  --skills_path data/clstr_dagger_expert_corrected_train_enriched/skills.jsonl \
  --output_dir outputs/clstr_dagger_success_train_enriched \
  --max_steps 1000 \
  --batch_size 2 \
  --skill_text_format clstr_enriched
```

Expected:

- `outputs/clstr_dagger_success_train_enriched/checkpoints/clstr_full_base-step1000.pt`
- `outputs/clstr_dagger_success_train_enriched/train_report.json`
- report includes `skill_text_format=clstr_enriched`.

- [ ] **Step 2: Run valid_seen gate eval**

Run the existing ALFWorld eval script for 20 episodes / 50 steps with the enriched checkpoint.

Expected:

- `outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen/metrics.json`
- `outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen/run.jsonl`
- `outputs/alfworld_eval/clstr_dagger_qsuccess_enriched_gate_valid_seen/controller_diagnostic.json`

- [ ] **Step 3: Resume remaining ablations**

Run ablations with the enriched skill table and `--resume_existing`, keeping old pre-fix ablations separate from post-fix ablations.

Expected:

- `outputs/clstr_loss_ablation_enriched/ablation_summary.json`
- `outputs/clstr_loss_ablation_enriched/ablation_table.md`
- `outputs/clstr_loss_ablation_enriched/recommended_slim_loss_config.json`

---

## Success Criteria

- Training/eval ALFWorld action-to-skill mapping imports one canonical implementation.
- Enriched skill table contains SkillNet body text and train-only action grounding.
- Controller transition/belief scores are produced by trained heads, not goal-cosine deltas.
- Ablation summary distinguishes old pre-fix diagnostics from post-fix aligned results.
- ALFWorld closed-loop metrics are reported as real environment metrics only.
- If metrics still fail, blocker report identifies whether the remaining bottleneck is Qwen planner, expert correction, Q_success, SkillNet grounding, controller calibration, or long-horizon ALFWorld planning.

## Relation To `docs/clstr_alfworld_issues_and_fixes.md`

This plan directly addresses:

- P2: transition/belief eval scoring detached from trained objectives.
- P4: training/eval vocabulary mismatch.
- P6: hardcoded penalty can dominate when real head signals are weak.

It also supports P1 by making verified-pair and next-skill supervision operate on a stable SkillNet-aligned action space.
