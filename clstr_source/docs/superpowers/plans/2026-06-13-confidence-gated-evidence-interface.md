# Confidence-Gated Evidence Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a benchmark-agnostic evidence gate so CLSTR retrieved skills are omitted, exposed as schema-only evidence, or exposed with workflow hints based on reliability diagnostics.

**Architecture:** Keep the existing per-skill verifier/compressor in `clstr/appworld_skill_handoff.py`, then add a small deterministic gate that consumes verifier output plus controller diagnostics. The official AppWorld executor receives an evidence prompt only when the gate allows it; `no_evidence_fallback` returns an empty prompt so `_build_user_prompt()` stays equivalent to qwen-only evidence-free prompting.

**Tech Stack:** Python 3, pytest, dataclasses, existing CLSTR AppWorld official executor path.

---

## File Structure

- Create `clstr/evidence_gate.py`: benchmark-agnostic gate dataclass, decision logic, and prompt formatter.
- Modify `clstr/appworld_skill_handoff.py`: keep verifier/compressor responsibilities, call the gate, and return gate diagnostics plus a gated prompt block.
- Modify `scripts/run_appworld_official_executor_eval.py`: pass CLSTR controller diagnostics into the handoff gate and preserve selected-score diagnostics.
- Test `tests/test_evidence_gate.py`: focused unit tests for decision and prompt contracts.
- Modify `tests/test_appworld_skill_handoff.py`: verify handoff fallback and schema/workflow behavior through the existing public function.
- Modify `tests/test_appworld_official_executor_cli.py`: verify `CLSTREvidenceBuilder` emits no prompt when all evidence is suppressed and records gate diagnostics.

## Task 1: Add Evidence Gate Unit Tests

**Files:**
- Create: `tests/test_evidence_gate.py`
- Create later: `clstr/evidence_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
from clstr.evidence_gate import EvidenceGateDecision, gate_verified_skill_handoff


def _decision(skill_id, decision, refs=None, action_refs=None, reasons=None):
    return {
        "skill_id": skill_id,
        "decision": decision,
        "valid_api_refs": refs or [],
        "state_changing_action_apis": action_refs or [],
        "reasons": reasons or [],
    }


def test_gate_falls_back_with_empty_prompt_when_all_selected_evidence_is_suppressed():
    result = gate_verified_skill_handoff(
        selected_skill_ids=["s1"],
        handoff_decisions=[
            _decision("s1", "suppress", refs=["apis.spotify.show_song"], reasons=["low_confidence"]),
        ],
        useful_apis=[],
        state_changing_action_apis=[],
        workflow_hints=[],
        controller_diagnostics={"scores": [0.7]},
    )

    assert isinstance(result.decision, EvidenceGateDecision)
    assert result.decision.decision == "no_evidence_fallback"
    assert result.prompt_block == ""
    assert result.diagnostics["exact_base_executor_fallback"] is True
    assert "all_selected_evidence_suppressed" in result.diagnostics["gate_reasons"]


def test_gate_schema_only_omits_workflow_hints_but_keeps_valid_apis():
    result = gate_verified_skill_handoff(
        selected_skill_ids=["s1"],
        handoff_decisions=[
            _decision("s1", "schema_only", refs=["apis.spotify.show_song_library"]),
        ],
        useful_apis=["apis.spotify.show_song_library"],
        state_changing_action_apis=[],
        workflow_hints=["This hint must not be visible."],
        controller_diagnostics={"scores": [0.8, 0.7], "transition_scores_available": False},
    )

    assert result.decision.decision == "schema_only_evidence"
    assert "[Optional Retrieved Evidence]" in result.prompt_block
    assert "decision: schema_only_evidence" in result.prompt_block
    assert "- apis.spotify.show_song_library" in result.prompt_block
    assert "This hint must not be visible." not in result.prompt_block
    assert "workflow_hints:" not in result.prompt_block
    assert "raw_skill_text_omitted: true" in result.prompt_block


def test_gate_workflow_hints_requires_schema_support_and_high_confidence_signal():
    result = gate_verified_skill_handoff(
        selected_skill_ids=["s1", "s2"],
        handoff_decisions=[
            _decision("s1", "inject_hint", refs=["apis.spotify.search_songs"], action_refs=["apis.spotify.add_to_queue"]),
            _decision("s2", "schema_only", refs=["apis.spotify.add_to_queue"]),
        ],
        useful_apis=["apis.spotify.search_songs", "apis.spotify.add_to_queue"],
        state_changing_action_apis=["apis.spotify.add_to_queue"],
        workflow_hints=["Add each matching song to the queue."],
        controller_diagnostics={"scores": [0.91, 0.86], "transition_scores_available": True},
    )

    assert result.decision.decision == "workflow_hint_evidence"
    assert result.decision.confidence == "high"
    assert "workflow_hints:" in result.prompt_block
    assert "- Add each matching song to the queue." in result.prompt_block
    assert "constraint_guard:" in result.prompt_block
    assert "- preserve user entity/date/source/count/ranking/action constraints" in result.prompt_block
```

- [ ] **Step 2: Run tests to verify they fail because the module does not exist**

Run:

```bash
pytest tests/test_evidence_gate.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'clstr.evidence_gate'`.

## Task 2: Implement `clstr/evidence_gate.py`

**Files:**
- Create: `clstr/evidence_gate.py`
- Test: `tests/test_evidence_gate.py`

- [ ] **Step 1: Add the minimal gate implementation**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_GATE_DECISIONS = {"no_evidence_fallback", "schema_only_evidence", "workflow_hint_evidence"}


@dataclass(frozen=True)
class EvidenceGateDecision:
    decision: str
    confidence: str
    reasons: list[str] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)
    selected_scores: list[float] = field(default_factory=list)
    score_margin: float | None = None
    transition_scores_available: bool = False
    exact_base_executor_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGateResult:
    decision: EvidenceGateDecision
    prompt_block: str
    diagnostics: dict[str, Any]


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    scores: list[float] = []
    for item in value:
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            continue
    return scores


def _selected_scores(controller_diagnostics: dict[str, Any] | None) -> list[float]:
    diagnostics = controller_diagnostics if isinstance(controller_diagnostics, dict) else {}
    for key in ("selected_scores", "scores", "top_scores", "candidate_scores"):
        scores = _as_float_list(diagnostics.get(key))
        if scores:
            return scores
    return []


def _score_margin(scores: list[float]) -> float | None:
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def _transition_available(controller_diagnostics: dict[str, Any] | None) -> bool:
    diagnostics = controller_diagnostics if isinstance(controller_diagnostics, dict) else {}
    for key in ("transition_scores_available", "transition_active", "has_transition_scores"):
        if key in diagnostics:
            return bool(diagnostics.get(key))
    mode = str(diagnostics.get("ranking_mode", ""))
    return "transition" in mode


def _format_prompt(
    *,
    decision: EvidenceGateDecision,
    useful_apis: list[str],
    state_changing_action_apis: list[str],
    workflow_hints: list[str],
    max_chars: int,
) -> str:
    if decision.decision == "no_evidence_fallback":
        return ""
    lines = [
        "[Optional Retrieved Evidence]",
        f"decision: {decision.decision}",
        f"confidence: {decision.confidence}",
        "valid_tools_or_apis:",
    ]
    lines.extend(f"- {api}" for api in useful_apis) if useful_apis else lines.append("- (none)")
    if state_changing_action_apis:
        lines.extend(["state_changing_apis:"])
        lines.extend(f"- {api}" for api in state_changing_action_apis)
    if decision.decision == "workflow_hint_evidence" and workflow_hints:
        lines.extend(["workflow_hints:"])
        lines.extend(f"- {hint}" for hint in workflow_hints)
    lines.extend(
        [
            "constraint_guard:",
            "- preserve user entity/date/source/count/ranking/action constraints",
            "raw_skill_text_omitted: true",
        ]
    )
    return "\n".join(lines)[: int(max_chars)]


def gate_verified_skill_handoff(
    *,
    selected_skill_ids: list[str],
    handoff_decisions: list[dict[str, Any]],
    useful_apis: list[str],
    state_changing_action_apis: list[str],
    workflow_hints: list[str],
    controller_diagnostics: dict[str, Any] | None = None,
    max_chars: int = 1600,
) -> EvidenceGateResult:
    counts = {"inject_hint": 0, "schema_only": 0, "suppress": 0}
    for row in handoff_decisions:
        value = str(row.get("decision", ""))
        if value in counts:
            counts[value] += 1

    reasons: list[str] = []
    schema_count = len(useful_apis)
    if not handoff_decisions:
        reasons.append("no_selected_skills")
    if handoff_decisions and counts["suppress"] == len(handoff_decisions):
        reasons.append("all_selected_evidence_suppressed")
    if schema_count == 0:
        reasons.append("no_schema_valid_evidence")

    scores = _selected_scores(controller_diagnostics)
    margin = _score_margin(scores)
    transition_available = _transition_available(controller_diagnostics)

    if "no_schema_valid_evidence" in reasons or "all_selected_evidence_suppressed" in reasons:
        public_decision = "no_evidence_fallback"
        confidence = "low"
    elif counts["inject_hint"] > 0 and workflow_hints and (transition_available or (scores and scores[0] >= 0.85)):
        public_decision = "workflow_hint_evidence"
        confidence = "high" if transition_available or (margin is not None and margin >= 0.03) else "medium"
        reasons.append("schema_supported_workflow_hints")
    else:
        public_decision = "schema_only_evidence"
        confidence = "medium" if schema_count else "low"
        reasons.append("schema_grounded_evidence_without_high_confidence_workflow")

    decision = EvidenceGateDecision(
        decision=public_decision,
        confidence=confidence,
        reasons=list(dict.fromkeys(reasons)),
        selected_skill_ids=list(selected_skill_ids),
        selected_scores=scores,
        score_margin=margin,
        transition_scores_available=transition_available,
        exact_base_executor_fallback=public_decision == "no_evidence_fallback",
    )
    prompt = _format_prompt(
        decision=decision,
        useful_apis=list(useful_apis),
        state_changing_action_apis=list(state_changing_action_apis),
        workflow_hints=list(workflow_hints),
        max_chars=max_chars,
    )
    return EvidenceGateResult(
        decision=decision,
        prompt_block=prompt,
        diagnostics={
            "evidence_gate": decision.to_dict(),
            "gate_decision": decision.decision,
            "gate_confidence": decision.confidence,
            "gate_reasons": decision.reasons,
            "exact_base_executor_fallback": decision.exact_base_executor_fallback,
            "selected_scores": decision.selected_scores,
            "score_margin": decision.score_margin,
            "transition_scores_available": decision.transition_scores_available,
            "gate_input_counts": counts,
            "prompt_visible_evidence_text": prompt,
        },
    )
```

- [ ] **Step 2: Run gate tests**

Run:

```bash
pytest tests/test_evidence_gate.py -q
```

Expected: PASS.

## Task 3: Integrate Gate Into Verified Skill Handoff

**Files:**
- Modify: `clstr/appworld_skill_handoff.py`
- Modify: `tests/test_appworld_skill_handoff.py`

- [ ] **Step 1: Write failing handoff integration tests**

Append to `tests/test_appworld_skill_handoff.py`:

```python
def test_verified_handoff_gate_omits_prompt_when_all_evidence_is_suppressed_without_schema_fallback():
    handoff = build_verified_skill_handoff(
        instruction="How many unique songs are there across my Spotify song library?",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-rank-play-count",
                "name": "spotify rank play count",
                "description": "Rank songs by play count.",
                "executor_desc": "apis.spotify.show_song_library",
            }
        ],
        valid_api_refs={("spotify", "show_song_library")},
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.4]},
    )

    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["exact_base_executor_fallback"] is True
    assert handoff["prompt_block"] == ""
    assert "[Optional Retrieved Hints]" not in handoff["prompt_block"]


def test_verified_handoff_gate_emits_structured_workflow_evidence_for_high_confidence_hints():
    handoff = build_verified_skill_handoff(
        instruction="Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-filtered-songs-to-queue-3",
                "name": "spotify add filtered songs to queue",
                "description": "Search matching songs and add them to queue.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.show_song, apis.spotify.add_to_queue",
            }
        ],
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.93, 0.84], "transition_scores_available": True},
    )

    assert handoff["gate_decision"] == "workflow_hint_evidence"
    assert "[Optional Retrieved Evidence]" in handoff["prompt_block"]
    assert "[Optional Retrieved Hints]" not in handoff["prompt_block"]
    assert "decision: workflow_hint_evidence" in handoff["prompt_block"]
    assert "Add each matching song to the queue" in handoff["prompt_block"]
```

- [ ] **Step 2: Run integration tests to verify failure before implementation**

Run:

```bash
pytest tests/test_appworld_skill_handoff.py::test_verified_handoff_gate_omits_prompt_when_all_evidence_is_suppressed_without_schema_fallback tests/test_appworld_skill_handoff.py::test_verified_handoff_gate_emits_structured_workflow_evidence_for_high_confidence_hints -q
```

Expected: FAIL with `TypeError: build_verified_skill_handoff() got an unexpected keyword argument 'controller_diagnostics'`.

- [ ] **Step 3: Modify `build_verified_skill_handoff()` signature and call the gate**

Change the function signature to include:

```python
    controller_diagnostics: dict[str, Any] | None = None,
```

Add import near the top:

```python
from clstr.evidence_gate import gate_verified_skill_handoff
```

Replace the current prompt formatting call:

```python
    prompt_block = _format_prompt_block(
        decisions=decisions,
        useful_refs=useful_refs,
        action_refs=action_refs,
        suppressed_refs=suppressed_refs,
        workflow_hints=workflow_hints,
        max_chars=max_chars,
    )
```

with:

```python
    gate = gate_verified_skill_handoff(
        selected_skill_ids=[str(skill.get("skill_id", "")) for skill in skills],
        handoff_decisions=decisions,
        useful_apis=_api_ref_list(useful_refs),
        state_changing_action_apis=_api_ref_list(action_refs),
        workflow_hints=workflow_hints,
        controller_diagnostics=controller_diagnostics,
        max_chars=max_chars,
    )
    prompt_block = gate.prompt_block
```

Add these keys to the returned dictionary:

```python
        "gate_decision": gate.decision.decision,
        "gate_confidence": gate.decision.confidence,
        "gate_reasons": gate.decision.reasons,
        "evidence_gate": gate.decision.to_dict(),
        "exact_base_executor_fallback": gate.decision.exact_base_executor_fallback,
        "prompt_visible_evidence_text": gate.prompt_block,
```

- [ ] **Step 4: Run focused handoff tests**

Run:

```bash
pytest tests/test_appworld_skill_handoff.py -q
```

Expected: Existing old tests that assert `[Optional Retrieved Hints]` need updating because the public prompt contract changed to `[Optional Retrieved Evidence]`; gate-specific tests should pass.

- [ ] **Step 5: Update old handoff expectations to the new prompt contract**

In `tests/test_appworld_skill_handoff.py`, replace prompt header assertions:

```python
assert "Suppressed retrieved hints" in handoff["prompt_block"]
```

with:

```python
assert handoff["gate_decision"] in {"no_evidence_fallback", "schema_only_evidence", "workflow_hint_evidence"}
```

For positive evidence tests, replace:

```python
assert "Potentially useful APIs:" in handoff["prompt_block"]
```

with:

```python
assert "[Optional Retrieved Evidence]" in handoff["prompt_block"]
assert "valid_tools_or_apis:" in handoff["prompt_block"]
```

- [ ] **Step 6: Run focused handoff tests again**

Run:

```bash
pytest tests/test_appworld_skill_handoff.py -q
```

Expected: PASS.

## Task 4: Pass Controller Diagnostics Through Official CLSTR Evidence Builder

**Files:**
- Modify: `scripts/run_appworld_official_executor_eval.py`
- Modify: `tests/test_appworld_official_executor_cli.py`

- [ ] **Step 1: Write failing builder tests**

Append to `tests/test_appworld_official_executor_cli.py`:

```python
def test_clstr_evidence_builder_omits_evidence_prompt_when_gate_falls_back():
    class SuppressedController(_FakeController):
        def select(self, *, task, state_text, steps, top_k):
            self.select_calls.append({"task": task, "state_text": state_text, "steps": steps, "top_k": top_k})
            return _FakeSelection(
                [
                    {
                        "skill_id": "skillx/appworld/spotify-rank-play-count",
                        "name": "spotify rank play count",
                        "description": "Rank songs by play count.",
                        "executor_desc": "apis.spotify.show_song_library",
                    }
                ],
                diagnostics={"selected_scores": [0.2], "transition_scores_available": False},
            )

    builder = cli.CLSTREvidenceBuilder(
        controller=SuppressedController(),
        appworld_root="unused",
        top_k=1,
        valid_api_refs_loader=lambda **_kwargs: {("spotify", "show_song_library")},
    )

    evidence = builder(
        task={"task_id": "task_1", "instruction_text": "How many unique songs are in my Spotify library?", "required_apps": ["spotify"]},
        history=[],
    )

    assert evidence.prompt_text == ""
    assert evidence.diagnostics["gate_decision"] == "no_evidence_fallback"
    assert evidence.diagnostics["exact_base_executor_fallback"] is True
    assert evidence.diagnostics["controller"]["selected_scores"] == [0.2]
```

- [ ] **Step 2: Run the new builder test to verify failure before implementation**

Run:

```bash
pytest tests/test_appworld_official_executor_cli.py::test_clstr_evidence_builder_omits_evidence_prompt_when_gate_falls_back -q
```

Expected: FAIL because `controller_diagnostics` is not passed into `build_verified_skill_handoff()`.

- [ ] **Step 3: Pass controller diagnostics into handoff**

In `CLSTREvidenceBuilder.__call__()`, assign diagnostics before calling handoff:

```python
        controller_diagnostics = getattr(selection, "diagnostics", {})
```

Then pass:

```python
            controller_diagnostics=controller_diagnostics if isinstance(controller_diagnostics, dict) else {},
```

Keep the existing returned diagnostics:

```python
        diagnostics["controller"] = getattr(selection, "diagnostics", {})
```

- [ ] **Step 4: Run official executor CLI tests**

Run:

```bash
pytest tests/test_appworld_official_executor_cli.py -q
```

Expected: PASS.

## Task 5: Verify Prompt Fallback Contract

**Files:**
- Modify if needed: `tests/test_appworld_official_executor_cli.py`
- Existing implementation: `clstr/appworld_official_executor.py`

- [ ] **Step 1: Add prompt-builder regression test**

Append to `tests/test_appworld_official_executor_cli.py`:

```python
def test_official_prompt_omits_retrieved_evidence_guard_when_evidence_is_empty():
    prompt = cli.run_official_react_task.__globals__["_build_user_prompt"](
        {"task_id": "task_1", "instruction_text": "Give me the answer.", "required_apps": []},
        [],
        "",
        api_docs_context="[Task-Relevant AppWorld API Docs]\n- apis.supervisor.complete_task(answer: string optional): Complete task.",
    )

    assert "[Retrieved Evidence Guard]" not in prompt
    assert "[Optional Retrieved Evidence]" not in prompt
    assert "Write the next Python code block." in prompt
```

- [ ] **Step 2: Run the prompt fallback test**

Run:

```bash
pytest tests/test_appworld_official_executor_cli.py::test_official_prompt_omits_retrieved_evidence_guard_when_evidence_is_empty -q
```

Expected: PASS because `_build_user_prompt()` already guards on `if evidence:`.

## Task 6: CPU Verification And Documentation Update

**Files:**
- Modify: `description.md`

- [ ] **Step 1: Run focused CPU tests**

Run:

```bash
pytest tests/test_evidence_gate.py tests/test_appworld_skill_handoff.py tests/test_appworld_official_executor_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader AppWorld prompt tests**

Run:

```bash
pytest tests/test_appworld_executor.py tests/test_appworld_multistep.py -q
```

Expected: PASS or only failures caused by old prompt header expectations. If header-only failures occur, update those assertions from `[Optional Retrieved Hints]` to the new structured `[Optional Retrieved Evidence]` only for paths using `verified_hints`; do not change unrelated executor modes.

- [ ] **Step 3: Record the change in `description.md`**

Add a short dated note:

```markdown
### 2026-06-13: Confidence-gated evidence interface

- Added a benchmark-agnostic step-level gate between CLSTR skill selection and executor-visible evidence.
- `no_evidence_fallback` now omits retrieved evidence entirely, preserving qwen-only prompt behavior for low-confidence or fully suppressed evidence.
- `schema_only_evidence` exposes only valid APIs/tools; `workflow_hint_evidence` additionally exposes short compressor hints when schema support and confidence are sufficient.
- This is a reliability layer before any RL gate policy; it avoids task-id or AppWorld-family-specific rules.
```

- [ ] **Step 4: Run diff checks**

Run:

```bash
git diff --check -- clstr/evidence_gate.py clstr/appworld_skill_handoff.py scripts/run_appworld_official_executor_eval.py tests/test_evidence_gate.py tests/test_appworld_skill_handoff.py tests/test_appworld_official_executor_cli.py description.md
```

Expected: no whitespace errors.

## Task 7: Optional Gate-1 Runtime Smoke, Only After CPU Tests Pass

**Files:**
- No code changes.

- [ ] **Step 1: Prepare a 9-task disagreement subset command**

Use the existing `--task_ids` support:

```bash
TASK_IDS=50e1ac9_1,50e1ac9_2,50e1ac9_3,57c3486_2,68ee2c9_3,383cbac_1,383cbac_3,6171bbc_3,396c5a2_1
```

- [ ] **Step 2: Submit only if the user approves a small GPU/AppWorld job**

Do not run this on the login/storage node. Submit via sbatch, one job only. The expected comparison is against:

- qwen-only dev57: `outputs/appworld_official_executor_dev57/qwen14b_api_compressor_constraints1`
- previous CLSTR Stage4 dev57: `outputs/appworld_official_executor_dev57/clstr_stage4_act_api_compressor_guard1`

- [ ] **Step 3: Gate criteria**

Proceed to dev10 or dev57 only if the 9-task subset shows one of:

- qwen-only-only regressions are reduced without losing most Stage4-only wins;
- execution failures decrease materially;
- success is no worse and evidence fallback decisions are correctly recorded.

## Self-Review

- Spec coverage: Tasks 1-5 implement fallback omission, schema-only evidence, workflow-hint evidence, diagnostics, controller score/margin/transition signals, and prompt contract. Task 7 covers the required disagreement-subset runtime gate without introducing task-specific code.
- Placeholder scan: No task uses TBD/TODO/fill-in placeholders. Runtime commands and expected outcomes are explicit.
- Type consistency: Public gate names are `EvidenceGateDecision`, `EvidenceGateResult`, and `gate_verified_skill_handoff`; these are used consistently in tests and integration tasks.
- Non-goals: No task adds task-id rules, app-family patches, Qwen training, or RL.
