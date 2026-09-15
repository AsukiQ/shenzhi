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
            _decision(
                "s1",
                "inject_hint",
                refs=["apis.spotify.search_songs"],
                action_refs=["apis.spotify.add_to_queue"],
            ),
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
