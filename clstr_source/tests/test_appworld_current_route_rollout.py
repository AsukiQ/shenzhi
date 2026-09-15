from pathlib import Path

from clstr.appworld_current_route_rollout import (
    CurrentRouteRolloutLogger,
    build_current_route_rollout_trace,
)
from clstr.appworld_routing import read_jsonl


def _step(**overrides):
    row = {
        "step_idx": 0,
        "code": "apis.spotify.show_song_library()",
        "execute_output": "[{}]",
        "execution_attempted": True,
        "execution_ok": True,
        "preflight_ok": True,
        "completion_precheck": {"ok": True, "reason": "ok"},
        "task_completed": False,
        "evaluation_success": False,
        "skill_evidence": {
            "selected_skill_ids": ["skill/a", "skill/b"],
            "evidence_gate": {"decision": "schema_only_evidence", "confidence": "medium"},
            "controller": {
                "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
                "ranking_mode": "transition_blend",
                "candidate_source": "routing",
                "selected_scores": [0.9, 0.7],
                "filtered_duplicate_skills": 2,
            },
        },
    }
    row.update(overrides)
    return row


def test_current_route_trace_marks_success_as_positive_but_not_gradient_ready_without_logprobs():
    trace = build_current_route_rollout_trace(
        {
            "method": "clstr_multistep",
            "task_id": "task_success",
            "query_id": "query_success",
            "user_goal": "Like songs from followed artists.",
            "success": True,
            "task_completed": True,
            "evaluation_success": True,
            "steps": [_step(task_completed=True, evaluation_success=True)],
        }
    )

    assert trace["schema_version"] == "current_route_rollout.v1"
    assert trace["outcome"]["label"] == "success"
    assert trace["outcome"]["policy_signal"] == "positive"
    assert trace["outcome"]["reward"] == 1.0
    assert trace["decisions"][0]["selected_skill_ids"] == ["skill/a", "skill/b"]
    assert trace["decisions"][0]["candidate_skill_ids"] == ["skill/a", "skill/b", "skill/c"]
    assert trace["decisions"][0]["training_ready"] is False
    assert "missing_policy_logprobs" in trace["decisions"][0]["training_blockers"]


def test_current_route_trace_keeps_wrong_completion_ambiguous_not_clstr_negative():
    trace = build_current_route_rollout_trace(
        {
            "method": "clstr_multistep",
            "task_id": "task_wrong",
            "success": False,
            "task_completed": True,
            "evaluation_success": False,
            "steps": [_step(task_completed=True, evaluation_success=False, execute_output="completed")],
        }
    )

    assert trace["outcome"]["label"] == "wrong_completion"
    assert trace["outcome"]["policy_signal"] == "ambiguous_do_not_penalize_clstr"
    assert trace["executor_summary"]["wrong_completion_count"] == 1
    assert trace["decisions"][0]["step_label"] == "wrong_completion"


def test_current_route_trace_marks_explicit_routing_miss_as_clstr_negative():
    trace = build_current_route_rollout_trace(
        {
            "method": "clstr_multistep",
            "task_id": "task_miss",
            "success": False,
            "task_completed": False,
            "evaluation_success": False,
            "steps": [
                _step(
                    skill_evidence={
                        "selected_skill_ids": ["skill/noise"],
                        "controller": {
                            "candidate_skill_ids": ["skill/noise"],
                            "positive_in_candidates": False,
                            "routing_miss": True,
                        },
                    },
                    execute_output="No useful API call was made.",
                )
            ],
        }
    )

    assert trace["outcome"]["label"] == "routing_miss"
    assert trace["outcome"]["policy_signal"] == "negative"
    assert trace["outcome"]["routing_miss_evidence"]["step_idx"] == 0


def test_current_route_trace_keeps_state_text_and_marks_ready_when_logprobs_present():
    trace = build_current_route_rollout_trace(
        {
            "method": "clstr_multistep",
            "task_id": "task_ready",
            "success": True,
            "task_completed": True,
            "evaluation_success": True,
            "steps": [
                _step(
                    skill_evidence={
                        "state_text": "[User Goal]\nFind songs.\n[Execution History]\nNo previous execution steps.",
                        "selected_skill_ids": ["skill/a"],
                        "controller": {
                            "candidate_skill_ids": ["skill/a", "skill/b"],
                            "selected_log_probs": [-0.2],
                            "policy_log_probs": [-0.2, -1.7],
                            "selected_candidate_local_indices": [0],
                        },
                    },
                    task_completed=True,
                    evaluation_success=True,
                )
            ],
        }
    )

    decision = trace["decisions"][0]
    assert decision["state_text"].startswith("[User Goal]")
    assert decision["selected_log_probs"] == [-0.2]
    assert decision["policy_log_probs"] == [-0.2, -1.7]
    assert decision["selected_candidate_local_indices"] == [0]
    assert decision["training_ready"] is True
    assert trace["training_ready"] is True
    assert trace["training_blockers"] == []


def test_current_route_trace_records_handoff_packing_diagnostics_for_stage4_audit():
    trace = build_current_route_rollout_trace(
        {
            "method": "clstr_multistep",
            "task_id": "task_visible_packing",
            "success": False,
            "task_completed": True,
            "evaluation_success": False,
            "steps": [
                _step(
                    task_completed=True,
                    evaluation_success=False,
                    skill_evidence={
                        "state_text": "state",
                        "selected_skill_ids": ["skill/create", "skill/read", "skill/add"],
                        "visible_skill_limit": 2,
                        "visible_skill_ids": ["skill/create", "skill/read"],
                        "rescued_action_skill_ids": ["skill/add"],
                        "useful_apis": ["apis.spotify.show_song", "apis.spotify.create_playlist"],
                        "state_changing_action_apis": [
                            "apis.spotify.create_playlist",
                            "apis.spotify.add_song_to_playlist",
                        ],
                        "controller": {
                            "candidate_skill_ids": ["skill/create", "skill/read", "skill/add"],
                            "selected_log_probs": [-0.1],
                            "policy_log_probs": [-0.1, -1.0, -1.2],
                            "selected_candidate_local_indices": [0],
                        },
                    },
                )
            ],
        }
    )

    decision = trace["decisions"][0]
    assert decision["visible_skill_limit"] == 2
    assert decision["visible_skill_ids"] == ["skill/create", "skill/read"]
    assert decision["rescued_action_skill_ids"] == ["skill/add"]
    assert decision["useful_apis"] == ["apis.spotify.show_song", "apis.spotify.create_playlist"]
    assert decision["state_changing_action_apis"] == [
        "apis.spotify.create_playlist",
        "apis.spotify.add_song_to_playlist",
    ]


def test_current_route_trace_carries_previous_action_and_observation_for_transition_training():
    trace = build_current_route_rollout_trace(
        {
            "method": "clstr_multistep",
            "task_id": "task_transition",
            "success": True,
            "task_completed": True,
            "evaluation_success": True,
            "steps": [
                _step(
                    step_idx=0,
                    execute_output="Execution successful.",
                    skill_evidence={
                        "state_text": "step 0 state",
                        "selected_skill_ids": ["skill/a"],
                        "controller": {
                            "candidate_skill_ids": ["skill/a", "skill/b"],
                            "selected_log_probs": [-0.2],
                            "policy_log_probs": [-0.2, -1.7],
                            "selected_candidate_local_indices": [0],
                        },
                    },
                ),
                _step(
                    step_idx=1,
                    task_completed=True,
                    evaluation_success=True,
                    skill_evidence={
                        "state_text": "step 1 state",
                        "selected_skill_ids": ["skill/b"],
                        "controller": {
                            "candidate_skill_ids": ["skill/a", "skill/b"],
                            "selected_log_probs": [-0.3],
                            "policy_log_probs": [-1.3, -0.3],
                            "selected_candidate_local_indices": [1],
                        },
                    },
                ),
            ],
        }
    )

    second = trace["decisions"][1]
    assert second["previous_selected_skill_ids"] == ["skill/a"]
    assert second["previous_execute_output"] == "Execution successful."


def test_current_route_rollout_logger_appends_and_flushes_jsonl(tmp_path):
    path = tmp_path / "current_route_rollouts.jsonl"
    logger = CurrentRouteRolloutLogger(path)

    logger.append(
        {
            "method": "clstr_multistep",
            "task_id": "task_1",
            "success": True,
            "task_completed": True,
            "evaluation_success": True,
            "steps": [_step(task_completed=True, evaluation_success=True)],
        }
    )
    logger.append(
        {
            "method": "qwen_only",
            "task_id": "task_2",
            "success": False,
            "task_completed": False,
            "evaluation_success": False,
            "steps": [],
            "error": "RuntimeError('reset failed')",
        }
    )

    assert Path(path).exists()
    rows = read_jsonl(path)
    assert [row["task_id"] for row in rows] == ["task_1", "task_2"]
    assert rows[0]["outcome"]["label"] == "success"
    assert rows[1]["outcome"]["label"] == "setup_error"
