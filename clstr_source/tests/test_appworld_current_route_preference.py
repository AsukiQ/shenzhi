import json

import torch

from clstr.appworld_current_route_preference import (
    build_current_route_disagreement_preference_dataset,
    build_current_route_preference_dataset,
    summarize_current_route_preference_signal,
    write_current_route_preference_dataset,
)
from clstr.appworld_current_route_preference_train import (
    compute_current_route_preference_batch_loss,
    configure_current_route_preference_trainable,
)
from clstr.appworld_routing import read_jsonl
from scripts.build_appworld_current_route_preference_dataset import main as preference_cli_main
from scripts.run_appworld_current_route_preference_train import build_parser as train_cli_build_parser


def _decision(**overrides):
    row = {
        "step_idx": 0,
        "state_text": "[User Goal]\nFind songs.",
        "selected_skill_ids": ["skill/a", "skill/c"],
        "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "selected_candidate_local_indices": [0, 2],
        "selected_log_probs": [-0.349012, -1.349012],
        "policy_log_probs": [-0.349012, -3.349012, -1.349012],
        "candidate_policy_logits": [3.0, 0.0, 2.0],
        "ranking_mode": "transition_blend",
        "candidate_source": "routing",
        "training_ready": True,
        "training_blockers": [],
    }
    row.update(overrides)
    return row


def _rollout(label="success", policy_signal="positive", decisions=None):
    return {
        "schema_version": "current_route_rollout.v1",
        "method": "clstr_multistep",
        "task_id": "task_1",
        "query_id": "task_1",
        "user_goal": "Find songs.",
        "success": label == "success",
        "outcome": {
            "label": label,
            "policy_signal": policy_signal,
            "reward": 1.0 if label == "success" else -0.5,
        },
        "decisions": decisions if decisions is not None else [_decision()],
        "training_ready": True,
        "training_blockers": [],
    }


def _run(task_id, success):
    return {
        "task_id": task_id,
        "query_id": task_id,
        "success": bool(success),
        "evaluation_success": bool(success),
    }


def test_current_route_preference_adapter_keeps_success_decisions_as_multi_positive_targets():
    dataset = build_current_route_preference_dataset([_rollout()])

    assert dataset.report["sample_count"] == 1
    assert dataset.report["positive_sample_count"] == 1
    assert dataset.report["skipped_ambiguous_rollout_count"] == 0
    sample = dataset.samples[0]
    assert sample["sample_type"] == "success_imitation"
    assert sample["task_id"] == "task_1"
    assert sample["decision_id"] == "task_1::0"
    assert sample["target_skill_ids"] == ["skill/a", "skill/c"]
    assert sample["target_local_indices"] == [0, 2]
    assert sample["state_text"].startswith("[User Goal]")
    assert sample["candidate_skill_ids"] == ["skill/a", "skill/b", "skill/c"]


def test_current_route_preference_adapter_keeps_previous_action_fields_for_transition_loss():
    dataset = build_current_route_preference_dataset(
        [
            _rollout(
                decisions=[
                    _decision(
                        step_idx=1,
                        previous_selected_skill_ids=["skill/a"],
                        previous_execute_output="Execution successful.",
                    )
                ]
            )
        ]
    )

    sample = dataset.samples[0]
    assert sample["previous_selected_skill_ids"] == ["skill/a"]
    assert sample["previous_execute_output"] == "Execution successful."


def test_current_route_preference_adapter_skips_wrong_completion_as_ambiguous():
    dataset = build_current_route_preference_dataset(
        [_rollout(label="wrong_completion", policy_signal="ambiguous_do_not_penalize_clstr")]
    )

    assert dataset.samples == []
    assert dataset.report["skipped_ambiguous_rollout_count"] == 1
    assert dataset.report["skipped_by_outcome"] == {"wrong_completion": 1}


def test_current_route_preference_adapter_skips_blocked_or_unaligned_decisions():
    blocked = _decision(training_ready=False, training_blockers=["missing_policy_logprobs"])
    unaligned = _decision(selected_skill_ids=["skill/z"], selected_candidate_local_indices=[99])

    dataset = build_current_route_preference_dataset([_rollout(decisions=[blocked, unaligned])])

    assert dataset.samples == []
    assert dataset.report["skipped_blocked_decision_count"] == 1
    assert dataset.report["skipped_unaligned_decision_count"] == 1


def test_current_route_preference_writer_emits_jsonl_and_report(tmp_path):
    output_jsonl = tmp_path / "samples.jsonl"
    report_json = tmp_path / "report.json"

    dataset = write_current_route_preference_dataset(
        [_rollout()],
        output_jsonl_path=output_jsonl,
        report_json_path=report_json,
    )

    assert dataset.report["sample_count"] == 1
    assert read_jsonl(output_jsonl)[0]["target_local_indices"] == [0, 2]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["sample_count"] == 1


def test_current_route_preference_cli_builds_dataset(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    output_jsonl = tmp_path / "samples.jsonl"
    report_json = tmp_path / "report.json"
    rollouts_path.write_text(json.dumps(_rollout(), ensure_ascii=False) + "\n", encoding="utf-8")

    report = preference_cli_main(
        [
            "--rollouts_path",
            str(rollouts_path),
            "--output_jsonl_path",
            str(output_jsonl),
            "--report_json_path",
            str(report_json),
        ]
    )

    assert report["sample_count"] == 1
    assert read_jsonl(output_jsonl)[0]["decision_id"] == "task_1::0"


def test_current_route_preference_cli_can_build_disagreement_dataset(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    current_runs_path = tmp_path / "current_runs.jsonl"
    qwen_runs_path = tmp_path / "qwen_runs.jsonl"
    reference_runs_path = tmp_path / "reference_runs.jsonl"
    output_jsonl = tmp_path / "samples.jsonl"
    report_json = tmp_path / "report.json"
    failed_current = _rollout(label="wrong_completion", policy_signal="ambiguous_do_not_penalize_clstr")
    failed_current["task_id"] = "qwen_rescue"
    failed_current["query_id"] = "qwen_rescue"
    rollouts_path.write_text(json.dumps(failed_current, ensure_ascii=False) + "\n", encoding="utf-8")
    current_runs_path.write_text(json.dumps(_run("qwen_rescue", False)) + "\n", encoding="utf-8")
    qwen_runs_path.write_text(json.dumps(_run("qwen_rescue", True)) + "\n", encoding="utf-8")
    reference_runs_path.write_text(json.dumps(_run("qwen_rescue", False)) + "\n", encoding="utf-8")

    report = preference_cli_main(
        [
            "--rollouts_path",
            str(rollouts_path),
            "--output_jsonl_path",
            str(output_jsonl),
            "--report_json_path",
            str(report_json),
            "--current_runs_path",
            str(current_runs_path),
            "--qwen_runs_path",
            str(qwen_runs_path),
            "--reference_runs_path",
            str(reference_runs_path),
            "--adapter_mode",
            "disagreement",
            "--negative_weight",
            "0.15",
            "--allow_ambiguous_disagreement",
        ]
    )

    assert report["schema_version"] == "current_route_disagreement_preference_report.v1"
    assert report["suppress_sample_count"] == 1
    assert read_jsonl(output_jsonl)[0]["sample_type"] == "fallback_to_qwen_suppress_current"
    assert read_jsonl(output_jsonl)[0]["weight"] == 0.15


def test_current_route_preference_cli_defaults_to_conservative_with_baseline_paths(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    current_runs_path = tmp_path / "current_runs.jsonl"
    qwen_runs_path = tmp_path / "qwen_runs.jsonl"
    output_jsonl = tmp_path / "samples.jsonl"
    report_json = tmp_path / "report.json"
    failed_current = _rollout(label="wrong_completion", policy_signal="ambiguous_do_not_penalize_clstr")
    failed_current["task_id"] = "qwen_rescue"
    failed_current["query_id"] = "qwen_rescue"
    rollouts_path.write_text(json.dumps(failed_current, ensure_ascii=False) + "\n", encoding="utf-8")
    current_runs_path.write_text(json.dumps(_run("qwen_rescue", False)) + "\n", encoding="utf-8")
    qwen_runs_path.write_text(json.dumps(_run("qwen_rescue", True)) + "\n", encoding="utf-8")

    report = preference_cli_main(
        [
            "--rollouts_path",
            str(rollouts_path),
            "--output_jsonl_path",
            str(output_jsonl),
            "--report_json_path",
            str(report_json),
            "--current_runs_path",
            str(current_runs_path),
            "--qwen_runs_path",
            str(qwen_runs_path),
        ]
    )

    assert report["schema_version"] == "current_route_preference_report.v1"
    assert report["sample_count"] == 0
    assert report["skipped_by_outcome"] == {"wrong_completion": 1}
    assert read_jsonl(output_jsonl) == []


def test_current_route_preference_signal_summary_uses_multi_positive_nll():
    dataset = build_current_route_preference_dataset([_rollout()])

    report = summarize_current_route_preference_signal(dataset.samples)

    assert report["sample_count"] == 1
    assert report["finite_loss_sample_count"] == 1
    assert report["mean_candidate_count"] == 3.0
    assert report["mean_target_count"] == 2.0
    assert 0.0 < report["mean_logged_multi_positive_nll"] < 1.0


class _TinySkillTable:
    def __init__(self):
        self.E = torch.eye(3)

    def retrieval_logits(self, h):
        del h
        return torch.zeros(1, 3)


class _TinyPreferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skills = [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}]
        self.skill_table = _TinySkillTable()
        self.skill_head = torch.nn.Linear(3, 1, bias=False)
        torch.nn.init.zeros_(self.skill_head.weight)

    @property
    def device(self):
        return torch.device("cpu")

    def encode_states(self, states):
        return torch.zeros(len(states), 3)

    def batch_cross_encode(self, states, candidate_rows):
        del states
        rows = []
        for candidates in candidate_rows:
            rows.append(self.skill_table.E.index_select(0, torch.tensor(candidates, dtype=torch.long)))
        return torch.stack(rows)

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        del h_t, m_t, routing_logits
        skill_logits = self.skill_head(candidate_embs).squeeze(-1)
        stop = torch.full((skill_logits.size(0), 1), -100.0)
        return torch.cat([skill_logits, stop], dim=-1)


class _TinyTransition(torch.nn.Module):
    def forward(self, m_t, action_input, obs_emb):
        del m_t, obs_emb
        return action_input


class _TinyTransHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, pred, candidate_embs):
        return self.scale * (candidate_embs * pred.unsqueeze(1)).sum(dim=-1)


class _TinyTransitionPreferenceModel(_TinyPreferenceModel):
    def __init__(self):
        super().__init__()
        self.transition = _TinyTransition()
        self.trans_head = _TinyTransHead()
        self.action_proj = torch.nn.Linear(3, 3, bias=False)
        torch.nn.init.eye_(self.action_proj.weight)

    def encode_observations(self, observations):
        return torch.zeros(len(observations), 3)

    def action_embeddings(self, labels):
        ids = labels.reshape(-1).to(dtype=torch.long)
        emb = self.skill_table.E.index_select(0, ids)
        return emb.view(*labels.shape, -1) if labels.ndim > 1 else emb


def test_current_route_preference_loss_backprops_to_policy_head():
    model = _TinyPreferenceModel()
    configure_current_route_preference_trainable(model)
    dataset = build_current_route_preference_dataset([_rollout()])
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        dataset.samples,
        skill_id_to_idx=skill_id_to_idx,
    )
    loss.backward()

    assert metrics["sample_count"] == 1
    assert metrics["decision_count"] == 1
    assert loss.item() > 0.0
    assert model.skill_head.weight.grad is not None
    assert float(model.skill_head.weight.grad.abs().sum().item()) > 0.0


def test_current_route_preference_transition_blend_loss_backprops_to_transition_head():
    model = _TinyTransitionPreferenceModel()
    configure_current_route_preference_trainable(model, train_transition=True)
    dataset = build_current_route_preference_dataset(
        [
            _rollout(
                decisions=[
                    _decision(
                        step_idx=1,
                        selected_skill_ids=["skill/c"],
                        selected_candidate_local_indices=[2],
                        previous_selected_skill_ids=["skill/a"],
                        previous_execute_output="Execution successful.",
                    )
                ]
            )
        ]
    )
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        dataset.samples,
        skill_id_to_idx=skill_id_to_idx,
        loss_score_mode="transition_blend",
    )
    loss.backward()

    assert metrics["transition_blend_sample_count"] == 1
    assert loss.item() > 0.0
    assert model.trans_head.scale.grad is not None
    assert float(model.trans_head.scale.grad.abs().item()) > 0.0


def test_current_route_preference_policy_transition_blend_backprops_to_policy_and_transition():
    model = _TinyTransitionPreferenceModel()
    with torch.no_grad():
        model.skill_head.weight.copy_(torch.tensor([[0.0, 0.5, 1.0]]))
    configure_current_route_preference_trainable(model, train_transition=True)
    dataset = build_current_route_preference_dataset(
        [
            _rollout(
                decisions=[
                    _decision(
                        step_idx=1,
                        selected_skill_ids=["skill/c"],
                        selected_candidate_local_indices=[2],
                        previous_selected_skill_ids=["skill/a"],
                        previous_execute_output="Execution successful.",
                    )
                ]
            )
        ]
    )
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        dataset.samples,
        skill_id_to_idx=skill_id_to_idx,
        loss_score_mode="policy_transition_blend",
    )
    loss.backward()

    assert metrics["policy_transition_blend_sample_count"] == 1
    assert loss.item() > 0.0
    assert model.skill_head.weight.grad is not None
    assert float(model.skill_head.weight.grad.abs().sum().item()) > 0.0
    assert model.trans_head.scale.grad is not None
    assert float(model.trans_head.scale.grad.abs().item()) > 0.0


def test_current_route_preference_policy_transition_blend_warms_up_low_confidence_policy_without_zscore_blend():
    model = _TinyPreferenceModel()
    with torch.no_grad():
        model.skill_head.weight.copy_(torch.tensor([[0.0, 0.02, 0.04]]))
    configure_current_route_preference_trainable(model)
    dataset = build_current_route_preference_dataset([_rollout()])
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        dataset.samples,
        skill_id_to_idx=skill_id_to_idx,
        loss_score_mode="policy_transition_blend",
        learned_component_min_range=0.1,
    )
    loss.backward()

    assert metrics["policy_transition_blend_sample_count"] == 1
    assert metrics["policy_component_enabled_count"] == 0
    assert metrics["policy_component_disabled_count"] == 1
    assert metrics["transition_component_enabled_count"] == 0
    assert metrics["low_confidence_warmup_count"] == 1
    assert metrics["skipped_learned_component_unavailable_count"] == 0
    assert metrics["decision_count"] == 1
    assert loss.item() > 0.0
    assert model.skill_head.weight.grad is not None
    assert float(model.skill_head.weight.grad.abs().sum().item()) > 0.0


def test_current_route_preference_policy_transition_blend_skips_target_outside_trust_region():
    model = _TinyTransitionPreferenceModel()
    with torch.no_grad():
        model.skill_head.weight.copy_(torch.tensor([[0.0, 0.5, 1.0]]))
    configure_current_route_preference_trainable(model, train_transition=True)
    dataset = build_current_route_preference_dataset(
        [
            _rollout(
                decisions=[
                    _decision(
                        step_idx=1,
                        selected_skill_ids=["skill/c"],
                        selected_candidate_local_indices=[2],
                        previous_selected_skill_ids=["skill/a"],
                        previous_execute_output="Execution successful.",
                    )
                ]
            )
        ]
    )
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        dataset.samples,
        skill_id_to_idx=skill_id_to_idx,
        loss_score_mode="policy_transition_blend",
        learned_component_min_range=0.1,
        learned_component_trust_top_k=2,
    )
    loss.backward()

    assert metrics["policy_transition_blend_sample_count"] == 1
    assert metrics["trust_region_active_count"] == 1
    assert metrics["skipped_target_outside_trust_region_count"] == 1
    assert metrics["decision_count"] == 0
    assert loss.item() == 0.0
    assert model.skill_head.weight.grad is None
    assert model.trans_head.scale.grad is None


def test_current_route_preference_train_cli_accepts_current_route_paths():
    parser = train_cli_build_parser()
    args = parser.parse_args(
        [
            "--preference_samples_path",
            "samples.jsonl",
            "--output_dir",
            "out",
            "--skill_pool_path",
            "data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl",
            "--base_skill_pool_path",
            "data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
            "--clstr_checkpoint_path",
            "checkpoint.pt",
            "--max_steps",
            "1",
            "--loss_score_mode",
            "policy_transition_blend",
            "--learned_component_min_range",
            "0.1",
            "--learned_component_trust_top_k",
            "80",
        ]
    )

    assert args.preference_samples_path == "samples.jsonl"
    assert args.max_steps == 1
    assert args.train_transition is False
    assert args.loss_score_mode == "policy_transition_blend"
    assert args.learned_component_min_range == 0.1
    assert args.learned_component_trust_top_k == 80


def test_current_route_disagreement_adapter_uses_success_and_route_split_signals():
    current_success = _rollout()
    current_success["task_id"] = "current_success"
    current_success["query_id"] = "current_success"
    current_success["decisions"] = [_decision(step_idx=0)]
    qwen_rescue = _rollout(label="routing_miss", policy_signal="negative")
    qwen_rescue["task_id"] = "qwen_rescue"
    qwen_rescue["query_id"] = "qwen_rescue"
    qwen_rescue["decisions"] = [_decision(step_idx=1, selected_skill_ids=["skill/b"], selected_candidate_local_indices=[1])]
    old_rescue = _rollout(label="routing_miss", policy_signal="negative")
    old_rescue["task_id"] = "old_rescue"
    old_rescue["query_id"] = "old_rescue"
    old_rescue["decisions"] = [_decision(step_idx=2, selected_skill_ids=["skill/c"], selected_candidate_local_indices=[2])]

    dataset = build_current_route_disagreement_preference_dataset(
        [current_success, qwen_rescue, old_rescue],
        current_runs=[_run("current_success", True), _run("qwen_rescue", False), _run("old_rescue", False)],
        qwen_runs=[_run("current_success", False), _run("qwen_rescue", True), _run("old_rescue", False)],
        reference_runs=[_run("current_success", False), _run("qwen_rescue", False), _run("old_rescue", True)],
        negative_weight=0.2,
    )

    assert dataset.report["route_split_counts"]["current_only_success"] == 1
    assert dataset.report["route_split_counts"]["qwen_only_success"] == 1
    assert dataset.report["route_split_counts"]["reference_only_success"] == 1
    sample_types = [sample["sample_type"] for sample in dataset.samples]
    assert sample_types == ["success_imitation", "fallback_to_qwen_suppress_current", "prefer_reference_suppress_current"]
    assert dataset.samples[1]["avoid_local_indices"] == [1]
    assert dataset.samples[1]["target_local_indices"] == []
    assert dataset.samples[1]["weight"] == 0.2
    assert dataset.samples[2]["reference_success"] is True


def test_current_route_disagreement_adapter_skips_ambiguous_wrong_completion_by_default():
    qwen_rescue = _rollout(label="wrong_completion", policy_signal="ambiguous_do_not_penalize_clstr")
    qwen_rescue["task_id"] = "qwen_rescue"
    qwen_rescue["query_id"] = "qwen_rescue"
    qwen_rescue["decisions"] = [_decision(step_idx=1, selected_skill_ids=["skill/b"], selected_candidate_local_indices=[1])]

    dataset = build_current_route_disagreement_preference_dataset(
        [qwen_rescue],
        current_runs=[_run("qwen_rescue", False)],
        qwen_runs=[_run("qwen_rescue", True)],
        reference_runs=[],
        negative_weight=0.2,
    )

    assert dataset.samples == []
    assert dataset.report["route_split_counts"]["qwen_only_success"] == 1
    assert dataset.report["skipped_ambiguous_disagreement_count"] == 1


def test_current_route_preference_loss_skips_suppress_samples_by_default():
    model = _TinyPreferenceModel()
    configure_current_route_preference_trainable(model)
    suppress_sample = {
        "sample_type": "fallback_to_qwen_suppress_current",
        "task_id": "qwen_rescue",
        "state_text": "[User Goal]\nFind songs.",
        "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "target_local_indices": [],
        "avoid_local_indices": [1],
        "previous_selected_skill_ids": [],
        "previous_execute_output": "",
        "weight": 0.25,
    }
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        [suppress_sample],
        skill_id_to_idx=skill_id_to_idx,
    )

    assert metrics["suppress_sample_count"] == 0
    assert metrics["skipped_suppress_disabled_count"] == 1
    assert metrics["decision_count"] == 0
    assert loss.item() == 0.0


def test_current_route_preference_loss_can_explicitly_train_low_weight_suppress_samples():
    model = _TinyPreferenceModel()
    configure_current_route_preference_trainable(model)
    suppress_sample = {
        "sample_type": "fallback_to_qwen_suppress_current",
        "task_id": "qwen_rescue",
        "state_text": "[User Goal]\nFind songs.",
        "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "target_local_indices": [],
        "avoid_local_indices": [1],
        "previous_selected_skill_ids": [],
        "previous_execute_output": "",
        "weight": 0.25,
    }
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(model.skills)}

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        [suppress_sample],
        skill_id_to_idx=skill_id_to_idx,
        enable_suppress_loss=True,
    )
    loss.backward()

    assert metrics["suppress_sample_count"] == 1
    assert metrics["positive_sample_count"] == 0
    assert metrics["mean_sample_weight"] == 0.25
    assert loss.item() > 0.0
    assert model.skill_head.weight.grad is not None
