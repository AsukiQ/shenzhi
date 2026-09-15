import json

import torch

from clstr.appworld_current_route_online_train import run_current_route_online_stage4_train_with_model
from scripts.run_appworld_current_route_online_stage4_train import build_parser


def _decision(step_idx=0):
    return {
        "step_idx": step_idx,
        "state_text": "[User Goal]\nFind songs.",
        "selected_skill_ids": ["skill/a"],
        "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "selected_candidate_local_indices": [0],
        "selected_log_probs": [-1.0],
        "policy_log_probs": [-1.0, -2.0, -3.0],
        "candidate_policy_logits": [2.0, 1.0, 0.0],
        "training_ready": True,
        "training_blockers": [],
    }


def _success_rollout(task_id, step_idx=0):
    return {
        "schema_version": "current_route_rollout.v1",
        "method": "clstr_multistep",
        "task_id": task_id,
        "query_id": task_id,
        "user_goal": "Find songs.",
        "success": True,
        "task_completed": True,
        "evaluation_success": True,
        "outcome": {
            "label": "success",
            "policy_signal": "positive",
            "reward": 1.0,
        },
        "decisions": [_decision(step_idx=step_idx)],
        "training_ready": True,
        "training_blockers": [],
    }


def _failure_rollout(task_id, step_idx=0):
    row = _success_rollout(task_id, step_idx=step_idx)
    row["success"] = False
    row["task_completed"] = False
    row["evaluation_success"] = False
    row["outcome"] = {
        "label": "wrong_completion",
        "policy_signal": "negative",
        "reward": 0.0,
    }
    return row


class _TinySkillTable:
    def __init__(self):
        self.E = torch.eye(3)

    def retrieval_logits(self, h):
        del h
        return torch.zeros(1, 3)


class _TinyOnlineModel(torch.nn.Module):
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


class _TrainableStage0SkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.eye(3)
        self.W = torch.nn.Linear(3, 3, bias=False)
        self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
        self.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
        with torch.no_grad():
            self.W.weight.copy_(torch.eye(3))

    def retrieval_logits(self, h):
        values = self.W(h) @ self.E.t()
        return values * self.logit_scale_retr.exp() + self.skill_bias_retr


class _TrainableStage0OnlineModel(_TinyOnlineModel):
    def __init__(self):
        super().__init__()
        self.skill_table = _TrainableStage0SkillTable()
        self.transition = torch.nn.Linear(3, 3, bias=False)
        self.trans_head = torch.nn.Linear(3, 3, bias=False)

    def encode_states(self, states):
        rows = []
        for text in states:
            if "missing skill" in str(text).lower():
                rows.append(torch.tensor([0.0, 1.0, 0.0]))
            else:
                rows.append(torch.zeros(3))
        return torch.stack(rows)


def test_online_stage4_updates_after_each_success_rollout_and_writes_latest_checkpoint(tmp_path):
    model = _TinyOnlineModel()
    initial_weight = model.skill_head.weight.detach().clone()
    rollouts = [_success_rollout("task_1"), _success_rollout("task_2", step_idx=1)]

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: rollouts[task_index - 1],
        tasks=[{"task_id": "task_1"}, {"task_id": "task_2"}],
        output_dir=tmp_path / "online",
        max_online_updates=2,
        batch_size=1,
        learning_rate=0.1,
    )

    assert report["status"] == "ok"
    assert report["on_policy_rollout_used"] is True
    assert report["rollout_count"] == 2
    assert report["online_update_count"] == 2
    assert report["checkpoint_save_count"] == 2
    assert not torch.allclose(initial_weight, model.skill_head.weight.detach())
    assert (tmp_path / "online" / "checkpoints" / "latest.pt").exists()
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["online_update"] for row in metrics] == [1, 2]


def test_online_stage4_can_take_multiple_optimizer_steps_per_clean_rollout(tmp_path):
    model = _TinyOnlineModel()

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: _success_rollout("task_1"),
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        online_update_epochs=3,
    )

    assert report["status"] == "ok"
    assert report["online_update_count"] == 1
    assert report["optimizer_step_count"] == 3
    assert report["last_metrics"]["optimizer_step_count"] == 3
    assert report["last_metrics"]["actual_online_update_epochs"] == 3
    assert len(report["last_metrics"]["epoch_losses"]) == 3
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["online_update_epochs"] == 3
    assert metrics[0]["optimizer_step_count"] == 3
    assert len(metrics[0]["epoch_losses"]) == 3


def test_online_stage4_records_stability_kl_regularization(tmp_path):
    model = _TinyOnlineModel()
    rollout = _success_rollout("task_1")
    # Real replay rows can carry the original top-350 distribution while the
    # Stage4 trust region trains on a shorter candidate prefix.
    rollout["decisions"][0]["policy_log_probs"] = [-0.01, -5.0, -6.0, -7.0, -8.0]

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: rollout,
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        stability_kl_weight=0.25,
    )

    assert report["status"] == "ok"
    assert report["stability_kl_weight"] == 0.25
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["stability_kl_weight"] == 0.25
    assert metrics[0]["stability_kl_sample_count"] == 1
    assert metrics[0]["stability_kl_loss"] > 0.0
    assert metrics[0]["preference_loss"] > 0.0


def test_online_stage4_can_warm_start_from_verified_success_replay_before_live_rollouts(tmp_path):
    model = _TinyOnlineModel()
    replay_path = tmp_path / "replay.jsonl"
    replay_rows = [_success_rollout("replay_1"), _success_rollout("replay_2", step_idx=1)]
    replay_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in replay_rows),
        encoding="utf-8",
    )

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: _failure_rollout("live_1"),
        tasks=[{"task_id": "live_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        replay_success_rollouts_path=replay_path,
        max_replay_updates=2,
        online_update_epochs=2,
        batch_size=1,
        learning_rate=0.1,
    )

    assert report["status"] == "ok"
    assert report["rollout_count"] == 1
    assert report["online_update_count"] == 0
    assert report["replay_update_count"] == 2
    assert report["total_stage4_update_count"] == 2
    assert report["optimizer_step_count"] == 4
    assert report["checkpoint_save_count"] == 2
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row.get("update_source") for row in metrics[:2]] == ["replay_success", "replay_success"]
    assert metrics[0]["actual_online_update_epochs"] == 2
    assert metrics[-1]["reason"] == "no_clean_preference_samples"


def test_online_stage4_replay_accepts_prebuilt_current_route_preference_samples(tmp_path):
    model = _TinyOnlineModel()
    replay_path = tmp_path / "replay_samples.jsonl"
    sample = {
        "schema_version": "current_route_preference.v1",
        "sample_type": "success_imitation",
        "task_id": "replay_sample_1",
        "query_id": "replay_sample_1",
        "decision_id": "replay_sample_1::0",
        "step_idx": 0,
        "state_text": "[User Goal]\nFind songs.",
        "user_goal": "Find songs.",
        "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "target_skill_ids": ["skill/a"],
        "target_local_indices": [0],
        "selected_log_probs": [-1.0],
        "policy_log_probs": [-1.0, -2.0, -3.0],
        "candidate_policy_logits": [2.0, 1.0, 0.0],
        "outcome_label": "success",
        "policy_signal": "positive",
        "reward": 1.0,
        "weight": 1.0,
    }
    replay_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: _failure_rollout("live_1"),
        tasks=[],
        output_dir=tmp_path / "online",
        replay_success_rollouts_path=replay_path,
        max_replay_updates=1,
        online_update_epochs=2,
        batch_size=1,
        learning_rate=0.1,
    )

    assert report["status"] == "ok"
    assert report["on_policy_rollout_used"] is False
    assert report["replay_update_count"] == 1
    assert report["optimizer_step_count"] == 2
    assert report["replay_sample_count"] == 1


def test_online_stage4_does_not_count_gated_zero_signal_as_update(tmp_path):
    model = _TinyOnlineModel()
    rollout = _success_rollout("task_1")
    rollout["decisions"][0].update(
        selected_skill_ids=["skill/c"],
        selected_candidate_local_indices=[2],
    )

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: rollout,
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        loss_score_mode="policy_transition_blend",
        learned_component_min_range=0.1,
        learned_component_trust_top_k=2,
    )

    assert report["status"] == "blocked"
    assert report["online_update_count"] == 0
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["updated"] is False
    assert metrics[0]["reason"] == "no_trainable_stage4_preference_signal"
    assert metrics[0]["decision_count"] == 0
    assert metrics[0]["skipped_target_outside_trust_region_count"] == 1


def test_online_stage4_warms_up_low_confidence_success_sample(tmp_path):
    model = _TinyOnlineModel()

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: _success_rollout("task_1"),
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        loss_score_mode="policy_transition_blend",
        learned_component_min_range=0.1,
        learned_component_trust_top_k=80,
    )

    assert report["status"] == "ok"
    assert report["online_update_count"] == 1
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["updated"] is True
    assert metrics[0]["decision_count"] == 1
    assert metrics[0]["low_confidence_warmup_count"] == 1


def test_online_stage4_retries_remaining_success_samples_when_first_batch_is_gated(tmp_path):
    model = _TinyOnlineModel()
    with torch.no_grad():
        model.skill_head.weight.copy_(torch.tensor([[0.0, 0.5, 1.0]]))
    rollout = _success_rollout("task_1")
    first = _decision(step_idx=0)
    first.update(selected_skill_ids=["skill/c"], selected_candidate_local_indices=[2])
    second = _decision(step_idx=1)
    second.update(selected_skill_ids=["skill/b"], selected_candidate_local_indices=[1])
    rollout["decisions"] = [first, second]

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: rollout,
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        loss_score_mode="policy_transition_blend",
        learned_component_min_range=0.1,
        learned_component_trust_top_k=2,
    )

    assert report["status"] == "ok"
    assert report["online_update_count"] == 1
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["updated"] is True
    assert metrics[0]["online_batch_retry_all_samples"] is True
    assert metrics[0]["decision_count"] == 1
    assert metrics[0]["skipped_target_outside_trust_region_count"] == 1


def test_online_stage4_routes_failed_missing_candidate_signal_to_stage0_without_update(tmp_path):
    model = _TinyOnlineModel()
    failure = _failure_rollout("task_1")

    def correction_builder(trace):
        assert trace["task_id"] == "task_1"
        return {
            "report": {"stage0_retrieval_correction_count": 1},
            "stage0_retrieval_rows": [
                {
                    "query_id": "task_1::stage0_api_correction::0",
                    "query_text": "Need missing skill",
                    "positive_skill_id": "skill/b",
                    "negative_skill_ids": ["skill/a"],
                    "source": "appworld_stage0_api_correction",
                }
            ],
        }

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: failure,
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        failure_correction_builder=correction_builder,
    )

    assert report["status"] == "blocked"
    assert report["online_update_count"] == 0
    assert report["stage0_correction_row_count"] == 1
    stage0_rows = [
        json.loads(line)
        for line in (tmp_path / "online" / "stage0_retrieval_corrections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert stage0_rows[0]["positive_skill_id"] == "skill/b"
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["updated"] is False
    assert metrics[0]["stage0_correction_row_count"] == 1


def test_online_stage4_can_update_from_failed_candidate_api_correction_sample(tmp_path):
    model = _TinyOnlineModel()
    failure = _failure_rollout("task_1")
    initial_weight = model.skill_head.weight.detach().clone()

    def correction_builder(trace):
        assert trace["task_id"] == "task_1"
        return {
            "report": {"sample_count": 1, "stage0_retrieval_correction_count": 0},
            "samples": [
                {
                    "schema_version": "current_route_preference.v1",
                    "sample_type": "counterfactual_api_correction",
                    "task_id": "task_1",
                    "decision_id": "task_1::0",
                    "state_text": "[User Goal]\nNeed the better candidate.",
                    "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
                    "target_skill_ids": ["skill/b"],
                    "target_local_indices": [1],
                    "rejected_skill_ids": ["skill/a"],
                    "rejected_local_indices": [0],
                    "policy_log_probs": [-1.0, -2.0, -3.0],
                }
            ],
            "stage0_retrieval_rows": [],
        }

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: failure,
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        loss_score_mode="policy_head",
        failure_correction_builder=correction_builder,
        enable_failure_stage4_correction=True,
    )

    assert report["status"] == "ok"
    assert report["online_update_count"] == 1
    assert not torch.allclose(initial_weight, model.skill_head.weight.detach())
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["updated"] is True
    assert metrics[0]["update_source"] == "failure_stage4_correction"
    assert metrics[0]["module_update_target"] == "stage4_correction"
    assert metrics[0]["pairwise_correction_sample_count"] == 1
    assert metrics[0]["failure_correction_report"]["sample_count"] == 1


def test_online_stage4_can_update_stage0_only_for_missing_candidate_failures(tmp_path):
    model = _TrainableStage0OnlineModel()
    failure = _failure_rollout("task_1")
    initial_stage0 = model.skill_table.W.weight.detach().clone()
    initial_stage1 = model.skill_head.weight.detach().clone()
    initial_transition = model.transition.weight.detach().clone()

    def correction_builder(trace):
        assert trace["task_id"] == "task_1"
        return {
            "report": {"stage0_retrieval_correction_count": 1},
            "stage0_retrieval_rows": [
                {
                    "query_id": "task_1::stage0_api_correction::0",
                    "query_text": "Need missing skill",
                    "positive_skill_id": "skill/b",
                    "negative_skill_ids": ["skill/a"],
                    "source": "appworld_stage0_api_correction",
                }
            ],
        }

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=lambda task, task_index, model: failure,
        tasks=[{"task_id": "task_1"}],
        output_dir=tmp_path / "online",
        max_online_updates=1,
        batch_size=1,
        learning_rate=0.1,
        failure_correction_builder=correction_builder,
        enable_online_stage0_correction_update=True,
        online_stage0_learning_rate=0.1,
    )

    assert report["status"] == "ok"
    assert report["online_update_count"] == 0
    assert report["online_stage0_update_count"] == 1
    assert report["stage0_correction_row_count"] == 1
    assert not torch.allclose(initial_stage0, model.skill_table.W.weight.detach())
    assert torch.allclose(initial_stage1, model.skill_head.weight.detach())
    assert torch.allclose(initial_transition, model.transition.weight.detach())
    metrics = [
        json.loads(line)
        for line in (tmp_path / "online" / "online_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics[0]["updated"] is False
    assert metrics[0]["stage0_updated"] is True
    assert metrics[0]["module_update_target"] == "stage0_retrieval"


def test_online_stage4_cli_accepts_current_route_official_executor_paths():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks_path",
            "data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl",
            "--output_dir",
            "outputs/online",
            "--skill_pool_path",
            "data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl",
            "--base_skill_pool_path",
            "data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
            "--clstr_checkpoint_path",
            "checkpoint.pt",
            "--max_online_updates",
            "2",
            "--online_update_epochs",
            "3",
            "--replay_success_rollouts_path",
            "outputs/replay/verified_success.jsonl",
            "--max_replay_updates",
            "4",
            "--completion_precheck_mode",
            "constraint_tokens",
            "--enable_failure_stage0_correction",
            "--enable_failure_stage4_correction",
            "--enable_online_stage0_correction_update",
            "--appworld_tasks_root",
            "/tmp/appworld/tasks",
            "--learned_component_min_range",
            "0.1",
            "--learned_component_trust_top_k",
            "80",
        ]
    )

    assert args.max_online_updates == 2
    assert args.online_update_epochs == 3
    assert args.replay_success_rollouts_path == "outputs/replay/verified_success.jsonl"
    assert args.max_replay_updates == 4
    assert args.method == "clstr_multistep"
    assert args.ranking_mode == "policy_transition_blend"
    assert args.loss_score_mode == "policy_transition_blend"
    assert args.train_transition is True
    assert args.completion_precheck_mode == "constraint_tokens"
    assert args.enable_failure_stage0_correction is True
    assert args.enable_failure_stage4_correction is True
    assert args.enable_online_stage0_correction_update is True
    assert args.appworld_tasks_root == "/tmp/appworld/tasks"
    assert args.learned_component_min_range == 0.1
    assert args.learned_component_trust_top_k == 80
    assert args.stability_kl_weight == 0.1
    assert args.top_k == 20
    assert args.handoff_visible_skill_limit == 5
    assert args.handoff_prompt_style == "legacy_hints"


def test_online_stage4_cli_defaults_to_stable_executor_surface():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tasks_path",
            "data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl",
            "--output_dir",
            "outputs/online",
            "--skill_pool_path",
            "data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl",
            "--base_skill_pool_path",
            "data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
            "--clstr_checkpoint_path",
            "checkpoint.pt",
        ]
    )

    assert args.top_k == 20
    assert args.handoff_visible_skill_limit == 5
    assert args.handoff_prompt_style == "legacy_hints"
    assert args.completion_precheck_mode == "constraint_tokens"
    assert args.stability_kl_weight == 0.1
    assert args.enable_failure_stage4_correction is False
