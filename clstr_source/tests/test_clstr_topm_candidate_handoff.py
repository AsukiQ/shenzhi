import json
import inspect
from pathlib import Path

import torch
import pytest

from clstr.full_base_train import (
    _attach_stage0_topm_candidates,
    _stage0_topk_with_explicit_inventory,
    _stage0_handoff_query_text,
    _compute_full_base_loss,
    run_clstr_full_base_train,
    run_clstr_stage1_heads_init,
    train_clstr_full_base_with_model,
    train_clstr_stage1_heads_with_model,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _Stage0SkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _IdentityTransition(torch.nn.Module):
    def forward(self, m_obs, labels, action_emb):
        del labels, action_emb
        return m_obs


class _Stage0TopMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _Stage0SkillTable()
        self.transition = _IdentityTransition()
        self.stop_head = None
        self.gate = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "state-a" in lowered:
                rows.append(torch.tensor([9.0, 1.0, 0.0]))
            elif "next-b" in lowered:
                rows.append(torch.tensor([0.0, 8.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0, 9.0]))
        return torch.stack(rows).to(self.device) + self.anchor * 0.0

    def initial_belief(self, h, top_k=None):
        del top_k
        return torch.zeros_like(h)

    def unified_route_full_logits(self, h, m):
        del m
        return h @ self.skill_table.E.t()


class _RecordingStage0TopMModel(_Stage0TopMModel):
    def __init__(self):
        super().__init__()
        self.encoded_texts = []
        self.encode_calls = []

    def encode_observations(self, texts):
        self.encode_calls.append([str(text) for text in texts])
        self.encoded_texts.extend(str(text) for text in texts)
        return super().encode_observations(texts)

class _CandidateOnlyTransHead(torch.nn.Module):
    def forward(self, pred, candidate_embs):
        del pred
        return candidate_embs[:, :, 1] * 5.0


class _CandidateOnlyModel(_Stage0TopMModel):
    def __init__(self):
        super().__init__()
        self.trans_head = _CandidateOnlyTransHead()

    def action_embeddings(self, action_ids):
        return self.skill_table.E.index_select(0, action_ids.reshape(-1)).view(*action_ids.shape, -1)


class _UnifiedStaticHandoffSkillTable(_Stage0SkillTable):
    def retrieval_logits(self, h):
        del h
        raise AssertionError("h-only retrieval must not be used by unified Stage0 handoff")


class _UnifiedStaticHandoffModel(_Stage0TopMModel):
    def __init__(self):
        super().__init__()
        self.skill_table = _UnifiedStaticHandoffSkillTable()
        self.initial_belief_calls = 0
        self.unified_full_calls = 0

    def initial_belief(self, h, top_k=None):
        del top_k
        self.initial_belief_calls += 1
        return torch.zeros_like(h)

    def unified_route_full_logits(self, h, m):
        del m
        self.unified_full_calls += 1
        return h @ self.skill_table.E.t()


def test_stage0_handoff_uses_initial_belief_and_unified_full_logits():
    model = _UnifiedStaticHandoffModel()
    rows = [
        {
            "state_text": "state-a",
            "action_text": "do",
            "next_observation_text": "next-b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        }
    ]
    skills = [
        {"skill_id": "skill/a"},
        {"skill_id": "skill/b"},
        {"skill_id": "skill/c"},
    ]

    retained, report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        top_m=2,
        positive_missing_policy="skip",
        query_mode="raw_state",
        encode_batch_size=16,
        device=torch.device("cpu"),
    )

    assert retained
    assert model.initial_belief_calls == 2
    assert model.unified_full_calls == 2
    assert report["static_candidate_scorer"] == "unified_static"


def test_row_sharded_handoff_reuses_only_complete_legacy_schedule_batches(tmp_path):
    from clstr.full_base_train import _prepare_stage0_topm_candidates_with_cache

    model = _RecordingStage0TopMModel()
    skills = [
        {"skill_id": "skill/a"},
        {"skill_id": "skill/b"},
        {"skill_id": "skill/c"},
    ]
    skill_id_to_idx = {"skill/a": 0, "skill/b": 1, "skill/c": 2}

    def row(row_id, state_text, next_state_text, skill_id, next_skill_id):
        return {
            "row_id": row_id,
            "state_text": state_text,
            "next_state_text": next_state_text,
            "skill_id": skill_id,
            "next_skill_id": next_skill_id,
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        }

    row_a = row("a", "state-a", "next-b", "skill/a", "skill/b")
    row_b = row("b", "next-b", "next-c", "skill/b", "skill/c")
    row_c = row("c", "next-c", "next-d", "skill/c", "skill/c")
    row_d = row("d", "next-d", "next-e", "skill/c", "skill/a")
    row_e = row("e", "state-e", "next-f", "skill/a", "skill/b")
    row_f = row("f", "next-f", "next-g", "skill/b", "skill/c")
    common = {
        "top_m": 2,
        "positive_missing_policy": "skip",
        "query_mode": "raw_state",
        "encode_batch_size": 2,
        "device": torch.device("cpu"),
        "cache_mode": "auto",
        "cache_dir": tmp_path / "cache",
        "cache_format": "row_sharded_v1",
        "cache_shard_size": 1,
    }

    first_rows, first_report = _prepare_stage0_topm_candidates_with_cache(
        model,
        [row_a, row_b, row_c, row_d],
        skills,
        skill_id_to_idx,
        manifest_path=tmp_path / "first.json",
        **common,
    )
    assert model.encode_calls == [
        ["state-a", "next-b"],
        ["next-b", "next-c"],
        ["next-c", "next-d"],
        ["next-d", "next-e"],
    ]
    model.encoded_texts = []
    model.encode_calls = []
    second_rows, second_report = _prepare_stage0_topm_candidates_with_cache(
        model,
        [row_a, row_b, row_e, row_f],
        skills,
        skill_id_to_idx,
        manifest_path=tmp_path / "second.json",
        **common,
    )

    assert first_report["source_rows"] == 4
    assert second_report["source_rows"] == 4
    assert model.encode_calls == [
        ["state-e", "next-f"],
        ["next-f", "next-g"],
    ]
    assert first_report["cache"]["hit_rows"] == 0
    assert first_report["cache"]["miss_rows"] == 4
    assert second_report["cache"]["hit_rows"] == 2
    assert second_report["cache"]["miss_rows"] == 2
    assert second_report["cache"]["computed_rows"] == 2


def test_full_pool_handoff_retains_natural_static_misses_without_gold_injection():
    model = _UnifiedStaticHandoffModel()
    rows = [
        {
            "state_text": "state-a",
            "action_text": "do",
            "next_observation_text": "next-c",
            "next_state_text": "next-c",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        }
    ]
    skills = [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}]

    retained, report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        top_m=1,
        positive_missing_policy="skip",
        query_mode="raw_state",
        next_skill_pool_mode="full_pool",
        device=torch.device("cpu"),
    )

    assert len(retained) == 1
    assert retained[0]["stage0_next_candidate_skill_indices"] == [2]
    assert retained[0]["stage0_next_positive_hit"] is False
    assert retained[0]["loss_mask"]["L_trans_skill_ce"] is True
    assert retained[0]["stage0_positive_injected"] is False
    assert retained[0]["stage0_current_skill_candidate_added"] is False
    assert report["stage0_topm_next_positive_covered_rows"] == 0


def test_full_pool_handoff_masks_only_candidate_limited_current_loss_on_static_miss():
    model = _UnifiedStaticHandoffModel()
    rows = [
        {
            "state_text": "state-missing-positive",
            "action_text": "do",
            "next_observation_text": "next-b",
            "next_state_text": "next-b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        }
    ]
    skills = [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}]

    retained, _report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        top_m=1,
        positive_missing_policy="skip",
        query_mode="raw_state",
        next_skill_pool_mode="full_pool",
        device=torch.device("cpu"),
    )

    assert len(retained) == 1
    assert retained[0]["stage0_current_positive_hit"] is False
    assert retained[0]["loss_mask"]["routing"] is False
    assert retained[0]["loss_mask"]["L_trans_skill_ce"] is True


def test_stage0_topk_with_explicit_inventory_keeps_global_topk_backfill():
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]], dtype=torch.float32)
    rows = [
        {
            "visible_inventory_skill_ids": ["skill/b", "skill/d"],
        }
    ]

    result = _stage0_topk_with_explicit_inventory(
        logits,
        rows,
        skill_id_to_idx={
            "skill/a": 0,
            "skill/b": 1,
            "skill/c": 2,
            "skill/d": 3,
            "skill/e": 4,
        },
        skill_count=5,
        k=5,
        min_candidates=4,
    )
    candidates, audit = result[0], result[-1]

    assert candidates == [[1, 3, 0, 2]]
    assert audit["inventory_candidate_rows"] == 1
    assert audit["inventory_candidate_topk_backfilled_rows"] == 1
    assert audit["inventory_candidate_topk_backfilled_candidates"] == 2


def test_stage0_topk_with_explicit_inventory_returns_scores_in_candidate_order():
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]], dtype=torch.float32)

    result = _stage0_topk_with_explicit_inventory(
        logits,
        [{"visible_inventory_skill_ids": ["skill/b", "skill/d"]}],
        skill_id_to_idx={
            "skill/a": 0,
            "skill/b": 1,
            "skill/c": 2,
            "skill/d": 3,
            "skill/e": 4,
        },
        skill_count=5,
        k=5,
        min_candidates=4,
    )

    assert len(result) == 3
    candidates, scores, _audit = result
    assert candidates == [[1, 3, 0, 2]]
    assert scores == [[9.0, 7.0, 10.0, 8.0]]


def test_vectorized_stage0_score_transfer_matches_legacy_inventory_topk_exactly():
    from clstr.full_base_train import _stage0_topk_with_explicit_inventory_vectorized_scores

    logits = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [1.0, 5.0, 3.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )
    rows = [
        {"visible_inventory_skill_ids": ["skill/b"]},
        {},
    ]
    skill_id_to_idx = {
        "skill/a": 0,
        "skill/b": 1,
        "skill/c": 2,
        "skill/d": 3,
        "skill/e": 4,
    }

    legacy = _stage0_topk_with_explicit_inventory(
        logits,
        rows,
        skill_id_to_idx=skill_id_to_idx,
        skill_count=5,
        k=3,
        min_candidates=2,
    )
    vectorized = _stage0_topk_with_explicit_inventory_vectorized_scores(
        logits,
        rows,
        skill_id_to_idx=skill_id_to_idx,
        skill_count=5,
        k=3,
        min_candidates=2,
    )

    assert vectorized == legacy


def test_exact_schedule_raw_candidates_preserve_legacy_calls_and_outputs():
    from clstr.full_base_train import _compute_stage0_raw_candidates_exact_schedule

    rows = [
        {
            "state_text": "state-a",
            "next_state_text": "next-b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        },
        {
            "state_text": "next-b",
            "next_state_text": "next-c",
            "skill_id": "skill/b",
            "next_skill_id": "skill/c",
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        },
    ]
    skills = [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}]
    skill_id_to_idx = {"skill/a": 0, "skill/b": 1, "skill/c": 2}
    legacy_model = _Stage0TopMModel()
    exact_model = _RecordingStage0TopMModel()

    legacy_rows, legacy_report = _attach_stage0_topm_candidates(
        legacy_model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=2,
        positive_missing_policy="skip",
        query_mode="raw_state",
        encode_batch_size=2,
        device=torch.device("cpu"),
    )
    raw_candidates, raw_report = _compute_stage0_raw_candidates_exact_schedule(
        exact_model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=2,
        query_mode="raw_state",
        encode_batch_size=2,
        device=torch.device("cpu"),
    )
    exact_rows, exact_report = _attach_stage0_topm_candidates(
        exact_model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=2,
        positive_missing_policy="skip",
        query_mode="raw_state",
        encode_batch_size=2,
        device=torch.device("cpu"),
        raw_candidates=raw_candidates,
        raw_candidate_report=raw_report,
    )

    assert exact_model.encode_calls == [["state-a", "next-b"], ["next-b", "next-c"]]
    assert raw_report["mode"] == "legacy_schedule_vectorized_scores"
    assert exact_rows == legacy_rows
    for key in (
        "retained_rows",
        "skipped_reasons",
        "current_positive_covered_rows",
        "next_positive_covered_rows",
        "next_positive_rank_bucket_counts",
    ):
        assert exact_report[key] == legacy_report[key]


def test_transition_skill_ce_scores_only_stage0_candidate_set():
    model = _CandidateOnlyModel()
    batch = [
        {
            "state_text": "state-z",
            "action_text": "do",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "stage0_candidate_skill_indices": [0, 1],
            "stage0_next_candidate_skill_indices": [1, 2],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
    )

    assert loss.item() == pytest.approx(metrics["transition_skill_ce_loss"])
    assert metrics["transition_skill_head_type"] == "stage0_rank_prior+native_trans_head_stage0_topm"
    assert metrics["transition_skill_ce_candidate_count"] == 2.0
    assert metrics["transition_skill_recall@1"] == 1.0


def test_transition_skill_ce_defaults_to_rank_prior_even_when_stage0_scores_are_available():
    model = _CandidateOnlyModel()
    batch = [
        {
            "state_text": "state-z",
            "action_text": "do",
            "skill_id": "skill/a",
            "next_skill_id": "skill/c",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_next_candidate_skill_scores": [0.0, 1.0, 9.0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_residual_lambda=0.0,
    )

    assert metrics["transition_skill_head_type"].startswith("stage0_rank_prior+")
    assert metrics["stage0_prior_transition_skill_recall@1"] == 0.0


def test_transition_skill_ce_uses_stage0_candidate_scores_when_explicitly_enabled():
    model = _CandidateOnlyModel()
    batch = [
        {
            "state_text": "state-z",
            "action_text": "do",
            "skill_id": "skill/a",
            "next_skill_id": "skill/c",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_next_candidate_skill_scores": [0.0, 1.0, 9.0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_residual_lambda=0.0,
        stage0_score_prior_calibration="rank_std",
    )

    assert metrics["transition_skill_head_type"].startswith("stage0_score_prior+")
    assert metrics["stage0_prior_transition_skill_recall@1"] == 1.0
    assert metrics["transition_skill_recall@1"] == 1.0


def test_stage2_handoff_skips_rows_when_stage0_topm_misses_current_positive(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a", "description": "a"},
            {"skill_id": "skill/b", "name": "b", "description": "b"},
            {"skill_id": "skill/c", "name": "c", "description": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "kept",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "task_id": "skipped",
                "state_text": "state-missing-positive",
                "action_text": "do",
                "next_observation_text": "next-missing-positive",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            },
        ],
    )

    report = train_clstr_full_base_with_model(
        model=_Stage0TopMModel(),
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"stage0_checkpoint": "stage0.pt"},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=3,
        loss_weights={"routing": 1.0, "L_trans_skill_ce": 1.0, "L_policy": 0.0, "L_trans": 0.0, "belief": 0.0, "STOP": 0.0},
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
        allow_full_pool_stage2_debug=False,
    )

    assert report["sample_count"] == 1
    assert report["stage0_candidate_handoff"]["candidate_source"] == "stage0_topm_online"
    assert report["stage0_candidate_handoff"]["top_m"] == 1
    assert report["stage0_candidate_handoff"]["positive_missing_policy"] == "skip"
    assert report["stage0_candidate_handoff"]["query_mode"] == "raw_state"
    assert report["stage0_candidate_handoff"]["skipped_rows"] == 1
    assert report["stage0_candidate_handoff"]["full_pool_stage2_debug_allowed"] is False
    assert Path(report["stage0_candidate_handoff"]["manifest_path"]).is_file()


def test_stage0_handoff_skillrouter_query_mode_wraps_sequential_state():
    row = {
        "state_text": "goal: inspect the invoice\nobservation: state-a\nhistory: look",
        "action_text": "open inbox",
        "next_observation_text": "next-b invoice is visible",
    }

    current = _stage0_handoff_query_text(row, target="current", mode="skillrouter_state")
    next_query = _stage0_handoff_query_text(row, target="next", mode="skillrouter_state")

    assert current.startswith("Instruct: Given a task description")
    assert "Query:goal: inspect the invoice" in current
    assert "history: look" not in current
    assert next_query.startswith("Instruct: Given a task description")
    assert "history: look" not in next_query
    assert "previous_action: open inbox" in next_query
    assert "next_observation: next-b invoice is visible" in next_query


def test_stage0_handoff_checkpoint_mode_passes_raw_state_to_model_prompt_boundary():
    model = _Stage0TopMModel()
    model.state_inputs = []

    def encode_states(texts):
        model.state_inputs.extend(str(text) for text in texts)
        return model.encode_observations(texts)

    model.encode_states = encode_states
    row = {
        "state_text": "goal: inspect invoice",
        "action_text": "open inbox",
        "next_observation_text": "invoice visible",
        "skill_id": "skill/c",
        "next_skill_id": "skill/c",
        "loss_mask": {"routing": True, "L_trans_skill_ce": True},
    }
    current = _stage0_handoff_query_text(
        row,
        target="current",
        mode="checkpoint_state_query",
    )
    next_text = _stage0_handoff_query_text(
        row,
        target="next",
        mode="checkpoint_state_query",
    )

    retained, _report = _attach_stage0_topm_candidates(
        model,
        [row],
        [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}],
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        top_m=3,
        positive_missing_policy="skip",
        query_mode="checkpoint_state_query",
        device=torch.device("cpu"),
    )

    assert retained
    assert current == "goal: inspect invoice"
    assert not current.startswith("Instruct:")
    assert "previous_action: open inbox" in next_text
    assert "next_observation: invoice visible" in next_text
    assert model.state_inputs == [current, next_text]
    assert all(not text.startswith("Instruct:") for text in model.state_inputs)


def test_stage2_handoff_masks_next_skill_ce_instead_of_dropping_row_when_next_positive_missing(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a", "description": "a"},
            {"skill_id": "skill/b", "name": "b", "description": "b"},
            {"skill_id": "skill/c", "name": "c", "description": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "keep-current-mask-next",
                "trajectory_id": "traj-mask-next",
                "step_index": 0,
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-c",
                "next_state_text": "next-c",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"source_id": "unit", "split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "task_id": "keep-current-mask-next-successor",
                "trajectory_id": "traj-mask-next",
                "step_index": 1,
                "state_text": "next-c",
                "skill_id": "skill/b",
                "loss_mask": {},
                "source_quality": "unit_test",
                "provenance": {"source_id": "unit", "split": "train"},
            },
        ],
    )

    report = train_clstr_full_base_with_model(
        model=_Stage0TopMModel(),
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"stage0_checkpoint": "stage0.pt"},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=3,
        loss_weights={"routing": 1.0, "L_trans_skill_ce": 1.0, "L_policy": 0.0, "L_trans": 0.0, "belief": 0.0, "STOP": 0.0},
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        allow_full_pool_stage2_debug=False,
    )

    handoff = report["stage0_candidate_handoff"]
    assert report["sample_count"] == 2
    assert handoff["skipped_rows"] == 0
    assert handoff["masked_next_skill_ce_rows"] == 1
    assert handoff["next_positive_coverage@M"] == 0.0
    assert report["loss_activation_counts"]["L_trans_skill_ce"] == 0


def test_stage2_handoff_adds_current_skill_as_keep_positive_candidate():
    skills = [
        {"skill_id": "skill/a", "name": "a", "description": "a"},
        {"skill_id": "skill/b", "name": "b", "description": "b"},
        {"skill_id": "skill/c", "name": "c", "description": "c"},
    ]
    rows = [
        {
            "benchmark": "traject_bench",
            "task_id": "keep-current",
            "state_text": "state-z",
            "action_text": "do",
            "next_observation_text": "next-b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/a",
            "loss_mask": {"routing": False, "L_trans_skill_ce": True},
        }
    ]

    retained, report = _attach_stage0_topm_candidates(
        _Stage0TopMModel(),
        rows,
        skills,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        top_m=1,
        positive_missing_policy="skip",
        query_mode="raw_state",
        device=torch.device("cpu"),
    )

    assert len(retained) == 1
    assert retained[0]["loss_mask"]["L_trans_skill_ce"] is True
    assert retained[0]["stage0_next_candidate_skill_indices"] == [1, 0]
    assert retained[0]["stage0_current_skill_candidate_added"] is True
    assert retained[0]["stage0_current_skill_candidate_role"] == "positive"
    assert report["current_skill_candidate_added_rows"] == 1
    assert report["current_skill_candidate_positive_rows"] == 1
    assert report["masked_next_skill_ce_rows"] == 0
    assert report["next_positive_rank_bucket_counts"]["2-5"] == 1
    assert report["learnable_correction_rows_rank_2_to_100"] == 1


def test_stage2_handoff_adds_current_skill_as_switch_hard_negative_candidate():
    skills = [
        {"skill_id": "skill/a", "name": "a", "description": "a"},
        {"skill_id": "skill/b", "name": "b", "description": "b"},
        {"skill_id": "skill/c", "name": "c", "description": "c"},
    ]
    rows = [
        {
            "benchmark": "traject_bench",
            "task_id": "switch-away",
            "state_text": "state-z",
            "action_text": "do",
            "next_observation_text": "next-b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "loss_mask": {"routing": False, "L_trans_skill_ce": True},
        }
    ]

    retained, report = _attach_stage0_topm_candidates(
        _Stage0TopMModel(),
        rows,
        skills,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        top_m=1,
        positive_missing_policy="skip",
        query_mode="raw_state",
        device=torch.device("cpu"),
    )

    assert len(retained) == 1
    assert retained[0]["stage0_next_candidate_skill_indices"] == [1, 0]
    assert retained[0]["stage0_current_skill_candidate_added"] is True
    assert retained[0]["stage0_current_skill_candidate_role"] == "hard_negative"
    assert report["current_skill_candidate_added_rows"] == 1
    assert report["current_skill_candidate_hard_negative_rows"] == 1
    assert report["masked_next_skill_ce_rows"] == 0
    assert report["next_positive_rank_bucket_counts"]["1"] == 1
    assert report["learnable_correction_rows_rank_2_to_100"] == 0


def test_stage2_handoff_ranks_within_explicit_tool_inventory(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a", "description": "a"},
            {"skill_id": "skill/b", "name": "b", "description": "b"},
            {"skill_id": "skill/c", "name": "c", "description": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "inventory-keep-gold",
                "trajectory_id": "traj-inventory",
                "step_index": 0,
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-missing-positive",
                "next_state_text": "next-missing-positive",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "visible_inventory_skill_ids": ["skill/a", "skill/b"],
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"source_id": "unit", "split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "task_id": "inventory-keep-gold-successor",
                "trajectory_id": "traj-inventory",
                "step_index": 1,
                "state_text": "next-missing-positive",
                "skill_id": "skill/b",
                "visible_inventory_skill_ids": ["skill/a", "skill/b"],
                "loss_mask": {},
                "source_quality": "unit_test",
                "provenance": {"source_id": "unit", "split": "train"},
            },
        ],
    )

    report = train_clstr_full_base_with_model(
        model=_Stage0TopMModel(),
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"stage0_checkpoint": "stage0.pt"},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=3,
        loss_weights={"routing": 1.0, "L_trans_skill_ce": 1.0, "L_policy": 0.0, "L_trans": 0.0, "belief": 0.0, "STOP": 0.0},
        stage0_top_m=2,
        stage0_positive_missing_policy="skip",
        allow_full_pool_stage2_debug=False,
    )

    handoff = report["stage0_candidate_handoff"]
    assert report["sample_count"] == 2
    assert handoff["next_positive_coverage@M"] == 1.0
    assert handoff["inventory_candidate_rows"] >= 1
    assert handoff["inventory_candidate_missing_rows"] == 0


def test_stage1_heads_init_writes_stage1_checkpoint_on_stage0_topm(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a", "description": "a"},
            {"skill_id": "skill/b", "name": "b", "description": "b"},
            {"skill_id": "skill/c", "name": "c", "description": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "stage1",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            }
        ],
    )

    report = train_clstr_stage1_heads_with_model(
        model=_CandidateOnlyModel(),
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"stage0_checkpoint": "stage0.pt"},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "stage1",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=3,
        loss_weights={"routing": 0.0, "L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0},
        stage0_top_m=2,
        stage0_positive_missing_policy="skip",
    )

    checkpoint_path = Path(report["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu")

    assert report["stage"] == "clstr_stage1_heads_init"
    assert report["training_objective"] == "stage1_heads_init_topm_supervised"
    assert checkpoint_path.name == "clstr_stage1_heads-step1.pt"
    assert payload["stage"] == "clstr_stage1_heads_init"
    assert payload["training_objective"] == "stage1_heads_init_topm_supervised"
    assert report["stage0_candidate_handoff"]["candidate_source"] == "stage0_topm_online"
    assert report["frozen_routing_foundation"] is True


def test_stage1_heads_init_passes_transition_ranking_config_into_training(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a", "description": "a"},
            {"skill_id": "skill/b", "name": "b", "description": "b"},
            {"skill_id": "skill/c", "name": "c", "description": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "stage1-transition-config",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "positive_next_skill_ids": ["skill/c"],
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            }
        ],
    )

    report = train_clstr_stage1_heads_with_model(
        model=_CandidateOnlyModel(),
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"stage0_checkpoint": "stage0.pt"},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "stage1",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=3,
        loss_weights={"routing": 0.0, "L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0},
        stage0_top_m=3,
        stage0_positive_missing_policy="skip",
        transition_inventory_mask_mode="auto",
        transition_loss_type="listwise_nll",
        transition_positive_mode="gold_plus_equivalent",
    )
    metrics = [json.loads(line) for line in Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()]

    assert report["transition_candidate_training"]["inventory_mask_mode"] == "auto"
    assert report["transition_candidate_training"]["loss_type"] == "listwise_nll"
    assert report["transition_candidate_training"]["positive_mode"] == "gold_plus_equivalent"
    assert report["transition_skill_ce"]["inventory_mask_mode"] == "auto"
    assert report["transition_skill_ce"]["positive_mode"] == "gold_plus_equivalent"
    assert metrics[-1]["transition_inventory_mask_mode"] == "auto"
    assert metrics[-1]["transition_loss_type"] == "listwise_nll"
    assert metrics[-1]["transition_positive_mode"] == "gold_plus_equivalent"


def test_stage1_handoff_can_prepare_only_rows_reached_by_training_budget_and_reports_progress(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    setup_status_path = tmp_path / "stage1" / "setup_status.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a", "description": "a"},
            {"skill_id": "skill/b", "name": "b", "description": "b"},
            {"skill_id": "skill/c", "name": "c", "description": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": f"budgeted-{idx}",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            }
            for idx in range(10)
        ],
    )

    report = train_clstr_stage1_heads_with_model(
        model=_CandidateOnlyModel(),
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"stage0_checkpoint": "stage0.pt"},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "stage1",
        max_steps=2,
        batch_size=2,
        learning_rate=1.0e-3,
        seed=3,
        loss_weights={"routing": 0.0, "L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0},
        stage0_top_m=2,
        stage0_positive_missing_policy="skip",
        stage0_handoff_sample_multiplier=1.0,
        stage0_handoff_cache_mode="off",
        stage0_candidate_encode_batch_size=2,
        stage0_candidate_progress_interval_batches=1,
        setup_status_path=setup_status_path,
    )

    subset_report = report["stage0_candidate_handoff_subset"]
    assert subset_report["enabled"] is True
    assert subset_report["source_rows"] == 10
    assert 0 < subset_report["selected_rows"] < 10
    assert report["stage0_candidate_handoff"]["source_rows"] == subset_report["selected_rows"]

    phases = [json.loads(line)["phase"] for line in setup_status_path.read_text(encoding="utf-8").splitlines()]
    assert "stage0_candidate_handoff_started" in phases
    assert "stage0_candidate_handoff_progress" in phases
    assert "training_started" in phases


def test_full_base_training_requires_stage0_topm_unless_explicit_debug(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "requires-stage0",
                "state_text": "state-a",
                "action_text": "do",
                "skill_id": "skill/a",
                "loss_mask": {"routing": True},
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            }
        ],
    )

    with pytest.raises(ValueError, match="Stage2 requires Stage0 top-M candidates"):
        train_clstr_full_base_with_model(
            model=_Stage0TopMModel(),
            model_config={"d": 3, "base_model_name": "tiny-test"},
            routing_report={"stage0_checkpoint": "stage0.pt"},
            train_path=train_path,
            skills_path=skills_path,
            output_dir=tmp_path / "out",
            max_steps=1,
            batch_size=1,
            learning_rate=1.0e-3,
            seed=3,
            loss_weights={"routing": 1.0, "L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        )


def test_generic_stage2_cli_exposes_stage0_topm_handoff_without_qwen():
    script = Path("scripts/run_clstr_stage2_full_base_train.py")

    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "run_clstr_full_base_train" in text
    assert "run_clstr_qwen3_full_base_train" not in text
    assert "qwen_model_path" not in text
    assert "--stage0_top_m" in text
    assert "--stage0_positive_missing_policy" in text
    assert "STAGE0_HANDOFF_QUERY_MODES" in text
    assert "choices=sorted(STAGE0_HANDOFF_QUERY_MODES)" in text
    assert "--stage0_handoff_cache_mode" in text
    assert "--stage0_handoff_cache_dir" in text
    assert "--allow_full_pool_stage2_debug" in text
    assert "--stage1_checkpoint_path" in text
    assert "--policy_hard_negative_margin_loss_weight" in text
    assert "--q_success_loss_weight" in text
    assert "--transition_inventory_min_candidates" in text
    assert "--next_skill_pool_mode" in text
    assert "--counterfactual_utility_loss_weight" in text
    assert "--counterfactual_gain_margin" in text
    assert "--counterfactual_safety_tolerance" in text
    assert "--counterfactual_gain_weight" in text
    assert "--counterfactual_safety_weight" in text
    assert "--counterfactual_warmup_fraction" in text
    assert 'default="full_pool"' in text
    assert 'default="explicit_only"' in text
    assert 'default="unified_memory"' in text
    assert 'default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl"' in text
    assert 'default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl"' in text
    assert 'default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025"' in text
    assert 'default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt"' in text
    assert 'default="outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init/checkpoints/clstr_stage1_heads-step3000.pt"' in text
    assert "data/clstr_unified_pretrain_v3_trajectory_retrieval_caps" not in text
    assert "clstr_unified_stage2_v3_traj_retrieval_full_base" not in text
    assert '"hard_negative_margin": args.policy_hard_negative_margin_loss_weight' in text
    assert '"Q_success": args.q_success_loss_weight' in text
    assert "transition_inventory_min_candidates=args.transition_inventory_min_candidates" in text
    assert '"counterfactual_utility": args.counterfactual_utility_loss_weight' in text
    assert "next_skill_pool_mode=args.next_skill_pool_mode" in text
    assert "counterfactual_warmup_fraction=args.counterfactual_warmup_fraction" in text
    assert "stage0_handoff_cache_mode=args.stage0_handoff_cache_mode" in text
    assert "stage0_handoff_cache_dir=Path(args.stage0_handoff_cache_dir)" in text
    assert "--stage0_handoff_cache_format" in text
    assert "--stage0_handoff_cache_shard_size" in text
    assert "stage0_handoff_cache_format=args.stage0_handoff_cache_format" in text
    assert "stage0_handoff_cache_shard_size=args.stage0_handoff_cache_shard_size" in text
    assert "--resume_checkpoint_path" in text
    assert "resume_checkpoint_path=Path(args.resume_checkpoint_path) if args.resume_checkpoint_path else None" in text
    assert "resume_checkpoint_path" in inspect.signature(run_clstr_full_base_train).parameters
    assert "next_skill_pool_mode" in inspect.signature(run_clstr_full_base_train).parameters
    assert "counterfactual_warmup_fraction" in inspect.signature(run_clstr_full_base_train).parameters


def test_stage1_heads_cli_exposes_sampling_and_benchmark_caps_for_unified_data():
    script = Path("scripts/run_clstr_stage1_heads_init.py")

    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "--benchmark_caps" in text
    assert "--stage0_handoff_query_mode" in text
    assert "STAGE0_HANDOFF_QUERY_MODES" in text
    assert "choices=sorted(STAGE0_HANDOFF_QUERY_MODES)" in text
    assert "--sampling_strategy" in text
    assert "--transition_inventory_mask_mode" in text
    assert "--transition_inventory_min_candidates" in text
    assert "--transition_loss_type" in text
    assert "--transition_positive_mode" in text
    assert "--transition_hard_negative_margin_loss_weight" in text
    assert "--transition_hard_negative_margin" in text
    assert "--route_scorer" in text
    assert "--resume_checkpoint_path" in text
    assert "SKILL_PRIOR_TRANSITION_SCORING_MODE" in text
    assert "default=SKILL_PRIOR_TRANSITION_SCORING_MODE" in text
    assert 'parser.add_argument("--transition_hard_negative_margin_loss_weight", type=float, default=0.0)' in text
    assert 'default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl"' in text
    assert 'default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl"' in text
    assert 'default="outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init"' in text
    assert "data/clstr_unified_pretrain_v3_trajectory_retrieval_caps" not in text
    assert "clstr_unified_stage1_v3_traj_retrieval_heads_init" not in text
    assert "benchmark_caps=_parse_benchmark_caps(args.benchmark_caps)" in text
    assert "sampling_strategy=args.sampling_strategy" in text
    assert "stage0_handoff_query_mode=args.stage0_handoff_query_mode" in text
    assert "transition_inventory_mask_mode=args.transition_inventory_mask_mode" in text
    assert "transition_inventory_min_candidates=args.transition_inventory_min_candidates" in text
    assert "transition_loss_type=args.transition_loss_type" in text
    assert "transition_positive_mode=args.transition_positive_mode" in text
    assert '"transition_hard_negative_margin": args.transition_hard_negative_margin_loss_weight' in text
    assert "transition_hard_negative_margin=args.transition_hard_negative_margin" in text
    assert "route_scorer=args.route_scorer" in text
    assert "--stage0_handoff_cache_mode" in text
    assert "--stage0_handoff_cache_dir" in text
    assert "--stage0_handoff_cache_format" in text
    assert "--stage0_handoff_cache_shard_size" in text
    assert "stage0_handoff_cache_format=args.stage0_handoff_cache_format" in text
    assert "stage0_handoff_cache_shard_size=args.stage0_handoff_cache_shard_size" in text
    assert "resume_checkpoint_path=Path(args.resume_checkpoint_path) if args.resume_checkpoint_path else None" in text


def test_stage4_cli_exposes_checkpoint_state_query_handoff_mode():
    text = Path("scripts/run_clstr_stage4_act_train.py").read_text(encoding="utf-8")

    assert "STAGE0_HANDOFF_QUERY_MODES" in text
    assert "choices=sorted(STAGE0_HANDOFF_QUERY_MODES)" in text
    assert "stage0_handoff_query_mode=args.stage0_handoff_query_mode" in text
    assert "--stage0_handoff_cache_mode" in text
    assert "--stage0_handoff_cache_dir" in text
    assert "--stage0_handoff_cache_format" in text
    assert "--stage0_handoff_cache_shard_size" in text
    assert "stage0_handoff_cache_format=args.stage0_handoff_cache_format" in text
    assert "stage0_handoff_cache_shard_size=args.stage0_handoff_cache_shard_size" in text


def test_stage1_heads_python_defaults_match_progressive_mainline():
    for func in (run_clstr_stage1_heads_init, train_clstr_stage1_heads_with_model):
        signature = inspect.signature(func)
        assert signature.parameters["stage0_top_m"].default == 350
        assert signature.parameters["transition_inventory_mask_mode"].default == "auto"
        assert signature.parameters["transition_inventory_min_candidates"].default == 64
        assert signature.parameters["transition_loss_type"].default == "listwise_nll"
        assert signature.parameters["transition_positive_mode"].default == "gold_plus_equivalent"
        assert signature.parameters["sampling_strategy"].default == "balanced_random"
        assert signature.parameters["stage0_handoff_cache_format"].default == "legacy_jsonl"
        assert signature.parameters["stage0_handoff_cache_shard_size"].default == 2048

    from clstr.stage4_act_train import train_stage4_act_with_model

    stage4_signature = inspect.signature(train_stage4_act_with_model)
    assert stage4_signature.parameters["stage0_handoff_cache_format"].default == "legacy_jsonl"
    assert stage4_signature.parameters["stage0_handoff_cache_shard_size"].default == 2048

    source = inspect.getsource(train_clstr_stage1_heads_with_model)
    assert '"L_policy": 0.7' in source
    assert '"L_trans": 0.2' in source
    assert '"L_trans_skill_ce": 0.5' in source
    assert '"STOP": 0.1' in source
    assert '"transition_hard_negative_margin": 0.0' in source
