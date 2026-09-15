from __future__ import annotations

from pathlib import Path

import pytest
import torch

from clstr.current_state_route_eval import (
    _build_current_state_route_batch,
    _compute_current_state_route_loss,
)


class _CurrentStateSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3), requires_grad=False)

    def belief_logits(self, h):
        return h @ self.E.to(device=h.device, dtype=h.dtype).t()

    def forward(self, h):
        return self.belief_logits(h)


class _RecordingTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, m_t, action_input, observation_embedding):
        self.calls.append(
            (
                m_t.detach().clone(),
                action_input.detach().clone(),
                observation_embedding.detach().clone(),
            )
        )
        return m_t + action_input + observation_embedding


class _ZeroGate(torch.nn.Module):
    def forward(self, predicted, observation_memory, observation_embedding):
        del observation_memory, observation_embedding
        return torch.zeros_like(predicted)


class _FixedResidualAdapter(torch.nn.Module):
    def forward(self, h_t, memory_delta):
        del memory_delta
        residual = torch.zeros_like(h_t)
        residual[:, 2] = 10.0
        return residual


class _FixedCandidateGate(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, features):
        return features.new_full((features.size(0),), self.alpha)


class _RecordingCandidateAdmissionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.widths = []

    def forward(self, h, memory_delta, candidate_embeddings, scalar_features):
        del h, memory_delta, scalar_features
        self.widths.append(int(candidate_embeddings.size(1)))
        admission = candidate_embeddings.new_full(
            candidate_embeddings.shape[:2],
            4.0,
        )
        residual = 4.0 * (
            candidate_embeddings[..., 2] - candidate_embeddings[..., 0]
        )
        return admission, residual


class _CurrentStateRouteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _CurrentStateSkillTable()
        self.action_proj = torch.nn.Linear(3, 3, bias=False)
        self.transition = _RecordingTransition()
        self.gate = _ZeroGate()
        self.initial_belief_calls = []
        self.route_calls = []
        self.route_outputs = []
        self.encoded_texts = []
        with torch.no_grad():
            self.action_proj.weight.copy_(torch.eye(3))
        self.route_alpha = 0.0

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            text = str(text)
            self.encoded_texts.append(text)
            lowered = text.lower()
            if "prefix action alpha" in lowered:
                row = [0.0, 0.5, 0.0]
            elif "prefix action beta" in lowered:
                row = [0.0, 0.0, 0.5]
            elif "prefix next alpha" in lowered:
                row = [0.0, 1.0, 0.0]
            elif "prefix next beta" in lowered:
                row = [0.0, 0.0, 1.0]
            else:
                row = [1.0, 0.0, 0.0]
            rows.append(torch.tensor(row, dtype=torch.float32, device=self.device))
        return torch.stack(rows) + self.anchor * 0.0

    def initial_belief(self, h_t, top_k=None):
        del top_k
        self.initial_belief_calls.append(h_t.detach().clone())
        return h_t

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = self.unified_route_full_logits(h_t, m_t)
        if candidate_rows is None:
            output = full_logits
        else:
            output = self.gather_unified_route_logits(full_logits, candidate_rows)
        self.route_outputs.append(output.detach().clone())
        return output

    def unified_route_full_logits(self, h_t, m_t):
        return h_t + 2.0 * m_t

    def gather_unified_route_logits(self, full_logits, candidate_rows):
        width = max((len(row) for row in candidate_rows), default=0)
        output = torch.full(
            (len(candidate_rows), width),
            torch.finfo(full_logits.dtype).min,
            dtype=full_logits.dtype,
            device=full_logits.device,
        )
        for row_idx, row in enumerate(candidate_rows):
            if row:
                ids = torch.tensor(row, dtype=torch.long, device=full_logits.device)
                output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output

    def route_memory_alpha(self, h_t, static_memory, dynamic_memory, causal_update_count):
        del h_t, static_memory, dynamic_memory
        raw = causal_update_count.new_full(causal_update_count.shape, float(self.route_alpha))
        return torch.where(causal_update_count > 0, raw, torch.zeros_like(raw))


def _route_row(**overrides):
    row = {
        "state_text": "current route state",
        "action_text": "current action alpha",
        "next_observation_text": "current observation alpha",
        "skill_id": "skill/a",
        "skill_idx": 0,
        "next_skill_id": "skill/b",
        "positive_next_skill_idx": 1,
        "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "candidate_next_skill_indices": [0, 1, 2],
        "replay_prefix": [
            {
                "observation_text": "prefix initial",
                "action_text": "prefix action alpha",
                "next_observation_text": "prefix next alpha",
                "skill_id": "skill/a",
                "skill_idx": 0,
            }
        ],
    }
    row.update(overrides)
    return row


def _evaluate_row(**overrides):
    model = _CurrentStateRouteModel()
    loss, metrics = _compute_current_state_route_loss(
        model,
        [_route_row(**overrides)],
        torch.device("cpu"),
    )
    return model, loss.detach(), metrics


def test_current_state_route_eval_uses_state_and_replayed_memory_without_current_action_update():
    model, _loss, metrics = _evaluate_row()

    assert len(model.transition.calls) == 1
    assert len(model.route_calls) == 2
    dynamic_h, dynamic_m, dynamic_candidates = model.route_calls[0]
    static_h, static_m, static_candidates = model.route_calls[1]
    assert torch.equal(dynamic_h, model.encode_observations(["current route state"]))
    assert torch.equal(static_h, dynamic_h)
    assert not torch.equal(dynamic_m, static_m)
    assert dynamic_candidates == static_candidates == [[0, 1, 2]]
    assert "current action alpha" not in model.encoded_texts
    assert "current observation alpha" not in model.encoded_texts
    assert metrics["current_state_route_count"] == 1.0
    assert metrics["current_state_replay_prefix_used_count"] == 1.0
    assert metrics["stage4_post_action_update_rows"] == 0.0
    assert metrics["stage4_next_state_rows"] == 0.0
    assert metrics["stage4_score_calibrator_enabled"] is False
    assert metrics["stage4_replay_prefix_trainable_enabled"] is False
    assert metrics["online_memory_weight"] == 0.0
    assert metrics["stage4_act_count"] == 1.0
    assert metrics["route_scorer"] == "unified_memory"


def test_current_state_route_eval_ignores_current_row_action_and_observation_counterfactuals():
    baseline_model, baseline_loss, _metrics = _evaluate_row(
        action_text="current action alpha",
        next_observation_text="current observation alpha",
    )
    changed_model, changed_loss, _metrics = _evaluate_row(
        action_text="current action beta",
        next_observation_text="current observation beta",
    )

    assert torch.equal(baseline_model.route_outputs[0], changed_model.route_outputs[0])
    assert torch.equal(baseline_model.route_outputs[1], changed_model.route_outputs[1])
    assert torch.equal(baseline_loss, changed_loss)


def test_current_state_route_eval_changes_dynamic_not_static_logits_when_replay_changes():
    baseline_model, _baseline_loss, _metrics = _evaluate_row()
    changed_model, _changed_loss, _metrics = _evaluate_row(
        replay_prefix=[
            {
                "observation_text": "prefix initial",
                "action_text": "prefix action beta",
                "next_observation_text": "prefix next beta",
                "skill_id": "skill/a",
                "skill_idx": 0,
            }
        ]
    )

    assert not torch.equal(baseline_model.route_outputs[0], changed_model.route_outputs[0])
    assert torch.equal(baseline_model.route_outputs[1], changed_model.route_outputs[1])


def test_current_state_route_eval_accepts_row_local_equivalent_positive():
    _model, loss, metrics = _evaluate_row(
        equivalent_next_skill_ids=["skill/a"],
    )

    assert torch.isfinite(loss)
    assert metrics["stage4_next_skill_recall@1"] == 1.0
    assert metrics["stage4_next_skill_mrr"] == 1.0


def test_current_state_route_eval_requires_unified_memory_interfaces():
    with pytest.raises(ValueError, match="initial_belief and model.unified_route_logits"):
        _compute_current_state_route_loss(
            torch.nn.Linear(3, 3),
            [_route_row()],
            torch.device("cpu"),
        )


def test_current_state_route_eval_candidate_union_rescues_static_miss_strictly():
    model = _CurrentStateRouteModel()

    loss, metrics = _compute_current_state_route_loss(
        model,
        [
            _route_row(
                replay_prefix=[
                    {
                        "observation_text": "prefix initial",
                        "action_text": "prefix action alpha",
                        "next_observation_text": "prefix next alpha",
                        "skill_id": "skill/a",
                        "skill_idx": 0,
                    },
                    {
                        "observation_text": "prefix next alpha",
                        "action_text": "prefix action alpha",
                        "next_observation_text": "prefix next alpha",
                        "skill_id": "skill/a",
                        "skill_idx": 0,
                    },
                ]
            )
        ],
        torch.device("cpu"),
        candidate_recall_mode="static_plus_dynamic_extra",
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=1,
        dynamic_extra_k=1,
        final_k=1,
    )

    assert torch.isfinite(loss)
    assert metrics["candidate_recall_all_source_rows"] == 1.0
    assert metrics["candidate_recall_all_static_recall"] == 0.0
    assert metrics["candidate_recall_all_union_recall"] == 1.0
    assert metrics["candidate_recall_all_dynamic_rescue_rows"] == 1.0
    assert metrics["candidate_recall_memory_active_union_recall"] == 1.0
    assert metrics["candidate_recall_memory_active_coverage"] == 1.0
    assert metrics["stage4_next_skill_recall@1"] == 1.0
    assert metrics["stage4_candidate_count"] == 1.0
    assert metrics["candidate_recall_mode"] == "static_plus_dynamic_extra"


def test_current_state_route_eval_candidate_union_rejects_inexact_static_fallback_budget():
    with pytest.raises(ValueError, match="final_k must not exceed static_k"):
        _compute_current_state_route_loss(
            _CurrentStateRouteModel(),
            [_route_row()],
            torch.device("cpu"),
            candidate_recall_mode="static_plus_dynamic_extra",
            skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
            static_k=1,
            dynamic_extra_k=1,
            final_k=2,
        )


def _build_reliability_output(
    *,
    reliability_mode: str,
    fixed_alpha: float = 1.0,
    replay_prefix=None,
):
    model = _CurrentStateRouteModel()
    row = _route_row()
    if replay_prefix is not None:
        row["replay_prefix"] = replay_prefix
    output = _build_current_state_route_batch(
        model,
        [row],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=1,
        dynamic_extra_k=1,
        final_k=1,
        reliability_mode=reliability_mode,
        fixed_alpha=fixed_alpha,
        memory_utility_gate=None,
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
    )
    return model, output


def test_current_state_reliability_modes_have_exact_endpoints_and_do_not_mutate_memory():
    replay_prefix = [
        {
            "observation_text": "prefix initial",
            "action_text": "prefix action alpha",
            "next_observation_text": "prefix next alpha",
            "skill_id": "skill/a",
            "skill_idx": 0,
        },
        {
            "observation_text": "prefix next alpha",
            "action_text": "prefix action alpha",
            "next_observation_text": "prefix next alpha",
            "skill_id": "skill/a",
            "skill_idx": 0,
        },
    ]
    _static_model, static_output = _build_reliability_output(
        reliability_mode="static",
        replay_prefix=replay_prefix,
    )
    _dynamic_model, dynamic_output = _build_reliability_output(
        reliability_mode="dynamic",
        replay_prefix=replay_prefix,
    )
    _fixed_model, fixed_output = _build_reliability_output(
        reliability_mode="fixed_alpha",
        fixed_alpha=0.25,
        replay_prefix=replay_prefix,
    )
    _heuristic_model, heuristic_output = _build_reliability_output(
        reliability_mode="heuristic",
        replay_prefix=replay_prefix,
    )

    assert torch.equal(
        static_output.fused_candidate_logits,
        static_output.static_candidate_logits,
    )
    assert torch.equal(
        dynamic_output.fused_candidate_logits,
        dynamic_output.dynamic_candidate_logits,
    )
    assert fixed_output.metrics["memory_utility_alpha_mean"] == pytest.approx(0.25)
    assert torch.equal(static_output.dynamic_memory, dynamic_output.dynamic_memory)
    assert torch.equal(static_output.dynamic_memory, fixed_output.dynamic_memory)
    assert torch.equal(static_output.dynamic_memory, heuristic_output.dynamic_memory)
    assert static_output.metrics["stage4_next_skill_recall@1"] == 0.0
    assert dynamic_output.metrics["stage4_next_skill_recall@1"] == 1.0
    assert heuristic_output.features.shape == (1, 11)
    assert heuristic_output.metrics["reliability_changes_memory_state"] is False


def test_causal_gate_keeps_raw_dynamic_candidate_recall_but_can_rank_exact_static() -> None:
    model, output = _build_reliability_output(
        reliability_mode="causal_gate",
        replay_prefix=[
            {
                "observation_text": "prefix initial",
                "action_text": "prefix action alpha",
                "next_observation_text": "prefix next alpha",
                "skill_id": "skill/a",
                "skill_idx": 0,
            }
        ],
    )

    assert model.route_alpha == 0.0
    assert output.metrics["candidate_recall_all_dynamic_rescue_rows"] == 1.0
    assert output.candidate_union.dynamic_extra_rows == [[1]]
    assert output.effective_alpha.tolist() == [0.0]
    assert torch.equal(output.fused_candidate_logits, output.static_candidate_logits)
    assert output.metrics["memory_utility_reliability_mode"] == "causal_gate"
    assert output.metrics["safe_memory_residual_bound"] == 2.0


def test_cmc_adapter_drives_dynamic_extra_recall_before_candidate_gate_fusion() -> None:
    row = _route_row(next_skill_id="skill/c", positive_next_skill_idx=2)
    model = _CurrentStateRouteModel()
    model.route_memory_residual_adapter = _FixedResidualAdapter()
    model.route_memory_candidate_utility_gate = _FixedCandidateGate(1.0)

    dynamic_output = _build_current_state_route_batch(
        model,
        [row],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=1,
        dynamic_extra_k=1,
        final_k=1,
        reliability_mode="cmc",
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
    )

    assert dynamic_output.candidate_union.dynamic_extra_rows == [[2]]
    assert not torch.equal(
        dynamic_output.raw_dynamic_full_logits,
        dynamic_output.dynamic_full_logits,
    )
    assert dynamic_output.effective_alpha.tolist() == [1.0]
    assert torch.equal(
        dynamic_output.fused_candidate_logits,
        dynamic_output.dynamic_candidate_logits,
    )
    assert dynamic_output.metrics["stage4_next_skill_recall@1"] == 1.0

    static_model = _CurrentStateRouteModel()
    static_model.route_memory_residual_adapter = _FixedResidualAdapter()
    static_model.route_memory_candidate_utility_gate = _FixedCandidateGate(0.0)
    static_output = _build_current_state_route_batch(
        static_model,
        [row],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=1,
        dynamic_extra_k=1,
        final_k=1,
        reliability_mode="cmc",
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
    )
    assert static_output.effective_alpha.tolist() == [0.0]
    assert torch.equal(
        static_output.fused_candidate_logits,
        static_output.static_candidate_logits,
    )


def test_cmc_external_gate_overrides_only_alpha_and_keeps_residual_endpoint() -> None:
    row = _route_row(next_skill_id="skill/c", positive_next_skill_idx=2)
    model = _CurrentStateRouteModel()
    model.route_memory_residual_adapter = _FixedResidualAdapter()
    model.route_memory_candidate_utility_gate = _FixedCandidateGate(0.0)

    output = _build_current_state_route_batch(
        model,
        [row],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=1,
        dynamic_extra_k=1,
        final_k=1,
        reliability_mode="cmc",
        memory_utility_gate=_FixedCandidateGate(1.0),
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
    )

    assert output.metrics["cmc_adapter_enabled"] is True
    assert output.metrics["memory_utility_external_gate_loaded"] is True
    assert output.raw_alpha.tolist() == [1.0]
    assert not torch.equal(output.dynamic_full_logits, output.raw_dynamic_full_logits)
    assert torch.equal(output.fused_candidate_logits, output.dynamic_candidate_logits)


def test_cmc_zero_history_forces_exact_static_even_with_active_gate() -> None:
    model = _CurrentStateRouteModel()
    model.route_memory_residual_adapter = _FixedResidualAdapter()
    model.route_memory_candidate_utility_gate = _FixedCandidateGate(1.0)
    output = _build_current_state_route_batch(
        model,
        [_route_row(replay_prefix=[])],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=1,
        dynamic_extra_k=1,
        final_k=1,
        reliability_mode="cmc",
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
    )

    assert output.effective_alpha.item() == 0.0
    assert torch.equal(output.fused_candidate_logits, output.static_candidate_logits)


def test_cmc_candidate_provenance_boosts_only_dynamic_extra_before_and_after_final_k() -> None:
    row = _route_row(next_skill_id="skill/c", positive_next_skill_idx=2)
    model = _CurrentStateRouteModel()
    model.route_memory_residual_adapter = _FixedResidualAdapter()
    model.route_memory_candidate_utility_gate = _FixedCandidateGate(0.0)

    output = _build_current_state_route_batch(
        model,
        [row],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=2,
        dynamic_extra_k=1,
        final_k=2,
        reliability_mode="cmc_candidate_provenance",
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
        safe_memory_residual_bound=2.0,
    )

    assert output.candidate_union.dynamic_extra_rows == [[2]]
    assert output.candidate_union.candidate_rows == [[0, 2]]
    assert output.fused_candidate_logits[0, 0] == output.static_candidate_logits[0, 0]
    assert output.fused_candidate_logits[0, 1] > output.static_candidate_logits[0, 1]
    assert (
        output.fused_candidate_logits[0, 1]
        - output.static_candidate_logits[0, 1]
        <= 2.0 + 1.0e-6
    )
    assert output.metrics["stage4_next_skill_recall@1"] == 0.0
    assert output.metrics["candidate_recall_all_dynamic_rescue_rows"] == 1.0
    assert output.metrics["memory_utility_reliability_mode"] == "cmc_candidate_provenance"
    assert output.metrics["cmc_adapter_enabled"] is True
    assert output.metrics["memory_utility_external_gate_loaded"] is False


def test_candidate_admission_scores_union_before_and_after_final_k() -> None:
    row = _route_row(next_skill_id="skill/c", positive_next_skill_idx=2)
    model = _CurrentStateRouteModel()
    model.route_memory_residual_adapter = _FixedResidualAdapter()
    model.route_memory_candidate_admission_residual = (
        _RecordingCandidateAdmissionHead()
    )

    output = _build_current_state_route_batch(
        model,
        [row],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=2,
        dynamic_extra_k=1,
        final_k=2,
        reliability_mode="candidate_admission_residual",
        feature_update_count_cap=16.0,
        feature_candidate_count_cap=256.0,
    )

    assert model.route_memory_candidate_admission_residual.widths == [3, 2]
    assert output.candidate_union.dynamic_extra_rows == [[2]]
    assert output.metrics["memory_utility_reliability_mode"] == (
        "candidate_admission_residual"
    )
    assert output.metrics["cmc_adapter_enabled"] is True
    assert output.metrics["stage4_next_skill_recall@1"] == 1.0


def test_candidate_admission_zero_history_is_exact_static() -> None:
    model = _CurrentStateRouteModel()
    model.route_memory_residual_adapter = _FixedResidualAdapter()
    model.route_memory_candidate_admission_residual = (
        _RecordingCandidateAdmissionHead()
    )

    output = _build_current_state_route_batch(
        model,
        [_route_row(replay_prefix=[])],
        torch.device("cpu"),
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={},
        static_k=2,
        dynamic_extra_k=1,
        final_k=2,
        reliability_mode="candidate_admission_residual",
        feature_update_count_cap=16.0,
        feature_candidate_count_cap=256.0,
    )

    assert torch.equal(
        output.fused_candidate_logits,
        output.static_candidate_logits,
    )


@pytest.mark.parametrize("mode", ["dynamic", "fixed_alpha", "heuristic"])
def test_current_state_reliability_zero_history_forces_exact_static(mode):
    _model, output = _build_reliability_output(
        reliability_mode=mode,
        fixed_alpha=0.8,
        replay_prefix=[],
    )

    assert output.causal_update_count.item() == 0.0
    assert output.effective_alpha.item() == 0.0
    assert torch.equal(output.fused_candidate_logits, output.static_candidate_logits)
    assert output.metrics["zero_history_fallback"] == "exact_static"


def test_current_state_reliability_learned_mode_requires_gate_checkpoint():
    with pytest.raises(ValueError, match="learned reliability mode requires"):
        _build_reliability_output(reliability_mode="learned")


def test_active_current_state_route_surfaces_do_not_default_to_legacy_scorer():
    paths = [
        Path("clstr/toolbench_full_clstr_route_eval.py"),
        Path("clstr/tau2_route_eval.py"),
        Path("clstr/toolsandbox_route_eval.py"),
        Path("clstr/apibank_route_eval.py"),
        Path("clstr/global_pool_route_eval.py"),
        Path("scripts/run_toolbench_g3_full_clstr_route_eval.py"),
        Path("scripts/run_tau2_full_clstr_route_eval.py"),
        Path("scripts/run_tau3_full_clstr_route_eval.py"),
        Path("scripts/run_toolsandbox_full_clstr_route_eval.py"),
        Path("scripts/run_apibank_full_clstr_route_eval.py"),
        Path("scripts/run_global_pool_clstr_route_eval.py"),
        Path("scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh"),
        Path("scripts/sbatch/run_tau2_full_clstr_route_eval.sh"),
        Path("scripts/sbatch/run_tau3_full_clstr_route_eval.sh"),
        Path("scripts/sbatch/run_toolsandbox_full_clstr_route_eval.sh"),
        Path("scripts/sbatch/run_apibank_full_clstr_route_eval.sh"),
        Path("scripts/sbatch/run_global_pool_clstr_route_eval.sh"),
        Path("scripts/sbatch/run_trajectbench_full_clstr_eval.sh"),
    ]
    forbidden = (
        "default=LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER",
        'default="legacy_prior_residual"',
        "or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER",
        ":-legacy_prior_residual",
    )

    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits.extend((str(path), token) for token in forbidden if token in text)

    assert hits == []
