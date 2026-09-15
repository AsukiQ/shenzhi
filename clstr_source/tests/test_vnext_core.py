from __future__ import annotations

import torch
import torch.nn.functional as F

from clstr.belief import subspace_obs
from clstr.model import UnifiedMemoryRetriever
from clstr.stage_checkpoint_init import load_compatible_state_dict
from clstr.vnext_core import CLSTRVNextCore, LatentTraceSynchronization


class _DummySkillTable:
    def __init__(self) -> None:
        self.E = torch.eye(3)
        self.belief_top_k = 3

    def belief_logits(self, h: torch.Tensor) -> torch.Tensor:
        return h


def test_subspace_obs_masks_unavailable_skills_before_belief() -> None:
    table = _DummySkillTable()
    h = torch.tensor([[9.0, 2.0, 1.0]])
    mask = torch.tensor([[False, True, False]])
    belief = subspace_obs(table, h, valid_mask=mask)
    assert torch.allclose(belief, torch.tensor([[0.0, 1.0, 0.0]]))


def test_subspace_obs_rejects_empty_runtime_inventory() -> None:
    table = _DummySkillTable()
    h = torch.tensor([[1.0, 2.0, 3.0]])
    try:
        subspace_obs(table, h, valid_mask=torch.zeros(1, 3, dtype=torch.bool))
    except ValueError as exc:
        assert "runtime-visible" in str(exc)
    else:
        raise AssertionError("empty runtime inventory must fail closed")


def test_vnext_zero_history_reuses_static_queries_exactly() -> None:
    torch.manual_seed(3)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    output = core.queries(
        h,
        dynamic_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(output.dynamic_recall, output.static_recall)
    assert torch.equal(output.dynamic_route, output.static_route)
    assert torch.count_nonzero(output.static_route_delta) == 0
    assert torch.count_nonzero(output.recall_delta) == 0
    assert torch.count_nonzero(output.route_delta) == 0


def test_vnext_history_memory_changes_recall_query() -> None:
    torch.manual_seed(4)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    output = core.queries(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
    )
    assert torch.count_nonzero(output.recall_delta) > 0
    assert torch.count_nonzero(output.route_delta) > 0
    assert not torch.equal(output.recall_delta, output.route_delta)
    assert not torch.equal(output.dynamic_recall, output.static_recall)


def test_unified_static_query_is_legacy_weight_and_logit_compatible() -> None:
    torch.manual_seed(5)
    legacy = UnifiedMemoryRetriever(8)
    core = CLSTRVNextCore(8, hidden_dim=4)
    core.static_query.load_state_dict(legacy.state_dict(), strict=True)
    h = torch.randn(3, 8)
    memory = torch.randn(3, 8)
    skills = F.normalize(torch.randn(11, 8), dim=-1)
    legacy_query = legacy(h, memory)
    recall_query, route_query = core.static_queries(h, memory)
    assert torch.equal(recall_query, route_query)
    assert torch.allclose(recall_query, legacy_query, atol=0.0, rtol=0.0)
    assert torch.allclose(
        core.full_pool_logits(
            recall_query,
            skills,
            head="recall",
            skill_embeddings_are_normalized=True,
        ),
        legacy_query @ skills.t(),
        atol=0.0,
        rtol=0.0,
    )


def test_candidate_compressor_initializes_to_exact_coarse_ordering() -> None:
    torch.manual_seed(31)
    core = CLSTRVNextCore(8, hidden_dim=4)
    causal_query = torch.randn(2, 8)
    current_state = torch.randn(2, 8)
    candidates = torch.randn(2, 5, 8)
    base_logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 2.0, 4.0, 5.0]]
    )
    valid = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    output = core.candidate_compression_scores(
        causal_query,
        current_state,
        candidates,
        base_logits,
        valid,
    )
    assert torch.equal(output.logits[valid], base_logits[valid])
    assert torch.count_nonzero(output.bounded_delta) == 0
    assert torch.isneginf(output.logits[~valid]).all() or bool(
        (output.logits[~valid] == torch.finfo(output.logits.dtype).min).all()
    )


def test_vnext_memory_update_is_finite_and_result_gate_is_scalar() -> None:
    torch.manual_seed(5)
    core = CLSTRVNextCore(8, hidden_dim=4)
    values = [torch.randn(3, 8) for _ in range(5)]
    without_result = core.update_memory(*values[:4])
    with_result = core.update_memory(*values[:4], result_embedding=values[4])
    assert torch.isfinite(without_result.effective_memory).all()
    assert torch.isfinite(with_result.effective_memory).all()
    assert without_result.correction_delta is None
    assert without_result.correction_beta is None
    assert with_result.correction_delta is not None
    assert with_result.correction_beta is not None
    assert with_result.correction_beta.shape == (3, 1)
    assert bool(((with_result.correction_beta > 0) & (with_result.correction_beta < 1)).all())
    assert torch.allclose(
        with_result.correction_beta.mean(),
        torch.tensor(0.5),
        atol=0.02,
    )


def test_native_synchronization_is_opt_in_and_keeps_legacy_updates_unchanged() -> None:
    torch.manual_seed(101)
    core = CLSTRVNextCore(8, hidden_dim=4)
    memory, state, skill, action = [torch.randn(2, 8) for _ in range(4)]
    update = core.update_memory(memory, state, skill, action)
    assert core.synchronization_enabled is False
    assert core.synchronization is None
    assert update.latent_trace is None


def test_native_synchronization_preserves_configured_initial_decay_on_reset() -> None:
    synchronization = LatentTraceSynchronization(
        8,
        pair_dim=5,
        trace_length=3,
        initial_decay=0.8,
    )
    assert synchronization.initial_decay == 0.8
    assert torch.allclose(synchronization.decay(), torch.full((5,), 0.8), atol=1.0e-6)
    synchronization.reset_parameters(scale_initial=0.05)
    assert torch.allclose(synchronization.decay(), torch.full((5,), 0.8), atol=1.0e-6)


def test_native_synchronization_keeps_a_fixed_recent_post_action_trace() -> None:
    torch.manual_seed(102)
    core = CLSTRVNextCore(
        8,
        hidden_dim=4,
        synchronization_enabled=True,
        synchronization_pair_dim=5,
        synchronization_trace_length=3,
    )
    memory, state, skill, action = [torch.randn(2, 8) for _ in range(4)]
    first = core.update_memory(memory, state, skill, action)
    assert first.latent_trace is not None
    assert first.latent_trace.shape == (2, 3, 8)
    assert torch.count_nonzero(first.latent_trace[:, :2]) == 0
    assert torch.equal(first.latent_trace[:, -1], first.effective_memory)

    second = core.update_memory(
        first.effective_memory,
        state,
        skill,
        action,
        latent_trace=first.latent_trace,
    )
    assert second.latent_trace is not None
    assert torch.equal(second.latent_trace[:, -2], first.effective_memory)
    assert torch.equal(second.latent_trace[:, -1], second.effective_memory)


def test_native_synchronization_changes_query_readout_and_receives_gradients() -> None:
    torch.manual_seed(103)
    core = CLSTRVNextCore(
        8,
        hidden_dim=4,
        synchronization_enabled=True,
        synchronization_pair_dim=6,
        synchronization_trace_length=3,
    )
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    trace = torch.randn(2, 3, 8)
    history = torch.ones(2, dtype=torch.bool)
    without_trace = core.queries(
        h,
        dynamic_memory,
        static_memory,
        history,
    )
    with_trace = core.queries(
        h,
        dynamic_memory,
        static_memory,
        history,
        latent_trace=trace,
    )
    assert not torch.equal(with_trace.dynamic_recall, without_trace.dynamic_recall)
    assert not torch.equal(with_trace.dynamic_route, without_trace.dynamic_route)

    (with_trace.dynamic_recall.sum() + with_trace.dynamic_route.sum()).backward()
    assert core.synchronization is not None
    synchronization_gradients = [
        parameter.grad
        for parameter in core.synchronization.parameters()
        if parameter.requires_grad
    ]
    assert synchronization_gradients
    assert all(gradient is not None for gradient in synchronization_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in synchronization_gradients)


def test_native_synchronization_zero_padding_is_safe_and_inert() -> None:
    torch.manual_seed(104)
    core = CLSTRVNextCore(
        8,
        hidden_dim=4,
        synchronization_enabled=True,
        synchronization_pair_dim=5,
        synchronization_trace_length=4,
    )
    memory = core.normalize_memory(torch.randn(2, 8))
    trace = torch.zeros(2, 4, 8)
    synchronized = core.synchronized_memory(memory, trace)
    assert torch.equal(synchronized, memory)
    assert torch.isfinite(synchronized).all()


def test_native_synchronization_is_finite_under_bfloat16_autocast() -> None:
    torch.manual_seed(106)
    core = CLSTRVNextCore(
        8,
        hidden_dim=4,
        synchronization_enabled=True,
        synchronization_pair_dim=5,
        synchronization_trace_length=3,
    )
    values = [torch.randn(2, 8) for _ in range(5)]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        update = core.update_memory(*values[:4])
        queries = core.queries(
            values[1],
            update.effective_memory,
            values[0],
            torch.ones(2, dtype=torch.bool),
            latent_trace=update.latent_trace,
        )
    assert update.effective_memory.dtype == torch.bfloat16
    assert queries.dynamic_recall.dtype == torch.bfloat16
    assert torch.isfinite(update.effective_memory).all()
    assert torch.isfinite(queries.dynamic_recall).all()


def test_legacy_core_state_can_warm_start_a_sync_enabled_core() -> None:
    torch.manual_seed(105)
    legacy = CLSTRVNextCore(8, hidden_dim=4)
    native = CLSTRVNextCore(
        8,
        hidden_dim=4,
        synchronization_enabled=True,
        synchronization_pair_dim=5,
        synchronization_trace_length=3,
    )
    report = load_compatible_state_dict(
        native,
        legacy.state_dict(),
        partial_load_mode="test_native_stage2_warm_start",
    )
    assert not report["unexpected_keys"]
    assert not report["skipped_keys"]
    assert any(
        key.startswith("synchronization.") for key in report["missing_keys"]
    )


def test_vnext_autocast_aligns_float32_frozen_features_to_recurrent_dtype() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = CLSTRVNextCore(8, hidden_dim=4).to(device)
    memory = core.normalize_memory(
        torch.randn(2, 8, device=device, dtype=torch.float32)
    )
    static_memory = core.normalize_memory(
        torch.randn(2, 8, device=device, dtype=torch.float32)
    )
    state, skill, action, result = [
        torch.randn(2, 8, device=device, dtype=torch.float32)
        for _ in range(4)
    ]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        update = core.update_memory(
            memory,
            state,
            skill,
            action,
            result_embedding=result,
        )
        queries = core.queries(
            state,
            update.effective_memory,
            static_memory,
            torch.ones(2, dtype=torch.bool, device=device),
        )
    assert update.effective_memory.dtype == torch.bfloat16
    assert queries.dynamic_recall.dtype == torch.bfloat16
    assert queries.dynamic_route.dtype == torch.bfloat16
    assert torch.isfinite(update.effective_memory).all()
    assert torch.isfinite(queries.dynamic_recall).all()
    assert torch.isfinite(queries.dynamic_route).all()


def test_vnext_residual_adapters_do_not_compound_tiny_output_initialization() -> None:
    torch.manual_seed(6)
    core = CLSTRVNextCore(32, hidden_dim=16, scale_initial=0.01)
    output_layers = (
        core.transition_delta.adapter.output,
        core.correction_delta.adapter.output,
        core.memory_recall_query.query.output,
        core.memory_route_query.query.output,
    )
    for layer in output_layers:
        assert float(layer.weight.detach().std()) > 0.02
    assert torch.allclose(core.transition_scale(), torch.tensor(0.01), atol=1.0e-6)
    assert torch.allclose(core.result_scale(), torch.tensor(0.01), atol=1.0e-6)
    assert torch.allclose(core.recall_scale(), torch.tensor(0.01), atol=1.0e-6)
    assert torch.allclose(core.route_scale(), torch.tensor(0.01), atol=1.0e-6)


def test_vnext_stage2_reset_copies_direct_router_and_preserves_zero_history_fallback() -> None:
    torch.manual_seed(61)
    core = CLSTRVNextCore(16, hidden_dim=8)
    report = core.reset_stage2_recurrent_parameters(seed=71)
    h = torch.randn(2, 16)
    static_memory = core.normalize_memory(torch.randn(2, 16))
    skill = torch.randn(2, 16)
    action = torch.randn(2, 16)
    update = core.update_memory(static_memory, h, skill, action)
    queries = core.queries(
        h,
        update.effective_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
    )
    assert report["zero_history_exact_static"] is True
    assert report["history_rows_use_unbounded_unified_residual_at_step0"] is True
    assert report["unified_route_initialized_from_static_query"] is True
    assert report["route_skill_adapter_zero_residual"] is True
    assert not torch.equal(update.effective_memory, static_memory)
    assert torch.equal(queries.dynamic_recall, queries.static_recall)
    assert not torch.equal(queries.dynamic_route, queries.static_route)
    zero_history = core.queries(
        h,
        update.effective_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(zero_history.dynamic_route, zero_history.static_route)
    assert torch.allclose(core.transition_scale(), torch.tensor(0.05), atol=1.0e-6)
    assert torch.allclose(core.result_scale(), torch.tensor(0.05), atol=1.0e-6)
    assert torch.allclose(core.recall_scale(), torch.tensor(0.10), atol=1.0e-6)
    assert report["route_expert_mixture_initial_probability"] == 0.1
    assert report["route_expert_mixture_feature_dim"] == 6
    assert report["route_expert_mixture_semantic_dim"] == 16
    assert report["route_expert_mixture_semantic_projection_dim"] == 16
    assert report["semantic_per_sample_route_selector"] is True
    assert report["candidate_semantic_expert_selector"] is True
    assert report["query_level_expert_mixture_enabled"] is True
    assert report["candidate_local_gate_enabled"] is False
    assert report["legacy_candidate_route_gate_frozen"] is True
    assert report["legacy_candidate_route_gate_reset_for_rng_compatibility"] is True
    assert report["depth_selector_reset_after_raw_expert"] is True
    assert report["route_expert_selection_threshold"] == 0.55


def test_depth_selector_reset_preserves_legacy_raw_expert_rng_stream() -> None:
    torch.manual_seed(62)
    observed = CLSTRVNextCore(16, hidden_dim=8)
    legacy_reference = CLSTRVNextCore(16, hidden_dim=8)
    legacy_reference.load_state_dict(observed.state_dict())
    seed = 72
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        legacy_reference.memory_recall_query.reset_parameters(zero_output=True)
        legacy_reference.memory_route_query.reset_parameters(zero_output=True)
        legacy_reference.candidate_route_gate.reset_parameters()
        legacy_reference.unified_route_query.load_state_dict(
            legacy_reference.static_query.state_dict()
        )
        with torch.no_grad():
            legacy_reference.route_skill_adapter.weight.zero_()
        legacy_reference.transition_delta.reset_parameters()
        legacy_reference.correction_delta.reset_parameters()
        legacy_reference.correction_gate.reset_parameters()
        with torch.no_grad():
            legacy_reference.action_adapter.weight.copy_(torch.eye(16))
            legacy_reference.result_adapter.weight.copy_(torch.eye(16))
        legacy_reference.transition_scale.reset_to(0.05)
        legacy_reference.result_scale.reset_to(0.05)
        legacy_reference.recall_scale.reset_to(0.10)
    observed.reset_stage2_recurrent_parameters(seed=seed)
    raw_prefixes = (
        "memory_recall_query.",
        "memory_route_query.",
        "candidate_route_gate.",
        "unified_route_query.",
        "route_skill_adapter.",
        "transition_delta.",
        "correction_delta.",
        "correction_gate.",
        "action_adapter.",
        "result_adapter.",
        "transition_scale.",
        "result_scale.",
        "recall_scale.",
    )
    observed_state = observed.state_dict()
    reference_state = legacy_reference.state_dict()
    compared = [
        name for name in observed_state if name.startswith(raw_prefixes)
    ]
    assert compared
    assert all(
        torch.equal(observed_state[name], reference_state[name])
        for name in compared
    )


def test_direct_router_reset_preserves_accepted_static_logits_exactly() -> None:
    torch.manual_seed(67)
    core = CLSTRVNextCore(8, hidden_dim=4)
    with torch.no_grad():
        core.static_route_query_delta.adapter.output.weight.normal_(std=0.1)
        core.static_route_query_delta.adapter.output.bias.normal_(std=0.1)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = F.normalize(torch.randn(2, 5, 8), p=2, dim=-1)
    static_recall, static_delta, _static_route = core.static_query_components(
        h,
        static_memory,
    )
    scored_candidates = F.normalize(candidates.float(), p=2, dim=-1).to(h.dtype)
    expected = torch.einsum(
        "bd,bcd->bc", static_recall, scored_candidates
    ) + torch.einsum(
        "bd,bcd->bc", static_delta, scored_candidates
    )
    core.reset_stage2_recurrent_parameters(seed=79)
    observed = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
        candidates,
        torch.ones(2, 5, dtype=torch.bool),
        temperature=torch.tensor(1.0),
        hard_fallback=False,
    )
    assert torch.equal(observed.static_logits, expected)
    assert torch.equal(observed.raw_dynamic_logits, expected)


def test_vnext_static_context_and_current_only_memory_state_are_separate() -> None:
    torch.manual_seed(73)
    core = CLSTRVNextCore(8, hidden_dim=4)
    static_context = torch.randn(2, 8)
    current = torch.randn(1, 8).expand(2, -1).clone()
    dynamic_memory = core.normalize_memory(torch.randn(1, 8)).expand(2, -1).clone()
    static_memory = core.normalize_memory(torch.randn(1, 8)).expand(2, -1).clone()
    queries = core.queries(
        static_context,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        memory_state=current,
    )
    assert not torch.equal(queries.static_recall[0], queries.static_recall[1])
    assert torch.equal(queries.recall_delta[0], queries.recall_delta[1])
    assert not torch.equal(queries.route_delta[0], queries.route_delta[1])


def test_vnext_full_pool_accepts_pre_normalized_frozen_skill_cache() -> None:
    torch.manual_seed(13)
    core = CLSTRVNextCore(8, hidden_dim=4)
    query = F.normalize(torch.randn(3, 8), dim=-1)
    skills = torch.randn(17, 8)
    reference = core.full_pool_logits(query, skills, head="recall")
    cached = F.normalize(skills.float(), dim=-1).to(query.dtype)
    accelerated = core.full_pool_logits(
        query,
        cached,
        head="recall",
        skill_embeddings_are_normalized=True,
    )
    assert torch.allclose(accelerated, reference, atol=1.0e-6)


def test_vnext_stage2_modules_receive_gradients() -> None:
    torch.manual_seed(7)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h, memory, static_memory, skill, action, result = [
        torch.randn(2, 8) for _ in range(6)
    ]
    memory = core.normalize_memory(memory)
    static_memory = core.normalize_memory(static_memory)
    update = core.update_memory(
        memory,
        h,
        skill,
        action,
        result_embedding=result,
    )
    route_scores = core.candidate_route_scores(
        h,
        update.effective_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    queries = core.queries(
        h,
        update.effective_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
    )
    loss = (
        update.effective_memory.square().mean()
        + queries.dynamic_recall.square().mean()
        + route_scores.mixed_logits.square().mean()
    )
    loss.backward()
    required_prefixes = (
        "transition_delta",
        "correction_delta",
        "memory_recall_query",
        "unified_route_query",
        "route_skill_adapter",
        "route_expert_mixture",
    )
    for prefix in required_prefixes:
        gradients = [
            parameter.grad
            for name, parameter in core.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        finite_norms = [
            float(gradient.detach().float().norm())
            for gradient in gradients
            if gradient is not None and torch.isfinite(gradient).all()
        ]
        assert max(finite_norms, default=0.0) > 1.0e-10


def test_vnext_query_expert_mixture_preserves_static_and_uses_one_row_probability() -> None:
    torch.manual_seed(29)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 4, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    static_support = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    zero_history = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
        candidates,
        valid,
        static_support_mask=static_support,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.equal(zero_history.mixed_logits, zero_history.static_logits)
    assert torch.equal(zero_history.mixture_probability, torch.zeros(2))
    assert torch.equal(zero_history.selector_probability, torch.zeros(2))
    assert bool(
        zero_history.static_logits[:, -1]
        .eq(torch.finfo(zero_history.static_logits.dtype).min)
        .all()
    )
    hard_history = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        static_support_mask=static_support,
        temperature=torch.tensor(10.0),
        hard_fallback=True,
    )
    soft_history = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        static_support_mask=static_support,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.equal(hard_history.mixed_logits, hard_history.static_logits)
    assert hard_history.mixture_probability.shape == (2,)
    assert not torch.equal(soft_history.mixed_logits, soft_history.static_logits)
    assert not torch.equal(soft_history.mixed_logits, soft_history.raw_dynamic_logits)
    assert soft_history.mixture_probability.shape == (2,)
    assert torch.equal(
        soft_history.selector_probability,
        soft_history.mixture_probability,
    )
    assert bool((soft_history.mixture_probability > 0.0).all())
    assert bool((soft_history.mixture_probability < 0.5).all())
    with torch.no_grad():
        core.route_expert_mixture.output.weight.zero_()
        core.route_expert_mixture.output.bias.fill_(10.0)
    hard_dynamic = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        static_support_mask=static_support,
        temperature=torch.tensor(10.0),
        hard_fallback=True,
    )
    assert torch.equal(hard_dynamic.mixed_logits, hard_dynamic.raw_dynamic_logits)
    assert torch.equal(
        hard_dynamic.mixture_probability,
        torch.ones_like(hard_dynamic.mixture_probability),
    )
    assert bool((hard_dynamic.selector_probability > 0.55).all())


def test_vnext_route_selector_uses_semantics_without_source_labels() -> None:
    torch.manual_seed(81)
    core = CLSTRVNextCore(8, hidden_dim=8)
    selector = core.route_expert_mixture
    confidence = torch.zeros(2, 6)
    route_state = torch.zeros(2, 8)
    route_state[:, 0] = 1.0
    current_state = route_state.clone()
    memory_delta = torch.zeros(2, 8)
    memory_delta[0, 0] = 1.0
    memory_delta[1, 0] = -1.0
    with torch.no_grad():
        selector.output.weight.normal_(mean=0.0, std=0.5)
        selector.output.bias.zero_()
    probability = selector(
        confidence,
        route_state=route_state,
        current_state=current_state,
        memory_delta=memory_delta,
        static_expert_state=route_state,
        dynamic_expert_state=current_state,
        expert_delta_state=memory_delta,
    )
    assert probability.shape == (2,)
    assert not torch.equal(probability[0], probability[1])


def test_vnext_query_expert_mixture_uses_history_depth_but_zero_history_stays_static() -> None:
    torch.manual_seed(30)
    core = CLSTRVNextCore(8, hidden_dim=4)
    with torch.no_grad():
        core.route_expert_mixture.hidden.weight.zero_()
        core.route_expert_mixture.hidden.bias.zero_()
        core.route_expert_mixture.hidden.weight[0, 5] = 1.0
        core.route_expert_mixture.output.weight.zero_()
        core.route_expert_mixture.output.weight[0, 0] = 1.0
        core.route_expert_mixture.output.bias.zero_()
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(1, 8)).expand(2, -1).clone()
    dynamic_memory = core.normalize_memory(torch.randn(1, 8)).expand(2, -1).clone()
    candidates = torch.randn(1, 4, 8).expand(2, -1, -1).clone()
    valid = torch.ones(2, 4, dtype=torch.bool)
    shallow_deep = core.candidate_route_scores(
        h[0:1].expand(2, -1),
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        history_depth=torch.tensor([1.0, 16.0]),
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert not torch.equal(
        shallow_deep.mixture_probability[0],
        shallow_deep.mixture_probability[1],
    )
    zero_history = core.candidate_route_scores(
        h[0:1].expand(2, -1),
        dynamic_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
        candidates,
        valid,
        history_depth=torch.tensor([1.0, 16.0]),
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.equal(zero_history.mixed_logits, zero_history.static_logits)
    assert torch.equal(
        zero_history.mixture_probability,
        torch.zeros_like(zero_history.mixture_probability),
    )


def test_static_route_query_residual_is_shared_before_memory_delta() -> None:
    torch.manual_seed(31)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    before = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    with torch.no_grad():
        core.static_route_query_delta.adapter.output.bias.fill_(0.5)
    after = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert not torch.equal(after.static_logits, before.static_logits)
    assert torch.allclose(
        after.raw_dynamic_logits - after.static_logits,
        before.raw_dynamic_logits - before.static_logits,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    zero_history = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.equal(zero_history.raw_dynamic_logits, zero_history.static_logits)


def test_static_route_scoring_uses_frozen_accepted_query() -> None:
    torch.manual_seed(47)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    temperature = torch.tensor(10.0)
    queries = core.queries(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
    )
    normalized_candidates = torch.nn.functional.normalize(
        candidates.float(), p=2, dim=-1
    ).to(h.dtype)
    frozen_static_base_logits = temperature * torch.einsum(
        "bd,bcd->bc",
        queries.static_recall,
        normalized_candidates,
    )
    route_delta_logits = temperature * torch.einsum(
        "bd,bcd->bc",
        queries.static_route_delta,
        normalized_candidates,
    )
    memory_residual_logits = temperature * torch.einsum(
        "bd,bcd->bc",
        queries.route_delta,
        normalized_candidates,
    )
    observed = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=temperature,
        hard_fallback=False,
    )
    assert torch.allclose(
        observed.static_logits,
        frozen_static_base_logits + route_delta_logits,
        atol=1.0e-6,
    )
    assert torch.allclose(
        observed.raw_dynamic_logits,
        observed.static_logits + memory_residual_logits,
        atol=1.0e-6,
    )


def test_route_skill_adapter_changes_only_unbounded_memory_residual() -> None:
    torch.manual_seed(49)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    before = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    with torch.no_grad():
        core.route_skill_adapter.weight.normal_(std=0.1)
    after = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.equal(after.static_logits, before.static_logits)
    assert not torch.equal(after.raw_residual, before.raw_residual)


def test_zero_static_route_adapter_preserves_base_candidate_logits_exactly() -> None:
    torch.manual_seed(51)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    queries = core.queries(
        h,
        static_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
    )
    normalized_candidates = F.normalize(candidates.float(), p=2, dim=-1).to(h.dtype)
    base = torch.tensor(10.0) * torch.einsum(
        "bd,bcd->bc",
        queries.static_route,
        normalized_candidates,
    )
    observed = core.candidate_route_scores(
        h,
        static_memory,
        static_memory,
        torch.zeros(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.count_nonzero(queries.static_route_delta) == 0
    assert torch.equal(observed.static_logits, base)
    assert torch.equal(observed.raw_dynamic_logits, base)


def test_static_route_scoring_is_candidate_order_equivariant() -> None:
    torch.manual_seed(53)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    permuted = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates.index_select(1, permutation),
        valid.index_select(1, permutation),
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.allclose(
        permuted.static_logits,
        original.static_logits.index_select(1, permutation),
        atol=1.0e-6,
    )
    assert torch.allclose(
        permuted.raw_dynamic_logits,
        original.raw_dynamic_logits.index_select(1, permutation),
        atol=1.0e-6,
    )


def test_candidate_mlp_is_not_a_hidden_canonical_route_scorer() -> None:
    torch.manual_seed(59)
    core = CLSTRVNextCore(8, hidden_dim=4)
    h = torch.randn(2, 8)
    static_memory = core.normalize_memory(torch.randn(2, 8))
    dynamic_memory = core.normalize_memory(torch.randn(2, 8))
    candidates = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    before = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    with torch.no_grad():
        core.candidate_compressor.output.weight.normal_(mean=0.0, std=0.2)
        core.candidate_compressor.output.bias.fill_(0.3)
    after = core.candidate_route_scores(
        h,
        dynamic_memory,
        static_memory,
        torch.ones(2, dtype=torch.bool),
        candidates,
        valid,
        temperature=torch.tensor(10.0),
        hard_fallback=False,
    )
    assert torch.equal(after.static_logits, before.static_logits)
    assert torch.equal(after.raw_dynamic_logits, before.raw_dynamic_logits)
