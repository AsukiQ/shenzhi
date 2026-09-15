import torch

from clstr.dynamic_skill_registry import (
    DynamicSkillRegistry,
    blend_skill_scores_with_coverage,
    skill_head_coverage_weights,
)
from clstr.encoders import SkillTable
from clstr.model import CLSTRModel
from clstr.stage_checkpoint_init import load_compatible_state_dict


def _deterministic_encoder(texts: list[str]) -> torch.Tensor:
    rows = []
    for text in texts:
        value = float(sum(ord(ch) for ch in text) % 17)
        rows.append([value, value + 1.0, value + 2.0])
    return torch.tensor(rows, dtype=torch.float32)


def _skill_text(skill: dict) -> str:
    return str(skill.get("name") or skill.get("skill_id"))


def test_skill_table_append_skills_preserves_existing_rows_and_marks_new_rows():
    table = SkillTable(
        [
            {"skill_id": "old/a", "name": "alpha"},
            {"skill_id": "old/b", "name": "beta"},
        ],
        encoder_fn=_deterministic_encoder,
        d=3,
        trainable=True,
        skill_text_fn=_skill_text,
    )
    with torch.no_grad():
        table.skill_bias_retr.copy_(torch.tensor([0.25, -0.5]))
        table.skill_bias_belief.copy_(torch.tensor([0.1, -0.2]))
    old_embeddings = table.E.detach().clone()

    report = table.append_skills(
        [
            {"skill_id": "old/b", "name": "duplicate beta"},
            {"skill_id": "new/c", "name": "gamma"},
        ]
    )

    assert report["old_count"] == 2
    assert report["new_count"] == 3
    assert report["appended_skill_ids"] == ["new/c"]
    assert report["skipped_duplicate_skill_ids"] == ["old/b"]
    assert [row["skill_id"] for row in table.skills] == ["old/a", "old/b", "new/c"]
    assert table.skills[-1]["is_appended_after_checkpoint"] is True
    assert table.skills[-1]["retrieval_seen_count"] == 0
    assert table.skills[-1]["transition_seen_count"] == 0
    assert table.skills[-1]["act_seen_count"] == 0
    assert table.E.requires_grad is True
    assert table.skill_bias_retr.requires_grad is True
    assert table.skill_bias_belief.requires_grad is True
    assert torch.allclose(table.E[:2], old_embeddings)
    assert torch.allclose(table.skill_bias_retr, torch.tensor([0.25, -0.5, 0.0]))
    assert torch.allclose(table.skill_bias_belief, torch.tensor([0.1, -0.2, 0.0]))
    assert not torch.allclose(table.E[2], torch.zeros(3))


def test_clstr_model_append_skills_keeps_model_and_skill_table_in_sync():
    table = SkillTable(
        [{"skill_id": "old/a", "name": "alpha"}],
        encoder_fn=_deterministic_encoder,
        d=3,
        trainable=False,
        skill_text_fn=_skill_text,
    )
    model = object.__new__(CLSTRModel)
    torch.nn.Module.__init__(model)
    model.skills = list(table.skills)
    model.skill_table = table
    model._cross_encoder_cache = {("task", 0): torch.ones(1)}

    report = model.append_skills([{"skill_id": "new/b", "name": "beta"}])

    assert report["appended_skill_ids"] == ["new/b"]
    assert [row["skill_id"] for row in model.skills] == ["old/a", "new/b"]
    assert [row["skill_id"] for row in model.skill_table.skills] == ["old/a", "new/b"]
    assert model.stop_idx == 2
    assert model._cross_encoder_cache is None


def test_coverage_weights_keep_old_skills_full_and_downweight_unseen_appended_skills():
    weights = skill_head_coverage_weights(
        [
            {"skill_id": "old/a"},
            {"skill_id": "new/b", "is_appended_after_checkpoint": True},
            {"skill_id": "new/c", "is_appended_after_checkpoint": True, "transition_seen_count": 2},
        ],
        head="transition",
        min_count=4,
    )

    assert torch.allclose(weights, torch.tensor([1.0, 0.0, 0.5]))


def test_blend_skill_scores_with_coverage_keeps_unseen_skill_on_stage0_score():
    blended, report = blend_skill_scores_with_coverage(
        stage0_scores=torch.tensor([[1.0, 1.0]]),
        skill_rows=[
            {"skill_id": "old/a"},
            {"skill_id": "new/b", "is_appended_after_checkpoint": True},
        ],
        transition_scores=torch.tensor([[2.0, 100.0]]),
        transition_weight=1.0,
    )

    assert torch.allclose(blended, torch.tensor([[3.0, 1.0]]))
    assert report["transition_head_weights"] == [1.0, 0.0]


def test_load_compatible_state_dict_can_prefix_expand_skill_table_rows():
    model = torch.nn.Module()
    model.skill_table = SkillTable(
        [
            {"skill_id": "old/a", "name": "alpha"},
            {"skill_id": "new/b", "name": "beta"},
        ],
        encoder_fn=_deterministic_encoder,
        d=3,
        trainable=False,
        skill_text_fn=_skill_text,
    )
    original_new_embedding = model.skill_table.E.detach()[1].clone()
    state = {
        "skill_table.E": torch.tensor([[10.0, 11.0, 12.0]]),
        "skill_table.skill_bias_retr": torch.tensor([0.75]),
        "skill_table.skill_bias_belief": torch.tensor([-0.25]),
    }

    report = load_compatible_state_dict(
        model,
        state,
        partial_load_mode="unit_test_dynamic_prefix",
        allow_skill_table_prefix_expansion=True,
    )

    assert torch.allclose(model.skill_table.E[0], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.allclose(model.skill_table.E[1], original_new_embedding)
    assert torch.allclose(model.skill_table.skill_bias_retr, torch.tensor([0.75, 0.0]))
    assert torch.allclose(model.skill_table.skill_bias_belief, torch.tensor([-0.25, 0.0]))
    assert report["skill_table_prefix_expanded_keys"] == [
        "skill_table.E",
        "skill_table.skill_bias_belief",
        "skill_table.skill_bias_retr",
    ]


def test_dynamic_skill_registry_appends_dedupes_and_filters_appworld_inventory():
    registry = DynamicSkillRegistry.from_rows(
        [
            {"skill_id": "toolbench/search", "name": "search", "executor_domain": "toolbench"},
            {"skill_id": "appworld/login", "name": "login", "appworld_executor_compatible": True},
        ]
    )

    report = registry.append_rows(
        [
            {"skill_id": "toolbench/search", "name": "duplicate"},
            {"skill_id": "appworld/count-songs", "name": "count songs", "appworld_executor_compatible": True},
            {"skill_id": "alfworld/take", "name": "take object", "executor_domain": "alfworld"},
        ]
    )

    assert report["old_count"] == 2
    assert report["new_count"] == 4
    assert report["appended_skill_ids"] == ["appworld/count-songs", "alfworld/take"]
    assert report["skipped_duplicate_skill_ids"] == ["toolbench/search"]
    assert registry.id_to_index["appworld/count-songs"] == 2
    assert registry.rows[2]["is_appended_after_checkpoint"] is True
    assert registry.available_skill_ids(executor_domain="appworld") == [
        "appworld/login",
        "appworld/count-songs",
    ]
