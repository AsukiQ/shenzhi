import json
from pathlib import Path

from clstr.model import _skill_text_serializer
from clstr.skill_embedding import enrich_skill_rows_for_embedding
from clstr.skillnet_aux_rebuild import _skill_rows, load_skillnet_skills


def _write_skill(root: Path, environment: str, name: str, description: str, body: str) -> None:
    path = root / environment / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_skillnet_skill_rows_preserve_skill_body_for_embedding(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "alfworld",
        "alfworld-clean-object",
        "Clean a held object using a sinkbasin.",
        "# Instructions\nClean with `clean {obj} with sinkbasin 1`.",
    )

    skills_by_env = load_skillnet_skills(tmp_path / "skills")
    rows = _skill_rows(skills_by_env)

    clean = next(row for row in rows if row["skill_id"] == "alfworld/alfworld-clean-object")
    assert "Clean with" in clean["body"]


def test_enriched_skill_serializer_includes_body_and_train_only_action_grounding():
    skill_rows = [
        {
            "skill_id": "alfworld/alfworld-clean-object",
            "name": "alfworld-clean-object",
            "description": "Clean a held object.",
            "body": "# Workflow\nUse clean commands after reaching a sinkbasin.",
            "environment": "alfworld",
        }
    ]
    train_rows = [
        {
            "skill_id": "alfworld/alfworld-clean-object",
            "expert_action": "clean apple 1 with sinkbasin 1",
            "hard_negative_action": "go to drawer 1",
            "provenance": {"split": "train"},
        },
        {
            "skill_id": "alfworld/alfworld-clean-object",
            "expert_action": "clean mug 2 with sinkbasin 1",
            "hard_negative_action": "look",
            "provenance": {"split": "valid_seen"},
        },
    ]

    enriched, report = enrich_skill_rows_for_embedding(skill_rows, train_rows)
    text = _skill_text_serializer("clstr_enriched")(enriched[0])

    assert report["train_only_action_example_count"] == 1
    assert "Use clean commands" in text
    assert "clean apple 1 with sinkbasin 1" in text
    assert "clean mug 2 with sinkbasin 1" not in text
    assert "clean <object> with <receptacle>" in json.dumps(enriched[0]["action_templates"])
