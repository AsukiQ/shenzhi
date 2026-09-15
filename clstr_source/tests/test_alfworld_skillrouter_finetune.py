import json
import inspect
from pathlib import Path

import torch

from clstr.alfworld_policy_train import build_goal_conditioned_state_text
from clstr.alfworld_skillrouter_finetune import train_alfworld_skillrouter_action_adapter_with_embeddings
from clstr.alfworld_eval import SkillRouterAdmissibleActionScorer, evaluate_alfworld_skillrouter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows() -> list[dict]:
    return [
        {
            "split": "train",
            "task_type": "pick_and_place_simple",
            "goal_text": "put the mug in the drawer",
            "observation_t": "You see a mug on a table.",
            "history_t": [],
            "admissible_commands_t": ["look", "take mug 1 from table 1", "inventory"],
            "expert_action_t": "take mug 1 from table 1",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": False,
        },
        {
            "split": "train",
            "task_type": "pick_and_place_simple",
            "goal_text": "put the mug in the drawer",
            "observation_t": "You are near drawer 1.",
            "history_t": ["take mug 1 from table 1"],
            "admissible_commands_t": ["look", "open drawer 1", "inventory"],
            "expert_action_t": "open drawer 1",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": False,
        },
        {
            "split": "valid_seen",
            "task_type": "pick_and_place_simple",
            "goal_text": "leak row",
            "observation_t": "Do not train on this.",
            "history_t": [],
            "admissible_commands_t": ["look"],
            "expert_action_t": "look",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": False,
        },
    ]


def _toy_text_embedding(text: str) -> torch.Tensor:
    lowered = text.lower()
    return torch.tensor(
        [
            1.0 if "look" in lowered else 0.0,
            1.0 if "mug" in lowered or "take" in lowered else 0.0,
            1.0 if "drawer" in lowered or "open" in lowered else 0.0,
            1.0 if "inventory" in lowered else 0.0,
        ],
        dtype=torch.float32,
    )


def test_train_skillrouter_action_adapter_writes_checkpoint_and_improves_ranking(tmp_path):
    replay_path = tmp_path / "train_replay.jsonl"
    _write_jsonl(replay_path, _rows())

    report = train_alfworld_skillrouter_action_adapter_with_embeddings(
        replay_path=replay_path,
        output_dir=tmp_path / "sr_ft",
        text_embedding_fn=_toy_text_embedding,
        max_steps=40,
        batch_size=2,
        learning_rate=0.1,
        eval_fraction=0.5,
        seed=7,
    )

    assert report["status"] == "ok"
    assert report["training_data"] == "ALFWorld train replay only"
    assert report["uses_alfworld_valid_or_test_for_training"] is False
    assert report["policy_sample_count"] == 2
    assert report["eval_metrics"]["expert_action_recall@1"] >= report["baseline_eval_metrics"]["expert_action_recall@1"]
    checkpoint = Path(report["checkpoint"])
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["stage"] == "alfworld_skillrouter_action_adapter"
    assert payload["adapter_state_dict"]


def test_skillrouter_admissible_scorer_can_apply_finetuned_adapter_checkpoint(tmp_path):
    checkpoint = tmp_path / "adapter.pt"
    report = train_alfworld_skillrouter_action_adapter_with_embeddings(
        replay_path=tmp_path / "missing.jsonl",
        output_dir=tmp_path / "sr_ft",
        examples=[
            {
                "state_text": build_goal_conditioned_state_text(
                    {
                        "goal_text": "put the mug in the drawer",
                        "task_type": "pick_and_place_simple",
                        "observation_t": "You see a mug.",
                        "history_t": [],
                    }
                ),
                "candidate_actions": ["look", "take mug 1 from table 1"],
                "label_index": 1,
                "done": False,
            }
        ],
        text_embedding_fn=_toy_text_embedding,
        output_checkpoint_path=checkpoint,
        max_steps=30,
        batch_size=1,
        learning_rate=0.1,
        eval_fraction=0.0,
        seed=11,
    )
    assert Path(report["checkpoint"]) == checkpoint

    scorer = SkillRouterAdmissibleActionScorer(
        model_name_or_path="unused-in-test",
        adapter_checkpoint_path=checkpoint,
    )
    scorer._encode = lambda texts: torch.stack([_toy_text_embedding(text) for text in texts], dim=0)  # type: ignore[method-assign]

    scores = scorer(["goal: put the mug in the drawer\nobservation: You see a mug."], [["look", "take mug 1 from table 1"]])

    assert int(torch.argmax(scores[0]).item()) == 1
    assert scorer.last_metadata[0]["adapter_checkpoint_path"] == str(checkpoint)
    assert scorer.last_metadata[0]["policy_family"] == "skillrouter_finetuned_admissible_action"


def test_alfworld_skillrouter_eval_exposes_adapter_checkpoint_parameter():
    assert "adapter_checkpoint_path" in inspect.signature(evaluate_alfworld_skillrouter).parameters
