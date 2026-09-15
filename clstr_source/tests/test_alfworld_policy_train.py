import json
from pathlib import Path

import torch

from clstr.alfworld_policy_train import (
    _stop_logits,
    compute_policy_diagnostic_metrics,
    load_policy_examples,
    train_alfworld_policy_head_with_model,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class _TinyTextModel(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            vec = torch.zeros(self.dim, device=self.device)
            for idx, token in enumerate(("mug", "apple", "drawer", "look", "take", "open", "put", "inventory")):
                if token in lowered:
                    vec[idx % self.dim] += 1.0
            if vec.sum() == 0:
                vec[0] = 0.1
            rows.append(vec + self.anchor * 0.0)
        return torch.stack(rows, dim=0)


def _replay_rows():
    return [
        {
            "split": "train",
            "gamefile": "/train/game1.tw-pddl",
            "task_type": "pick_and_place_simple",
            "goal_text": "put the mug in the drawer",
            "initial_observation": "Your task is to: put the mug in the drawer",
            "observation_t": "You see a mug on a table.",
            "history_t": [],
            "admissible_commands_t": ["take mug 1 from table 1", "look", "inventory"],
            "expert_action_t": "take mug 1 from table 1",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": False,
        },
        {
            "split": "train",
            "gamefile": "/train/game1.tw-pddl",
            "task_type": "pick_and_place_simple",
            "goal_text": "put the mug in the drawer",
            "initial_observation": "Your task is to: put the mug in the drawer",
            "observation_t": "You are near drawer 1.",
            "history_t": ["take mug 1 from table 1"],
            "admissible_commands_t": ["open drawer 1", "look"],
            "expert_action_t": "open drawer 1",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": False,
        },
        {
            "split": "train",
            "gamefile": "/train/game1.tw-pddl",
            "task_type": "pick_and_place_simple",
            "goal_text": "put the mug in the drawer",
            "initial_observation": "Your task is to: put the mug in the drawer",
            "observation_t": "The drawer is open.",
            "history_t": ["take mug 1 from table 1", "open drawer 1"],
            "admissible_commands_t": ["put mug 1 in/on drawer 1", "look"],
            "expert_action_t": "put mug 1 in/on drawer 1",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": True,
        },
        {
            "split": "valid_seen",
            "gamefile": "/valid_seen/leak.tw-pddl",
            "goal_text": "should not train on this",
            "observation_t": "leak",
            "history_t": [],
            "admissible_commands_t": ["look"],
            "expert_action_t": "look",
            "expert_action_in_admissible": True,
            "usable_for_policy": True,
            "done_t": False,
        },
    ]


def test_stop_logits_uses_sparse_belief_memory_not_dense_retrieval():
    class _SkillTable:
        def __init__(self):
            self.E = torch.eye(3)

        def retrieval_logits(self, h):
            return torch.tensor([[9.0, 0.0, 0.0]], dtype=h.dtype, device=h.device)

        def belief_logits(self, h):
            return torch.tensor([[0.0, 9.0, 0.0]], dtype=h.dtype, device=h.device)

    class _StopHead(torch.nn.Module):
        def forward(self, h, m):
            del h
            return m[:, 1:2]

    class _Model:
        def __init__(self):
            self.skill_table = _SkillTable()
            self.stop_head = _StopHead()

    logits = _stop_logits(_Model(), torch.zeros(1, 3))

    assert float(logits.item()) > 0.9


def test_load_policy_examples_uses_only_train_usable_rows_and_builds_goal_conditioned_state(tmp_path):
    replay_path = tmp_path / "train_replay.jsonl"
    _write_jsonl(replay_path, _replay_rows())

    examples, skipped = load_policy_examples(replay_path)

    assert len(examples) == 3
    assert skipped["non_train_split"] == 1
    assert all(example.split == "train" for example in examples)
    assert examples[1].label_index == 0
    assert "goal: put the mug in the drawer" in examples[1].state_text
    assert "observation: You are near drawer 1." in examples[1].state_text
    assert "history: take mug 1 from table 1" in examples[1].state_text
    assert examples[1].candidate_actions == ["open drawer 1", "look"]


def test_compute_policy_diagnostic_metrics_reports_recall_mrr_and_stop_accuracy():
    scores = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.5, 2.0, 1.0],
            [0.2, 0.1, 4.0],
        ]
    )
    labels = torch.tensor([0, 2, 1])
    done = torch.tensor([0.0, 0.0, 1.0])
    stop_logits = torch.tensor([-2.0, -1.0, 2.0])

    metrics = compute_policy_diagnostic_metrics(scores, labels, done, stop_logits)

    assert metrics["expert_action_recall@1"] == 1 / 3
    assert metrics["expert_action_recall@5"] == 1.0
    assert round(metrics["expert_action_mrr"], 6) == round((1.0 + 0.5 + 1 / 3) / 3, 6)
    assert metrics["stop_accuracy"] == 1.0


def test_train_alfworld_policy_head_with_model_writes_checkpoint_and_reports(tmp_path):
    replay_path = tmp_path / "train_replay.jsonl"
    _write_jsonl(replay_path, _replay_rows())
    model = _TinyTextModel(dim=8)

    report = train_alfworld_policy_head_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False, "manifest": "routing_manifest.json"},
        replay_path=replay_path,
        output_dir=tmp_path / "policy_train",
        max_steps=4,
        batch_size=2,
        learning_rate=1.0e-2,
        seed=3,
        eval_fraction=0.34,
    )

    checkpoint_path = Path(report["checkpoint"])
    train_report_path = tmp_path / "policy_train" / "train_report.json"
    eval_report_path = tmp_path / "policy_train" / "eval_report.json"

    assert checkpoint_path.exists()
    assert train_report_path.exists()
    assert eval_report_path.exists()
    assert report["training_objective"] == "L_policy_admissible_action_ce"
    assert report["frozen_routing_foundation"] is True
    assert report["qdoc_adapter_used"] is False
    assert report["uses_alfworld_valid_or_test_for_training"] is False
    assert report["policy_sample_count"] == 3
    assert report["expert_action_coverage"] == 1.0
    assert "universal_action_adapter" in report["trainable_modules"]
    assert report["candidate_length_stats"]["min"] == 2
    assert report["candidate_length_stats"]["max"] == 3
    assert "expert_action_recall@1" in report["eval_metrics"]

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["stage"] == "alfworld_policy_l_policy"
    assert payload["training_data"] == "ALFWorld train replay only"
    assert payload["qdoc_adapter_used"] is False
    assert payload["universal_action_adapter_state_dict"]
