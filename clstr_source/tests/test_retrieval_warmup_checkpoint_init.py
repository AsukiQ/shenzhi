import json

import torch

from clstr.retrieval_warmup import _load_warmup_init_checkpoint


def _write_skill_pool(path, skill_ids):
    path.write_text(
        "".join(json.dumps({"skill_id": skill_id, "name": skill_id}) + "\n" for skill_id in skill_ids),
        encoding="utf-8",
    )


def test_retrieval_warmup_init_checkpoint_skips_shape_mismatches(tmp_path):
    checkpoint_path = tmp_path / "expanded_pool_init.pt"

    class TinyWarmupModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.zeros(3, 2))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))

    model = TinyWarmupModel()
    torch.save(
        {
            "step": 5000,
            "model_state_dict": {
                "linear.weight": torch.eye(2),
                "skill_table.E": torch.ones(2, 2),
                "skill_table.skill_bias_retr": torch.ones(2),
            },
        },
        checkpoint_path,
    )

    report = _load_warmup_init_checkpoint(model, checkpoint_path)

    assert report["loaded"] is True
    assert report["checkpoint_step"] == 5000
    assert report["loaded_state_key_count"] == 1
    assert report["loaded_state_keys"] == ["linear.weight"]
    assert report["shape_mismatched"] == {
        "skill_table.E": {"checkpoint": [2, 2], "model": [3, 2]},
        "skill_table.skill_bias_retr": {"checkpoint": [2], "model": [3]},
    }
    assert sorted(report["skipped_state_keys"]) == ["skill_table.E", "skill_table.skill_bias_retr"]
    assert torch.allclose(model.linear.weight, torch.eye(2))
    assert torch.allclose(model.skill_table.E, torch.zeros(3, 2))


def test_retrieval_warmup_init_checkpoint_transplants_skill_table_by_skill_id(tmp_path):
    checkpoint_path = tmp_path / "old_pool_init.pt"
    old_skills = tmp_path / "old_skill_pool.jsonl"
    new_skills = tmp_path / "new_skill_pool.jsonl"
    _write_skill_pool(old_skills, ["old/a", "old/b"])
    _write_skill_pool(new_skills, ["new/x", "old/b", "old/a"])

    class TinyWarmupModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.full((3, 2), -1.0))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.full((3,), -2.0))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.full((3,), -3.0))

    model = TinyWarmupModel()
    torch.save(
        {
            "step": 400,
            "model_state_dict": {
                "linear.weight": torch.eye(2),
                "skill_table.E": torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
                "skill_table.skill_bias_retr": torch.tensor([0.1, 0.2]),
                "skill_table.skill_bias_belief": torch.tensor([0.3, 0.4]),
            },
        },
        checkpoint_path,
    )

    report = _load_warmup_init_checkpoint(
        model,
        checkpoint_path,
        init_checkpoint_skills_path=old_skills,
        current_skills_path=new_skills,
    )

    assert report["loaded"] is True
    assert report["skill_table_id_aligned_transplant"]["matched_skill_count"] == 2
    assert sorted(report["skill_table_id_aligned_transplant"]["transplanted_keys"]) == [
        "skill_table.E",
        "skill_table.skill_bias_belief",
        "skill_table.skill_bias_retr",
    ]
    assert torch.allclose(model.skill_table.E[0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(model.skill_table.E[1], torch.tensor([2.0, 2.5]))
    assert torch.allclose(model.skill_table.E[2], torch.tensor([1.0, 1.5]))
    assert torch.allclose(model.skill_table.skill_bias_retr, torch.tensor([-2.0, 0.2, 0.1]))
    assert torch.allclose(model.skill_table.skill_bias_belief, torch.tensor([-3.0, 0.4, 0.3]))
