import json
from pathlib import Path

import torch

from clstr.appworld_skillrouter_base import (
    SkillRouterBaseScorer,
    evaluate_skillrouter_base_from_embeddings,
    load_skillrouter_base_checkpoint,
    multi_positive_nll,
    save_skillrouter_base_checkpoint,
    train_skillrouter_base_from_embeddings,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_multi_positive_nll_uses_probability_mass_over_all_positive_skills():
    logits = torch.tensor([[0.0, 2.0, 1.0], [1.0, 0.0, 3.0]])
    positives = [[1, 2], [2]]

    loss = multi_positive_nll(logits, positives)

    assert loss.ndim == 0
    assert 0.0 < float(loss) < 0.6


def test_skillrouter_base_checkpoint_round_trips_scores(tmp_path):
    model = SkillRouterBaseScorer(dim=2, skill_count=3)
    query_embs = torch.tensor([[1.0, 0.0]])
    skill_embs = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    before = model.score(query_embs, skill_embs)

    checkpoint_path = tmp_path / "model.pt"
    save_skillrouter_base_checkpoint(
        checkpoint_path,
        model=model,
        skill_ids=["s0", "s1", "s2"],
        report={"status": "ok", "epochs": 1},
    )
    loaded, metadata = load_skillrouter_base_checkpoint(checkpoint_path)
    after = loaded.score(query_embs, skill_embs)

    assert metadata["skill_ids"] == ["s0", "s1", "s2"]
    assert torch.allclose(before, after)


def test_train_and_eval_skillrouter_base_from_embeddings_learns_appworld_qrels(tmp_path):
    query_ids = ["q0", "q1"]
    skill_ids = ["bad", "skill_for_q0", "skill_for_q1"]
    query_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    skill_embs = torch.tensor([[0.2, 0.2], [0.0, 1.0], [1.0, 0.0]])
    qrels = {
        "q0": {"skill_for_q0"},
        "q1": {"skill_for_q1"},
    }

    train_report = train_skillrouter_base_from_embeddings(
        query_ids=query_ids,
        skill_ids=skill_ids,
        query_embs=query_embs,
        skill_embs=skill_embs,
        qrels=qrels,
        output_dir=tmp_path / "train",
        epochs=120,
        learning_rate=0.08,
        seed=7,
        top_k=3,
    )

    assert train_report["status"] == "ok"
    assert train_report["train_query_count"] == 2
    assert Path(train_report["checkpoint_path"]).exists()
    assert train_report["final_loss"] < train_report["initial_loss"]

    qrels_path = tmp_path / "dev_qrels.jsonl"
    _write_jsonl(
        qrels_path,
        [
            {"query_id": "q0", "skill_id": "skill_for_q0", "relevance": 1},
            {"query_id": "q1", "skill_id": "skill_for_q1", "relevance": 1},
        ],
    )
    eval_report = evaluate_skillrouter_base_from_embeddings(
        query_ids=query_ids,
        skill_ids=skill_ids,
        query_embs=query_embs,
        skill_embs=skill_embs,
        checkpoint_path=train_report["checkpoint_path"],
        qrels_path=qrels_path,
        output_dir=tmp_path / "eval",
        top_k=3,
    )

    assert eval_report["status"] == "ok"
    assert eval_report["method"] == "skillrouter_base_trainable_scorer"
    assert eval_report["metrics"]["recall@1"] == 1.0
    predictions = [json.loads(line) for line in Path(eval_report["predictions_path"]).read_text().splitlines()]
    assert predictions[0]["ranked_skill_ids"][0] == "skill_for_q0"
