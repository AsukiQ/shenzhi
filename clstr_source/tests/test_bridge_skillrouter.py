import json
from pathlib import Path
import types

from clstr.bridges.skillrouter import checkpoints
from clstr.bridges.skillrouter.datasets import load_eval_pool, load_eval_tasks
from clstr.bridges.skillrouter.evaluation import write_retrieval_predictions
from clstr.bridges.skillrouter.serialization import serialize_skill_text


def test_serialize_skill_text_contains_body_and_schema():
    skill = {
        "name": "weather.lookup",
        "description": "lookup weather",
        "input_schema": {"z": "last", "a": "first"},
        "output_schema": {"temp": "float", "city": "str"},
        "executor_desc": "calls weather API",
        "failure_modes": [{"code": "timeout", "retryable": True}],
        "body": "def run(city): return city",
    }
    text = serialize_skill_text(skill)
    assert "name:weather.lookup" in text
    assert 'in:{"a": "first", "z": "last"}' in text
    assert 'out:{"city": "str", "temp": "float"}' in text
    assert 'fail:[{"code": "timeout", "retryable": true}]' in text
    assert "in:{'z': 'last', 'a': 'first'}" not in text
    assert "out:{'temp': 'float', 'city': 'str'}" not in text
    assert "body:def run(city): return city" in text


def test_load_eval_files_and_write_predictions(tmp_path: Path):
    tasks = load_eval_tasks(Path("tests/fixtures/skillrouter/tasks.jsonl"))
    pool = load_eval_pool(Path("tests/fixtures/skillrouter/easy.jsonl"))
    output = tmp_path / "retrieval.json"
    write_retrieval_predictions({tasks[0].task_id: [pool[0].skill_id]}, output)
    assert output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == {str(tasks[0].task_id): [pool[0].skill_id]}


def test_load_eval_tasks_prefers_enriched_query_over_instruction_text(tmp_path: Path):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "appworld-task-1",
                "instruction_text": "short instruction only",
                "query": "Instruction: enriched query\nRequired apps: spotify\nObserved train-only API refs: spotify.show_song",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = load_eval_tasks(task_path)

    assert len(tasks) == 1
    assert tasks[0].query.startswith("Instruction: enriched query")


def test_map_backbone_state_dict_strips_known_prefixes():
    fake_torch = types.SimpleNamespace(
        ones=lambda size: [1.0] * size,
        zeros=lambda size: [0.0] * size,
    )
    state_dict = {
        "model.encoder.layer.weight": fake_torch.ones(1),
        "model.encoder.layer.bias": fake_torch.zeros(1),
        "reranker.head.weight": fake_torch.ones(1),
    }
    mapped = checkpoints.map_backbone_state_dict(state_dict, prefixes=("model.",))
    assert "encoder.layer.weight" in mapped
    assert "encoder.layer.bias" in mapped
    assert "reranker.head.weight" not in mapped


def test_map_backbone_state_dict_raises_on_stripped_key_conflict():
    state_dict = {
        "model.layer.weight": [1.0],
        "module.layer.weight": [2.0],
    }
    try:
        checkpoints.map_backbone_state_dict(state_dict, prefixes=("model.", "module."))
    except ValueError as exc:
        assert "layer.weight" in str(exc)
    else:
        raise AssertionError("expected ValueError for stripped key collision")


def test_load_checkpoint_state_extracts_state_dict(monkeypatch):
    fake_torch = types.SimpleNamespace(
        load=lambda path, map_location: {"state_dict": {"encoder.weight": [1.0]}},
    )
    monkeypatch.setattr(checkpoints, "torch", fake_torch)

    loaded = checkpoints.load_checkpoint_state("checkpoint.pt")

    assert loaded == {"encoder.weight": [1.0]}


def test_load_checkpoint_state_returns_top_level_dict(monkeypatch):
    payload = {"encoder.weight": [1.0], "encoder.bias": [0.0]}
    fake_torch = types.SimpleNamespace(load=lambda path, map_location: payload)
    monkeypatch.setattr(checkpoints, "torch", fake_torch)

    loaded = checkpoints.load_checkpoint_state("checkpoint.pt")

    assert loaded is payload


def test_load_checkpoint_state_rejects_non_dict(monkeypatch):
    fake_torch = types.SimpleNamespace(load=lambda path, map_location: ["not", "a", "dict"])
    monkeypatch.setattr(checkpoints, "torch", fake_torch)

    try:
        checkpoints.load_checkpoint_state("checkpoint.pt")
    except TypeError as exc:
        assert "dict" in str(exc)
    else:
        raise AssertionError("expected TypeError for non-dict checkpoint payload")


def test_load_checkpoint_state_raises_without_torch(monkeypatch):
    monkeypatch.setattr(checkpoints, "torch", None)

    try:
        checkpoints.load_checkpoint_state("checkpoint.pt")
    except ModuleNotFoundError as exc:
        assert "torch" in str(exc)
    else:
        raise AssertionError("expected ModuleNotFoundError when torch is unavailable")
