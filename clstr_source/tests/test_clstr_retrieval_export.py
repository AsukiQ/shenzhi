import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.clstr_retrieval_export import export_clstr_retrieval_run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class FakeSkillTable:
    def logits(self, h):
        del h
        return torch.tensor([[0.2, 2.0, 1.0], [3.0, 0.1, 0.0]], dtype=torch.float32)


class StreamingProbeSkillTable:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.calls = 0

    def logits(self, h):
        self.calls += 1
        if self.calls == 2:
            progress = json.loads((self.output_dir / "progress.json").read_text(encoding="utf-8"))
            assert progress["status"] == "running"
            assert progress["phase"] == "retrieving"
            assert progress["processed_queries"] == 1
            assert (self.output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 1
            assert (self.output_dir / "run.tsv").read_text(encoding="utf-8").count("\n") == 2
        return torch.tensor([[float(self.calls), 0.5, 0.1]], dtype=torch.float32)


class FakeModel:
    def __init__(self, skill_table=None):
        self.skill_table = skill_table or FakeSkillTable()
        self.skills = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def rebuild_skill_table(self):
        self.rebuilt = True

    def load_state_dict(self, state_dict, strict=False):
        del state_dict, strict
        return [], []

    def encode_states(self, states):
        self.encoded_states = states
        return torch.zeros(len(states), 2)


class RebuildForbiddenModel(FakeModel):
    def rebuild_skill_table(self):
        raise AssertionError("checkpoint skill_table.E must not be rebuilt during export")


def test_export_clstr_retrieval_run_writes_jsonl_trec_and_report(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "export"
    write_jsonl(
        queries_path,
        [
            {"query_id": "q1", "query_text": "Need a weather API.", "split": "test"},
            {"query_id": "q2", "query": "Need a finance API.", "split": "test"},
        ],
    )
    write_jsonl(
        skills_path,
        [
            {"skill_id": "s0", "name": "weather", "description": "weather tool"},
            {"skill_id": "s1", "name": "finance", "description": "finance tool"},
            {"skill_id": "s2", "name": "calendar", "description": "calendar tool"},
        ],
    )
    monkeypatch.setattr("clstr.clstr_retrieval_export._build_model", lambda **_kwargs: (FakeModel(), {"fake": True}))

    report = export_clstr_retrieval_run(
        queries_path=queries_path,
        skills_path=skills_path,
        output_dir=output_dir,
        base_model_name="fake-model",
        top_k=2,
        batch_size=2,
        run_name="clstr_fake",
    )

    assert report["status"] == "ok"
    assert report["query_count"] == 2
    assert report["skill_count"] == 3
    assert report["top_k"] == 2
    assert report["model"]["fake"] is True
    assert read_jsonl(output_dir / "predictions.jsonl") == [
        {
            "query_id": "q1",
            "ranked_skill_ids": ["s1", "s2"],
            "scores": [2.0, 1.0],
            "split": "test",
        },
        {
            "query_id": "q2",
            "ranked_skill_ids": ["s0", "s1"],
            "scores": [3.0, 0.10000000149011612],
            "split": "test",
        },
    ]
    assert (output_dir / "run.tsv").read_text(encoding="utf-8").splitlines() == [
        "q1\tQ0\ts1\t1\t2.00000000\tclstr_fake",
        "q1\tQ0\ts2\t2\t1.00000000\tclstr_fake",
        "q2\tQ0\ts0\t1\t3.00000000\tclstr_fake",
        "q2\tQ0\ts1\t2\t0.10000000\tclstr_fake",
    ]


def test_export_clstr_retrieval_run_streams_batch_outputs_and_progress(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "export"
    write_jsonl(
        queries_path,
        [
            {"query_id": "q1", "query_text": "one"},
            {"query_id": "q2", "query_text": "two"},
            {"query_id": "q3", "query_text": "three"},
        ],
    )
    write_jsonl(
        skills_path,
        [
            {"skill_id": "s0", "name": "zero"},
            {"skill_id": "s1", "name": "one"},
            {"skill_id": "s2", "name": "two"},
        ],
    )
    monkeypatch.setattr(
        "clstr.clstr_retrieval_export._build_model",
        lambda **_kwargs: (FakeModel(StreamingProbeSkillTable(output_dir)), {"fake": True}),
    )

    report = export_clstr_retrieval_run(
        queries_path=queries_path,
        skills_path=skills_path,
        output_dir=output_dir,
        base_model_name="fake-model",
        top_k=2,
        batch_size=1,
        run_name="clstr_stream",
    )

    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "ok"
    assert progress["phase"] == "done"
    assert progress["processed_queries"] == 3
    assert progress["total_queries"] == 3
    assert report["progress_path"] == str(output_dir / "progress.json")
    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert (output_dir / "run.tsv").read_text(encoding="utf-8").count("\n") == 6


def test_export_clstr_retrieval_run_keeps_checkpoint_skill_table_embeddings(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "export"
    write_jsonl(queries_path, [{"query_id": "q1", "query_text": "one"}])
    write_jsonl(
        skills_path,
        [
            {"skill_id": "s0", "name": "zero"},
            {"skill_id": "s1", "name": "one"},
            {"skill_id": "s2", "name": "two"},
        ],
    )
    monkeypatch.setattr(
        "clstr.clstr_retrieval_export._build_model",
        lambda **_kwargs: (
            RebuildForbiddenModel(),
            {"fake": True, "loaded_skill_table_embeddings": True},
        ),
    )

    report = export_clstr_retrieval_run(
        queries_path=queries_path,
        skills_path=skills_path,
        output_dir=output_dir,
        base_model_name="fake-model",
        top_k=2,
        batch_size=1,
        run_name="clstr_checkpoint",
    )

    assert report["status"] == "ok"
    assert report["model"]["skill_table_rebuild"] == "skipped_checkpoint_embeddings"
    assert read_jsonl(output_dir / "predictions.jsonl")[0]["ranked_skill_ids"] == ["s1", "s2"]


def test_export_clstr_retrieval_run_cli_accepts_minimal_local_inputs(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "export"
    model_dir = tmp_path / "tiny-model"
    write_jsonl(queries_path, [{"query_id": "q1", "query_text": "weather lookup"}])
    write_jsonl(skills_path, [{"skill_id": "s0", "name": "weather", "description": "weather lookup"}])
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/export_clstr_retrieval_run.py",
            "--queries_path",
            str(queries_path),
            "--skills_path",
            str(skills_path),
            "--output_dir",
            str(output_dir),
            "--base_model_name",
            str(model_dir),
            "--model_dim",
            "16",
            "--top_k",
            "1",
            "--batch_size",
            "1",
            "--freeze_backbone",
            "--local_files_only",
            "--disable_cross_encoder",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["status"] == "ok"
    assert report["predictions_path"] == str(output_dir / "predictions.jsonl")
    assert read_jsonl(output_dir / "predictions.jsonl")[0]["ranked_skill_ids"] == ["s0"]
    assert (output_dir / "run.tsv").exists()
