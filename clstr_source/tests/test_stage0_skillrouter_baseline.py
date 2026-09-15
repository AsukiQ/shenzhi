from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

import clstr.stage0_skillrouter_baseline as baseline
from clstr.stage0_skillrouter_baseline import run_stage0_skillrouter_frozen_baseline


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_stage0_skillrouter_baseline_uses_canonical_skill_pool_not_original_pool(tmp_path, monkeypatch):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    output_dir = tmp_path / "outputs/stage0_baseline"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "canonical/weather",
                "canonical_skill_id": "canonical/weather",
                "alias_skill_ids": ["canonical/weather", "original/weather"],
                "name": "Weather",
                "description": "Weather forecast.",
            },
            {
                "skill_id": "canonical/news",
                "canonical_skill_id": "canonical/news",
                "name": "News",
                "description": "Search news.",
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "query_id": "q1",
                "query_text": "Need weather.",
                "positive_skill_id": "original/weather",
            }
        ],
    )
    seen_texts: list[str] = []

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        seen_texts.extend(texts)
        if len(texts) == 1:
            return torch.tensor([[1.0, 0.0]])
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(baseline, "_encode_skillrouter_texts", fake_encode)

    report = run_stage0_skillrouter_frozen_baseline(
        data_root=data_root,
        output_dir=output_dir,
        model_name_or_path="fake-skillrouter",
        top_k=2,
    )

    assert report["status"] == "ok"
    assert report["skill_pool_path"] == str(data_root / "skill_pool.jsonl")
    assert report["skill_count"] == 2
    assert report["uses_canonical_skill_pool"] is True
    assert "original/weather" not in report["skill_ids"]
    assert set(report["skill_ids"]) == {"canonical/weather", "canonical/news"}
    assert all("original/weather" not in text for text in seen_texts)


def test_stage0_skillrouter_baseline_outputs_run_and_qrels_in_canonical_id_space(tmp_path, monkeypatch):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    output_dir = tmp_path / "outputs/stage0_baseline"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "canonical/weather",
                "canonical_skill_id": "canonical/weather",
                "alias_skill_ids": ["canonical/weather", "alias/weather"],
                "name": "Weather",
                "description": "Weather forecast.",
            },
            {
                "skill_id": "canonical/news",
                "canonical_skill_id": "canonical/news",
                "name": "News",
                "description": "Search news.",
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "query_id": "q1",
                "query_text": "Need weather.",
                "positive_skill_ids": ["canonical/weather", "alias/weather"],
                "negative_skill_ids": ["canonical/news"],
            },
            {
                "query_id": "q2",
                "query_text": "Need news.",
                "positive_skill_id": "canonical/news",
            },
        ],
    )

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        if len(texts) == 2 and "Need weather." in texts[0]:
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(baseline, "_encode_skillrouter_texts", fake_encode)

    report = run_stage0_skillrouter_frozen_baseline(
        data_root=data_root,
        output_dir=output_dir,
        model_name_or_path="fake-skillrouter",
        top_k=2,
    )

    run_lines = (output_dir / "run.tsv").read_text(encoding="utf-8").splitlines()
    qrels = read_jsonl(output_dir / "qrels.jsonl")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert report["run_path"] == str(output_dir / "run.tsv")
    assert report["qrels_path"] == str(output_dir / "qrels.jsonl")
    assert {row["skill_id"] for row in qrels} == {"canonical/weather", "canonical/news"}
    assert all("alias/weather" not in line for line in run_lines)
    assert all("alias/weather" not in json.dumps(row) for row in qrels)
    assert set(manifest["canonical_skill_ids"]) == {"canonical/weather", "canonical/news"}
    assert manifest["qrels_id_space"] == "canonical_skill_id"
    assert metrics["method"] == "stage0_skillrouter_frozen_baseline"
    assert metrics["run_format"] == "trec"
    assert metrics["qrel_query_count"] == 2


def test_stage0_skillrouter_baseline_batched_trec_run_matches_full_matrix(tmp_path):
    query_ids = ["q1", "q2", "q3"]
    skill_ids = ["s1", "s2", "s3", "s4"]
    query_embs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.4, 0.2, 0.8],
        ]
    )
    skill_embs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.7, 0.1],
        ]
    )
    full_path = tmp_path / "full.tsv"
    batched_path = tmp_path / "batched.tsv"

    baseline._write_trec_run(
        path=full_path,
        query_ids=query_ids,
        skill_ids=skill_ids,
        scores=query_embs @ skill_embs.t(),
        top_k=3,
        run_name="baseline",
    )
    report = baseline._write_trec_run_batched(
        path=batched_path,
        query_ids=query_ids,
        skill_ids=skill_ids,
        query_embs=query_embs,
        skill_embs=skill_embs,
        top_k=3,
        run_name="baseline",
        query_batch_size=2,
        progress_path=tmp_path / "progress.json",
    )

    assert batched_path.read_text(encoding="utf-8") == full_path.read_text(encoding="utf-8")
    assert report == {"processed_queries": 3, "processed_batches": 2}
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "done"
    assert progress["processed_queries"] == 3
    assert progress["processed_batches"] == 2


def test_stage0_skillrouter_baseline_sbatch_wraps_cli():
    script = Path("scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh").read_text(encoding="utf-8")
    env_script = Path("scripts/sbatch/_clstr_gpu_env.sh").read_text(encoding="utf-8")

    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert "module load miniforge3" in env_script
    assert "module load cuda" in env_script
    assert "CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/scyb713/run/miniconda3/envs/xzf}" in env_script
    assert 'source activate "${CONDA_ENV_PATH}"' in env_script
    assert "export PYTHONUNBUFFERED=1" in env_script
    assert "scripts/run_clstr_stage0_skillrouter_frozen_baseline.py" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final" in script
    assert ".cache/hf_models/SkillRouter-Embedding-0.6B" in script
    assert "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak" in script
    assert "SCORE_BATCH_SIZE" in script
    assert "--score_batch_size" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in env_script


def test_stage0_skillrouter_baseline_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "scripts/run_clstr_stage0_skillrouter_frozen_baseline.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--data_root" in proc.stdout
    assert "--model_name_or_path" in proc.stdout
    assert "--score_batch_size" in proc.stdout
