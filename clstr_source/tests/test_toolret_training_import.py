from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_importer():
    path = Path("scripts/import_toolret_training.py")
    spec = importlib.util.spec_from_file_location("import_toolret_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_downloader():
    path = Path("scripts/download_toolret_training_parquet.py")
    spec = importlib.util.spec_from_file_location("download_toolret_training_parquet", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_toolret_training_normalizes_raw_hf_jsonl(tmp_path):
    raw_path = tmp_path / "raw/toolret_train.jsonl"
    _write_jsonl(
        raw_path,
        [
            {
                "query": "Is https://www.apple.com available in the Wayback Machine?",
                "prompt": "Given a URL availability task, retrieve tools that check archive availability.",
                "pos": [
                    "{'name': 'availability', 'description': 'Checks if a URL is archived.', "
                    "'parameters': {'url': {'type': 'str'}}}"
                ],
                "neg": [
                    "{'name': 'top_grossing_mac_apps', 'description': 'Fetches top-grossing Mac apps.', "
                    "'parameters': {'country': {'type': 'str'}}}"
                ],
            }
        ],
    )

    importer = _load_importer()
    manifest = importer.import_toolret_training(
        source_path=raw_path,
        output_dir=tmp_path / "data/toolret_training",
        max_rows=None,
    )
    out_dir = tmp_path / "data/toolret_training"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skills = [json.loads(line) for line in (out_dir / "skills.jsonl").read_text().splitlines()]

    assert manifest["status"] == "ok"
    assert manifest["source_dataset"] == "mangopy/ToolRet-Training-20w"
    assert manifest["retrieval_pairs"] == 1
    assert manifest["skills"] == 2
    assert retrieval == [
        {
            "query_id": "toolret-train-0",
            "query_text": "Given a URL availability task, retrieve tools that check archive availability.\n"
            "Task: Is https://www.apple.com available in the Wayback Machine?",
            "positive_skill_id": "toolret/availability",
            "negative_skill_ids": ["toolret/top-grossing-mac-apps"],
            "provenance": {
                "source_dataset": "mangopy/ToolRet-Training-20w",
                "source_path": str(raw_path),
                "row_index": 0,
                "split": "train",
            },
        }
    ]
    skills_by_id = {row["skill_id"]: row for row in skills}
    assert skills_by_id["toolret/availability"]["name"] == "availability"
    assert skills_by_id["toolret/availability"]["description"] == "Checks if a URL is archived."
    assert skills_by_id["toolret/availability"]["input_schema"] == {"url": {"type": "str"}}
    assert skills_by_id["toolret/top-grossing-mac-apps"]["description"] == "Fetches top-grossing Mac apps."


def test_import_toolret_training_normalizes_local_parquet_shards(tmp_path):
    pd = __import__("pandas")
    source_dir = tmp_path / "raw/ToolRet-Training-20w"
    source_dir.mkdir(parents=True)
    parquet_path = source_dir / "train-00000-of-00006.parquet"
    pd.DataFrame(
        [
            {
                "id": "train_0",
                "query": "What is the stock straddle data?",
                "prompt": "Given a stock market data task, retrieve tools for straddle data.",
                "positive": [
                    '{"name": "straddle", "description": "Retrieves straddle data.", '
                    '"parameters": {"ticker": {"type": "str"}}}'
                ],
                "negative": [
                    '{"name": "ideas_get_view_count", "description": "Fetches a view count.", '
                    '"parameters": {"uuid": {"type": "str"}}}'
                ],
            }
        ]
    ).to_parquet(parquet_path)

    importer = _load_importer()
    manifest = importer.import_toolret_training(
        source_path=source_dir,
        output_dir=tmp_path / "data/toolret_training",
        max_rows=None,
    )
    out_dir = tmp_path / "data/toolret_training"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skills = [json.loads(line) for line in (out_dir / "skills.jsonl").read_text().splitlines()]

    assert manifest["status"] == "ok"
    assert manifest["source_format"] == "local"
    assert manifest["retrieval_pairs"] == 1
    assert retrieval[0]["query_id"] == "train_0"
    assert retrieval[0]["positive_skill_id"] == "toolret/straddle"
    assert retrieval[0]["negative_skill_ids"] == ["toolret/ideas-get-view-count"]
    assert {row["skill_id"] for row in skills} == {"toolret/straddle", "toolret/ideas-get-view-count"}


def test_import_toolret_training_skips_rows_without_positive_tool(tmp_path):
    raw_path = tmp_path / "raw/toolret_train.jsonl"
    _write_jsonl(raw_path, [{"query": "bad row", "pos": [], "neg": []}])

    importer = _load_importer()
    manifest = importer.import_toolret_training(
        source_path=raw_path,
        output_dir=tmp_path / "data/toolret_training",
        max_rows=None,
    )

    assert manifest["status"] == "ok"
    assert manifest["retrieval_pairs"] == 0
    assert manifest["skipped_rows"] == 1


def test_import_toolret_training_sanitizes_lone_unicode_surrogates(tmp_path):
    raw_path = tmp_path / "raw/toolret_train.jsonl"
    _write_jsonl(
        raw_path,
        [
            {
                "id": "bad-unicode",
                "query": "Need a tool with malformed text \ud800.",
                "positive": [
                    {
                        "name": "bad_unicode_tool",
                        "description": "Description contains a lone surrogate \ud800.",
                        "parameters": {"text": {"type": "str", "description": "Bad field \udfff"}},
                    }
                ],
                "negative": [],
            }
        ],
    )

    importer = _load_importer()
    manifest = importer.import_toolret_training(
        source_path=raw_path,
        output_dir=tmp_path / "data/toolret_training",
        max_rows=None,
    )
    out_dir = tmp_path / "data/toolret_training"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skills = [json.loads(line) for line in (out_dir / "skills.jsonl").read_text().splitlines()]

    assert manifest["status"] == "ok"
    assert retrieval[0]["query_text"] == "Need a tool with malformed text ?."
    assert skills[0]["description"] == "Description contains a lone surrogate ?."
    assert skills[0]["input_schema"]["text"]["description"] == "Bad field ?"


def test_toolret_training_import_sbatch_entrypoint_uses_repo_local_paths_and_hf_mirror():
    script = Path("scripts/sbatch/run_import_toolret_training.sh").read_text(encoding="utf-8")

    assert "scripts/import_toolret_training.py" in script
    assert "data/toolret_training" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface" in script
    assert "HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets" in script
    assert "envs/reasoning_trap/bin/python" in script
    assert "SOURCE_PATH is required on compute nodes" in script
    assert "ALLOW_HF_STREAMING" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_download_toolret_training_parquet_dry_run_uses_hf_mirror_and_six_shards(tmp_path):
    downloader = _load_downloader()
    manifest = downloader.download_toolret_training_parquet(
        output_dir=tmp_path / "raw/toolret_training/ToolRet-Training-20w",
        dry_run=True,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["source_dataset"] == "mangopy/ToolRet-Training-20w"
    assert manifest["hf_endpoint"] == "https://hf-mirror.com"
    assert manifest["output_dir"] == str(tmp_path / "raw/toolret_training/ToolRet-Training-20w")
    assert len(manifest["remote_files"]) == 6
    assert manifest["remote_files"][0] == "ToolRet-Training-20w/train-00000-of-00006.parquet"
    assert manifest["remote_files"][-1] == "ToolRet-Training-20w/train-00005-of-00006.parquet"
    assert manifest["local_files"][0].endswith("train-00000-of-00006.parquet")
