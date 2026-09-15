#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SOURCE_DATASET = "mangopy/ToolRet-Training-20w"
DEFAULT_OUTPUT_DIR = Path(
    "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_training/ToolRet-Training-20w"
)
REMOTE_FILES = [
    f"ToolRet-Training-20w/train-{idx:05d}-of-00006.parquet"
    for idx in range(6)
]


def _local_path(output_dir: Path, remote_file: str) -> Path:
    return output_dir / Path(remote_file).name


def download_toolret_training_parquet(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_endpoint = os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache")

    local_files = [str(_local_path(out_dir, remote_file)) for remote_file in REMOTE_FILES]
    manifest = {
        "status": "dry_run" if dry_run else "ok",
        "source_dataset": SOURCE_DATASET,
        "hf_endpoint": hf_endpoint,
        "output_dir": str(out_dir),
        "remote_files": REMOTE_FILES,
        "local_files": local_files,
        "downloaded_files": [],
        "skipped_existing_files": [],
    }
    if dry_run:
        return manifest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download ToolRet parquet shards. "
            "Use the reasoning_trap env on the login node."
        ) from exc

    for remote_file in REMOTE_FILES:
        target = _local_path(out_dir, remote_file)
        if target.exists() and target.stat().st_size > 0:
            manifest["skipped_existing_files"].append(str(target))
            continue
        downloaded = hf_hub_download(
            repo_id=SOURCE_DATASET,
            repo_type="dataset",
            filename=remote_file,
            local_dir=str(out_dir.parent),
            endpoint=hf_endpoint,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != target and downloaded_path.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.replace(target)
        manifest["downloaded_files"].append(str(target))

    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download ToolRet-Training-20w parquet shards on the login node. "
            "Do not run this from compute nodes."
        )
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest = download_toolret_training_parquet(args.output_dir, dry_run=args.dry_run)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
