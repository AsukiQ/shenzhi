#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote


SOURCE_DATASET = "Adorg/ToolBench"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUTPUT_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data")
DEFAULT_ALLOWED_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp")


def _ensure_within_allowed(path: Path, allowed_root: Path) -> Path:
    resolved = path.resolve()
    root = allowed_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay under allowed root {root}: {resolved}") from exc
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_api_manifest(path: Path | None, *, hf_endpoint: str, timeout: int) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to fetch the Hugging Face dataset manifest.") from exc
    url = f"{hf_endpoint.rstrip('/')}/api/datasets/{SOURCE_DATASET}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _selected_g3_files(api_manifest: dict[str, Any], max_answer_files: int | None = None) -> list[str]:
    siblings = [
        str(item.get("rfilename") or "")
        for item in api_manifest.get("siblings", [])
        if isinstance(item, dict) and item.get("rfilename")
    ]
    answer_files = sorted(
        name
        for name in siblings
        if name.startswith("answer/G3_answer/") and name.endswith(".json")
    )
    if max_answer_files is not None:
        answer_files = answer_files[:max_answer_files]
    selected = []
    if "instruction/G3_query.json" in siblings:
        selected.append("instruction/G3_query.json")
    selected.extend(answer_files)
    return selected


def _download_file(
    remote_file: str,
    target_path: Path,
    *,
    hf_endpoint: str,
    repo_id: str,
    timeout: int,
) -> dict[str, Any]:
    if target_path.exists() and target_path.stat().st_size > 0:
        return {
            "remote_file": remote_file,
            "target_path": str(target_path),
            "bytes": target_path.stat().st_size,
            "skipped": True,
        }
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download ToolBench-G3 files.") from exc
    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_suffix(target_path.suffix + ".part")
    encoded = quote(remote_file)
    url = f"{hf_endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{encoded}"
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    bytes_written = 0
    with part_path.open("wb") as fout:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fout.write(chunk)
                bytes_written += len(chunk)
    try:
        shutil.move(str(part_path), str(target_path))
    except FileNotFoundError:
        if target_path.exists() and target_path.stat().st_size > 0:
            return {
                "remote_file": remote_file,
                "target_path": str(target_path),
                "bytes": target_path.stat().st_size,
                "skipped": False,
            }
        raise
    return {
        "remote_file": remote_file,
        "target_path": str(target_path),
        "bytes": bytes_written,
        "skipped": False,
    }


def _download_file_with_retries(
    remote_file: str,
    target_path: Path,
    *,
    hf_endpoint: str,
    repo_id: str,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float = 0.0,
    retry_backoff_multiplier: float = 1.0,
) -> dict[str, Any]:
    attempts = 0
    last_error = ""
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            result = _download_file(
                remote_file,
                target_path,
                hf_endpoint=hf_endpoint,
                repo_id=repo_id,
                timeout=timeout,
            )
            result["attempts"] = attempts
            return result
        except Exception as exc:  # pragma: no cover - exact network exceptions vary by environment.
            last_error = str(exc)
            if attempt < max_retries and retry_backoff_seconds > 0:
                delay = retry_backoff_seconds * (retry_backoff_multiplier ** attempt)
                time.sleep(delay)
    return {
        "remote_file": remote_file,
        "target_path": str(target_path),
        "bytes": 0,
        "skipped": False,
        "failed": True,
        "attempts": attempts,
        "error": last_error,
    }


def _verify_g3_output(output_root: Path, *, expected_answer_files: int | None = None) -> dict[str, Any]:
    instruction_path = output_root / "instruction" / "G3_query.json"
    answer_dir = output_root / "answer" / "G3_answer"
    answer_files = sorted(answer_dir.glob("*.json")) if answer_dir.exists() else []
    missing = []
    if not instruction_path.is_file():
        missing.append("instruction/G3_query.json")
    if not answer_files:
        missing.append("answer/G3_answer/*.json")
    elif expected_answer_files is not None and len(answer_files) < expected_answer_files:
        missing.append(f"answer/G3_answer/*.json: expected {expected_answer_files}, found {len(answer_files)}")
    if missing:
        status = "missing_required_files" if not answer_files or not instruction_path.is_file() else "incomplete_answer_files"
    else:
        status = "ok"
    return {
        "status": status,
        "data_root": str(output_root),
        "has_instruction_g3": instruction_path.is_file(),
        "has_answer_g3": bool(answer_files),
        "g3_answer_files": len(answer_files),
        "expected_g3_answer_files": expected_answer_files,
        "missing": missing,
    }


def download_toolbench_g3_from_hf(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    api_manifest_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    hf_endpoint: str = DEFAULT_HF_ENDPOINT,
    allowed_root: str | Path = DEFAULT_ALLOWED_ROOT,
    dry_run: bool = False,
    max_answer_files: int | None = None,
    timeout: int = 60,
    workers: int = 1,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.0,
    retry_backoff_multiplier: float = 1.0,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if retry_backoff_seconds < 0:
        raise ValueError(f"retry_backoff_seconds must be >= 0, got {retry_backoff_seconds}")
    if retry_backoff_multiplier < 1:
        raise ValueError(f"retry_backoff_multiplier must be >= 1, got {retry_backoff_multiplier}")
    allowed = Path(allowed_root)
    output = _ensure_within_allowed(Path(output_root), allowed)
    manifest_out = _ensure_within_allowed(
        Path(manifest_path) if manifest_path is not None else output.parent / "hf_g3_download_manifest.json",
        allowed,
    )
    api_path = _ensure_within_allowed(Path(api_manifest_path), allowed) if api_manifest_path is not None else None
    api_manifest = _load_api_manifest(api_path, hf_endpoint=hf_endpoint, timeout=timeout)
    selected_files = _selected_g3_files(api_manifest, max_answer_files=max_answer_files)
    selected_answer_file_count = sum(1 for name in selected_files if name.startswith("answer/G3_answer/"))
    missing_remote = []
    if "instruction/G3_query.json" not in selected_files:
        missing_remote.append("instruction/G3_query.json")
    if not any(name.startswith("answer/G3_answer/") for name in selected_files):
        missing_remote.append("answer/G3_answer/*.json")

    manifest: dict[str, Any] = {
        "status": "dry_run" if dry_run else "ok",
        "source_dataset": SOURCE_DATASET,
        "paper_role": "official_g3_mirror",
        "hf_endpoint": hf_endpoint,
        "output_root": str(output),
        "api_manifest_path": str(api_path) if api_path is not None else None,
        "selected_file_count": len(selected_files),
        "selected_answer_file_count": selected_answer_file_count,
        "selected_files": selected_files,
        "missing_remote_required_files": missing_remote,
        "downloaded_file_count": 0,
        "downloaded_bytes": 0,
        "downloads": [],
        "max_answer_files": max_answer_files,
        "workers": workers,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "retry_backoff_multiplier": retry_backoff_multiplier,
    }
    if missing_remote:
        manifest["status"] = "missing_remote_required_files"
        _write_json(manifest_out, manifest)
        return manifest
    if dry_run:
        _write_json(manifest_out, manifest)
        return manifest

    def download_one(remote_file: str) -> dict[str, Any]:
        target_path = output / remote_file
        return _download_file_with_retries(
            remote_file,
            target_path,
            hf_endpoint=hf_endpoint,
            repo_id=SOURCE_DATASET,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_backoff_multiplier=retry_backoff_multiplier,
        )

    if workers == 1:
        downloads = [download_one(remote_file) for remote_file in selected_files]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            downloads = list(executor.map(download_one, selected_files))
    successful_downloads = [item for item in downloads if not item.get("failed")]
    failed_downloads = [item for item in downloads if item.get("failed")]
    manifest["downloads"] = successful_downloads[:100]
    manifest["downloaded_file_count"] = len(successful_downloads)
    manifest["downloaded_bytes"] = sum(int(item.get("bytes") or 0) for item in successful_downloads)
    manifest["skipped_existing_file_count"] = sum(1 for item in successful_downloads if item.get("skipped"))
    manifest["failed_download_count"] = len(failed_downloads)
    manifest["failed_downloads"] = failed_downloads[:100]
    verification = _verify_g3_output(output, expected_answer_files=selected_answer_file_count)
    manifest["verification"] = verification
    if failed_downloads:
        manifest["status"] = "download_failed"
    elif verification["status"] != "ok":
        manifest["status"] = verification["status"]
    _write_json(manifest_out, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official ToolBench-G3 files from the Adorg/ToolBench HF mirror.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--api_manifest_path", default=None)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--allowed_root", default=str(DEFAULT_ALLOWED_ROOT))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_answer_files", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_backoff_seconds", type=float, default=5.0)
    parser.add_argument("--retry_backoff_multiplier", type=float, default=2.0)
    args = parser.parse_args()
    manifest = download_toolbench_g3_from_hf(
        output_root=args.output_root,
        api_manifest_path=args.api_manifest_path,
        manifest_path=args.manifest_path,
        hf_endpoint=args.hf_endpoint,
        allowed_root=args.allowed_root,
        dry_run=args.dry_run,
        max_answer_files=args.max_answer_files,
        timeout=args.timeout,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        retry_backoff_multiplier=args.retry_backoff_multiplier,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest.get("status") in {"ok", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
