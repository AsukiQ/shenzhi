#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SOURCE_DATASET = "OpenBMB/ToolBench data.zip"
DEFAULT_ALLOWED_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp")
DEFAULT_EXTRACT_ROOT = DEFAULT_ALLOWED_ROOT / "ToolBench"
DEFAULT_ZIP_PATH = DEFAULT_EXTRACT_ROOT / "data.zip"
GOOGLE_DRIVE_FILE_ID = "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm=yes"
TSINGHUA_CLOUD_URL = "https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/"
TSINGHUA_DOWNLOAD_URL = "https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/?dl=1"
MODELSCOPE_TOOLBENCH_STATIC_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/toolbench-static/data.zip"


def _official_sources() -> dict[str, str]:
    return {
        "google_drive_file_id": GOOGLE_DRIVE_FILE_ID,
        "google_drive_url": GOOGLE_DRIVE_URL,
        "tsinghua_cloud_url": TSINGHUA_CLOUD_URL,
        "tsinghua_download_url": TSINGHUA_DOWNLOAD_URL,
        "toolbench_repo": "https://github.com/OpenBMB/ToolBench",
    }


def _manual_fallback() -> dict[str, Any]:
    return {
        "reason": "google_drive_or_tsinghua_cloud_unavailable",
        "target_zip": str(DEFAULT_ZIP_PATH),
        "target_data_root": str(DEFAULT_EXTRACT_ROOT / "data"),
        "instructions": [
            "Download the official ToolBench data.zip from Google Drive or Tsinghua Cloud on the login node.",
            f"Place it at {DEFAULT_ZIP_PATH}.",
            "Then rerun scripts/download_toolbench_data.py without --download to extract and verify it.",
        ],
    }


def _static_smoke_fallback() -> dict[str, Any]:
    return {
        "url": MODELSCOPE_TOOLBENCH_STATIC_URL,
        "dataset": "ToolBench-Static",
        "paper_role": "smoke_only_not_official_g3",
        "may_replace_official_toolbench_g3": False,
        "expected_members": [
            "data/toolbench_static/in_domain.json",
            "data/toolbench_static/out_of_domain.json",
        ],
        "note": (
            "This fallback can test static retrieval/schema plumbing only. It lacks "
            "official ToolBench-G3 instruction/G3_query.json and answer/G3_answer trees."
        ),
    }


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_within_allowed(path: Path, allowed_root: Path) -> Path:
    resolved = path.resolve()
    root = allowed_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay under allowed root {root}: {resolved}") from exc
    return path


def _resolve_data_root(path: Path) -> Path:
    if (path / "instruction" / "G3_query.json").is_file():
        return path
    if (path / "data" / "instruction" / "G3_query.json").is_file():
        return path / "data"
    return path


def verify_toolbench_data(data_root: str | Path, expected_answer_files: int | None = None) -> dict[str, Any]:
    root = _resolve_data_root(Path(data_root))
    instruction_path = root / "instruction" / "G3_query.json"
    answer_dir = root / "answer" / "G3_answer"
    answer_files = sorted(answer_dir.glob("*.json")) if answer_dir.exists() else []
    toolenv_files = sorted((root / "toolenv").glob("**/*.json")) if (root / "toolenv").exists() else []
    retrieval_files = sorted((root / "retrieval").glob("**/*.json")) if (root / "retrieval").exists() else []
    static_in_domain = root / "toolbench_static" / "in_domain.json"
    static_out_of_domain = root / "toolbench_static" / "out_of_domain.json"
    static_available = static_in_domain.is_file() and static_out_of_domain.is_file()

    missing: list[str] = []
    if not instruction_path.is_file():
        missing.append("instruction/G3_query.json")
    if not answer_files:
        missing.append("answer/G3_answer/*.json")
    elif expected_answer_files is not None and len(answer_files) < expected_answer_files:
        missing.append(f"answer/G3_answer/*.json: expected {expected_answer_files}, found {len(answer_files)}")

    if not missing:
        status = "ok"
    elif answer_files and expected_answer_files is not None and len(answer_files) < expected_answer_files:
        status = "incomplete_answer_files"
    else:
        status = "missing_required_files"

    return {
        "status": status,
        "source_dataset": SOURCE_DATASET,
        "data_root": str(root),
        "has_instruction_g3": instruction_path.is_file(),
        "has_answer_g3": bool(answer_files),
        "has_toolenv_assets": bool(toolenv_files),
        "has_retrieval_assets": bool(retrieval_files),
        "g3_answer_files": len(answer_files),
        "expected_g3_answer_files": expected_answer_files,
        "toolenv_json_files": len(toolenv_files),
        "retrieval_json_files": len(retrieval_files),
        "static_smoke_assets": {
            "available": static_available,
            "paper_role": "smoke_only_not_official_g3",
            "may_replace_official_toolbench_g3": False,
            "in_domain_path": str(static_in_domain) if static_in_domain.is_file() else None,
            "out_of_domain_path": str(static_out_of_domain) if static_out_of_domain.is_file() else None,
        },
        "missing": missing,
    }


def _safe_extract_zip(zip_path: Path, extract_root: Path, allowed_root: Path) -> list[str]:
    extracted: list[str] = []
    root = _ensure_within_allowed(extract_root, allowed_root)
    resolved_root = root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(f"unsafe zip member outside extract root: {member.filename}") from exc
        zf.extractall(root)
        extracted = zf.namelist()
    return extracted


def _stream_response_to_file(response, output_path: Path) -> int:
    bytes_written = 0
    with output_path.open("wb") as fout:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fout.write(chunk)
                bytes_written += len(chunk)
    return bytes_written


def _stream_url_to_file(url: str, output_path: Path) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for URL download; install it on the login node.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    bytes_written = _stream_response_to_file(response, output_path)
    return {
        "downloaded_bytes": bytes_written,
        "download_url": url,
        "content_type": response.headers.get("content-type", ""),
    }


def _download_google_drive_file(file_id: str, output_path: Path) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for Google Drive download; install it on the login node.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    session = requests.Session()
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id, "confirm": "yes"},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break
    parsed = parse_qs(urlparse(response.url).query)
    confirm_token = confirm_token or (parsed.get("confirm") or [None])[0]
    if confirm_token:
        response.close()
        response = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    bytes_written = _stream_response_to_file(response, part_path)
    shutil.move(str(part_path), str(output_path))
    return {"downloaded_bytes": bytes_written, "download_url": GOOGLE_DRIVE_URL}


def _download_plain_url(url: str, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    manifest = _stream_url_to_file(url, part_path)
    shutil.move(str(part_path), str(output_path))
    return manifest


def prepare_toolbench_data(
    zip_path: str | Path = DEFAULT_ZIP_PATH,
    extract_root: str | Path = DEFAULT_EXTRACT_ROOT,
    dry_run: bool = False,
    download: bool = False,
    source: str = "google",
    allowed_root: str | Path = DEFAULT_ALLOWED_ROOT,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    allowed = Path(allowed_root)
    zip_file = _ensure_within_allowed(Path(zip_path), allowed)
    extract_dir = _ensure_within_allowed(Path(extract_root), allowed)
    data_root = extract_dir / "data"
    manifest = {
        "status": "dry_run" if dry_run else "ok",
        "source_dataset": SOURCE_DATASET,
        "zip_path": str(zip_file),
        "extract_root": str(extract_dir),
        "data_root": str(data_root),
        "official_sources": _official_sources(),
        "manual_fallback": _manual_fallback(),
        "static_smoke_fallback": _static_smoke_fallback(),
        "download_requested": bool(download),
        "download_source": source,
        "download_url": TSINGHUA_DOWNLOAD_URL if source == "tsinghua" else GOOGLE_DRIVE_URL,
        "zip_exists_before": zip_file.is_file(),
        "extracted_members": [],
    }
    out_manifest = Path(manifest_path) if manifest_path is not None else extract_dir / "download_manifest.json"
    out_manifest = _ensure_within_allowed(out_manifest, allowed)
    if dry_run:
        _json_dump(out_manifest, manifest)
        return manifest

    downloaded_this_run = False
    if not zip_file.is_file() and download:
        try:
            if source == "tsinghua":
                manifest["download"] = _download_plain_url(TSINGHUA_DOWNLOAD_URL, zip_file)
            else:
                manifest["download"] = _download_google_drive_file(GOOGLE_DRIVE_FILE_ID, zip_file)
            downloaded_this_run = True
        except Exception as exc:  # pragma: no cover - network failures are environment-specific.
            manifest["status"] = "download_failed"
            manifest["error"] = str(exc)
            _json_dump(out_manifest, manifest)
            return manifest

    if not zip_file.is_file():
        manifest["status"] = "missing_zip"
        manifest["error"] = f"ToolBench data.zip not found: {zip_file}"
        _json_dump(out_manifest, manifest)
        return manifest

    try:
        manifest["extracted_members"] = _safe_extract_zip(zip_file, extract_dir, allowed)
    except zipfile.BadZipFile as exc:
        manifest["status"] = "invalid_zip"
        manifest["error"] = str(exc)
        if downloaded_this_run and zip_file.exists():
            invalid_path = zip_file.with_suffix(zip_file.suffix + ".invalid")
            if invalid_path.exists():
                invalid_path.unlink()
            shutil.move(str(zip_file), str(invalid_path))
            manifest["invalid_zip_path"] = str(invalid_path)
        _json_dump(out_manifest, manifest)
        return manifest
    verification = verify_toolbench_data(data_root)
    manifest["verification"] = verification
    if verification["status"] != "ok":
        manifest["status"] = verification["status"]
    _json_dump(out_manifest, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the official ToolBench data.zip on the login node, then verify "
            "that full G3 assets are available for CLSTR import."
        )
    )
    parser.add_argument("--zip_path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--extract_root", default=str(DEFAULT_EXTRACT_ROOT))
    parser.add_argument("--allowed_root", default=str(DEFAULT_ALLOWED_ROOT))
    parser.add_argument("--manifest_path")
    parser.add_argument("--download", action="store_true", help="Try Google Drive direct download if data.zip is missing.")
    parser.add_argument("--source", choices=("google", "tsinghua"), default="google")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--expected_answer_files", type=int, default=None)
    args = parser.parse_args()

    if args.verify_only:
        manifest = verify_toolbench_data(Path(args.extract_root) / "data", expected_answer_files=args.expected_answer_files)
        manifest_path = Path(args.manifest_path) if args.manifest_path else Path(args.extract_root) / "verify_manifest.json"
        _json_dump(_ensure_within_allowed(manifest_path, Path(args.allowed_root)), manifest)
    else:
        manifest = prepare_toolbench_data(
            zip_path=args.zip_path,
            extract_root=args.extract_root,
            dry_run=args.dry_run,
            download=args.download,
            source=args.source,
            allowed_root=args.allowed_root,
            manifest_path=args.manifest_path,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest.get("status") in {"ok", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
