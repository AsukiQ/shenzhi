#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.external_data import write_json
from clstr.qwen_external_encoder import write_qwen_init_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen3-8B under the CLSTR workspace.")
    parser.add_argument("--repo_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--local_dir", default="models/Qwen3-8B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--output_dir", default="outputs/clstr_qwen3_8b_init")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(ROOT)
    hardware = {
        "status": "ok",
        "repo_id": args.repo_id,
        "local_dir": str(Path(args.local_dir)),
        "cache_dir": str(Path(args.cache_dir)),
        "disk": {
            "root": str(ROOT),
            "total_gb": round(total / 1024**3, 3),
            "used_gb": round(used / 1024**3, 3),
            "free_gb": round(free / 1024**3, 3),
        },
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hardware["gpu"] = {"gpu_name": props.name, "total_memory_gb": round(props.total_memory / 1024**3, 3)}
    write_json(output_dir / "hardware_report.json", hardware)

    if not args.download:
        report = write_qwen_init_manifest(
            output_dir=output_dir,
            model_name_or_path=args.repo_id,
            cache_dir=str(Path(args.cache_dir)),
            status="needs_download",
        )
        print(json.dumps(report, ensure_ascii=False))
        return

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(Path(args.local_dir)),
            cache_dir=str(Path(args.cache_dir)),
            local_files_only=args.local_files_only,
        )
        report = write_qwen_init_manifest(
            output_dir=output_dir,
            model_name_or_path=local_dir,
            cache_dir=str(Path(args.cache_dir)),
            status="ok",
        )
    except BaseException as exc:
        report = write_qwen_init_manifest(
            output_dir=output_dir,
            model_name_or_path=args.repo_id,
            cache_dir=str(Path(args.cache_dir)),
            status="blocked",
            error=str(exc),
        )
        report["repro_command"] = (
            "python scripts/prepare_qwen3_8b.py --download "
            f"--repo_id {args.repo_id} --local_dir {args.local_dir} --cache_dir {args.cache_dir}"
        )
        report["network_turbo_reference"] = "https://www.autodl.com/docs/network_turbo/"
        write_json(output_dir / "blocker_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
