#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path


PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")


def clear_proxy_env() -> dict[str, str | None]:
    previous = {name: os.environ.get(name) for name in PROXY_VARS}
    for name in PROXY_VARS:
        os.environ.pop(name, None)
    return previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="Qwen/Qwen3-14B")
    parser.add_argument("--local_dir", default="models/Qwen3-14B")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    previous_proxy = clear_proxy_env()
    os.environ["HF_ENDPOINT"] = str(args.endpoint)
    local_dir = Path(args.local_dir)
    print(
        {
            "repo_id": args.repo_id,
            "local_dir": str(local_dir),
            "HF_ENDPOINT": os.environ["HF_ENDPOINT"],
            "cleared_proxy_vars": previous_proxy,
        }
    )
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(args.repo_id),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
