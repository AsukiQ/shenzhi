#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.agentgym_data import audit_agentgym_agenttraj_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AgentGym/AgentTraj-L train trajectories for CLSTR use.")
    parser.add_argument("--repo_id", default="AgentGym/AgentTraj-L")
    parser.add_argument("--filename", default="alfworld_train.json")
    parser.add_argument("--benchmark", default="alfworld")
    parser.add_argument("--raw_dir", default="data/agentgym_agenttraj_l/raw")
    parser.add_argument("--output_dir", default="data/agentgym_agenttraj_l/alfworld")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_path = raw_dir / args.filename
    if args.download or not source_path.exists():
        from huggingface_hub import hf_hub_download

        source_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                repo_type="dataset",
                local_dir=raw_dir,
                local_files_only=args.local_files_only,
            )
        )
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found; set HF_ENDPOINT=https://hf-mirror.com or source /etc/network_turbo if download is slow"
        )
    report = audit_agentgym_agenttraj_file(
        source_path=source_path,
        output_dir=Path(args.output_dir),
        benchmark=args.benchmark,
    )
    report["network_turbo_reference"] = "https://www.autodl.com/docs/network_turbo/"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
