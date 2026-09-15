#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_preprocess import build_clstr_full_base_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified CLSTR full-base training JSONL.")
    parser.add_argument("--registry_path", default="data/clstr_full_base_registry/sources.jsonl")
    parser.add_argument("--output_dir", default="data/clstr_full_base_train")
    parser.add_argument("--report_dir", default="outputs/clstr_full_base_train")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--max_records", type=int)
    args = parser.parse_args()

    manifest = build_clstr_full_base_data(
        registry_path=Path(args.registry_path),
        output_dir=Path(args.output_dir),
        report_dir=Path(args.report_dir),
        repo_root=Path(args.repo_root),
        max_records=args.max_records,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
