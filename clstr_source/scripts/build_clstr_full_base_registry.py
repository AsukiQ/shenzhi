#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.dataset_registry import build_clstr_full_base_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLSTR full-base dataset registry and audit report.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="data/clstr_full_base_registry")
    parser.add_argument("--report_dir", default="outputs/clstr_full_base_registry")
    args = parser.parse_args()

    manifest = build_clstr_full_base_registry(
        repo_root=Path(args.repo_root),
        output_dir=Path(args.output_dir),
        report_dir=Path(args.report_dir),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
