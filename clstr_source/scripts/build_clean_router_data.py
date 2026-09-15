#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.external_data import build_clean_router_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/check clean_router_data manifest from non-leaking SkillsBench inputs.")
    parser.add_argument("--skillsbench_root", default="/root/autodl-tmp/skillsbench")
    parser.add_argument("--audit_dir", default="outputs/leakage_audit")
    parser.add_argument("--output_dir", default="data/clean_router")
    args = parser.parse_args()

    manifest = build_clean_router_manifest(
        skillsbench_root=Path(args.skillsbench_root),
        audit_dir=Path(args.audit_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
