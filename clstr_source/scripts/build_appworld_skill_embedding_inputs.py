#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bridges.skillx.appworld_adapter import build_appworld_embedding_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-free embedding input text for AppWorld skills.")
    parser.add_argument("--pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_path", default="data/appworld_skill_pool/embedding_inputs.jsonl")
    parser.add_argument("--manifest_path", default="data/appworld_skill_pool/embedding_manifest.json")
    args = parser.parse_args()

    manifest = build_appworld_embedding_inputs(
        pool_path=args.pool_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
