from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillret import import_skillret_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ThakiCloud/SKILLRET into CLSTR retrieval warmup JSONL files.")
    parser.add_argument("--source_root", default="/root/autodl-tmp/skillret")
    parser.add_argument("--output_dir", default="data/skillret")
    parser.add_argument("--report_dir", default="outputs/skillret_import")
    args = parser.parse_args()
    manifest = import_skillret_dataset(
        source_root=args.source_root,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
