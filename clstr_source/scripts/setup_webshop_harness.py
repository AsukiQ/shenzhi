#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.envs.webshop_official_adapter import DEFAULT_WEBSHOP_REPO_PATH, write_webshop_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check official WebShop harness dependencies.")
    parser.add_argument("--repo_path", default=str(DEFAULT_WEBSHOP_REPO_PATH))
    parser.add_argument("--output_path", default="outputs/webshop_eval/harness_smoke_report.json")
    args = parser.parse_args()

    report = write_webshop_smoke_report(
        output_path=Path(args.output_path),
        repo_path=Path(args.repo_path) if args.repo_path else None,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
