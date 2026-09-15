#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.external_data import check_alfworld_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ALFWorld text-only data/environment prerequisites.")
    parser.add_argument("--alfworld_root", default="/root/autodl-tmp/alfworld")
    args = parser.parse_args()

    report = check_alfworld_root(Path(args.alfworld_root))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
