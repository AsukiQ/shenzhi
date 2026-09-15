#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage_preflight import validate_stage2_routing_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CLSTR Stage 2 routing checkpoint inputs before loading large models.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    report = validate_stage2_routing_checkpoint(
        checkpoint_path=Path(args.checkpoint_path),
        skills_path=Path(args.skills_path),
        model_dim=args.model_dim,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
