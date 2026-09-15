#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_clstr_multibench_submit import (  # noqa: E402
    load_accepted_qwen_clstr_smoke_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the accepted Qwen CLSTR multibench smoke gate at runtime."
    )
    parser.add_argument("--smoke_gate_path", required=True)
    parser.add_argument("--expected_smoke_gate_sha256", required=True)
    parser.add_argument("--expected_checkpoint_chain_digest", required=True)
    parser.add_argument("--expected_final_chain_manifest_sha256", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        loaded = load_accepted_qwen_clstr_smoke_gate(
            args.smoke_gate_path,
            checkpoint_chain_digest=args.expected_checkpoint_chain_digest,
            final_chain_manifest_sha256=args.expected_final_chain_manifest_sha256,
            expected_report_sha256=args.expected_smoke_gate_sha256,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "path": str(loaded["path"]),
                    "report_sha256": loaded["report_sha256"],
                    "checkpoint_chain_digest": args.expected_checkpoint_chain_digest,
                    "final_chain_manifest_sha256": args.expected_final_chain_manifest_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
