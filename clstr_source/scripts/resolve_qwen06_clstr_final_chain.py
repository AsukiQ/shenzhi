#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain  # noqa: E402


def _write_json_atomic(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve one release-safe Qwen06 CLSTR final checkpoint chain.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--reliability_gate_report_path")
    parser.add_argument("--candidate_provenance_overlay", action="store_true")
    parser.add_argument("--output_path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = resolve_qwen_clstr_final_chain(
            args.run_root,
            reliability_gate_report_path=args.reliability_gate_report_path,
            candidate_provenance_overlay=args.candidate_provenance_overlay,
        )
        _write_json_atomic(args.output_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
