#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.history_channel import audit_history_channel_rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit for separated CLSTR current-state and replay-history channels."
    )
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require_actual_replay_observation", action="store_true")
    args = parser.parse_args()

    report = audit_history_channel_rows(
        _read_jsonl(args.rows),
        require_explicit_current=True,
        require_actual_replay_observation=bool(
            args.require_actual_replay_observation
        ),
    )
    report["rows_path"] = str(args.rows)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
