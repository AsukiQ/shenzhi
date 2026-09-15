#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.memory_utility_gate_train import train_memory_utility_gate
from clstr.qwen_clstr_lineage import sha256_path


def _json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an audit-bound CLSTR memory-utility gate."
    )
    parser.add_argument("--route_records_path", required=True)
    parser.add_argument("--audit_report_path", required=True)
    parser.add_argument("--dynamic_selection_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--fused_rank_weight", type=float, default=1.0)
    parser.add_argument("--static_no_regret_weight", type=float, default=1.0)
    parser.add_argument("--direct_gate_bce_weight", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    dynamic = _json(args.dynamic_selection_path)
    if sha256_path(args.route_records_path)["sha256"] != str(
        dynamic.get("validation_route_records", {}).get("sha256") or ""
    ):
        raise ValueError("selected validation route-record file SHA-256 mismatch")
    report = train_memory_utility_gate(
        route_records=_jsonl(args.route_records_path),
        audit_report=_json(args.audit_report_path),
        dynamic_selection=dynamic,
        output_dir=args.output_dir,
        temperature=args.temperature,
        fused_rank_weight=args.fused_rank_weight,
        static_no_regret_weight=args.static_no_regret_weight,
        direct_gate_bce_weight=args.direct_gate_bce_weight,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ok", "not_recommended"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, KeyError) as exc:
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
        raise SystemExit(2)
