from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.data import verified_pair_to_dict
from clstr.validator import (
    CachedLLMValidator,
    build_verified_templates,
    openai_validator_client,
    parse_validator_output,
)


def _candidate_width(row: dict) -> int:
    widths = []
    for step in row.get("steps", []):
        for field in ("raw_topk", "raw_candidates"):
            if field in step and step[field] is not None:
                widths.append(len(step[field]))
                break
    return max(widths) if widths else 1


def _missing_candidate_error(row: dict):
    def _raise(_state, _k):
        raise ValueError(
            "missing raw_topk/raw_candidates for "
            f"task_id={row.get('task_id')}；请在输入轨迹中提供 raw_topk/raw_candidates，"
            "或先用当前 CLSTR 模型离线补齐真实 TopK 候选"
        )

    return _raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--call_llm_validator", action="store_true")
    parser.add_argument("--validator_model", default="gpt-4o")
    parser.add_argument("--validator_cache", default="cache/validator_cache.jsonl")
    args = parser.parse_args()

    rows = []
    for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    output_lines = []
    validator = None
    if args.call_llm_validator:
        validator = CachedLLMValidator(
            cache_path=args.validator_cache,
            client=openai_validator_client(args.validator_model),
        )

    for row in rows:
        if validator is not None:
            parsed = validator.validate(row)
            row["validator_output"] = json.dumps(parsed, ensure_ascii=False)
        else:
            parsed = parse_validator_output(row["validator_output"])
        if "steps" in row and "query" in row:
            topk_k = _candidate_width(row)
            verified = build_verified_templates(
                [row],
                _missing_candidate_error(row),
                K=topk_k,
            )
            output_lines.extend(json.dumps(verified_pair_to_dict(pair), ensure_ascii=False) for pair in verified)
        else:
            payload = {"task_id": row["task_id"], "essential_steps": parsed["essential_steps"]}
            if "reason" in parsed:
                payload["reason"] = parsed["reason"]
            output_lines.append(json.dumps(payload, ensure_ascii=False))

    Path(args.output_jsonl).write_text("\n".join(output_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
