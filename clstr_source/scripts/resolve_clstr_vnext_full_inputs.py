#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


FILE_ENV = {
    "TRAINING_SKILLS": "training_skills",
    "RETRIEVAL_ROWS": "retrieval_rows",
    "RETRIEVAL_DEV_ROWS": "retrieval_dev_rows",
    "STATIC_ROUTE_ROWS": "static_route_rows",
    "STATIC_ROUTE_DEV_ROWS": "static_route_dev_rows",
    "TRAJECTORY_ROWS": "trajectory_rows",
    "TRAJECTORY_DEV_ROWS": "trajectory_dev_rows",
    "PAIR_SUPPORT_ROWS": "causal_pair_support_rows",
    "PAIR_SUPPORT_DEV_ROWS": "causal_pair_support_dev_rows",
    "INVENTORY_CATALOGS": "inventory_catalogs",
    "CAUSAL_BRANCH_PAIRS": "causal_branch_pairs",
    "CAUSAL_BRANCH_DEV_PAIRS": "causal_branch_dev_pairs",
    "CAUSAL_ORDER_PAIRS": "causal_order_pairs",
    "CAUSAL_ORDER_DEV_PAIRS": "causal_order_dev_pairs",
    "CAUSAL_OUTCOME_PAIRS": "causal_outcome_pairs",
    "CAUSAL_OUTCOME_DEV_PAIRS": "causal_outcome_dev_pairs",
    "ONE_ERROR_PREFIX_ROWS": "one_error_prefix_rows",
    "ONE_ERROR_PREFIX_DEV_ROWS": "one_error_prefix_dev_rows",
    "TWO_ERROR_PREFIX_ROWS": "two_or_more_error_prefix_rows",
    "TWO_ERROR_PREFIX_DEV_ROWS": "two_or_more_error_prefix_dev_rows",
    "RECOVERY_PREFIX_ROWS": "recovery_prefix_rows",
    "RECOVERY_PREFIX_DEV_ROWS": "recovery_prefix_dev_rows",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok" or payload.get("blockers"):
        raise SystemExit("vNext full manifest is not approved")
    files = payload.get("files") or {}
    resolved: dict[str, str] = {"DATA_MANIFEST_PATH": str(manifest_path)}
    for environment_name, manifest_name in FILE_ENV.items():
        entry = files.get(manifest_name)
        if not isinstance(entry, dict):
            raise SystemExit(f"manifest lacks required file: {manifest_name}")
        path = Path(str(entry.get("path") or "")).resolve()
        if not path.is_file():
            raise SystemExit(f"manifest file does not exist: {manifest_name} -> {path}")
        resolved[environment_name] = str(path)
    if args.shell:
        for name, value in sorted(resolved.items()):
            print(f"export {name}={shlex.quote(value)}")
    else:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
