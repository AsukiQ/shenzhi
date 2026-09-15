#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_clstr_lineage import (  # noqa: E402
    create_derived_lineage_manifest,
    create_lineage_manifest,
    validate_lineage_manifest,
    sha256_path,
)
from clstr.memory_utility_records import canonical_digest


def _parse_role_paths(values: list[str] | None) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values or []:
        if "=" not in str(value):
            raise ValueError(f"lineage parent must use role=path format: {value}")
        role, raw_path = str(value).split("=", 1)
        role = role.strip()
        raw_path = raw_path.strip()
        if not role or not raw_path:
            raise ValueError(f"lineage parent must use non-empty role=path format: {value}")
        if role in output:
            raise ValueError(f"duplicate lineage parent role: {role}")
        output[role] = Path(raw_path)
    return output


def _write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or validate frozen-Qwen CLSTR lineage manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--stage", required=True)
    create.add_argument("--checkpoint_path", required=True)
    create.add_argument("--expected_checkpoint_stage", required=True)
    create.add_argument("--model_path", required=True)
    create.add_argument("--skill_pool_path", required=True)
    create.add_argument("--data_manifest_path", required=True)
    create.add_argument("--parent", action="append", default=[])
    create.add_argument("--output_path", required=True)

    create_derived = subparsers.add_parser("create-derived")
    create_derived.add_argument("--stage", required=True)
    create_derived.add_argument("--checkpoint_path", required=True)
    create_derived.add_argument("--expected_checkpoint_stage", required=True)
    create_derived.add_argument("--model_path", required=True)
    create_derived.add_argument("--skill_pool_path", required=True)
    create_derived.add_argument("--data_manifest_path", required=True)
    create_derived.add_argument("--parent", action="append", default=[])
    create_derived.add_argument("--identity_parent_role", required=True)
    create_derived.add_argument("--stage4_selection_path")
    create_derived.add_argument("--output_path", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest_path", required=True)
    validate.add_argument("--expected_stage", required=True)
    validate.add_argument("--expected_parent", action="append", default=[])
    validate.add_argument("--expected_model_path")
    validate.add_argument("--expected_skill_pool_path")
    validate.add_argument("--expected_data_manifest_path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "create":
            manifest = create_lineage_manifest(
                stage=args.stage,
                checkpoint_path=args.checkpoint_path,
                expected_checkpoint_stage=args.expected_checkpoint_stage,
                model_path=args.model_path,
                skill_pool_path=args.skill_pool_path,
                data_manifest_path=args.data_manifest_path,
                parent_manifests=_parse_role_paths(args.parent),
            )
            output_path = _write_json(args.output_path, manifest)
            report = {
                "status": "ok",
                "command": "create",
                "stage": args.stage,
                "output_path": str(output_path.resolve()),
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            }
        elif args.command == "create-derived":
            stage_metadata = None
            if args.stage4_selection_path is not None:
                selection_path = Path(args.stage4_selection_path).resolve()
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                digest_payload = dict(selection)
                recorded = str(digest_payload.pop("manifest_sha256", ""))
                if not recorded or canonical_digest(digest_payload) != recorded:
                    raise ValueError("Stage4 selection self-hash mismatch")
                if selection.get("status") != "ok":
                    raise ValueError("Stage4 selection is not complete")
                stage_metadata = {
                    "stage4_selection": sha256_path(selection_path),
                    "router_integrity": dict(selection.get("router_integrity") or {}),
                    "reliability": dict(selection.get("reliability") or {}),
                }
            manifest = create_derived_lineage_manifest(
                stage=args.stage,
                checkpoint_path=args.checkpoint_path,
                expected_checkpoint_stage=args.expected_checkpoint_stage,
                model_path=args.model_path,
                skill_pool_path=args.skill_pool_path,
                data_manifest_path=args.data_manifest_path,
                parent_manifests=_parse_role_paths(args.parent),
                identity_parent_role=args.identity_parent_role,
                stage_metadata=stage_metadata,
            )
            output_path = _write_json(args.output_path, manifest)
            report = {
                "status": "ok",
                "command": args.command,
                "stage": args.stage,
                "output_path": str(output_path.resolve()),
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            }
        else:
            report = validate_lineage_manifest(
                args.manifest_path,
                expected_stage=args.expected_stage,
                expected_parent_manifests=_parse_role_paths(args.expected_parent),
                expected_model_path=args.expected_model_path,
                expected_skill_pool_path=args.expected_skill_pool_path,
                expected_data_manifest_path=args.expected_data_manifest_path,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "command": args.command,
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
