#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stabletoolbench_root", required=True)
    parser.add_argument("--runtime_dir", required=True)
    parser.add_argument("--api_pool_file", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    source = Path(args.stabletoolbench_root).resolve() / "toolbench" / "tooleval"
    runtime = Path(args.runtime_dir).resolve()
    api_pool = Path(args.api_pool_file).resolve()
    if not (source / "eval_pass_rate.py").is_file():
        raise FileNotFoundError(source / "eval_pass_rate.py")
    if not api_pool.is_file():
        raise FileNotFoundError(api_pool)
    shutil.copytree(source, runtime, dirs_exist_ok=True)
    config_path = runtime / "evaluators" / args.evaluator / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid evaluator config: {config_path}")
    config["apis_json"] = str(api_pool)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "runtime_dir": str(runtime),
        "evaluator": args.evaluator,
        "api_pool_file": str(api_pool),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
