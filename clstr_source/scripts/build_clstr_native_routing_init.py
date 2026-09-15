#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.native_rerank import (
    build_clstr_native_rerank_decision_report,
    build_native_routing_init_manifest,
)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLSTR-native routing init manifest.")
    parser.add_argument("--base_metrics_path", default="outputs/skillret_official_eval/clstr_skillrouter_init_full/metrics.json")
    parser.add_argument("--native_metrics_path", default="outputs/skillret_official_eval/clstr_native_rerank_full/metrics.json")
    parser.add_argument("--qdoc_metrics_path", default="outputs/skillret_official_eval/clstr_qdoc_rerank_full/metrics.json")
    parser.add_argument("--base_clstr_checkpoint", default="outputs/skillret_warmup_skillrouter_init/checkpoints/skillret_retrieval-step512.pt")
    parser.add_argument("--native_rerank_checkpoint", default="outputs/skillret_warmup_clstr_native_rerank/checkpoints/clstr_native_rerank-step512.pt")
    parser.add_argument("--decision_output_path", default="outputs/skillret_official_eval/clstr_native_rerank_decision_report.json")
    parser.add_argument("--manifest_output_path", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args()
    decision = build_clstr_native_rerank_decision_report(
        base_metrics=_load_json(args.base_metrics_path),
        native_metrics=_load_json(args.native_metrics_path),
        qdoc_metrics=_load_json(args.qdoc_metrics_path),
        output_path=args.decision_output_path,
        tolerance=args.tolerance,
    )
    manifest = build_native_routing_init_manifest(
        base_clstr_checkpoint=args.base_clstr_checkpoint,
        native_rerank_checkpoint=args.native_rerank_checkpoint,
        native_rerank_metrics=_load_json(args.native_metrics_path),
        decision_report=decision,
        output_path=args.manifest_output_path,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
