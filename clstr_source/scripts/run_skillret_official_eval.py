#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillret_official import (
    build_official_protocol_report,
    evaluate_official_run,
    export_hf_encoder_official_run,
    export_lexical_official_run,
    write_skillret_serialization_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SKILLRET official-protocol report/export/eval helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    protocol = sub.add_parser("protocol-report")
    protocol.add_argument("--official_repo", default="/root/autodl-tmp/skillret_repo")
    protocol.add_argument("--output_path", default="outputs/skillret_official_eval/protocol_report.json")

    serialization = sub.add_parser("serialization-report")
    serialization.add_argument("--official_repo", default="/root/autodl-tmp/skillret_repo")
    serialization.add_argument("--output_path", default="outputs/skillret_official_eval/serialization_report.json")

    lexical = sub.add_parser("lexical-baseline")
    lexical.add_argument("--data_root", default="data/skillret")
    lexical.add_argument("--output_dir", default="outputs/skillret_official_eval/baselines/lexical_official")
    lexical.add_argument("--split", default="test")
    lexical.add_argument("--top_k", type=int, default=50)
    lexical.add_argument("--max_queries", type=int)

    hf = sub.add_parser("hf-encoder-baseline")
    hf.add_argument("--data_root", default="data/skillret")
    hf.add_argument("--model_name_or_path", required=True)
    hf.add_argument("--output_dir", required=True)
    hf.add_argument("--split", default="test")
    hf.add_argument("--top_k", type=int, default=50)
    hf.add_argument("--batch_size", type=int, default=16)
    hf.add_argument("--max_length", type=int, default=1024)
    hf.add_argument("--max_queries", type=int)
    hf.add_argument("--max_skills", type=int)
    hf.add_argument("--run_name", default="hf_encoder_official")

    evaluate = sub.add_parser("evaluate-run")
    evaluate.add_argument("--data_root", default="data/skillret")
    evaluate.add_argument("--run_path", required=True)
    evaluate.add_argument("--output_dir", required=True)
    evaluate.add_argument("--split", default="test")

    args = parser.parse_args()
    if args.cmd == "protocol-report":
        payload = build_official_protocol_report(args.official_repo, args.output_path)
    elif args.cmd == "serialization-report":
        payload = write_skillret_serialization_report(args.output_path, args.official_repo)
    elif args.cmd == "lexical-baseline":
        report = export_lexical_official_run(
            data_root=args.data_root,
            output_dir=args.output_dir,
            split=args.split,
            top_k=args.top_k,
            max_queries=args.max_queries,
        )
        payload = evaluate_official_run(
            data_root=args.data_root,
            run_path=report["run_path"],
            output_dir=args.output_dir,
            split=args.split,
        )
        payload.update(
            {
                "method": "lexical_official",
                "train_data": "none",
                "model_or_checkpoint": "lexical sanity baseline",
            }
        )
        Path(args.output_dir, "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif args.cmd == "hf-encoder-baseline":
        report = export_hf_encoder_official_run(
            data_root=args.data_root,
            model_name_or_path=args.model_name_or_path,
            output_dir=args.output_dir,
            split=args.split,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_queries=args.max_queries,
            max_skills=args.max_skills,
            run_name=args.run_name,
        )
        payload = evaluate_official_run(
            data_root=args.data_root,
            run_path=report["run_path"],
            output_dir=args.output_dir,
            split=args.split,
        )
        payload.update(
            {
                "method": args.run_name,
                "train_data": "none",
                "model_or_checkpoint": args.model_name_or_path,
                "init": "frozen pretrained encoder",
                "trainable_params": "none",
                "frozen_backbone": True,
                "paper_role": "paper_static_retrieval" if args.max_queries is None and args.max_skills is None else "pilot",
            }
        )
        Path(args.output_dir, "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif args.cmd == "evaluate-run":
        payload = evaluate_official_run(args.data_root, args.run_path, args.output_dir, args.split)
    else:  # pragma: no cover
        raise ValueError(args.cmd)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
