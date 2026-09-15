#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_online import ControlledMatchedHistorySelector  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and compare the three real E3 selectors without an LLM API."
    )
    parser.add_argument("--foundation_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--e1_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _learned_paths(e1_root: Path, method: str) -> tuple[Path | None, Path | None]:
    if method == "transformer":
        root = e1_root / "pilot_v2_seed23" / "transformer"
        return root / "best.pt", root / "train_report.json"
    if method == "lstr":
        root = e1_root / "full_seed31" / "lstr"
        return root / "best.pt", root / "train_report.json"
    return None, None


def _compact_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise RuntimeError("selector smoke requires exactly Top-8 outputs")
    first = rows[0]
    return {
        "skill_ids": [str(row["skill_id"]) for row in rows],
        "scores": [float(row["score"]) for row in rows],
        "static_support_sha256": str(first["static_support_sha256"]),
        "static_support_count": int(first["static_support_count"]),
        "candidate_count": int(first["candidate_count"]),
        "history_depth": int(first["history_depth"]),
        "history_window_depth": int(first["history_window_depth"]),
        "selected_expert": str(first["selected_expert"]),
    }


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("the real E3 selector smoke requires CUDA")
    e1_root = Path(args.e1_root).resolve()
    initial_state = (
        "goal: Check an airline reservation.\n"
        "task: Tau2 customer-service domain: airline\n"
        "observation: user: Please inspect my reservation."
    )
    next_state = (
        "goal: Check an airline reservation.\n"
        "task: Tau2 customer-service domain: airline\n"
        "observation: tool: {\"status\": \"ok\"}"
    )
    records: dict[str, Any] = {}
    shared_executed_skill = ""
    zero_reference: dict[str, Any] | None = None
    support_after_reference = ""
    for method in ("static", "transformer", "lstr"):
        checkpoint_path, report_path = _learned_paths(e1_root, method)
        selector = ControlledMatchedHistorySelector(
            method=method,
            foundation_checkpoint_path=args.foundation_checkpoint_path,
            skills_path=args.skills_path,
            e1_checkpoint_path=checkpoint_path,
            e1_report_path=report_path,
            device=args.device,
        )
        candidate_skill_ids = [
            skill_id
            for skill_id in selector.skill_ids
            if skill_id.startswith("tau2/airline/")
            and not skill_id.endswith("/__start__")
        ]
        if len(candidate_skill_ids) < 8:
            raise RuntimeError("locked E3 inventory has fewer than eight airline tools")
        session = selector.new_session()
        route_mode = "static" if method == "static" else "dynamic"
        zero = session.select(
            initial_state,
            candidate_skill_ids=candidate_skill_ids,
            top_k=8,
            route_mode=route_mode,
        )
        zero_record = _compact_selection(zero)
        if zero_reference is None:
            zero_reference = zero_record
            shared_executed_skill = str(zero[0]["skill_id"])
        else:
            if zero_record["skill_ids"] != zero_reference["skill_ids"]:
                raise RuntimeError("E3 zero-history rankings differ across arms")
            if zero_record["scores"] != zero_reference["scores"]:
                raise RuntimeError("E3 zero-history scores differ across arms")
            if (
                zero_record["static_support_sha256"]
                != zero_reference["static_support_sha256"]
            ):
                raise RuntimeError("E3 zero-history Static supports differ across arms")
        if zero_record["selected_expert"] != "static":
            raise RuntimeError("E3 zero-history endpoint is not exactly Static")

        tool_name = shared_executed_skill.rsplit("/", 1)[-1]
        session.observe(
            state_text_before=initial_state,
            skill_id=shared_executed_skill,
            action_text=f'{tool_name}({{"smoke": true}})',
            result_text='{ "status": "ok" }',
        )
        after = session.select(
            next_state,
            candidate_skill_ids=candidate_skill_ids,
            top_k=8,
            route_mode=route_mode,
        )
        after_record = _compact_selection(after)
        if support_after_reference:
            if after_record["static_support_sha256"] != support_after_reference:
                raise RuntimeError("E3 history-bearing Static supports differ across arms")
        else:
            support_after_reference = str(after_record["static_support_sha256"])
        if method == "static" and after_record["selected_expert"] != "static":
            raise RuntimeError("controlled Static selected a learned history expert")
        if method != "static" and after_record["selected_expert"] != "dynamic":
            raise RuntimeError("controlled learned arm did not use its history encoder")
        binding = selector.checkpoint_binding
        records[method] = {
            "zero_history": zero_record,
            "one_event_history": after_record,
            "foundation_checkpoint_sha256": binding["foundation_checkpoint_sha256"],
            "skills_sha256": binding["skills_sha256"],
            "e1_checkpoint_sha256": binding["e1_checkpoint_sha256"],
            "e1_seed": binding["e1_seed"],
            "e1_best_step": binding["e1_best_step"],
        }
        del session, selector
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "schema_version": "clstr_matched_history_e3_selector_smoke_v1",
        "status": "ok",
        "methods": records,
        "checks": {
            "real_locked_artifacts_loaded": True,
            "top8_every_arm": True,
            "zero_history_scores_bit_exact": True,
            "zero_history_support_equal": True,
            "history_bearing_support_equal": True,
            "one_factual_result_visible": True,
            "no_executor_or_user_model_called": True,
        },
        "cuda_device": torch.cuda.get_device_name(torch.device(args.device)),
        "torch_version": torch.__version__,
    }
    _write_json(Path(args.output_path).resolve(), report)
    return report


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
