from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clstr.stage4_quality_gate import audit_stage4_act_quality
from clstr.stabletoolbench_pass_rate import audit_stabletoolbench_pass_rate
from clstr.toolbench_g3_audit import audit_toolbench_g3_data
from clstr.toolret_eval_audit import audit_toolret_eval_data
from clstr.traject_eval_audit import audit_traject_eval_data


TRAJECT_REQUIRED_OFFICIAL_METRICS = ("EM", "Inclusion", "Usage", "Traj-Satisfy", "Acc")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid_json", "error": str(exc)}
    return payload if isinstance(payload, dict) else {"status": "invalid_json_type"}


def _file_ready(path: str | Path | None) -> bool:
    return bool(path) and Path(path).is_file() and Path(path).stat().st_size > 0


def _artifact_report(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "bytes": 0}
    path = Path(path)
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }


def _status_ok(path: str | Path | None) -> bool:
    payload = _read_json(path)
    return payload.get("status") == "ok"


def _stage4_output_dir_from_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _metric_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _traject_official_report(path: str | Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    artifact = _artifact_report(path)
    if not artifact["exists"]:
        return {
            **artifact,
            "status": "missing",
            "ready": False,
            "missing_metrics": list(TRAJECT_REQUIRED_OFFICIAL_METRICS),
        }
    if payload.get("status") == "invalid_json":
        return {
            **artifact,
            "status": "invalid_json",
            "ready": False,
            "missing_metrics": list(TRAJECT_REQUIRED_OFFICIAL_METRICS),
            "error": payload.get("error"),
        }
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    metric_keys = {_metric_key(str(key)) for key in metrics}
    missing = [
        name
        for name in TRAJECT_REQUIRED_OFFICIAL_METRICS
        if _metric_key(name) not in metric_keys
    ]
    status = str(payload.get("status") or "ok")
    ready = status == "ok" and not missing
    return {
        **artifact,
        "status": "ok" if ready else "action_required",
        "ready": ready,
        "required_metrics": list(TRAJECT_REQUIRED_OFFICIAL_METRICS),
        "missing_metrics": missing,
        "metric_scope": "Official TRAJECT EM/Inclusion/Usage/Traj-Satisfy/Acc, not selection-only proxy.",
    }


def _proxy_report(path: str | Path | None, *, expected_scope: str) -> dict[str, Any]:
    artifact = _artifact_report(path)
    payload = _read_json(path)
    ready = artifact["exists"] and str(payload.get("status") or "ok") == "ok"
    return {
        **artifact,
        "ready": ready,
        "status": "ok" if ready else "missing",
        "metric_scope": payload.get("metric_scope") or expected_scope,
        "may_satisfy_official_main_table": False,
    }


def _appworld_secondary_report(path: str | Path | None) -> dict[str, Any]:
    artifact = _artifact_report(path)
    payload = _read_json(path)
    ready = artifact["exists"] and str(payload.get("status") or "ok") == "ok"
    return {
        **artifact,
        "ready": ready,
        "status": "ok" if ready else "missing",
        "section": "§15.5.5 secondary combination plot",
        "not_main_table": True,
        "payload_summary": {
            key: payload.get(key)
            for key in ("token_cost_reduction", "task_success_drop", "task_success_not_regressed")
            if key in payload
        },
    }


def audit_eval_matrix_readiness(
    *,
    stage4_checkpoint: str | Path,
    stage4_output_dir: str | Path | None = None,
    min_stage4_steps: int = 2000,
    toolret_eval_dir: str | Path = "data/toolret_eval",
    toolret_run_path: str | Path | None = "outputs/toolret_eval/clstr_retrieval/run.tsv",
    traject_eval_dir: str | Path = "data/traject_eval_traject_split_test",
    traject_sequence_proxy_path: str | Path | None = "outputs/traject_eval_traject_split_test/traject_sequence_proxy_metrics.json",
    traject_official_metrics_path: str | Path | None = "outputs/traject_eval_traject_split_test/official_metrics.json",
    toolbench_g3_data_dir: str | Path = "data/toolbench_g3",
    toolbench_g3_source_root: str | Path | None = "../ToolBench/data",
    toolbench_g3_routing_report: str | Path | None = "outputs/toolbench_g3/clstr_routing_eval/metrics.json",
    stabletoolbench_root: str | Path = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench",
    converted_answer_path: str | Path = "outputs/toolbench_g3/stabletoolbench_converted",
    candidate_model: str = "clstr_toolbench_g3",
    test_set: str = "G3_instruction",
    api_pool_file: str | Path | None = None,
    appworld_combination_report: str | Path | None = "outputs/appworld_combination_plot/report.json",
    expected_toolbench_g3_answer_files: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    stage4_checkpoint_ready = _file_ready(stage4_checkpoint)
    stage4_quality = audit_stage4_act_quality(
        output_dir=stage4_output_dir or _stage4_output_dir_from_checkpoint(stage4_checkpoint),
        checkpoint_path=stage4_checkpoint,
        min_steps=min_stage4_steps,
    )
    stage4_quality_ready = stage4_quality.get("status") == "ok"
    toolret_data = audit_toolret_eval_data(data_dir=toolret_eval_dir)
    traject_data = audit_traject_eval_data(data_dir=traject_eval_dir)
    toolbench_data = audit_toolbench_g3_data(
        data_dir=toolbench_g3_data_dir,
        source_root=toolbench_g3_source_root,
        expected_answer_files=expected_toolbench_g3_answer_files,
    )
    stabletoolbench = audit_stabletoolbench_pass_rate(
        stabletoolbench_root=stabletoolbench_root,
        converted_answer_path=converted_answer_path,
        api_pool_file=api_pool_file,
        candidate_model=candidate_model,
        test_set=test_set,
    )

    toolret_run_ready = _file_ready(toolret_run_path)
    toolret_official_ready = toolret_data.get("status") == "ok" and toolret_run_ready

    traject_proxy = _proxy_report(
        traject_sequence_proxy_path,
        expected_scope="TRAJECT selection-only sequence proxy; not Usage/Traj-Satisfy/Acc.",
    )
    traject_official = _traject_official_report(traject_official_metrics_path)
    traject_official_ready = traject_data.get("status") == "ok" and bool(traject_official["ready"])

    toolbench_routing_proxy = _proxy_report(
        toolbench_g3_routing_report,
        expected_scope="ToolBench-G3 static routing/retrieval recall; not StableToolBench pass rate.",
    )
    toolbench_official_ready = toolbench_data.get("status") == "ok" and stabletoolbench.get("status") == "ok"

    appworld_secondary = _appworld_secondary_report(appworld_combination_report)

    blockers: list[str] = []
    if not stage4_checkpoint_ready:
        blockers.append("missing_stage4_checkpoint")
    if not stage4_quality_ready:
        blockers.append("stage4_quality_gate_not_ok")
    if toolret_data.get("status") != "ok":
        blockers.append("toolret_eval_data_not_ready")
    if not toolret_run_ready:
        blockers.append("toolret_run_missing")
    if traject_data.get("status") != "ok":
        blockers.append("traject_eval_data_not_ready")
    if not traject_official["ready"]:
        blockers.append("traject_official_outputs_missing")
    if toolbench_data.get("status") != "ok":
        blockers.append("toolbench_g3_data_not_ready")
    if stabletoolbench.get("status") != "ok":
        blockers.append("toolbench_g3_pass_rate_not_ready")
    if not appworld_secondary["ready"]:
        blockers.append("appworld_secondary_combination_report_missing")

    official_main_table_ready = (
        stage4_checkpoint_ready
        and stage4_quality_ready
        and toolret_official_ready
        and traject_official_ready
        and toolbench_official_ready
    )
    proxy_or_routing_eval_ready = any(
        [
            bool(traject_proxy["ready"]),
            bool(toolbench_routing_proxy["ready"]),
            toolret_data.get("status") == "ok",
        ]
    )
    appworld_secondary_ready = bool(appworld_secondary["ready"])
    phase_h_ready = official_main_table_ready and appworld_secondary_ready

    report = {
        "status": "ready" if phase_h_ready and not blockers else "action_required",
        "official_main_table_ready": official_main_table_ready,
        "proxy_or_routing_eval_ready": proxy_or_routing_eval_ready,
        "appworld_secondary_ready": appworld_secondary_ready,
        "phase_h_ready": phase_h_ready and not blockers,
        "blockers": blockers,
        "stage4_checkpoint": {
            **_artifact_report(stage4_checkpoint),
            "ready": stage4_checkpoint_ready,
            "required_for_phase_h": True,
        },
        "stage4_quality_gate": stage4_quality,
        "main_table": {
            "toolret": {
                "benchmark": "ToolRet",
                "official_ready": toolret_official_ready,
                "metric_scope": "Official ToolRet retrieval metrics: NDCG@5/10, Recall@5/10, MAP@10.",
                "data_audit": toolret_data,
                "run": {
                    **_artifact_report(toolret_run_path),
                    "ready": toolret_run_ready,
                },
            },
            "traject": {
                "benchmark": "TRAJECT-Bench",
                "official_ready": traject_official_ready,
                "proxy_ready": bool(traject_proxy["ready"]),
                "data_audit": traject_data,
                "sequence_proxy": traject_proxy,
                "official_metrics": traject_official,
            },
            "toolbench_g3": {
                "benchmark": "ToolBench-G3",
                "official_ready": toolbench_official_ready,
                "routing_proxy_ready": bool(toolbench_routing_proxy["ready"]),
                "data_audit": toolbench_data,
                "stabletoolbench_pass_rate": stabletoolbench,
                "routing_proxy": toolbench_routing_proxy,
            },
        },
        "secondary": {
            "appworld": appworld_secondary,
        },
        "metric_boundary": {
            "proxy_metrics_may_satisfy_official_main_table": False,
            "toolbench_g3_static_routing_is_not_pass_rate": True,
            "traject_sequence_proxy_is_not_usage_traj_satisfy_or_acc": True,
            "appworld_is_secondary_not_main_table": True,
        },
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
