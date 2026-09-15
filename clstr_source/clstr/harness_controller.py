from __future__ import annotations

from pathlib import Path
from typing import Any

from clstr.alfworld_eval import (
    _load_clstr_alfworld_model,
    make_candidate_scorer,
    make_controller_component_scorer,
)
from clstr.closed_loop_controller import ClosedLoopControllerConfig, ClstrTextActionController


def build_clstr_text_action_controller(
    *,
    routing_init_manifest: str | Path,
    checkpoint_path: str | Path | None = None,
    stage4_checkpoint_path: str | Path | None = None,
    skill_rows_path_override: str | Path | None = None,
    aux_data_root: str | Path = "data/clstr_full_base_train",
    output_dir: str | Path = "outputs/clstr_controller_model_cache",
    controller_mode: str = "policy_plus_transition_belief_stop_loop_penalty",
) -> tuple[ClstrTextActionController, dict[str, Any]]:
    model, action_adapter, routing_report = _load_clstr_alfworld_model(
        routing_init_manifest=Path(routing_init_manifest),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        stage4_checkpoint_path=Path(stage4_checkpoint_path) if stage4_checkpoint_path else None,
        skill_rows_path_override=Path(skill_rows_path_override) if skill_rows_path_override else None,
        output_dir=Path(output_dir),
        data_root=Path(aux_data_root),
    )
    candidate_scorer = make_candidate_scorer(model, action_adapter)
    component_scorer = None
    if controller_mode != "policy_only":
        component_scorer = make_controller_component_scorer(model)
    controller = ClstrTextActionController(
        candidate_scorer=candidate_scorer,
        component_scorer=component_scorer,
        config=ClosedLoopControllerConfig(mode=controller_mode),
    )
    report = {
        "controller_class": "ClstrTextActionController",
        "controller_mode": controller_mode,
        "routing_init_manifest": str(routing_init_manifest),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "stage4_checkpoint_path": str(stage4_checkpoint_path) if stage4_checkpoint_path else None,
        "skill_rows_path_override": str(skill_rows_path_override) if skill_rows_path_override else None,
        "stage4_overlay_loaded": bool(routing_report.get("stage4_checkpoint")),
        "aux_data_root": str(aux_data_root),
        "model_cache_dir": str(output_dir),
        "component_scorer_enabled": component_scorer is not None,
        "routing_report": routing_report,
    }
    return controller, report
