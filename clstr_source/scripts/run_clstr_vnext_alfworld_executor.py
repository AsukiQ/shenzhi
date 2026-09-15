#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_eval import _make_env_config, run_alfworld_closed_loop_eval
from clstr.alfworld_qwen_clstr_gate import (
    QwenClstrAbstractGroundedActionScorer,
    QwenClstrHybridActionScorer,
    QwenClstrSkillPromptActionScorer,
)
from clstr.closed_loop_controller import ClosedLoopControllerConfig
from clstr.external_data import write_json
from clstr.qwen_direct_policy import (
    QwenDirectAdmissibleActionScorer,
    QwenDirectLikelihoodActionScorer,
    QwenDirectPolicyConfig,
)
from clstr.vnext_alfworld import VNextAlfworldAdmissibleActionScorer
from clstr.vnext_online_selector import VNextOnlineSelector
from clstr.vnext_online_selector import VNEXT_ROUTE_MODES


class _LoopGuardComponentScorer:
    def __init__(self) -> None:
        self.last_metadata: list[dict[str, Any]] = []

    def __call__(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
        policy_scores,
        action_histories: list[list[str]],
    ) -> dict[str, Any]:
        del state_texts, action_histories
        zeros = policy_scores.new_zeros(policy_scores.shape)
        self.last_metadata = [
            {
                "component_score_source": "loop_guard_only_zero_components",
                "candidate_count": len(candidates),
            }
            for candidates in candidate_rows
        ]
        return {
            "transition_scores": zeros,
            "belief_scores": zeros,
            "stop_logits": zeros,
        }


def _env_factory_for(official_repo: Path, train_eval_name: str):
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from alfworld.agents.environment import get_environment

    def env_factory(cfg, train_eval=None, **_):
        del train_eval
        return get_environment(cfg["env"]["type"])(
            cfg,
            train_eval=train_eval_name,
        )

    return env_factory


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _trace_summary(run_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_path)
    total_steps = 0
    recurrent_steps = 0
    result_correction_steps = 0
    qwen_hybrid_disagreements = 0
    guidance_steps = 0
    empty_guidance_steps = 0
    maximum_runtime_skill_count = 0
    for row in rows:
        for item in row.get("policy_metadata_trace", []) or []:
            if not isinstance(item, dict):
                continue
            total_steps += 1
            recurrent_steps += int(bool(item.get("uses_recurrent_m_t", False)))
            result_correction_steps += int(
                bool(item.get("clstr_post_action_result_correction", False))
            )
            qwen_action = str(item.get("qwen_chosen_action") or "")
            hybrid_action = str(item.get("hybrid_chosen_action") or "")
            qwen_hybrid_disagreements += int(
                bool(qwen_action and hybrid_action and qwen_action != hybrid_action)
            )
            if str(item.get("executor_interface") or "") in {
                "skill_prompt",
                "abstract_grounded",
                "guided_exact_prior",
                "guided_static_exact_prior",
                "legacy_guided_exact_prior",
            }:
                guidance_steps += int(bool(item.get("uses_clstr_skill_guidance", False)))
                empty_guidance_steps += int(
                    not bool(item.get("uses_clstr_skill_guidance", False))
                )
            maximum_runtime_skill_count = max(
                maximum_runtime_skill_count,
                int(item.get("runtime_appended_skill_count") or 0),
            )
    total_updates = sum(int(row.get("causal_update_count_final") or 0) for row in rows)
    return {
        "episode_count": len(rows),
        "total_steps": total_steps,
        "causal_update_count": total_updates,
        "uses_recurrent_m_t": bool(total_updates > 0),
        "recurrent_step_rate": round(recurrent_steps / max(1, total_steps), 6),
        "post_action_result_correction_step_rate": round(
            result_correction_steps / max(1, total_steps), 6
        ),
        "qwen_hybrid_disagreement_rate": round(
            qwen_hybrid_disagreements / max(1, total_steps), 6
        ),
        "skill_guidance_step_rate": round(guidance_steps / max(1, total_steps), 6),
        "empty_skill_guidance_step_rate": round(
            empty_guidance_steps / max(1, total_steps), 6
        ),
        "maximum_runtime_appended_skill_count": maximum_runtime_skill_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clean matched vNext CLSTR release with recurrent abstract "
            "routing and exact legal-action grounding for the Qwen3-14B ALFWorld "
            "executor."
        )
    )
    parser.add_argument("--matched_release_selection_path", required=True)
    parser.add_argument("--stage2_checkpoint_path")
    parser.add_argument("--training_skills_path")
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="valid_seen")
    parser.add_argument("--run_name", default="clstr_vnext_qwen14b_alfworld")
    parser.add_argument("--qwen_model_name_or_path", required=True)
    parser.add_argument("--cache_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--scoring_method", choices=("generate", "likelihood"), default="generate")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--likelihood_batch_size", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--qwen_weight", type=float, default=1.0)
    parser.add_argument("--clstr_weight", type=float, default=0.25)
    parser.add_argument(
        "--executor_interface",
        choices=(
            "abstract_grounded",
            "guided_exact_prior",
            "guided_static_exact_prior",
            "legacy_guided_exact_prior",
            "skill_prompt",
            "exact_prior",
        ),
        default="abstract_grounded",
        help=(
            "Ground checkpoint-known abstract CLSTR skills onto exact legal actions "
            "(default), combine the same grounding prior with abstract guidance, "
            "use prompt-only guidance, or retain the runtime pseudo-skill "
            "exact-prior ablation."
        ),
    )
    parser.add_argument("--skill_guidance_top_k", type=int, default=2)
    parser.add_argument(
        "--clstr_route_mode",
        choices=tuple(sorted(VNEXT_ROUTE_MODES)),
        default="adaptive",
        help="Use the learned adaptive expert selector or force a static/dynamic CLSTR ablation.",
    )
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument("--loop_guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selector = VNextOnlineSelector(
        matched_release_selection_path=args.matched_release_selection_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        training_skills_path=args.training_skills_path,
        device=args.device,
        coarse_k=args.coarse_k,
        dynamic_extra_k=args.dynamic_extra_k,
    )
    clstr_scorer = VNextAlfworldAdmissibleActionScorer(
        selector,
        route_mode=str(args.clstr_route_mode),
        allow_runtime_action_skills=(
            str(args.executor_interface)
            not in {
                "abstract_grounded",
                "guided_exact_prior",
                "guided_static_exact_prior",
            }
        ),
        grounding_score_mode=(
            "memory_skill_head"
            if str(args.executor_interface)
            in {
                "abstract_grounded",
                "guided_exact_prior",
                "guided_static_exact_prior",
            }
            and str(args.scoring_method) == "generate"
            else "route_query"
        ),
        grounding_expert_mode=(
            "static"
            if str(args.executor_interface) == "guided_static_exact_prior"
            else "selected"
        ),
        memory_update_skill_mode=(
            "exact_action"
            if str(args.executor_interface) in {
                "exact_prior",
                "legacy_guided_exact_prior",
            }
            else "mapped_abstract"
        ),
    )
    scorer_class = (
        QwenDirectLikelihoodActionScorer
        if args.scoring_method == "likelihood"
        else QwenDirectAdmissibleActionScorer
    )
    qwen_scorer = scorer_class(
        QwenDirectPolicyConfig(
            model_name_or_path=str(args.qwen_model_name_or_path),
            cache_dir=args.cache_dir,
            local_files_only=bool(args.local_files_only),
            torch_dtype=str(args.torch_dtype),
            max_new_tokens=int(args.max_new_tokens),
            likelihood_batch_size=int(args.likelihood_batch_size),
            device=str(args.device),
            use_chat_template=True,
            enable_thinking=False,
        )
    )
    if str(args.executor_interface) in {
        "abstract_grounded",
        "guided_exact_prior",
        "guided_static_exact_prior",
    }:
        guided_interface = str(args.executor_interface) in {
            "guided_exact_prior",
            "guided_static_exact_prior",
        }
        abstract_qwen_score_mode = (
            "proposal_bonus"
            if str(args.scoring_method) == "generate"
            else "dense_likelihood"
        )
        policy_scorer = QwenClstrAbstractGroundedActionScorer(
            qwen_scorer=qwen_scorer,
            clstr_retriever=clstr_scorer,
            guidance_top_k=int(args.skill_guidance_top_k),
            qwen_weight=float(args.qwen_weight),
            clstr_weight=float(args.clstr_weight),
            qwen_score_mode=abstract_qwen_score_mode,
            inject_skill_guidance=(
                guided_interface
                or str(args.scoring_method) == "likelihood"
            ),
            executor_interface=str(args.executor_interface),
            policy_family=(
                f"qwen_clstr_{args.executor_interface}_executor"
                if guided_interface
                else "qwen_clstr_abstract_grounded_executor"
            ),
        )
        policy_family = (
            f"qwen3_14b_plus_clstr_vnext_{args.executor_interface}"
            if guided_interface
            else "qwen3_14b_plus_clstr_vnext_abstract_grounded"
        )
    elif str(args.executor_interface) == "skill_prompt":
        policy_scorer = QwenClstrSkillPromptActionScorer(
            qwen_scorer=qwen_scorer,
            clstr_retriever=clstr_scorer,
            guidance_top_k=int(args.skill_guidance_top_k),
        )
        policy_family = "qwen3_14b_plus_clstr_vnext_abstract_skill_guidance"
    else:
        legacy_guided = (
            str(args.executor_interface) == "legacy_guided_exact_prior"
        )
        hybrid_policy_family = (
            "qwen3_14b_plus_clstr_vnext_legacy_guided_exact_prior"
            if legacy_guided
            else "qwen3_14b_plus_clstr_vnext_recurrent_exact_prior"
        )
        policy_scorer = QwenClstrHybridActionScorer(
            qwen_scorer=qwen_scorer,
            clstr_scorer=clstr_scorer,
            qwen_weight=float(args.qwen_weight),
            clstr_weight=float(args.clstr_weight),
            normalize_scores=True,
            qwen_score_mode=(
                "proposal_bonus" if args.scoring_method == "generate" else "raw"
            ),
            prior_name="clstr",
            policy_family=hybrid_policy_family,
            executor_interface=str(args.executor_interface),
            inject_skill_guidance=legacy_guided,
            guidance_top_k=int(args.skill_guidance_top_k),
        )
        policy_family = hybrid_policy_family
    component_scorer = _LoopGuardComponentScorer() if args.loop_guard else None
    controller_config = (
        ClosedLoopControllerConfig(
            mode="policy_plus_transition_belief_stop_loop_penalty",
            transition_weight=0.0,
            belief_weight=0.0,
            stop_weight=0.0,
            planner_weight=0.0,
            q_success_weight=0.0,
        )
        if args.loop_guard
        else None
    )
    config, train_eval_name = _make_env_config(
        Path(args.official_repo),
        Path(args.data_dir),
        str(args.split),
    )
    result = run_alfworld_closed_loop_eval(
        env_factory=_env_factory_for(Path(args.official_repo), train_eval_name),
        config=config,
        split=str(args.split),
        candidate_scorer=policy_scorer,
        output_dir=output_dir,
        max_episodes=int(args.max_episodes),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        run_name=str(args.run_name),
        component_scorer=component_scorer,
        controller_config=controller_config,
    )
    metrics_path = output_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    trace = _trace_summary(output_dir / "run.jsonl")
    checkpoint_report = selector.checkpoint_binding["checkpoint"]
    metrics.update(
        {
            "policy_family": policy_family,
            "method_release": selector.method,
            "matched_release_selection_path": str(
                Path(args.matched_release_selection_path).resolve()
            ),
            "stage2_checkpoint_path": checkpoint_report["checkpoint_path"],
            "stage2_checkpoint_sha256": checkpoint_report["checkpoint_sha256"],
            "stage2_checkpoint_step": checkpoint_report["checkpoint_step"],
            "training_skills_path": checkpoint_report["training_skills_path"],
            "training_skills_sha256": checkpoint_report["training_skills_sha256"],
            "executor_model": str(Path(args.qwen_model_name_or_path).resolve()),
            "executor_is_qwen3_14b": True,
            "executor_interface": str(args.executor_interface),
            "scoring_method": str(args.scoring_method),
            "grounding_score_mode": (
                "memory_skill_head"
                if str(args.executor_interface)
                in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                }
                and str(args.scoring_method) == "generate"
                else (
                    "route_query"
                    if str(args.executor_interface)
                    in {
                        "abstract_grounded",
                        "guided_exact_prior",
                        "guided_static_exact_prior",
                    }
                    else None
                )
            ),
            "qwen_weight": (
                float(args.qwen_weight)
                if str(args.executor_interface) in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                    "legacy_guided_exact_prior",
                    "exact_prior",
                }
                else None
            ),
            "clstr_weight": (
                float(args.clstr_weight)
                if str(args.executor_interface) in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                    "legacy_guided_exact_prior",
                    "exact_prior",
                }
                else None
            ),
            "skill_guidance_top_k": (
                int(args.skill_guidance_top_k)
                if str(args.executor_interface) in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                    "legacy_guided_exact_prior",
                    "skill_prompt",
                }
                else None
            ),
            "clstr_route_mode": str(args.clstr_route_mode),
            "qwen_score_mode": (
                (
                    "proposal_bonus"
                    if str(args.scoring_method) == "generate"
                    else "dense_likelihood"
                )
                if str(args.executor_interface)
                in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                }
                else (
                    "proposal_bonus"
                    if args.scoring_method == "generate"
                    else "raw"
                )
                if str(args.executor_interface) in {
                    "exact_prior",
                    "legacy_guided_exact_prior",
                }
                else "direct_qwen_scores"
            ),
            "score_normalization": (
                (
                    "qwen:proposal_bonus;clstr:memory_skill_head_legal_action_row_zscore"
                    if str(args.scoring_method) == "generate"
                    else "qwen:row_zscore;clstr:route_query_legal_action_row_zscore"
                )
                if str(args.executor_interface)
                in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                }
                else (
                    "qwen:proposal_bonus;clstr:row_zscore"
                    if str(args.executor_interface) in {
                        "exact_prior",
                        "legacy_guided_exact_prior",
                    }
                    else "none_no_score_fusion"
                )
            ),
            "uses_clstr_score_fusion": str(args.executor_interface) in {
                "abstract_grounded",
                "guided_exact_prior",
                "guided_static_exact_prior",
                "legacy_guided_exact_prior",
                "exact_prior",
            },
            "uses_clstr_skill_guidance": str(args.executor_interface) in {
                "guided_exact_prior",
                "guided_static_exact_prior",
                "legacy_guided_exact_prior",
                "skill_prompt",
            } or (
                str(args.executor_interface) == "abstract_grounded"
                and str(args.scoring_method) == "likelihood"
            ),
            "guidance_conditions_qwen": str(args.executor_interface) in {
                "guided_exact_prior",
                "guided_static_exact_prior",
                "legacy_guided_exact_prior",
                "skill_prompt",
            } or (
                str(args.executor_interface) == "abstract_grounded"
                and str(args.scoring_method) == "likelihood"
            ),
            "loop_guard": bool(args.loop_guard),
            "candidate_skill_schema": (
                "alfworld_mapped_abstract_skill_v1"
                if str(args.executor_interface) in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                    "skill_prompt",
                }
                else "alfworld_exact_admissible_action_v1"
            ),
            "guidance_candidate_skill_schema": (
                "alfworld_mapped_abstract_skill_v1"
                if str(args.executor_interface) in {
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                    "legacy_guided_exact_prior",
                    "skill_prompt",
                }
                else None
            ),
            "grounding_candidate_schema": (
                "alfworld_exact_legal_action_v1"
                if str(args.executor_interface)
                in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                }
                else None
            ),
            "runtime_pseudo_skills_allowed": (
                str(args.executor_interface)
                not in {
                    "abstract_grounded",
                    "guided_exact_prior",
                    "guided_static_exact_prior",
                }
            ),
            "grounding_expert_mode": (
                "static"
                if str(args.executor_interface) == "guided_static_exact_prior"
                else (
                    "selected"
                    if str(args.executor_interface)
                    in {"abstract_grounded", "guided_exact_prior"}
                    else None
                )
            ),
            "history_removed_from_clstr_current_state": True,
            "memory_protocol": "native_factual_stage2_recurrent_v1",
            "memory_update_skill_mode": (
                "exact_action"
                if str(args.executor_interface) in {
                    "exact_prior",
                    "legacy_guided_exact_prior",
                }
                else "mapped_abstract"
            ),
            "post_action_result_correction": True,
            "runtime_appended_skill_count": int(
                selector.checkpoint_binding.get("runtime_appended_skill_count", 0)
            ),
            "runtime_appended_skill_ids": list(
                selector.checkpoint_binding.get("runtime_appended_skill_ids", [])
            ),
            "trace_summary": trace,
        }
    )
    write_json(metrics_path, metrics)
    report = {
        "status": "ok",
        "metrics": metrics,
        "output_dir": str(output_dir),
        "trace_summary": trace,
        "result": result,
    }
    write_json(output_dir / "clstr_vnext_alfworld_report.json", report)
    return report


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
