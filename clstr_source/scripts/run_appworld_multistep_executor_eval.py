#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_clstr_eval import _load_checkpoint_model
from clstr.appworld_dynamic_routing_audit import _extract_appended_appworld_rows, _read_jsonl as _read_dynamic_jsonl
from clstr.appworld_executor import (
    NullSkillProvider,
    QwenCodeGenerator,
    QwenCodeGeneratorConfig,
    VALID_SKILL_CONTEXT_MODES,
)
from clstr.appworld_multistep import (
    CLSTRMultiStepController,
    HybridCLSTRSkillRouterController,
    LiveEmbeddingStepController,
    StaticSkillProviderStepController,
    run_appworld_multistep_executor_eval,
)
from clstr.appworld_routing import read_jsonl
from clstr.appworld_skillrouter_base import (
    load_skillrouter_base_checkpoint,
    skillrouter_base_query_text,
    skillrouter_base_skill_text,
)
from clstr.device_utils import resolve_device
from clstr.full_base_train import TRANSITION_SCORING_MODES, V4_1B_TRANSITION_SCORING_MODE
from clstr.skillret_official import _last_token_pool
from clstr.stage_checkpoint_init import (
    build_clstr_model_from_stage0_checkpoint,
    checkpoint_payload,
    checkpoint_state,
    load_compatible_state_dict,
    load_routing_and_head_checkpoints,
)
from scripts.run_appworld_qwen_executor_eval import DEFAULT_PREDICTIONS, _build_skill_provider


SKILL_CONTEXT_MODE_CHOICES = sorted(VALID_SKILL_CONTEXT_MODES)


def _resolve_auto_ranking_mode(method: str, requested_mode: str | None, load_report: dict | None) -> str:
    mode = str(requested_mode or "auto").lower()
    if mode != "auto":
        return mode
    if method == "clstr_mt_fusion":
        return "policy_head"

    report = load_report or {}
    stage = str(report.get("stage") or "").lower()
    if stage in {"appworld_current_route_preference", "appworld_current_route_online_stage4"}:
        return "policy_transition_blend"
    if "stage4" in stage or "transition_conditioned_next_skill" in stage:
        return "transition_blend"
    train_config = report.get("train_config") or {}
    if isinstance(train_config, dict):
        if bool(train_config.get("multi_step")):
            return "policy_blend"

    metrics = report.get("metrics") or {}
    if isinstance(metrics, dict):
        if "policy_loss" in metrics:
            return "policy_blend"

    return "skill_table"


def _load_dynamic_checkpoint_model(
    *,
    checkpoint_path: str,
    base_skill_pool_path: str,
    dynamic_skill_pool_path: str,
) -> tuple[object, dict]:
    base_rows = _read_dynamic_jsonl(base_skill_pool_path)
    dynamic_rows = _read_dynamic_jsonl(dynamic_skill_pool_path)
    appended_rows, extraction_report = _extract_appended_appworld_rows(
        base_rows=base_rows,
        dynamic_rows=dynamic_rows,
    )
    checkpoint_path_obj = Path(checkpoint_path)
    payload = checkpoint_payload(checkpoint_path_obj, "dynamic CLSTR")
    train_report = payload.get("train_report") if isinstance(payload, dict) else {}
    checkpoint_init_report = train_report.get("checkpoint_init_report") if isinstance(train_report, dict) else {}
    stage = str(payload.get("stage") or "")
    current_route_full_dynamic_stages = {
        "appworld_current_route_preference": "current_route_preference_full_dynamic_checkpoint",
        "appworld_current_route_online_stage4": "current_route_online_stage4_full_dynamic_checkpoint",
    }
    if stage in current_route_full_dynamic_stages:
        model, model_config, checkpoint_report = build_clstr_model_from_stage0_checkpoint(
            checkpoint_path=checkpoint_path_obj,
            skills_path=Path(dynamic_skill_pool_path),
        )
        device = resolve_device()
        if callable(getattr(model, "to", None)):
            model.to(device)
        if callable(getattr(model, "eval", None)):
            model.eval()
        return model, {
            "checkpoint_path": str(checkpoint_path),
            "base_skill_pool_path": str(base_skill_pool_path),
            "dynamic_skill_pool_path": str(dynamic_skill_pool_path),
            "device": str(device),
            "dynamic_checkpoint_prefix_load": False,
            "dynamic_checkpoint_load_mode": current_route_full_dynamic_stages[stage],
            "stage": stage,
            "model_config": model_config,
            "checkpoint_report": checkpoint_report,
            "extraction_report": extraction_report,
            "append_report": {"appended_count": 0, "appended_skill_ids": []},
        }
    is_stage4_delta = bool(payload.get("checkpoint_excludes_frozen_routing_foundation")) and (
        "stage4" in stage.lower() or "act" in stage.lower()
    )
    routing_checkpoint_path = None
    head_checkpoint_path = None
    if is_stage4_delta and isinstance(checkpoint_init_report, dict):
        routing_checkpoint_path = checkpoint_init_report.get("routing_checkpoint_path")
        head_checkpoint_path = checkpoint_init_report.get("head_checkpoint_path")

    if routing_checkpoint_path and head_checkpoint_path:
        routing_path = Path(str(routing_checkpoint_path))
        head_path = Path(str(head_checkpoint_path))
        model, model_config, routing_build_report = build_clstr_model_from_stage0_checkpoint(
            checkpoint_path=routing_path,
            skills_path=Path(base_skill_pool_path),
        )
        restored_init_report = load_routing_and_head_checkpoints(
            model,
            routing_checkpoint_path=routing_path,
            head_checkpoint_path=head_path,
            partial_load_mode="stage4_recorded_routing_plus_head_init",
            protect_routing_foundation=True,
        )
        stage4_delta_report = load_compatible_state_dict(
            model,
            checkpoint_state(payload, "stage4 delta"),
            partial_load_mode="stage4_delta_after_recorded_routing_head_init",
        )
        checkpoint_report = {
            "dynamic_checkpoint_load_mode": "stage4_delta_with_recorded_routing_head",
            "stage4_checkpoint_path": str(checkpoint_path_obj),
            "stage4_checkpoint_stage": payload.get("stage"),
            "routing_build_report": routing_build_report,
            "restored_init_report": restored_init_report,
            "stage4_delta_load_report": stage4_delta_report,
            "stage0_checkpoint_stage": routing_build_report.get("stage0_checkpoint_stage"),
        }
        dynamic_checkpoint_load_mode = "stage4_delta_with_recorded_routing_head"
    else:
        model, model_config, checkpoint_report = build_clstr_model_from_stage0_checkpoint(
            checkpoint_path=checkpoint_path_obj,
            skills_path=Path(base_skill_pool_path),
        )
        dynamic_checkpoint_load_mode = "single_checkpoint_prefix_load"
    device = resolve_device()
    if callable(getattr(model, "to", None)):
        model.to(device)
    append_report = model.append_skills(appended_rows)
    if callable(getattr(model, "eval", None)):
        model.eval()
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "base_skill_pool_path": str(base_skill_pool_path),
        "dynamic_skill_pool_path": str(dynamic_skill_pool_path),
        "device": str(device),
        "dynamic_checkpoint_prefix_load": True,
        "dynamic_checkpoint_load_mode": dynamic_checkpoint_load_mode,
        "stage": payload.get("stage") or checkpoint_report.get("stage0_checkpoint_stage"),
        "model_config": model_config,
        "checkpoint_report": checkpoint_report,
        "extraction_report": extraction_report,
        "append_report": append_report,
    }


class _HFTextEncoder:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        batch_size: int,
        max_length: int,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(torch, str(torch_dtype), torch.bfloat16)
        if self.device.type == "cpu":
            dtype = torch.float32
        kwargs = {"trust_remote_code": True, "local_files_only": bool(local_files_only)}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype, **kwargs)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device).eval()

    def __call__(self, texts: list[str]) -> torch.Tensor:
        encoded: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            tok = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {key: value.to(self.device) for key, value in tok.items()}
            with torch.no_grad():
                out = self.model(**tok)
                pooled = _last_token_pool(out.last_hidden_state, tok["attention_mask"])
                encoded.append(F.normalize(pooled.float(), p=2, dim=-1).cpu())
        return torch.cat(encoded, dim=0)


def _build_controller(
    *,
    method: str,
    skill_pool_path: str,
    base_skill_pool_path: str | None = None,
    predictions_path: str | None,
    clstr_model_config_path: str | None = None,
    clstr_checkpoint_path: str | None = None,
    ranking_mode: str = "auto",
    candidate_top_k: int | None = None,
    candidate_source: str = "routing",
    policy_blend_alpha: float = 0.5,
    allow_legacy_policy_skill_router: bool = False,
    transition_scoring_mode: str = V4_1B_TRANSITION_SCORING_MODE,
    transition_residual_lambda: float = 0.0,
    clstr_alpha: float = 0.25,
    recurrent_belief: bool = True,
    skillrouter_model_name_or_path: str = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    skillrouter_checkpoint_path: str = "outputs/appworld_skillrouter_base_train_v1/model.pt",
    skillrouter_batch_size: int = 8,
    skillrouter_max_length: int = 1024,
    skillrouter_device: str | None = None,
    skillrouter_torch_dtype: str = "bfloat16",
    skillrouter_local_files_only: bool = True,
    dedupe_canonical_skills: bool = False,
    max_auth_like_skills: int | None = None,
    appworld_executor_compatible_only: bool = False,
) -> StaticSkillProviderStepController | CLSTRMultiStepController:
    if method == "qwen_only":
        return StaticSkillProviderStepController(NullSkillProvider())
    if method in {"clstr_multistep", "clstr_mt_fusion"}:
        if not clstr_checkpoint_path:
            raise ValueError(f"--clstr_checkpoint_path is required for method {method!r}")
        if base_skill_pool_path:
            model, load_report = _load_dynamic_checkpoint_model(
                checkpoint_path=clstr_checkpoint_path,
                base_skill_pool_path=base_skill_pool_path,
                dynamic_skill_pool_path=skill_pool_path,
            )
            skills = list(getattr(model, "skills", None) or _read_dynamic_jsonl(skill_pool_path))
        else:
            model_config_path = clstr_model_config_path
            if not model_config_path:
                model_config_path = (
                    "configs/model/appworld_skillrouter_init_mt_fusion.yaml"
                    if method == "clstr_mt_fusion"
                    else "configs/model/appworld_skillrouter_init.yaml"
                )
            model, load_report = _load_checkpoint_model(
                model_config_path=model_config_path,
                skill_pool_path=skill_pool_path,
                checkpoint_path=clstr_checkpoint_path,
            )
            skills = read_jsonl(skill_pool_path)
        resolved_ranking_mode = _resolve_auto_ranking_mode(method, ranking_mode, load_report)
        return CLSTRMultiStepController(
            model=model,
            skills=skills,
            ranking_mode=resolved_ranking_mode,
            candidate_top_k=candidate_top_k,
            candidate_source=candidate_source,
            policy_blend_alpha=policy_blend_alpha,
            allow_legacy_policy_skill_router=allow_legacy_policy_skill_router,
            transition_scoring_mode=transition_scoring_mode,
            transition_residual_lambda=transition_residual_lambda,
            recurrent_belief=bool(recurrent_belief),
            dedupe_canonical_skills=dedupe_canonical_skills,
            max_auth_like_skills=max_auth_like_skills,
            appworld_executor_compatible_only=appworld_executor_compatible_only,
        )
    if method == "clstr_skillrouter_hybrid_live":
        if not clstr_checkpoint_path:
            raise ValueError(f"--clstr_checkpoint_path is required for method {method!r}")
        model, _load_report = _load_checkpoint_model(
            model_config_path=clstr_model_config_path or "configs/model/appworld_skillrouter_init.yaml",
            skill_pool_path=skill_pool_path,
            checkpoint_path=clstr_checkpoint_path,
        )
        encoder = _HFTextEncoder(
            model_name_or_path=skillrouter_model_name_or_path,
            batch_size=skillrouter_batch_size,
            max_length=skillrouter_max_length,
            device=skillrouter_device,
            torch_dtype=skillrouter_torch_dtype,
            local_files_only=skillrouter_local_files_only,
        )
        scorer, _metadata = load_skillrouter_base_checkpoint(skillrouter_checkpoint_path)
        retriever = LiveEmbeddingStepController(
            skills=read_jsonl(skill_pool_path),
            encode_texts=encoder,
            query_text_fn=lambda state_text: skillrouter_base_query_text({"query": state_text}),
            skill_text_fn=skillrouter_base_skill_text,
            scorer=scorer,
            controller_name="skillrouter_base_live_prior",
        )
        return HybridCLSTRSkillRouterController(
            model=model,
            skills=read_jsonl(skill_pool_path),
            retriever=retriever,
            clstr_alpha=clstr_alpha,
            candidate_top_k=candidate_top_k,
            recurrent_belief=bool(recurrent_belief),
            dedupe_canonical_skills=dedupe_canonical_skills,
            max_auth_like_skills=max_auth_like_skills,
            appworld_executor_compatible_only=appworld_executor_compatible_only,
        )
    if method in {"skillrouter_embedding_live", "skillrouter_base_live"}:
        encoder = _HFTextEncoder(
            model_name_or_path=skillrouter_model_name_or_path,
            batch_size=skillrouter_batch_size,
            max_length=skillrouter_max_length,
            device=skillrouter_device,
            torch_dtype=skillrouter_torch_dtype,
            local_files_only=skillrouter_local_files_only,
        )
        scorer = None
        controller_name = "skillrouter_embedding_live"
        if method == "skillrouter_base_live":
            scorer, _metadata = load_skillrouter_base_checkpoint(skillrouter_checkpoint_path)
            controller_name = "skillrouter_base_live"
        return LiveEmbeddingStepController(
            skills=read_jsonl(skill_pool_path),
            encode_texts=encoder,
            query_text_fn=lambda state_text: skillrouter_base_query_text({"query": state_text}),
            skill_text_fn=skillrouter_base_skill_text,
            scorer=scorer,
            controller_name=controller_name,
            dedupe_canonical_skills=dedupe_canonical_skills,
            max_auth_like_skills=max_auth_like_skills,
            appworld_executor_compatible_only=appworld_executor_compatible_only,
        )
    provider = _build_skill_provider(
        method,
        skill_pool_path,
        predictions_path,
        dedupe_canonical_skills=dedupe_canonical_skills,
        max_auth_like_skills=max_auth_like_skills,
    )
    return StaticSkillProviderStepController(provider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-step Qwen3 AppWorld executor with optional per-step skill context.")
    parser.add_argument(
        "--method",
        choices=[
            "qwen_only",
            "skillrouter_embedding",
            "skillrouter_base",
            "skillrouter_embedding_live",
            "skillrouter_base_live",
            "clstr_base",
            "clstr_skillrouter_init",
            "clstr_act",
            "clstr_multistep",
            "clstr_mt_fusion",
            "clstr_skillrouter_hybrid_live",
        ],
        default="qwen_only",
    )
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument(
        "--base_skill_pool_path",
        default=None,
        help=(
            "Optional checkpoint-prefix skill pool. When set for CLSTR methods, "
            "the checkpoint is loaded on this base pool and AppWorld skills from "
            "--skill_pool_path are appended dynamically."
        ),
    )
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--clstr_model_config", default=None)
    parser.add_argument("--clstr_checkpoint_path", default=None)
    parser.add_argument(
        "--ranking_mode",
        choices=[
            "auto",
            "skill_table",
            "policy_head",
            "policy_blend",
            "transition_head",
            "transition_blend",
            "policy_transition_blend",
        ],
        default="auto",
    )
    parser.add_argument("--candidate_top_k", type=int, default=None)
    parser.add_argument(
        "--candidate_source",
        choices=["routing", "routing_belief_union", "routing_belief_union_after_update"],
        default="routing",
    )
    parser.add_argument("--policy_blend_alpha", type=float, default=0.5)
    parser.add_argument("--allow_legacy_policy_skill_router", action="store_true")
    parser.add_argument(
        "--transition_scoring_mode",
        choices=sorted(TRANSITION_SCORING_MODES),
        default=V4_1B_TRANSITION_SCORING_MODE,
    )
    parser.add_argument("--transition_residual_lambda", type=float, default=0.0)
    parser.add_argument("--clstr_alpha", type=float, default=0.25)
    parser.add_argument("--recurrent_belief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skillrouter_model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--skillrouter_checkpoint_path", default="outputs/appworld_skillrouter_base_train_v1/model.pt")
    parser.add_argument("--skillrouter_batch_size", type=int, default=8)
    parser.add_argument("--skillrouter_max_length", type=int, default=1024)
    parser.add_argument("--skillrouter_device", default=None)
    parser.add_argument("--skillrouter_torch_dtype", default="bfloat16")
    parser.add_argument("--skillrouter_local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--appworld_cache", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-8B")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--dedupe_canonical_skills", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_auth_like_skills", type=int, default=None)
    parser.add_argument("--appworld_executor_compatible_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skill_context_mode", choices=SKILL_CONTEXT_MODE_CHOICES, default="safe_metadata")
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--max_apis_per_app", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout_seconds", type=int, default=60)
    parser.add_argument("--max_interactions", type=int, default=10)
    parser.add_argument("--use_stop_head", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_dir = args.output_dir or f"outputs/appworld_multistep_executor_smoke/{args.method}"
    controller = _build_controller(
        method=args.method,
        skill_pool_path=args.skill_pool_path,
        base_skill_pool_path=args.base_skill_pool_path,
        predictions_path=args.predictions_path or DEFAULT_PREDICTIONS.get(args.method),
        clstr_model_config_path=args.clstr_model_config,
        clstr_checkpoint_path=args.clstr_checkpoint_path,
        ranking_mode=args.ranking_mode,
        candidate_top_k=args.candidate_top_k,
        candidate_source=args.candidate_source,
        policy_blend_alpha=args.policy_blend_alpha,
        allow_legacy_policy_skill_router=bool(args.allow_legacy_policy_skill_router),
        transition_scoring_mode=args.transition_scoring_mode,
        transition_residual_lambda=args.transition_residual_lambda,
        clstr_alpha=args.clstr_alpha,
        recurrent_belief=bool(args.recurrent_belief),
        skillrouter_model_name_or_path=args.skillrouter_model_name_or_path,
        skillrouter_checkpoint_path=args.skillrouter_checkpoint_path,
        skillrouter_batch_size=args.skillrouter_batch_size,
        skillrouter_max_length=args.skillrouter_max_length,
        skillrouter_device=args.skillrouter_device,
        skillrouter_torch_dtype=args.skillrouter_torch_dtype,
        skillrouter_local_files_only=bool(args.skillrouter_local_files_only),
        dedupe_canonical_skills=bool(args.dedupe_canonical_skills),
        max_auth_like_skills=args.max_auth_like_skills,
        appworld_executor_compatible_only=bool(args.appworld_executor_compatible_only),
    )
    generator = QwenCodeGenerator(
        QwenCodeGeneratorConfig(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            local_files_only=bool(args.local_files_only),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            enable_thinking=bool(args.enable_thinking),
        )
    )
    report = run_appworld_multistep_executor_eval(
        tasks_path=args.tasks_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=output_dir,
        appworld_root=args.appworld_root,
        appworld_cache=args.appworld_cache,
        method=args.method,
        controller=controller,
        generator=generator,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        top_k=args.top_k,
        max_apis_per_app=args.max_apis_per_app,
        skill_context_mode=args.skill_context_mode,
        timeout_seconds=args.timeout_seconds,
        max_interactions=args.max_interactions,
        use_stop_head=bool(args.use_stop_head),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
