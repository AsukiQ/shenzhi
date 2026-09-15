from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Iterable

import torch

from clstr.history_channel import serialize_compact_causal_state
from clstr.matched_history_encoders import build_matched_history_encoder
from clstr.matched_history_train import (
    MATCHED_HISTORY_TRAIN_SCHEMA,
    MatchedHistoryRouteScorer,
)
from clstr.vnext_candidates import masked_topk_tensor
from clstr.vnext_eval import load_vnext_stage2_for_evaluation
from clstr.vnext_training import file_sha256


CONTROLLED_HISTORY_METHODS = frozenset({"static", "transformer", "lstr"})
CONTROLLED_HISTORY_ONLINE_SCHEMA = "clstr_matched_history_e3_online_v1"
E3_FOUNDATION_SHA256 = "b35510cfea43addc1e962b9643d7000ee2504a20ae9b1ac836a15888c9c6a603"
E3_SKILLS_SHA256 = "3ec6d0d6d9f79775897ed4c7fd62e44d2a23859461e265e51d816d1789eb430d"
E3_SELECTED_CHECKPOINTS = {
    "transformer": {
        "sha256": "d80497d69c6bfade883d5e73e311a6e20c6037e7cb9d829282526c0f7721d759",
        "seed": 23,
        "step": 400,
    },
    "lstr": {
        "sha256": "7863548ce0a2ee740e606df9f977c98df3dc1718e16260a6cb83299c532eb481",
        "seed": 31,
        "step": 500,
    },
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "").strip()


def _deduplicate(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def canonical_tau2_action_text(skill_id: str, action_text: str) -> str:
    """Convert the online Tau2 call string to the exact E1 action channel."""

    resolved_skill_id = str(skill_id or "").strip()
    tool_name = resolved_skill_id.rsplit("/", 1)[-1]
    raw = str(action_text or "").strip()
    if not resolved_skill_id or not tool_name or not raw:
        raise ValueError("controlled Tau2 action requires a skill ID and action text")
    canonical_prefix = f"tool: {tool_name} arguments: "
    if raw.startswith(canonical_prefix):
        arguments_text = raw[len(canonical_prefix) :].strip()
    else:
        call_prefix = f"{tool_name}("
        if not raw.startswith(call_prefix) or not raw.endswith(")"):
            raise ValueError(
                "online Tau2 action does not match its selected tool: "
                f"skill={resolved_skill_id}, action={raw[:160]}"
            )
        arguments_text = raw[len(call_prefix) : -1].strip()
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise ValueError("online Tau2 action arguments are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("online Tau2 action arguments must be a JSON object")
    return canonical_prefix + json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
    )


def _support_sha256(skill_ids: list[str]) -> str:
    payload = json.dumps(
        skill_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_e1_report(
    *,
    report_path: Path,
    checkpoint_path: Path,
    encoder_kind: str,
    foundation_checkpoint_path: Path,
    skills_path: Path,
) -> dict[str, Any]:
    report = _read_json(report_path)
    if report.get("schema_version") != MATCHED_HISTORY_TRAIN_SCHEMA:
        raise ValueError("controlled selector E1 report has the wrong schema")
    if report.get("status") != "ok" or str(report.get("encoder_kind")) != encoder_kind:
        raise ValueError("controlled selector E1 report method/status differs")
    if Path(str(report.get("best_checkpoint_path") or "")).resolve() != checkpoint_path:
        raise ValueError("controlled selector checkpoint differs from the E1 report")
    if str(report.get("best_checkpoint_sha256") or "") != file_sha256(checkpoint_path):
        raise ValueError("controlled selector E1 checkpoint hash differs")
    contract = report.get("contract") or {}
    expected_contract = {
        "max_horizon": 16,
        "support_k": 500,
        "belief_top_k": 64,
        "candidate_support": "bounded_last8_skill_action_frozen_static_top500",
        "same_shared_scorer": True,
        "frozen_foundation": True,
        "dynamic_candidate_extras": False,
        "selector": False,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"controlled selector E1 contract differs for {key}")
    foundation = (report.get("inputs") or {}).get("foundation") or {}
    if Path(str(foundation.get("checkpoint_path") or "")).resolve() != foundation_checkpoint_path:
        raise ValueError("controlled selector foundation path differs from E1")
    if str(foundation.get("checkpoint_sha256") or "") != file_sha256(
        foundation_checkpoint_path
    ):
        raise ValueError("controlled selector foundation hash differs from E1")
    if Path(str(foundation.get("training_skills_path") or "")).resolve() != skills_path:
        raise ValueError("controlled selector skill table differs from E1")
    if str(foundation.get("training_skills_sha256") or "") != file_sha256(skills_path):
        raise ValueError("controlled selector skill-table hash differs from E1")
    return report


class ControlledMatchedHistorySelector:
    """Frozen E1 foundation with one fixed-support online history arm."""

    def __init__(
        self,
        *,
        method: str,
        foundation_checkpoint_path: str | Path,
        skills_path: str | Path,
        e1_checkpoint_path: str | Path | None = None,
        e1_report_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        max_horizon: int = 16,
        support_k: int = 500,
        belief_top_k: int = 64,
    ) -> None:
        encoder_kind = str(method).strip().lower()
        if encoder_kind not in CONTROLLED_HISTORY_METHODS:
            raise ValueError(f"unsupported controlled history method: {method}")
        if int(max_horizon) != 16 or int(support_k) != 500 or int(belief_top_k) != 64:
            raise ValueError("canonical E3 requires horizon 16, Top-500, and belief Top-64")
        foundation_path = Path(foundation_checkpoint_path).resolve()
        resolved_skills_path = Path(skills_path).resolve()
        if not foundation_path.is_file() or not resolved_skills_path.is_file():
            raise ValueError("controlled selector is missing its foundation or skill table")
        if file_sha256(foundation_path) != E3_FOUNDATION_SHA256:
            raise ValueError("controlled selector foundation is not the locked E3 artifact")
        if file_sha256(resolved_skills_path) != E3_SKILLS_SHA256:
            raise ValueError("controlled selector skill table is not the locked E3 artifact")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for controlled history selection")

        e1_report: dict[str, Any] | None = None
        checkpoint_path: Path | None = None
        report_path: Path | None = None
        if encoder_kind == "static":
            if e1_checkpoint_path is not None or e1_report_path is not None:
                raise ValueError("the controlled Static arm must not load an E1 encoder")
        else:
            if e1_checkpoint_path is None or e1_report_path is None:
                raise ValueError("a learned controlled history arm requires E1 artifacts")
            checkpoint_path = Path(e1_checkpoint_path).resolve()
            report_path = Path(e1_report_path).resolve()
            if not checkpoint_path.is_file() or not report_path.is_file():
                raise ValueError("controlled selector is missing an E1 artifact")
            e1_report = _validate_e1_report(
                report_path=report_path,
                checkpoint_path=checkpoint_path,
                encoder_kind=encoder_kind,
                foundation_checkpoint_path=foundation_path,
                skills_path=resolved_skills_path,
            )
            selected = E3_SELECTED_CHECKPOINTS[encoder_kind]
            if file_sha256(checkpoint_path) != selected["sha256"]:
                raise ValueError("controlled selector checkpoint is not the locked E3 selection")
            if int(e1_report["seed"]) != int(selected["seed"]) or int(
                e1_report["best_step"]
            ) != int(selected["step"]):
                raise ValueError("controlled selector seed/step is not the locked E3 selection")

        foundation, skills, skill_id_to_idx, foundation_report = (
            load_vnext_stage2_for_evaluation(
                checkpoint_path=foundation_path,
                training_skills_path=resolved_skills_path,
                benchmark_skills=[],
                device=resolved_device,
            )
        )
        foundation.eval()
        for parameter in foundation.parameters():
            parameter.requires_grad_(False)
        self.foundation = foundation
        self.skills = skills
        self.skill_ids = [_skill_id(row) for row in skills]
        self.skill_index = skill_id_to_idx
        self.skill_id_to_idx = skill_id_to_idx
        self.device = resolved_device
        self.encoder_kind = encoder_kind
        self.max_horizon = int(max_horizon)
        self.support_k = int(support_k)
        self.belief_top_k = int(belief_top_k)
        self.method = f"matched_history_e3_{encoder_kind}"
        self._model_lock = threading.RLock()

        self.encoder: torch.nn.Module | None = None
        self.scorer: MatchedHistoryRouteScorer | None = None
        checkpoint_payload: dict[str, Any] | None = None
        if encoder_kind != "static":
            checkpoint_payload = torch.load(
                checkpoint_path,
                map_location=resolved_device,
                weights_only=False,
            )
            if checkpoint_payload.get("schema_version") != MATCHED_HISTORY_TRAIN_SCHEMA:
                raise ValueError("controlled selector E1 checkpoint has the wrong schema")
            if str(checkpoint_payload.get("encoder_kind")) != encoder_kind:
                raise ValueError("controlled selector E1 checkpoint method differs")
            if int(checkpoint_payload.get("seed")) != int(e1_report["seed"]):
                raise ValueError("controlled selector E1 checkpoint seed differs")
            if int(checkpoint_payload.get("step")) != int(e1_report["best_step"]):
                raise ValueError("controlled selector E1 checkpoint step differs")
            dimension = int(foundation.vnext.d)
            self.encoder = build_matched_history_encoder(
                encoder_kind,
                dimension,
                max_horizon=self.max_horizon,
            ).to(resolved_device)
            self.scorer = MatchedHistoryRouteScorer(dimension).to(resolved_device)
            self.encoder.load_state_dict(checkpoint_payload["encoder_state_dict"])
            self.scorer.load_state_dict(checkpoint_payload["scorer_state_dict"])
            self.encoder.eval()
            self.scorer.eval()
            for module in (self.encoder, self.scorer):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        self.checkpoint_binding = {
            "schema_version": CONTROLLED_HISTORY_ONLINE_SCHEMA,
            "method": encoder_kind,
            "foundation": foundation_report,
            "foundation_checkpoint_path": str(foundation_path),
            "foundation_checkpoint_sha256": file_sha256(foundation_path),
            "expected_foundation_checkpoint_sha256": E3_FOUNDATION_SHA256,
            "skills_path": str(resolved_skills_path),
            "skills_sha256": file_sha256(resolved_skills_path),
            "expected_skills_sha256": E3_SKILLS_SHA256,
            "e1_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "e1_checkpoint_sha256": (
                None if checkpoint_path is None else file_sha256(checkpoint_path)
            ),
            "e1_report_path": None if report_path is None else str(report_path),
            "e1_report_sha256": None if report_path is None else file_sha256(report_path),
            "e1_seed": None if e1_report is None else int(e1_report["seed"]),
            "e1_best_step": None if e1_report is None else int(e1_report["best_step"]),
            "e1_source_commit": (
                None if e1_report is None else str(e1_report.get("source_commit") or "")
            ),
            "online_protocol": "bounded_static_support_history_reranking_tau2_v1",
            "max_horizon": self.max_horizon,
            "support_k": self.support_k,
            "belief_top_k": self.belief_top_k,
            "top_k_executor_tools": 8,
            "dynamic_candidate_extras": False,
            "selector": False,
            "shared_foundation_and_static_support": True,
        }

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def new_session(self) -> "ControlledMatchedHistorySession":
        return ControlledMatchedHistorySession(self)


class ControlledMatchedHistorySession:
    """Rebuild the bounded E1 prefix at every online Tau2 decision."""

    def __init__(self, selector: ControlledMatchedHistorySelector) -> None:
        self.selector = selector
        self.events: list[dict[str, Any]] = []
        self.initial_state_text: str | None = None
        self.initial_candidate_skill_ids: list[str] = []
        self.last_selection_state_text: str | None = None

    @property
    def history_depth(self) -> int:
        return len(self.events)

    def _legal_mask(self, candidate_skill_ids: list[str]) -> torch.Tensor:
        selector = self.selector
        mask = torch.zeros(
            (1, len(selector.skills)),
            dtype=torch.bool,
            device=selector.device,
        )
        indices = torch.tensor(
            [selector.skill_id_to_idx[item] for item in candidate_skill_ids],
            dtype=torch.long,
            device=selector.device,
        )
        mask[0, indices] = True
        return mask

    def _event_tensors(
        self,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        selector = self.selector
        events = self.events[-selector.max_horizon :]
        dimension = int(selector.foundation.vnext.d)
        time = len(events)
        shape = (1, time, dimension)
        event_states = torch.zeros(shape, device=selector.device, dtype=dtype)
        event_skills = torch.zeros_like(event_states)
        event_actions = torch.zeros_like(event_states)
        event_results = torch.zeros_like(event_states)
        event_mask = torch.ones(1, time, device=selector.device, dtype=torch.bool)
        result_mask = torch.zeros_like(event_mask)
        if not events:
            return (
                event_states,
                event_skills,
                event_actions,
                event_results,
                event_mask,
                result_mask,
            )
        foundation = selector.foundation
        state_values = foundation.encode_states(
            [str(event["state_text"]) for event in events]
        ).to(dtype=dtype)
        action_values = foundation.encode_observations(
            [str(event["action_text"]) for event in events]
        ).to(dtype=dtype)
        skill_indices = torch.tensor(
            [selector.skill_id_to_idx[str(event["skill_id"])] for event in events],
            device=selector.device,
            dtype=torch.long,
        )
        skill_values = foundation.vnext_normalized_skill_embeddings(
            dtype=dtype
        ).index_select(0, skill_indices)
        event_states[0] = state_values
        event_actions[0] = action_values
        event_skills[0] = skill_values
        result_positions = [
            index for index, event in enumerate(events) if bool(event["result_executed"])
        ]
        if result_positions:
            result_values = foundation.encode_observations(
                [str(events[index]["result_text"]) for index in result_positions]
            ).to(dtype=dtype)
            for result_index, event_index in enumerate(result_positions):
                event_results[0, event_index] = result_values[result_index]
                result_mask[0, event_index] = True
        return (
            event_states,
            event_skills,
            event_actions,
            event_results,
            event_mask,
            result_mask,
        )

    def select(
        self,
        current_state_text: str,
        *,
        candidate_skill_ids: Iterable[str],
        top_k: int,
        route_mode: str = "dynamic",
        **_unused: Any,
    ) -> list[dict[str, Any]]:
        selector = self.selector
        state_text = str(current_state_text or "").strip()
        candidates = _deduplicate(candidate_skill_ids)
        if not state_text or not candidates:
            raise ValueError("controlled online selection requires state and candidates")
        unknown = [item for item in candidates if item not in selector.skill_id_to_idx]
        if unknown:
            raise ValueError(
                "controlled candidate skills are absent from the E1 inventory: "
                + ", ".join(unknown[:10])
            )
        expected_route_mode = "static" if selector.encoder_kind == "static" else "dynamic"
        if str(route_mode).strip().lower() != expected_route_mode:
            raise ValueError(
                f"controlled {selector.encoder_kind} requires route_mode={expected_route_mode}"
            )
        if self.initial_state_text is None:
            self.initial_state_text = state_text
            self.initial_candidate_skill_ids = list(candidates)
        elif candidates != self.initial_candidate_skill_ids:
            raise ValueError("controlled Tau2 candidate inventory changed within a session")

        requested_top_k = max(1, min(int(top_k), len(candidates)))
        foundation = selector.foundation
        with selector._model_lock, torch.inference_mode(), selector._autocast():
            legal = self._legal_mask(candidates)
            static_text, _canonical = serialize_compact_causal_state(
                state_text,
                self.events,
                max_events=8,
                max_action_chars=256,
            )
            initial_state, current_state, static_state = foundation.encode_states(
                [self.initial_state_text, state_text, static_text]
            )
            initial_state = initial_state.unsqueeze(0)
            current_state = current_state.unsqueeze(0)
            static_state = static_state.unsqueeze(0)
            initial_memory = foundation.vnext_initial_belief(
                initial_state,
                legal,
                top_k=selector.belief_top_k,
            )
            static_memory = foundation.vnext_initial_belief(
                static_state,
                legal,
                top_k=selector.belief_top_k,
            )
            static_recall, _static_delta, base_query = (
                foundation.vnext.static_query_components(static_state, static_memory)
            )
            full_static = foundation.vnext_full_pool_logits(static_recall, head="recall")
            support_ids, support_valid = masked_topk_tensor(
                full_static,
                legal,
                k=selector.support_k,
            )
            skill_embeddings = foundation.vnext_normalized_skill_embeddings(
                dtype=current_state.dtype
            )
            support_embeddings = skill_embeddings.index_select(
                0,
                support_ids.reshape(-1),
            ).view(support_ids.size(0), support_ids.size(1), -1)
            static_logits = foundation.vnext.static_query.temperature().to(
                dtype=base_query.dtype
            ) * torch.einsum("bd,bcd->bc", base_query, support_embeddings)
            query = base_query
            if selector.encoder_kind != "static" and self.events:
                if selector.encoder is None or selector.scorer is None:
                    raise RuntimeError("controlled learned history modules are unavailable")
                tensors = self._event_tensors(dtype=current_state.dtype)
                history_memory = selector.encoder(
                    initial_memory,
                    *tensors,
                    serialized_history_embedding=None,
                )
                query = selector.scorer(
                    base_query,
                    current_state,
                    history_memory,
                    static_memory,
                    tensors[-2].any(dim=-1),
                )
            logits = foundation.vnext.static_query.temperature().to(
                dtype=query.dtype
            ) * torch.einsum("bd,bcd->bc", query, support_embeddings)
            valid_positions = support_valid[0].nonzero(as_tuple=False).flatten().tolist()
            ordered_support_ids = [
                selector.skill_ids[int(support_ids[0, position].item())]
                for position in valid_positions
            ]
            support_digest = _support_sha256(ordered_support_ids)
            selected_expert = (
                "dynamic"
                if selector.encoder_kind != "static" and bool(self.events)
                else "static"
            )
            mixture_probability = 1.0 if selected_expert == "dynamic" else 0.0
            ranked: list[dict[str, Any]] = []
            for position in valid_positions:
                skill_index = int(support_ids[0, position].item())
                ranked.append(
                    {
                        "skill_id": selector.skill_ids[skill_index],
                        "skill": selector.skills[skill_index],
                        "score": float(logits[0, position].float().item()),
                        "static_score": float(static_logits[0, position].float().item()),
                        "dynamic_score": float(logits[0, position].float().item()),
                        "selector_probability": 0.0,
                        "mixture_probability": mixture_probability,
                        "adaptive_mixture_probability": mixture_probability,
                        "selected_expert": selected_expert,
                        "route_mode": expected_route_mode,
                        "history_depth": len(self.events),
                        "history_window_depth": min(
                            len(self.events), selector.max_horizon
                        ),
                        "static_support_sha256": support_digest,
                        "static_support_count": len(ordered_support_ids),
                        "candidate_count": len(candidates),
                        "natural_support_count": len(ordered_support_ids),
                        "encoder_kind": selector.encoder_kind,
                    }
                )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["skill_id"])))
        selected = ranked[:requested_top_k]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        self.last_selection_state_text = state_text
        return selected

    def observe(
        self,
        *,
        state_text_before: str,
        skill_id: str,
        action_text: str,
        result_text: str | None = None,
    ) -> dict[str, Any]:
        state_text = str(state_text_before or "").strip()
        resolved_skill_id = str(skill_id or "").strip()
        if self.last_selection_state_text is None:
            raise RuntimeError("controlled history update requires a preceding selection")
        if state_text != self.last_selection_state_text:
            raise ValueError("controlled history update state differs from its selection")
        if resolved_skill_id not in self.selector.skill_id_to_idx:
            raise ValueError("controlled history update skill is outside the inventory")
        canonical_action = canonical_tau2_action_text(resolved_skill_id, action_text)
        result = str(result_text or "").strip()
        event = {
            "step_index": len(self.events),
            "state_text": state_text,
            "skill_id": resolved_skill_id,
            "action_text": canonical_action,
            "result_text": result,
            "result_executed": bool(result),
        }
        self.events.append(event)
        return {
            "history_depth": len(self.events),
            "history_window_depth": min(
                len(self.events), self.selector.max_horizon
            ),
            "skill_id": resolved_skill_id,
            "event_action_text": canonical_action,
            "result_observed": bool(result),
            "result_sha256": (
                hashlib.sha256(result.encode("utf-8")).hexdigest() if result else ""
            ),
            "result_correction_applied": bool(
                result and self.selector.encoder_kind == "lstr"
            ),
            "correction_beta": None,
            "prefix_reconstructed_at_next_selection": True,
        }
