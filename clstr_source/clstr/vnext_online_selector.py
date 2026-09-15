from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence

import torch

from clstr.history_channel import serialize_compact_causal_state
from clstr.vnext_eval import load_vnext_stage2_for_evaluation
from clstr.vnext_matched_release import validate_matched_release_selection


VNEXT_ROUTE_MODES = frozenset({"adaptive", "static", "dynamic"})


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


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


def _selected_route_logits(
    route: Any,
    *,
    route_mode: str,
    has_history: bool,
) -> tuple[torch.Tensor, str, float]:
    """Select one complete route expert without candidate-level switching."""

    mode = str(route_mode or "adaptive").strip().lower()
    if mode not in VNEXT_ROUTE_MODES:
        raise ValueError(f"unsupported vNext route mode: {route_mode}")
    adaptive_probability = float(route.mixture_probability[0].float().item())
    if mode == "static" or not has_history:
        return route.static_logits, "static", 0.0
    if mode == "dynamic":
        return route.raw_dynamic_logits, "dynamic", 1.0
    selected_expert = "dynamic" if adaptive_probability >= 0.5 else "static"
    return route.mixed_logits, selected_expert, adaptive_probability


def _selected_route_support_mask(
    route: Any,
    *,
    support_valid: torch.Tensor,
    selected_expert: str,
) -> torch.Tensor:
    """Return the natural support owned by the selected complete expert."""

    if selected_expert == "static":
        return support_valid & route.static_support_mask
    return support_valid


class VNextOnlineSelector:
    """One release-bound CLSTR model shared by independent episode sessions."""

    method = "clstr_vnext_stage2_matched_release"

    def __init__(
        self,
        *,
        matched_release_selection_path: str | Path,
        stage2_checkpoint_path: str | Path | None = None,
        training_skills_path: str | Path | None = None,
        benchmark_skills_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        belief_top_k: int | None = None,
        coarse_k: int = 500,
        dynamic_extra_k: int = 64,
    ) -> None:
        release_payload = _read_json(matched_release_selection_path)
        checkpoint = Path(
            stage2_checkpoint_path
            or str(release_payload.get("selected_checkpoint_path") or "")
        ).resolve()
        skills_path = Path(
            training_skills_path
            or str(release_payload.get("training_skills_path") or "")
        ).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"release-selected Stage2 checkpoint is missing: {checkpoint}")
        if not skills_path.is_file():
            raise ValueError(f"release-bound training skills are missing: {skills_path}")
        release_contract = validate_matched_release_selection(
            matched_release_selection_path,
            stage2_checkpoint_path=checkpoint,
        )
        if Path(release_contract["selected_checkpoint_path"]).resolve() != checkpoint:
            raise ValueError("online selector checkpoint differs from matched release")

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for CLSTR online selection but is unavailable")
        benchmark_skills = (
            _read_jsonl(benchmark_skills_path)
            if benchmark_skills_path is not None
            else []
        )
        model, skills, skill_id_to_idx, load_report = (
            load_vnext_stage2_for_evaluation(
                checkpoint_path=checkpoint,
                training_skills_path=skills_path,
                benchmark_skills=benchmark_skills,
                device=resolved_device,
            )
        )
        self.model = model
        self.skills = skills
        self.skill_id_to_idx = skill_id_to_idx
        self.skill_index = skill_id_to_idx
        self.skill_ids = [_skill_id(row) for row in skills]
        self.skill_by_id = {
            _skill_id(row): row for row in skills if _skill_id(row)
        }
        self.device = resolved_device
        self.belief_top_k = belief_top_k
        self.coarse_k = max(1, int(coarse_k))
        self.dynamic_extra_k = max(1, int(dynamic_extra_k))
        self._model_lock = threading.RLock()
        self.checkpoint_binding = {
            "release": release_contract,
            "checkpoint": load_report,
            "benchmark_skills_path": (
                None
                if benchmark_skills_path is None
                else str(Path(benchmark_skills_path).resolve())
            ),
            "online_protocol": "native_factual_stage2_recurrent_v1",
            "coarse_k": self.coarse_k,
            "dynamic_extra_k": self.dynamic_extra_k,
            "hard_fallback": True,
            "uses_benchmark_identity_for_dispatch": False,
        }

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def new_session(self) -> "VNextOnlineSession":
        return VNextOnlineSession(self)

    def encode_external_candidate_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode temporary candidates without mutating the release inventory."""

        values = [str(text) for text in texts]
        if not values or any(not value.strip() for value in values):
            raise ValueError("external candidate texts must be nonempty")
        with self._model_lock, torch.inference_mode(), self._autocast():
            encoded = self.model.encode_observations(values)
        return encoded.detach().float().cpu()

    def ensure_skills(
        self,
        skill_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Append unseen evaluation skills without changing the release prefix."""

        pending: list[dict[str, Any]] = []
        pending_by_id: dict[str, dict[str, Any]] = {}
        with self._model_lock:
            serializer = self.model.skill_table.skill_text_fn
            for raw_row in skill_rows:
                row = dict(raw_row)
                skill_id = _skill_id(row)
                if not skill_id:
                    raise ValueError("runtime skill append requires a nonempty skill_id")
                row["skill_id"] = skill_id
                existing = self.skill_by_id.get(skill_id) or pending_by_id.get(skill_id)
                if existing is not None:
                    if serializer(existing) != serializer(row):
                        raise ValueError(
                            f"runtime skill definition conflicts with release inventory: {skill_id}"
                        )
                    continue
                pending.append(row)
                pending_by_id[skill_id] = row

            if not pending:
                return {
                    "old_count": len(self.skills),
                    "new_count": len(self.skills),
                    "appended_count": 0,
                    "appended_skill_ids": [],
                    "checkpoint_prefix_bit_exact_after_append": True,
                }

            old_count = len(self.skills)
            prefix_before = self.model.skill_table.E.detach()
            with torch.no_grad(), self._autocast():
                report = self.model.append_skills(pending)
            prefix_after = self.model.skill_table.E[:old_count].detach()
            if not torch.equal(prefix_before, prefix_after):
                raise RuntimeError("runtime skill append changed the release skill prefix")

            self.skills = list(self.model.skills)
            self.skill_ids = [_skill_id(row) for row in self.skills]
            self.skill_id_to_idx = {
                skill_id: index
                for index, skill_id in enumerate(self.skill_ids)
                if skill_id
            }
            self.skill_index = self.skill_id_to_idx
            self.skill_by_id = {
                _skill_id(row): row for row in self.skills if _skill_id(row)
            }
            appended_ids = list(report.get("appended_skill_ids") or [])
            runtime_ids = self.checkpoint_binding.setdefault(
                "runtime_appended_skill_ids", []
            )
            runtime_ids.extend(
                skill_id for skill_id in appended_ids if skill_id not in runtime_ids
            )
            self.checkpoint_binding["runtime_appended_skill_count"] = len(runtime_ids)
            return {
                **report,
                "checkpoint_prefix_bit_exact_after_append": True,
            }


class VNextOnlineSession:
    """Episode-local recurrent memory over one shared frozen-backbone model."""

    def __init__(self, selector: VNextOnlineSelector) -> None:
        self.selector = selector
        self.reset()

    @property
    def history_depth(self) -> int:
        return len(self.events)

    def reset(self) -> None:
        self.memory: torch.Tensor | None = None
        self.latent_trace: torch.Tensor | None = None
        self.events: list[dict[str, Any]] = []
        self.last_state_text: str | None = None
        self.last_candidate_skill_ids: list[str] = []

    def _legal_mask(self, candidate_skill_ids: list[str]) -> torch.Tensor:
        mask = torch.zeros(
            (1, len(self.selector.skills)),
            dtype=torch.bool,
            device=self.selector.device,
        )
        indices = torch.tensor(
            [self.selector.skill_id_to_idx[item] for item in candidate_skill_ids],
            dtype=torch.long,
            device=self.selector.device,
        )
        mask[0, indices] = True
        return mask

    def select(
        self,
        current_state_text: str,
        *,
        candidate_skill_ids: Iterable[str],
        top_k: int,
        coarse_k: int | None = None,
        dynamic_extra_k: int | None = None,
        route_mode: str = "adaptive",
    ) -> list[dict[str, Any]]:
        state_text = str(current_state_text or "").strip()
        if not state_text:
            raise ValueError("CLSTR online selection requires a nonempty current state")
        candidates = _deduplicate(candidate_skill_ids)
        if not candidates:
            raise ValueError("CLSTR online selection requires at least one candidate skill")
        unknown = [item for item in candidates if item not in self.selector.skill_id_to_idx]
        if unknown:
            raise ValueError(
                "online candidate skills are absent from the release inventory: "
                + ", ".join(unknown[:10])
            )
        requested_top_k = max(1, min(int(top_k), len(candidates)))
        resolved_coarse_k = max(
            requested_top_k,
            min(
                int(coarse_k or self.selector.coarse_k),
                len(candidates),
            ),
        )
        resolved_extra_k = max(
            1,
            min(
                int(
                    self.selector.dynamic_extra_k
                    if dynamic_extra_k is None
                    else dynamic_extra_k
                ),
                len(candidates),
            ),
        )
        selector = self.selector
        with selector._model_lock, torch.inference_mode(), selector._autocast():
            legal = self._legal_mask(candidates)
            causal_text, _ = serialize_compact_causal_state(state_text, self.events)
            if causal_text == state_text:
                current_state = selector.model.encode_states([state_text])
                causal_state = current_state
            else:
                encoded_states = selector.model.encode_states(
                    [state_text, causal_text]
                )
                current_state = encoded_states[:1]
                causal_state = encoded_states[1:]
            if self.memory is None:
                self.memory = selector.model.vnext_initial_belief(
                    current_state,
                    legal,
                    top_k=selector.belief_top_k,
                ).detach()
            static_memory = selector.model.vnext_initial_belief(
                causal_state,
                legal,
                top_k=selector.belief_top_k,
            )
            history_mask = torch.tensor(
                [bool(self.events)],
                dtype=torch.bool,
                device=selector.device,
            )
            history_depth = torch.tensor(
                [len(self.events)],
                dtype=torch.long,
                device=selector.device,
            )
            queries = selector.model.vnext_queries(
                causal_state,
                self.memory.to(device=selector.device, dtype=causal_state.dtype),
                static_memory,
                history_mask,
                memory_state=current_state,
                latent_trace=self.latent_trace,
            )
            path = selector.model.vnext_natural_candidate_union_path(
                queries.static_recall,
                queries.dynamic_recall,
                current_state,
                legal,
                history_mask,
                coarse_k=resolved_coarse_k,
                dynamic_extra_k=resolved_extra_k,
            )
            support_ids = path.support.candidate_ids
            support_valid = path.support.valid_mask
            route = selector.model.vnext_safe_candidate_route_scores(
                causal_state,
                self.memory.to(device=selector.device, dtype=causal_state.dtype),
                static_memory,
                history_mask,
                support_ids,
                support_valid,
                static_candidate_ids=path.coarse_candidate_ids,
                static_valid_mask=path.coarse_valid_mask,
                history_depth=history_depth,
                memory_state=current_state,
                latent_trace=self.latent_trace,
                hard_fallback=True,
            )
            selected_logits, selected_expert, effective_mixture_probability = (
                _selected_route_logits(
                    route,
                    route_mode=route_mode,
                    has_history=bool(self.events),
                )
            )
            # The learned Static expert is defined on its own natural coarse
            # support.  Dynamic-only recall additions remain available for
            # diagnostics, but must never leak into an always-Static (or
            # adaptive Static-fallback) output merely because the caller asks
            # for more items than the static support contains.
            ranking_valid = _selected_route_support_mask(
                route,
                support_valid=support_valid,
                selected_expert=selected_expert,
            )
            positions = ranking_valid[0].nonzero(as_tuple=False).flatten().tolist()
            selector_probability = float(route.selector_probability[0].float().item())
            adaptive_mixture_probability = float(
                route.mixture_probability[0].float().item()
            )
            ranked: list[dict[str, Any]] = []
            for position in positions:
                skill_index = int(support_ids[0, position].item())
                skill_id = selector.skill_ids[skill_index]
                ranked.append(
                    {
                        "skill_id": skill_id,
                        "skill": selector.skills[skill_index],
                        "score": float(selected_logits[0, position].float().item()),
                        "static_score": float(
                            route.static_logits[0, position].float().item()
                        ),
                        "dynamic_score": float(
                            route.raw_dynamic_logits[0, position].float().item()
                        ),
                        "route_residual": float(
                            route.raw_residual[0, position].float().item()
                        ),
                        "on_static_support": bool(
                            route.static_support_mask[0, position].item()
                        ),
                        "selector_probability": selector_probability,
                        "mixture_probability": effective_mixture_probability,
                        "adaptive_mixture_probability": adaptive_mixture_probability,
                        "route_mode": str(route_mode).strip().lower(),
                        "selected_expert": selected_expert,
                        "history_depth": len(self.events),
                    }
                )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["skill_id"])))
        selected = ranked[:requested_top_k]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
            item["candidate_count"] = len(candidates)
            item["natural_support_count"] = len(ranked)
        self.last_state_text = state_text
        self.last_candidate_skill_ids = candidates
        return selected

    def score_external_candidate_embeddings(
        self,
        current_state_text: str,
        candidate_embeddings: torch.Tensor,
        *,
        selected_expert: str,
        route_mode: str = "adaptive",
        scoring_head: str = "route_query",
    ) -> dict[str, torch.Tensor | str]:
        """Score temporary candidates without adding them to the skill table.

        The expert choice must already have been made on checkpoint-known skill
        candidates by :meth:`select`.  This method reuses that choice while
        projecting the corresponding route query onto external candidate
        embeddings.  It never lets the temporary candidate distribution alter
        the reliability gate.
        """

        state_text = str(current_state_text or "").strip()
        if not state_text:
            raise ValueError("external candidate scoring requires a nonempty state")
        mode = str(route_mode or "adaptive").strip().lower()
        if mode not in VNEXT_ROUTE_MODES:
            raise ValueError(f"unsupported vNext route mode: {route_mode}")
        expert = str(selected_expert or "static").strip().lower()
        if expert not in {"static", "dynamic"}:
            raise ValueError(f"unsupported selected route expert: {selected_expert}")
        score_head = str(scoring_head or "route_query").strip().lower()
        if score_head not in {"route_query", "memory_skill_head"}:
            raise ValueError(f"unsupported external candidate scoring head: {scoring_head}")
        if self.memory is None or not self.last_candidate_skill_ids:
            raise ValueError(
                "external candidates require preceding abstract skill selection"
            )
        embeddings = candidate_embeddings
        if embeddings.ndim != 2 or int(embeddings.size(-1)) != int(
            self.memory.size(-1)
        ):
            raise ValueError("external candidate embeddings must have shape [candidates, d]")
        if int(embeddings.size(0)) <= 0:
            raise ValueError("external candidate scoring requires at least one candidate")

        selector = self.selector
        with selector._model_lock, torch.inference_mode(), selector._autocast():
            legal = self._legal_mask(self.last_candidate_skill_ids)
            current_state = selector.model.encode_states([state_text])
            external = embeddings.to(
                device=selector.device,
                dtype=current_state.dtype,
            ).unsqueeze(0)
            use_dynamic = bool(self.events) and (
                mode == "dynamic" or (mode == "adaptive" and expert == "dynamic")
            )
            if score_head == "memory_skill_head":
                static_memory = selector.model.vnext_initial_belief(
                    current_state,
                    legal,
                    top_k=selector.belief_top_k,
                )
                dynamic_memory = self.memory.to(
                    device=selector.device,
                    dtype=current_state.dtype,
                )
                static_scores = selector.model.skill_head(
                    external,
                    static_memory,
                )
                dynamic_scores = selector.model.skill_head(
                    external,
                    dynamic_memory,
                )
                selected = dynamic_scores if use_dynamic else static_scores
                return {
                    "scores": selected[0].detach().float().cpu(),
                    "static_scores": static_scores[0].detach().float().cpu(),
                    "dynamic_scores": dynamic_scores[0].detach().float().cpu(),
                    "route_residual": (
                        dynamic_scores[0] - static_scores[0]
                    ).detach().float().cpu(),
                    "selected_expert": "dynamic" if use_dynamic else "static",
                    "scoring_head": score_head,
                }
            causal_text, _ = serialize_compact_causal_state(state_text, self.events)
            if causal_text == state_text:
                causal_state = current_state
            else:
                causal_state = selector.model.encode_states([causal_text])
            static_memory = selector.model.vnext_initial_belief(
                causal_state,
                legal,
                top_k=selector.belief_top_k,
            )
            history_mask = torch.tensor(
                [bool(self.events)],
                dtype=torch.bool,
                device=selector.device,
            )
            history_depth = torch.tensor(
                [len(self.events)],
                dtype=torch.long,
                device=selector.device,
            )
            external = external.to(dtype=causal_state.dtype)
            valid = torch.ones(
                (1, int(external.size(1))),
                dtype=torch.bool,
                device=selector.device,
            )
            route = selector.model.vnext.candidate_route_scores(
                causal_state,
                self.memory.to(
                    device=selector.device,
                    dtype=causal_state.dtype,
                ),
                static_memory,
                history_mask,
                external,
                valid,
                static_support_mask=valid,
                history_depth=history_depth,
                memory_state=current_state,
                latent_trace=self.latent_trace,
                temperature=selector.model.vnext.static_query.temperature(),
                hard_fallback=False,
            )
            selected = (
                route.raw_dynamic_logits if use_dynamic else route.static_logits
            )
        return {
            "scores": selected[0].detach().float().cpu(),
            "static_scores": route.static_logits[0].detach().float().cpu(),
            "dynamic_scores": route.raw_dynamic_logits[0].detach().float().cpu(),
            "route_residual": route.raw_residual[0].detach().float().cpu(),
            "selected_expert": "dynamic" if use_dynamic else "static",
            "scoring_head": score_head,
        }

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
        action = str(action_text or "").strip()
        result = str(result_text or "").strip()
        if not state_text or not action:
            raise ValueError("memory update requires pre-action state and action text")
        if resolved_skill_id not in self.selector.skill_id_to_idx:
            raise ValueError(f"memory update skill is not in the release: {resolved_skill_id}")
        if self.memory is None:
            raise RuntimeError("memory update requires a preceding online selection")
        selector = self.selector
        with selector._model_lock, torch.inference_mode(), selector._autocast():
            state = selector.model.encode_states([state_text])
            skill_index = selector.skill_id_to_idx[resolved_skill_id]
            skill = selector.model.vnext_normalized_skill_embeddings(
                dtype=state.dtype
            )[skill_index].unsqueeze(0)
            if result:
                encoded_updates = selector.model.encoder([action, result])
                action_embedding = encoded_updates[:1]
                result_embedding = encoded_updates[1:]
            else:
                action_embedding = selector.model.encoder([action])
                result_embedding = None
            update = selector.model.vnext_update_memory(
                self.memory.to(device=selector.device, dtype=state.dtype),
                state,
                skill,
                action_embedding,
                result_embedding=result_embedding,
                latent_trace=self.latent_trace,
            )
            self.memory = update.effective_memory.detach()
            self.latent_trace = (
                None
                if update.latent_trace is None
                else update.latent_trace.detach()
            )
            correction_beta = (
                None
                if update.correction_beta is None
                else float(update.correction_beta.float().mean().item())
            )
        event = {
            "step_index": len(self.events),
            "skill_id": resolved_skill_id,
            "action_text": action,
            "result_text": result,
            "result_executed": bool(result),
        }
        self.events.append(event)
        return {
            "history_depth": len(self.events),
            "skill_id": resolved_skill_id,
            "result_correction_applied": bool(result),
            "correction_beta": correction_beta,
        }
