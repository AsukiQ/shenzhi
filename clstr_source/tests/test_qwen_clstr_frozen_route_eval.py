import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clstr.memory_utility_records import canonical_digest, checkpoint_chain_digest
from clstr.qwen_clstr_final_chain import FINAL_CHAIN_SCHEMA_VERSION
from clstr.qwen_clstr_lineage import sha256_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _frozen_corpus(tmp_path: Path, *, benchmark: str = "toolsandbox") -> dict[str, Path]:
    skills = [
        {"skill_id": f"{benchmark}/a", "description": "skill a"},
        {"skill_id": f"{benchmark}/b", "description": "skill b"},
        {"skill_id": f"{benchmark}/c", "description": "skill c"},
    ]
    rows = [
        {
            "row_id": "row-0",
            "raw_state": "query-0",
            "positive_skill_id": f"{benchmark}/a",
            "candidate_skill_ids": [f"{benchmark}/a", f"{benchmark}/b", f"{benchmark}/c"],
        },
        {
            "row_id": "row-1",
            "raw_state": "query-1",
            "positive_skill_id": f"{benchmark}/c",
            "candidate_skill_ids": [f"{benchmark}/a", f"{benchmark}/c"],
        },
    ]
    manifest = {
        "schema_version": 1,
        "multibench_protocol_version": "qwen06_method_multibench_v1",
        "formatting_version": "qwen06_method_multibench_format_v1",
        "benchmark": benchmark,
        "split": "eval",
        "canonical_skills": skills,
        "canonical_source_rows": rows,
        "skill_count": len(skills),
        "source_row_count": len(rows),
        "positive_coverage": {
            "status": "ok",
            "source_row_count": len(rows),
            "covered_row_count": len(rows),
            "missing_positive_row_ids": [],
        },
        "source_sha256": {},
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    root = tmp_path / benchmark
    root.mkdir(parents=True)
    paths = {
        "manifest": root / "manifest.json",
        "rows": root / "source_rows.jsonl",
        "skills": root / "skills.jsonl",
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(paths["rows"], rows)
    _write_jsonl(paths["skills"], skills)
    return paths


def _final_chain(tmp_path: Path, *, benchmark: str = "toolsandbox") -> dict[str, Path]:
    base_skills_path = tmp_path / "training" / "skill_pool.jsonl"
    _write_jsonl(
        base_skills_path,
        [{"skill_id": f"{benchmark}/a", "description": "trained skill a"}],
    )
    checkpoint_paths: dict[str, Path] = {}
    for role in ("stage0", "stage1", "stage2", "stage4"):
        path = tmp_path / "checkpoints" / f"{role}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "stage4":
            torch.save(
                {
                    "model_state_dict": {
                        "transition.weight": torch.zeros(1, 1),
                        "gate.weight": torch.zeros(1, 1),
                        "action_proj.weight": torch.zeros(1, 1),
                        "route_memory_utility_gate.output.weight": torch.zeros(1, 1),
                    }
                },
                path,
            )
        else:
            path.write_bytes(f"{role}-checkpoint".encode("utf-8"))
        checkpoint_paths[role] = path
    backbone_path = tmp_path / "models" / "Qwen3-Embedding-0.6B"
    backbone_path.mkdir(parents=True)
    (backbone_path / "config.json").write_text("{}\n", encoding="utf-8")
    selection_path = tmp_path / "stage4_selection.json"
    selection_path.write_text('{"status":"ok"}\n', encoding="utf-8")
    reliability = {
        "mode": "fixed_alpha",
        "fixed_alpha": 0.5,
        "gate_checkpoint": None,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    payload = {
        "schema_version": FINAL_CHAIN_SCHEMA_VERSION,
        "status": "ok",
        "final_checkpoint_role": "stage4",
        "checkpoint_chain_digest": checkpoint_chain_digest(checkpoint_paths),
        "checkpoints": {
            role: {**sha256_path(path), "stage": role}
            for role, path in checkpoint_paths.items()
        },
        "backbone": {**sha256_path(backbone_path), "frozen": True},
        "skill_pool": sha256_path(base_skills_path),
        "stage4_selection": sha256_path(selection_path),
        "reliability": reliability,
        "route_scorer": "unified_memory",
        "state_query": {
            "prompt_version": "clstr_causal_state_v1",
            "instruction": "retrieve the next skill",
            "max_chars": 2000,
            "truncation": "head_tail_v1",
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    manifest_path = tmp_path / "final_chain.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "base_skills": base_skills_path,
        "stage4_selection": selection_path,
        **{role: path for role, path in checkpoint_paths.items()},
    }


def test_load_frozen_route_corpus_validates_self_hash_jsonl_and_positive_coverage(tmp_path):
    from clstr.frozen_clstr_route_eval import load_frozen_route_corpus

    paths = _frozen_corpus(tmp_path)
    corpus = load_frozen_route_corpus(
        manifest_path=paths["manifest"],
        source_rows_path=paths["rows"],
        skills_path=paths["skills"],
    )

    assert corpus.benchmark == "toolsandbox"
    assert len(corpus.source_rows) == 2
    assert len(corpus.skills) == 3
    assert corpus.manifest_sha256

    enriched_paths = _frozen_corpus(tmp_path / "enriched")
    enriched_rows = [
        {
            **row,
            "split": "valid_unseen",
            "step_index": index + 3,
            "action_history": ["look"],
        }
        for index, row in enumerate(
            json.loads(enriched_paths["manifest"].read_text(encoding="utf-8"))["canonical_source_rows"]
        )
    ]
    _write_jsonl(enriched_paths["rows"], enriched_rows)
    enriched = load_frozen_route_corpus(
        manifest_path=enriched_paths["manifest"],
        source_rows_path=enriched_paths["rows"],
        skills_path=enriched_paths["skills"],
    )
    assert enriched.source_rows[0]["split"] == "valid_unseen"
    assert enriched.source_rows[0]["step_index"] == 3
    assert enriched.source_rows[0]["action_history"] == ["look"]

    paths["rows"].write_text('{"row_id":"drift"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source row"):
        load_frozen_route_corpus(
            manifest_path=paths["manifest"],
            source_rows_path=paths["rows"],
            skills_path=paths["skills"],
        )

    duplicate_rows = _frozen_corpus(tmp_path / "duplicate_rows")
    duplicate_manifest = json.loads(duplicate_rows["manifest"].read_text(encoding="utf-8"))
    duplicate_source = list(duplicate_manifest["canonical_source_rows"])
    duplicate_source[1] = {**duplicate_source[1], "row_id": duplicate_source[0]["row_id"]}
    duplicate_manifest["canonical_source_rows"] = duplicate_source
    duplicate_manifest["manifest_sha256"] = canonical_digest(
        {key: value for key, value in duplicate_manifest.items() if key != "manifest_sha256"}
    )
    duplicate_rows["manifest"].write_text(
        json.dumps(duplicate_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(duplicate_rows["rows"], duplicate_source)
    with pytest.raises(ValueError, match="unique non-empty row IDs"):
        load_frozen_route_corpus(
            manifest_path=duplicate_rows["manifest"],
            source_rows_path=duplicate_rows["rows"],
            skills_path=duplicate_rows["skills"],
        )

    duplicate_candidates = _frozen_corpus(tmp_path / "duplicate_candidates")
    candidate_manifest = json.loads(duplicate_candidates["manifest"].read_text(encoding="utf-8"))
    candidate_source = list(candidate_manifest["canonical_source_rows"])
    candidate_source[0] = {
        **candidate_source[0],
        "candidate_skill_ids": [
            candidate_source[0]["candidate_skill_ids"][0],
            candidate_source[0]["candidate_skill_ids"][0],
            *candidate_source[0]["candidate_skill_ids"][1:],
        ],
    }
    candidate_manifest["canonical_source_rows"] = candidate_source
    candidate_manifest["manifest_sha256"] = canonical_digest(
        {key: value for key, value in candidate_manifest.items() if key != "manifest_sha256"}
    )
    duplicate_candidates["manifest"].write_text(
        json.dumps(candidate_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(duplicate_candidates["rows"], candidate_source)
    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        load_frozen_route_corpus(
            manifest_path=duplicate_candidates["manifest"],
            source_rows_path=duplicate_candidates["rows"],
            skills_path=duplicate_candidates["skills"],
        )


def test_load_final_chain_manifest_rejects_chain_digest_or_backbone_drift(tmp_path):
    from clstr.frozen_clstr_route_eval import load_final_chain_manifest

    bad_chain = _final_chain(tmp_path / "bad_chain")
    bad_chain_payload = json.loads(bad_chain["manifest"].read_text(encoding="utf-8"))
    bad_chain_payload["checkpoint_chain_digest"] = "0" * 64
    bad_chain_payload["manifest_sha256"] = canonical_digest(
        {key: value for key, value in bad_chain_payload.items() if key != "manifest_sha256"}
    )
    bad_chain["manifest"].write_text(
        json.dumps(bad_chain_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint-chain digest mismatch"):
        load_final_chain_manifest(bad_chain["manifest"])

    backbone_drift = _final_chain(tmp_path / "backbone_drift")
    backbone_config = tmp_path / "backbone_drift" / "models" / "Qwen3-Embedding-0.6B" / "config.json"
    backbone_config.write_text('{"drift":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Qwen backbone sha256 mismatch"):
        load_final_chain_manifest(backbone_drift["manifest"])


def test_load_final_chain_manifest_accepts_cmc_direct_utility_overlay(tmp_path):
    from clstr.frozen_clstr_route_eval import load_final_chain_manifest

    paths = _final_chain(tmp_path)
    gate_checkpoint = tmp_path / "memory_utility_gate.pt"
    gate_checkpoint.write_bytes(b"gate")
    gate_report = tmp_path / "gate_report.json"
    gate_report.write_text("{}\n", encoding="utf-8")
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": sha256_path(gate_checkpoint),
        "gate_report": sha256_path(gate_report),
        "audit_manifest_sha256": "a" * 64,
        "selected_stage4_checkpoint_sha256": payload["checkpoints"]["stage4"][
            "sha256"
        ],
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 64.0,
        "base_gate_source": "stage4_checkpoint",
        "deployed_gate_source": "direct_utility_overlay",
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    payload["reliability"] = reliability
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_digest(payload)
    paths["manifest"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = load_final_chain_manifest(paths["manifest"])

    assert loaded["reliability"]["deployed_gate_source"] == (
        "direct_utility_overlay"
    )


def test_load_final_chain_manifest_rejects_invalid_fixed_alpha(tmp_path):
    from clstr.frozen_clstr_route_eval import load_final_chain_manifest

    paths = _final_chain(tmp_path)
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    payload["reliability"]["fixed_alpha"] = 1.5
    reliability_without_hash = dict(payload["reliability"])
    reliability_without_hash.pop("reliability_sha256")
    payload["reliability"]["reliability_sha256"] = canonical_digest(
        reliability_without_hash
    )
    payload["manifest_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    paths["manifest"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed alpha must be in"):
        load_final_chain_manifest(paths["manifest"])


def test_adapt_frozen_route_rows_preserves_candidates_and_never_fabricates_replay():
    from clstr.frozen_clstr_route_eval import adapt_frozen_route_rows

    local = adapt_frozen_route_rows(
        benchmark="tau2",
        source_rows=[
            {
                "row_id": "r0",
                "raw_state": "state",
                "positive_skill_id": "tau2/a",
                "candidate_skill_ids": ["tau2/a", "tau2/b"],
                "domain": "retail",
                "split": "test",
                "step_index": 7,
            }
        ],
        benchmark_skill_ids=["tau2/a", "tau2/b", "tau2/c"],
    )
    global_pool = adapt_frozen_route_rows(
        benchmark="toolbench_g3",
        source_rows=[
            {
                "row_id": "r1",
                "raw_state": "toolbench state",
                "positive_skill_id": "toolbench/a",
            }
        ],
        benchmark_skill_ids=["toolbench/a", "toolbench/b"],
    )

    assert local[0]["state_text"] == "state"
    assert local[0]["next_skill_id"] == "tau2/a"
    assert local[0]["visible_inventory_skill_ids"] == ["tau2/a", "tau2/b"]
    assert local[0]["domain"] == "retail"
    assert local[0]["split"] == "test"
    assert local[0]["step_index"] == 7
    assert "replay_prefix" not in local[0]
    assert global_pool[0]["visible_inventory_skill_ids"] == ["toolbench/a", "toolbench/b"]


def test_merge_frozen_benchmark_skills_preserves_training_prefix_and_reports_unseen():
    from clstr.frozen_clstr_route_eval import merge_frozen_benchmark_skills

    merged, report = merge_frozen_benchmark_skills(
        [{"skill_id": "known", "description": "trained"}],
        [
            {"skill_id": "known", "description": "benchmark copy"},
            {"skill_id": "new", "description": "zero shot"},
        ],
    )

    assert [row["skill_id"] for row in merged] == ["known", "new"]
    assert merged[0]["description"] == "trained"
    assert report == {
        "base_skill_count": 1,
        "benchmark_skill_count": 2,
        "known_benchmark_skill_count": 1,
        "appended_benchmark_skill_count": 1,
        "merged_skill_count": 2,
    }


class _FrozenRouteModel:
    def __init__(self):
        self.config = SimpleNamespace(state_query_prompt_version="clstr_causal_state_v1")
        self.encoded_texts: list[str] = []

    def encode_states(self, texts):
        self.encoded_texts.extend(texts)
        values = [0.0 if "query-0" in text else 1.0 for text in texts]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def initial_belief(self, h):
        return h

    def unified_route_logits(self, h, _m, candidate_rows=None):
        full = torch.stack(
            [
                torch.tensor([3.0, 2.0, 1.0]) if int(value.item()) == 0 else torch.tensor([2.0, 1.0, 4.0])
                for value in h[:, 0]
            ]
        )
        if candidate_rows is None:
            return full
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), torch.finfo(full.dtype).min)
        for row_idx, row in enumerate(candidate_rows):
            output[row_idx, : len(row)] = full[row_idx, torch.tensor(row, dtype=torch.long)]
        return output


def test_score_frozen_route_rows_reports_strict_metrics_and_complete_predictions():
    from clstr.frozen_clstr_route_eval import score_frozen_route_rows

    rows = [
        {
            "row_id": "row-0",
            "state_text": "query-0",
            "next_skill_id": "skill/a",
            "visible_inventory_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "domain": "retail",
            "split": "test",
            "step_index": 3,
        },
        {
            "row_id": "row-1",
            "state_text": "query-1",
            "next_skill_id": "skill/c",
            "visible_inventory_skill_ids": ["skill/a", "skill/c"],
        },
    ]
    model = _FrozenRouteModel()
    predictions, report = score_frozen_route_rows(
        model,
        rows,
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        batch_size=2,
        top_k=3,
        device=torch.device("cpu"),
    )

    assert report["status"] == "ok"
    assert report["source_rows"] == 2
    assert report["prediction_rows"] == 2
    assert report["recall@1"] == 1.0
    assert report["recall@5"] == 1.0
    assert report["mrr"] == 1.0
    assert report["memory_active_rows"] == 0
    assert report["zero_history_fallback"] == "exact_static"
    assert [row["positive_rank"] for row in predictions] == [1, 1]
    assert predictions[0]["domain"] == "retail"
    assert predictions[0]["split"] == "test"
    assert predictions[0]["step_index"] == 3
    assert predictions[1]["ranked_skill_ids"] == ["skill/c", "skill/a"]
    assert len(model.encoded_texts) == 2


def test_load_final_chain_model_reconstructs_chain_then_text_encodes_unseen_skill_suffix(
    monkeypatch,
    tmp_path,
):
    import clstr.frozen_clstr_route_eval as route_eval

    paths = _final_chain(tmp_path)
    final_chain = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    appended_skills = [
        {"skill_id": "toolsandbox/b", "description": "appended b"},
        {"skill_id": "toolsandbox/c", "description": "appended c"},
    ]
    calls: list[tuple] = []
    model = _FrozenRouteModel()
    model.to = lambda device: calls.append(("to", str(device))) or model
    model.append_skills = lambda rows: calls.append(
        ("append", [str(row["skill_id"]) for row in rows])
    ) or {
        "old_count": 1,
        "new_count": 3,
        "appended_count": 2,
        "appended_skill_ids": ["toolsandbox/b", "toolsandbox/c"],
        "skipped_duplicate_skill_ids": [],
    }
    model.eval = lambda: calls.append(("eval",)) or model

    def fake_build(
        checkpoint_path,
        skills_path,
        model_cache_dir,
        *,
        allow_skill_table_prefix_expansion=False,
    ):
        calls.append(
            (
                "stage0",
                str(Path(checkpoint_path).resolve()),
                str(Path(skills_path).resolve()),
                bool(allow_skill_table_prefix_expansion),
            )
        )
        return model, {"freeze_backbone": True}, {"stage0_loaded": True}

    def fake_overlay(
        loaded_model,
        checkpoint_path,
        partial_load_mode,
        *,
        protect_routing_foundation,
        allow_skill_table_prefix_expansion,
    ):
        assert loaded_model is model
        calls.append(
            (
                "overlay",
                str(Path(checkpoint_path).resolve()),
                partial_load_mode,
                bool(protect_routing_foundation),
                bool(allow_skill_table_prefix_expansion),
            )
        )
        return {"loaded": True, "partial_load_mode": partial_load_mode}

    monkeypatch.setattr(route_eval, "build_clstr_model_from_stage0_checkpoint", fake_build)
    monkeypatch.setattr(route_eval, "load_head_checkpoint_into_model", fake_overlay)
    router_digests = iter(["stage2-router-digest", "stage2-router-digest"])
    monkeypatch.setattr(
        route_eval,
        "router_state_digest",
        lambda _model, *, scope: next(router_digests),
    )

    loaded, report = route_eval.load_final_chain_model(
        final_chain=final_chain,
        base_skills_path=paths["base_skills"],
        appended_skills=appended_skills,
        model_cache_dir=tmp_path / "model_cache",
        device=torch.device("cpu"),
    )

    assert loaded is model
    assert [call[0] for call in calls] == ["stage0", "overlay", "overlay", "to", "append", "eval"]
    assert calls[0][1] == str(paths["stage0"].resolve())
    assert calls[0][2] == str(paths["base_skills"].resolve())
    assert calls[0][3] is False
    assert calls[1][1] == str(paths["stage2"].resolve())
    assert calls[2][1] == str(paths["stage4"].resolve())
    assert calls[1][3:] == (True, False)
    assert calls[2][3:] == (True, False)
    assert calls[4] == ("append", ["toolsandbox/b", "toolsandbox/c"])
    assert report["checkpoint_load_order"] == ["stage0", "stage2", "stage4"]
    assert report["stage2_router_digest"] == "stage2-router-digest"
    assert report["stage4_router_digest"] == "stage2-router-digest"
    assert report["freeze_backbone"] is True
    assert report["skill_append"]["appended_skill_ids"] == ["toolsandbox/b", "toolsandbox/c"]


def test_load_final_chain_model_rejects_router_key_in_stage4_delta(tmp_path):
    import clstr.frozen_clstr_route_eval as route_eval

    paths = _final_chain(tmp_path)
    stage4_payload = torch.load(paths["stage4"], map_location="cpu")
    stage4_payload["model_state_dict"]["initial_belief_head.weight"] = torch.zeros(
        1,
        1,
    )
    torch.save(stage4_payload, paths["stage4"])
    final_chain = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    final_chain["checkpoints"]["stage4"] = {
        **sha256_path(paths["stage4"]),
        "stage": "stage4",
    }
    final_chain["checkpoint_chain_digest"] = checkpoint_chain_digest(
        {
            role: paths[role]
            for role in ("stage0", "stage1", "stage2", "stage4")
        }
    )
    final_chain["manifest_sha256"] = canonical_digest(
        {
            key: value
            for key, value in final_chain.items()
            if key != "manifest_sha256"
        }
    )

    with pytest.raises(ValueError, match="forbidden Stage4 delta key"):
        route_eval.load_final_chain_model(
            final_chain=final_chain,
            base_skills_path=paths["base_skills"],
            appended_skills=[],
            model_cache_dir=tmp_path / "model_cache",
            device=torch.device("cpu"),
        )


def test_load_final_chain_model_rejects_an_unloaded_stage2_overlay(monkeypatch, tmp_path):
    import clstr.frozen_clstr_route_eval as route_eval

    paths = _final_chain(tmp_path)
    final_chain = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    model = _FrozenRouteModel()
    model.to = lambda _device: model
    model.append_skills = lambda _rows: {
        "appended_count": 0,
        "appended_skill_ids": [],
    }
    model.eval = lambda: model
    monkeypatch.setattr(
        route_eval,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **_kwargs: (model, {"freeze_backbone": True}, {"stage0_loaded": True}),
    )

    def fake_overlay(_model, checkpoint_path, *args, **kwargs):
        return {"loaded": Path(checkpoint_path).resolve() == paths["stage4"].resolve()}

    monkeypatch.setattr(route_eval, "load_head_checkpoint_into_model", fake_overlay)

    with pytest.raises(ValueError, match="Stage2 checkpoint overlay did not load"):
        route_eval.load_final_chain_model(
            final_chain=final_chain,
            base_skills_path=paths["base_skills"],
            appended_skills=[],
            model_cache_dir=tmp_path / "model_cache",
            device=torch.device("cpu"),
        )

    def mismatched_overlay(_model, checkpoint_path, *args, **kwargs):
        if Path(checkpoint_path).resolve() == paths["stage2"].resolve():
            return {
                "loaded": True,
                "shape_mismatched": {
                    "transition.cell.weight_ih": {
                        "checkpoint": [1, 1],
                        "model": [2, 2],
                    }
                },
            }
        return {"loaded": True, "shape_mismatched": {}}

    monkeypatch.setattr(route_eval, "load_head_checkpoint_into_model", mismatched_overlay)
    with pytest.raises(ValueError, match="Stage2 checkpoint overlay has shape mismatches"):
        route_eval.load_final_chain_model(
            final_chain=final_chain,
            base_skills_path=paths["base_skills"],
            appended_skills=[],
            model_cache_dir=tmp_path / "model_cache",
            device=torch.device("cpu"),
        )


def test_run_frozen_route_eval_writes_self_hashed_artifacts_and_resumes_fail_closed(
    monkeypatch,
    tmp_path,
):
    import clstr.frozen_clstr_route_eval as route_eval

    corpus_paths = _frozen_corpus(tmp_path)
    chain_paths = _final_chain(tmp_path)
    output_dir = tmp_path / "eval"
    loader_calls: list[dict] = []

    def fake_load_model(*, final_chain, base_skills_path, appended_skills, model_cache_dir, device):
        loader_calls.append(
            {
                "base": [
                    str(row["skill_id"])
                    for row in route_eval._read_jsonl(base_skills_path, label="base skills")
                ],
                "appended": [str(row["skill_id"]) for row in appended_skills],
            }
        )
        return _FrozenRouteModel(), {
            "checkpoint_load_order": ["stage0", "stage2", "stage4"],
            "freeze_backbone": True,
        }

    monkeypatch.setattr(route_eval, "load_final_chain_model", fake_load_model)
    first = route_eval.run_frozen_clstr_route_eval(
        final_chain_manifest_path=chain_paths["manifest"],
        benchmark_manifest_path=corpus_paths["manifest"],
        source_rows_path=corpus_paths["rows"],
        skills_path=corpus_paths["skills"],
        output_dir=output_dir,
        batch_size=2,
        top_k=3,
        device="cpu",
    )

    assert first["status"] == "ok"
    assert first["metric_scope"] == "next_tool_action_routing"
    assert first["task_success"] is False
    assert first["skill_merge"]["known_benchmark_skill_count"] == 1
    assert first["skill_merge"]["appended_benchmark_skill_count"] == 2
    assert loader_calls == [
        {
            "base": ["toolsandbox/a"],
            "appended": ["toolsandbox/b", "toolsandbox/c"],
        }
    ]
    assert first["checkpoint_load"]["checkpoint_load_order"] == ["stage0", "stage2", "stage4"]

    identity = json.loads((output_dir / "evaluation_identity.json").read_text(encoding="utf-8"))
    identity_sha256 = identity.pop("identity_sha256")
    assert canonical_digest(identity) == identity_sha256
    report = json.loads((output_dir / "frozen_route_eval_report.json").read_text(encoding="utf-8"))
    report_sha256 = report.pop("report_sha256")
    assert canonical_digest(report) == report_sha256
    prediction_rows = route_eval._read_jsonl(
        output_dir / "frozen_route_predictions.jsonl",
        label="frozen route predictions",
    )
    assert len(prediction_rows) == 2
    assert [row["row_id"] for row in prediction_rows] == ["row-0", "row-1"]
    for row in prediction_rows:
        prediction_sha256 = row.pop("prediction_sha256")
        assert row["evaluation_identity_sha256"] == identity_sha256
        assert canonical_digest(row) == prediction_sha256

    monkeypatch.setattr(
        route_eval,
        "load_final_chain_model",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("resume must not reload the model")),
    )
    resumed = route_eval.run_frozen_clstr_route_eval(
        final_chain_manifest_path=chain_paths["manifest"],
        benchmark_manifest_path=corpus_paths["manifest"],
        source_rows_path=corpus_paths["rows"],
        skills_path=corpus_paths["skills"],
        output_dir=output_dir,
        batch_size=2,
        top_k=3,
        device="cpu",
    )
    assert resumed["resume"]["used"] is True
    assert resumed["resume"]["prediction_rows"] == 2

    with pytest.raises(ValueError, match="existing evaluation identity does not match"):
        route_eval.run_frozen_clstr_route_eval(
            final_chain_manifest_path=chain_paths["manifest"],
            benchmark_manifest_path=corpus_paths["manifest"],
            source_rows_path=corpus_paths["rows"],
            skills_path=corpus_paths["skills"],
            output_dir=output_dir,
            batch_size=2,
            top_k=2,
            device="cpu",
        )

    with pytest.raises(ValueError, match="final-chain manifest identity mismatch"):
        route_eval.run_frozen_clstr_route_eval(
            final_chain_manifest_path=chain_paths["manifest"],
            benchmark_manifest_path=corpus_paths["manifest"],
            source_rows_path=corpus_paths["rows"],
            skills_path=corpus_paths["skills"],
            output_dir=tmp_path / "wrong_chain_identity",
            batch_size=2,
            top_k=3,
            device="cpu",
            expected_final_chain_manifest_sha256="0" * 64,
            expected_checkpoint_chain_digest="0" * 64,
        )


def test_frozen_route_eval_cli_forwards_pinned_chain_and_corpus_inputs(monkeypatch):
    import scripts.run_frozen_clstr_route_eval as cli

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "benchmark": "tau2"}

    monkeypatch.setattr(cli, "run_frozen_clstr_route_eval", fake_run)
    args = cli.build_parser().parse_args(
        [
            "--final_chain_manifest_path",
            "/run/final_chain.json",
            "--benchmark_manifest_path",
            "/frozen/tau2/manifest.json",
            "--source_rows_path",
            "/frozen/tau2/source_rows.jsonl",
            "--skills_path",
            "/frozen/tau2/skills.jsonl",
            "--output_dir",
            "/outputs/tau2",
            "--expected_benchmark_manifest_sha256",
            "abc123",
            "--expected_final_chain_manifest_sha256",
            "def456",
            "--expected_checkpoint_chain_digest",
            "ghi789",
            "--batch_size",
            "8",
            "--top_k",
            "20",
            "--max_eval_rows",
            "64",
            "--device",
            "cuda",
        ]
    )

    report = cli.run_from_args(args)

    assert report["status"] == "ok"
    assert captured == {
        "final_chain_manifest_path": "/run/final_chain.json",
        "benchmark_manifest_path": "/frozen/tau2/manifest.json",
        "source_rows_path": "/frozen/tau2/source_rows.jsonl",
        "skills_path": "/frozen/tau2/skills.jsonl",
        "output_dir": "/outputs/tau2",
        "batch_size": 8,
        "top_k": 20,
        "max_eval_rows": 64,
        "device": "cuda",
        "expected_benchmark_manifest_sha256": "abc123",
        "expected_final_chain_manifest_sha256": "def456",
        "expected_checkpoint_chain_digest": "ghi789",
    }
