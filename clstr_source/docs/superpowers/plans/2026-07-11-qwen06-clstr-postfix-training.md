# Qwen3-Embedding-0.6B CLSTR Post-Fix Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking. Do not delegate this plan unless the user explicitly requests subagents.

**Goal:** Build a safe, test-covered Qwen3-Embedding-0.6B CLSTR training chain with a frozen backbone, unified-memory Stage0, segmented quality gates, fresh Stage1/2/4 checkpoints, and post-fix reliability evaluation.

**Architecture:** Keep the existing CLSTR trainers as the implementation core. Add a stable Stage0 step-0 checkpoint, small experiment-specific lineage and promotion-gate modules, and thin Qwen launchers that hard-code the approved method choices while rejecting backbone training and stale checkpoints. All model/data-heavy commands run through Slurm; the storage node runs only focused tests and syntax checks.

**Tech Stack:** Python 3.10, PyTorch, Transformers, pytest, Bash, Slurm, JSON/JSONL, existing CLSTR Stage0/1/2/4 and evaluation modules.

---

## Required Prerequisite

Before Task 1, execute `docs/superpowers/plans/2026-07-11-clstr-qwen-prompt-consistency.md` completely. The Qwen run must use `clstr_causal_state_v1` with `state_query_max_chars=2000`, `state_query_truncation=head_tail_v1`, and Stage0 handoff mode `checkpoint_state_query`. No SR-prompt control run is part of this experiment.

---

## Fixed Paths and Run Layout

Use these paths throughout the implementation:

~~~text
WORKTREE=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix
ASSET_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
PYTHON=/data/home/scyb713/run/miniconda3/envs/xzf/bin/python
MODEL=$ASSET_ROOT/models/Qwen3-Embedding-0.6B
DATA=$ASSET_ROOT/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2
RUN_ROOT=$WORKTREE/outputs/qwen06_clstr_postfix
~~~

Canonical outputs:

~~~text
$RUN_ROOT/stage0_smoke
$RUN_ROOT/stage0_full
$RUN_ROOT/stage0_audits/step0
$RUN_ROOT/stage0_audits/step1200
$RUN_ROOT/stage0_audits/step2400
$RUN_ROOT/stage0_audits/step3600
$RUN_ROOT/stage0_audits/step5000
$RUN_ROOT/stage1_smoke
$RUN_ROOT/stage1_full
$RUN_ROOT/stage2_smoke
$RUN_ROOT/stage2_full
$RUN_ROOT/stage4_smoke
$RUN_ROOT/stage4_full
$RUN_ROOT/eval
$RUN_ROOT/memory_audit
~~~

## File Map

- clstr/retrieval_warmup.py: persist the post-rebuild Stage0 step-0 checkpoint.
- clstr/qwen_clstr_lineage.py: deterministic model/data/checkpoint lineage and frozen-backbone validation.
- clstr/qwen_clstr_stage0_gate.py: smoke, segment-promotion, release-floor, and best-checkpoint decisions.
- scripts/audit_qwen06_clstr_lineage.py: create and validate lineage manifests.
- scripts/audit_qwen06_clstr_stage0_gate.py: produce machine-readable Stage0 gate reports.
- scripts/sbatch/run_qwen06_clstr_stage0_train.sh: safe Stage0 smoke and segmented full training.
- scripts/sbatch/run_qwen06_clstr_stage0_audit.sh: fixed-row handoff audits and promotion gates.
- scripts/sbatch/run_qwen06_clstr_stage1_train.sh: Stage1 smoke/full plus quality and lineage gates.
- scripts/sbatch/run_qwen06_clstr_stage2_train.sh: Stage2 smoke/full plus quality and lineage gates.
- scripts/sbatch/run_qwen06_clstr_stage4_train.sh: Stage4 smoke/full plus quality and lineage gates.
- clstr/toolbench_full_clstr_route_eval.py and clstr/global_pool_route_eval.py: reliability-mode pass-through.
- scripts/run_toolbench_g3_full_clstr_route_eval.py and scripts/run_global_pool_clstr_route_eval.py: expose safe reliability modes.
- scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh and scripts/sbatch/run_global_pool_clstr_route_eval.sh: expose matching environment variables.
- tests/test_stage0_biencoder_protocol.py: stable step-0 checkpoint test.
- tests/test_qwen_clstr_lineage.py: identity and mismatch tests.
- tests/test_qwen_clstr_stage0_gate.py: promotion/release tests.
- tests/test_qwen_clstr_training_launchers.py: launcher defaults and hard guards.
- tests/test_toolbench_full_clstr_route_eval.py and tests/test_global_pool_route_eval.py: reliability pass-through tests.

---

### Task 1: Persist a Stable Post-Rebuild Stage0 Step-0 Checkpoint

**Files:**
- Modify: clstr/retrieval_warmup.py
- Modify: tests/test_stage0_biencoder_protocol.py

- [x] **Step 1: Write the failing named-checkpoint test**

Add this import and test:

~~~python
from clstr.retrieval_warmup import _save_named_stage0_checkpoint


def test_stage0_named_checkpoint_persists_step_zero_optimizer_and_frozen_metadata(tmp_path):
    payload = {
        "stage": "clstr_unified_retrieval_v2",
        "step": 0,
        "setup_phase": "training_started",
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "checkpoint_excludes_frozen_backbone": True,
        "config": {
            "base_model_name": "models/Qwen3-Embedding-0.6B",
            "freeze_backbone": True,
            "route_scorer": "unified_memory",
            "state_query_prompt_version": "clstr_causal_state_v1",
            "state_query_instruction": (
                "Given an agent task or current execution state and interaction history, "
                "retrieve the skill or tool document most useful for the next action."
            ),
            "state_query_max_chars": 2000,
            "state_query_truncation": "head_tail_v1",
        },
    }

    path = _save_named_stage0_checkpoint(
        checkpoint_dir=tmp_path,
        stage_name="clstr_unified_retrieval_v2",
        step=0,
        payload=payload,
    )

    loaded = torch.load(path, map_location="cpu")
    assert path.name == "clstr_unified_retrieval_v2-step0.pt"
    assert loaded["step"] == 0
    assert loaded["optimizer_state_dict"] == payload["optimizer_state_dict"]
    assert loaded["checkpoint_excludes_frozen_backbone"] is True
~~~

- [x] **Step 2: Run the focused test and confirm RED**

Run:

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_biencoder_protocol.py::test_stage0_named_checkpoint_persists_step_zero_optimizer_and_frozen_metadata
~~~

Expected: import failure because _save_named_stage0_checkpoint does not exist.

- [x] **Step 3: Add the named-checkpoint helper**

Add near the existing Stage0 checkpoint helpers:

~~~python
def _save_named_stage0_checkpoint(
    *,
    checkpoint_dir: str | Path,
    stage_name: str,
    step: int,
    payload: dict[str, Any],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{stage_name}-step{int(step)}.pt"
    torch.save(payload, path)
    return path
~~~

- [x] **Step 4: Reuse one payload for latest and stable step zero**

Lift the existing repeated step/final config dictionary into checkpoint_config before the initial payload:

~~~python
    checkpoint_config = {
        "base_model_name": base_model_name,
        "d": model_dim,
        "d_a": max(4, model_dim // 4),
        "top_k": top_k,
        "encoder_pooling": encoder_pooling,
        "cross_encoder_pooling": cross_encoder_pooling,
        "tokenizer_padding_side": tokenizer_padding_side,
        "torch_dtype": torch_dtype,
        "freeze_backbone": freeze_backbone,
        "max_length": max_length,
        "projection_init": projection_init,
        "normalize_embeddings": normalize_embeddings,
        "defer_skill_table_init": True,
        "skill_text_format": skill_text_format,
        "skill_table_batch_size": skill_table_batch_size,
        "skill_table_adapter_init": skill_table_adapter_init,
        "use_cross_encoder": use_cross_encoder,
        "query_text_format": query_text_format,
        "data_format": data_format,
        "metric_recall_ks": list(recall_ks),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "retrieval_loss_mode": retrieval_loss_mode,
        "route_scorer": route_scorer,
        "belief_top_k": belief_top_k,
        "state_query_prompt_version": state_query_contract["state_query_prompt_version"],
        "state_query_instruction": state_query_contract["state_query_instruction"],
        "state_query_max_chars": state_query_contract["state_query_max_chars"],
        "state_query_truncation": state_query_contract["state_query_truncation"],
        "sampling_strategy": sampling_strategy,
        "train_skill_embeddings": bool(train_skill_embeddings),
        "train_skill_bias": bool(train_skill_bias),
        "train_encoder_backbone": bool(train_encoder_backbone),
        "train_encoder_projection": bool(train_encoder_projection),
        "train_skill_adapter": bool(train_skill_adapter),
        "train_retrieval_scale": bool(train_retrieval_scale),
        "explicit_negative_loss_weight": explicit_negative_loss_weight,
        "mined_hard_negative_loss_weight": mined_hard_negative_loss_weight,
        "mined_hard_negative_margin": mined_hard_negative_margin,
        "mined_hard_negative_top_k": mined_hard_negative_top_k,
    }
~~~

Use checkpoint_config in the initial, per-step, and final checkpoint payloads so the three formats cannot drift. Replace the inline initial monitor payload with:

~~~python
    initial_checkpoint_payload = {
        "stage": stage_name,
        "step": int(start_step - 1),
        "setup_phase": "training_started",
        "config": checkpoint_config,
        "model_state_dict": initial_checkpoint_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "train_safety": train_safety,
        "setup_status_path": str(setup_status_path),
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "retrieval_loss_mode": retrieval_loss_mode,
        "route_scorer": route_scorer,
        "belief_top_k": belief_top_k,
        "resume": {
            "enabled": bool(resume_state is not None),
            "checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "start_step": int(start_step),
        },
        "stage0_protocol": protocol_metadata if protocol_metadata.get("role") else {},
        "protocol_metadata": protocol_metadata,
        "sampling_strategy": sampling_strategy,
        "tempered_correction_fraction": tempered_correction_fraction,
        "training_bucket_count": len(training_bucket_counts),
        "training_bucket_counts": training_bucket_counts,
        "shuffle_queries": bool(shuffle_queries),
        "alias_positive_expansion": alias_positive_report,
        "explicit_negative_supervision": explicit_negative_report,
        "mined_hard_negative_supervision": mined_hard_negative_report,
        "init_checkpoint": init_checkpoint_report,
        "preservation_anchor": anchor_setup_report,
        "trainable_parameter_policy": {
            "trains_encoder_backbone": bool(
                any(param.requires_grad for param in model.encoder.backbone.parameters())
            ),
            "optimizer_parameter_names": optimizer_parameter_names,
        },
        **initial_checkpoint_state_report,
    }
    initial_checkpoint_path = None
    if resume_state is None and start_step == 1:
        initial_checkpoint_path = _save_named_stage0_checkpoint(
            checkpoint_dir=checkpoint_dir,
            stage_name=stage_name,
            step=0,
            payload=initial_checkpoint_payload,
        )
    monitor.save_latest(initial_checkpoint_payload)
~~~

Add to the final report:

~~~python
"initial_checkpoint_path": (
    None if initial_checkpoint_path is None else str(initial_checkpoint_path)
),
~~~

- [x] **Step 5: Run Stage0 protocol tests**

Run:

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_biencoder_protocol.py
~~~

Expected: all Stage0 protocol tests pass.

- [x] **Step 6: Commit**

~~~bash
git add clstr/retrieval_warmup.py tests/test_stage0_biencoder_protocol.py
git commit -m "feat: persist stage0 initialization checkpoint"
~~~

---

### Task 2: Add Qwen CLSTR Lineage and Frozen-Backbone Validation

**Files:**
- Create: clstr/qwen_clstr_lineage.py
- Create: scripts/audit_qwen06_clstr_lineage.py
- Create: tests/test_qwen_clstr_lineage.py

- [x] **Step 1: Write lineage RED tests**

Create these fixtures and tests:

~~~python
import json
from pathlib import Path

import pytest
import torch

from clstr.qwen_clstr_lineage import (
    create_lineage_manifest,
    validate_lineage_manifest,
    validate_qwen_clstr_checkpoint,
)


def _write_identity_inputs(tmp_path: Path, *, frozen: bool = True):
    model = tmp_path / "Qwen3-Embedding-0.6B"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}\n')
    pool = tmp_path / "skill_pool.jsonl"
    pool.write_text('{"skill_id":"skill/a"}\n')
    data_manifest = tmp_path / "manifest.json"
    data_manifest.write_text('{"status":"ok","skill_count":1}\n')
    checkpoint = tmp_path / "stage0.pt"
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "config": {
                "base_model_name": str(model),
                "freeze_backbone": frozen,
                "route_scorer": "unified_memory",
                "state_query_prompt_version": "clstr_causal_state_v1",
                "state_query_instruction": (
                    "Given an agent task or current execution state and interaction history, "
                    "retrieve the skill or tool document most useful for the next action."
                ),
                "state_query_max_chars": 2000,
                "state_query_truncation": "head_tail_v1",
            },
            "trainable_parameter_policy": {
                "trains_encoder_backbone": not frozen,
            },
            "model_state_dict": {
                "skill_table.E": torch.zeros(1, 4),
            },
        },
        checkpoint,
    )
    return model, pool, data_manifest, checkpoint


def test_lineage_manifest_records_qwen_model_pool_data_and_checkpoint_hashes(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    manifest = create_lineage_manifest(
        stage="stage0",
        checkpoint_path=checkpoint,
        expected_checkpoint_stage="clstr_unified_retrieval_v2",
        model_path=model,
        skill_pool_path=pool,
        data_manifest_path=data_manifest,
        parent_manifests={},
    )
    assert manifest["schema_version"] == "qwen06_clstr_lineage_v1"
    assert manifest["backbone"]["frozen"] is True
    assert manifest["route_scorer"] == "unified_memory"
    assert manifest["state_query"]["prompt_version"] == "clstr_causal_state_v1"
    assert manifest["checkpoint"]["sha256"]


def test_lineage_rejects_trainable_backbone_checkpoint(tmp_path):
    model, _pool, _data_manifest, checkpoint = _write_identity_inputs(
        tmp_path,
        frozen=False,
    )
    with pytest.raises(ValueError, match="backbone must remain frozen"):
        validate_qwen_clstr_checkpoint(
            checkpoint,
            expected_stage="clstr_unified_retrieval_v2",
            expected_model_path=model,
        )


def test_lineage_rejects_skill_pool_mismatch(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={},
            )
        )
    )
    pool.write_text('{"skill_id":"skill/changed"}\n')
    with pytest.raises(ValueError, match="skill pool digest mismatch"):
        validate_lineage_manifest(
            manifest_path,
            expected_stage="stage0",
            expected_parent_manifests={},
        )


def test_lineage_rejects_prompt_contract_mismatch(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["config"]["state_query_max_chars"] = 1500
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="state query prompt contract mismatch"):
        create_lineage_manifest(
            stage="stage0",
            checkpoint_path=checkpoint,
            expected_checkpoint_stage="clstr_unified_retrieval_v2",
            model_path=model,
            skill_pool_path=pool,
            data_manifest_path=data_manifest,
            parent_manifests={},
        )
~~~

Use tiny files and tiny torch checkpoints so the tests do not load any real model.

- [x] **Step 2: Run the new tests and confirm RED**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_lineage.py
~~~

Expected: module import failure.

- [x] **Step 3: Implement deterministic path identities**

Create these public functions:

~~~python
SCHEMA_VERSION = "qwen06_clstr_lineage_v1"
QWEN_MODEL_BASENAME = "Qwen3-Embedding-0.6B"


def sha256_path(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    files = [resolved] if resolved.is_file() else sorted(
        item for item in resolved.rglob("*") if item.is_file()
    )
    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.name if resolved.is_file() else str(item.relative_to(resolved))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size": size,
        "file_count": len(files),
    }
~~~

- [x] **Step 4: Implement checkpoint safety extraction**

~~~python
def validate_qwen_clstr_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_stage: str,
    expected_model_path: str | Path,
    require_unified_memory: bool = True,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dict")
    config = dict(payload.get("config") or payload.get("model_config") or {})
    protocol = dict(payload.get("stage0_protocol") or payload.get("protocol_metadata") or {})
    model_path = str(config.get("base_model_name") or "")
    if Path(model_path).resolve() != Path(expected_model_path).resolve():
        raise ValueError("Qwen backbone identity mismatch")
    if config.get("freeze_backbone") is not True:
        raise ValueError("Qwen backbone must remain frozen")
    trainable = dict(payload.get("trainable_parameter_policy") or {})
    if trainable.get("trains_encoder_backbone") is True:
        raise ValueError("Qwen backbone must remain frozen")
    route_scorer = str(
        payload.get("route_scorer")
        or config.get("route_scorer")
        or protocol.get("route_scorer")
        or ""
    )
    if require_unified_memory and route_scorer != "unified_memory":
        raise ValueError("Qwen CLSTR requires route_scorer=unified_memory")
    state = payload.get("model_state_dict") or {}
    if any(str(key).startswith("encoder.backbone.") for key in state):
        raise ValueError("frozen encoder backbone unexpectedly stored in checkpoint")
    if str(payload.get("stage") or "") != expected_stage:
        raise ValueError("checkpoint stage mismatch")
    prompt_contract = resolve_state_query_prompt_contract(
        prompt_version=str(config.get("state_query_prompt_version") or ""),
        max_chars=config.get("state_query_max_chars"),
        truncation=config.get("state_query_truncation"),
        recorded_instruction=config.get("state_query_instruction"),
    )
    expected_prompt = resolve_state_query_prompt_contract(
        prompt_version="clstr_causal_state_v1",
        max_chars=2000,
        truncation="head_tail_v1",
    )
    if prompt_contract != expected_prompt:
        raise ValueError("state query prompt contract mismatch")
    return {
        "stage": expected_stage,
        "route_scorer": route_scorer,
        "backbone_frozen": True,
        **prompt_contract,
        "checkpoint": sha256_path(checkpoint_path),
    }
~~~

- [x] **Step 5: Implement create and validate lineage manifests**

The manifest must have this exact shape:

~~~python
{
    "schema_version": SCHEMA_VERSION,
    "stage": stage,
    "route_scorer": "unified_memory",
    "state_query": {
        "prompt_version": checkpoint_report["state_query_prompt_version"],
        "instruction": checkpoint_report["state_query_instruction"],
        "max_chars": checkpoint_report["state_query_max_chars"],
        "truncation": checkpoint_report["state_query_truncation"],
    },
    "backbone": {**sha256_path(model_path), "frozen": True},
    "skill_pool": sha256_path(skill_pool_path),
    "data_manifest": sha256_path(data_manifest_path),
    "checkpoint": sha256_path(checkpoint_path),
    "parents": {
        role: {
            "manifest_path": str(path.resolve()),
            "manifest_sha256": sha256_path(path)["sha256"],
            "checkpoint_sha256": loaded["checkpoint"]["sha256"],
        }
        for role, path in parent_manifests.items()
    },
}
~~~

Validation must compare actual files with all recorded hashes and reject:

- non-Qwen model identity;
- non-unified route scorer;
- any prompt-version, instruction, maximum-length, or truncation mismatch;
- frozen=false;
- changed skill pool;
- changed checkpoint;
- wrong parent role or parent checkpoint digest;
- unsupported schema version.

- [x] **Step 6: Add CLI create and validate subcommands**

The CLI surfaces are:

~~~text
audit_qwen06_clstr_lineage.py create
  --stage
  --checkpoint_path
  --expected_checkpoint_stage
  --model_path
  --skill_pool_path
  --data_manifest_path
  --parent role=manifest.json
  --output_path

audit_qwen06_clstr_lineage.py validate
  --manifest_path
  --expected_stage
  --expected_parent role=manifest.json
~~~

Return 0 for status ok and 2 for validation failure.

- [x] **Step 7: Run tests and commit**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_lineage.py
git add clstr/qwen_clstr_lineage.py scripts/audit_qwen06_clstr_lineage.py \
  tests/test_qwen_clstr_lineage.py
git commit -m "feat: validate qwen clstr checkpoint lineage"
~~~

---

### Task 3: Implement Stage0 Smoke, Promotion, Release, and Selection Gates

**Files:**
- Create: clstr/qwen_clstr_stage0_gate.py
- Create: scripts/audit_qwen06_clstr_stage0_gate.py
- Create: tests/test_qwen_clstr_stage0_gate.py

- [x] **Step 1: Write RED tests for all decision branches**

Create these helpers and tests:

~~~python
from clstr.qwen_clstr_stage0_gate import (
    audit_stage0_promotion,
    audit_stage0_release,
    audit_stage0_smoke,
    select_stage0_checkpoint,
)


def _handoff(global_100, global_500, tool_100, tool_500, traj_100, traj_200, traj_500):
    return {
        "query_modes": {
            "checkpoint_state_query": {
                "global": {
                    "next_recall@100": global_100,
                    "next_recall@500": global_500,
                },
                "benchmarks": {
                    "toolbench_g3": {
                        "next_recall@100": tool_100,
                        "next_recall@500": tool_500,
                    },
                    "traject_bench": {
                        "next_recall@100": traj_100,
                        "next_recall@200": traj_200,
                        "next_recall@500": traj_500,
                    },
                },
            }
        }
    }


def test_smoke_gate_requires_finite_loss_mined_pairs_and_frozen_unified_route():
    report = {
        "status": "ok",
        "metrics": {
            "loss": 1.0,
            "mined_hard_negative_rows": 8,
            "mined_hard_negative_pair_count": 256,
        },
        "stage0_protocol": {
            "route_scorer": "unified_memory",
            "state_query_prompt_version": "clstr_causal_state_v1",
            "state_query_max_chars": 2000,
            "state_query_truncation": "head_tail_v1",
        },
        "trainable_parameter_policy": {
            "trains_encoder_backbone": False,
            "optimizer_parameter_names": [
                "initial_belief_head.weight",
                "unified_retriever.query.weight",
            ],
        },
    }
    gate = audit_stage0_smoke(report, {"status": "ok"})
    assert gate["status"] == "ok"


def test_step1200_gate_requires_global_gain_and_hard_domain_safety():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    current = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.68)
    gate = audit_stage0_promotion(
        baseline_report=baseline,
        current_report=current,
        previous_report=baseline,
        target_step=1200,
    )
    assert gate["status"] == "ok"


def test_later_segment_gate_uses_composite_or_hard_domain_gain():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    previous = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.69)
    current = _handoff(0.75, 0.90, 0.58, 0.83, 0.37, 0.53, 0.69)
    gate = audit_stage0_promotion(
        baseline_report=baseline,
        current_report=current,
        previous_report=previous,
        target_step=2400,
    )
    assert gate["status"] == "ok"


def test_release_gate_marks_small_floor_miss_borderline_and_large_miss_failed():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    borderline = _handoff(0.76, 0.91, 0.60, 0.83, 0.40, 0.51, 0.69)
    failed = _handoff(0.76, 0.91, 0.60, 0.80, 0.40, 0.48, 0.65)
    assert audit_stage0_release(baseline, borderline, "step1200.pt")["status"] == "borderline"
    assert audit_stage0_release(baseline, failed, "step1200.pt")["status"] == "action_required"


def test_checkpoint_selection_chooses_highest_safe_composite():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    step1200 = _handoff(0.76, 0.91, 0.60, 0.85, 0.42, 0.54, 0.72)
    step2400 = _handoff(0.78, 0.92, 0.62, 0.86, 0.44, 0.56, 0.74)
    selection = select_stage0_checkpoint(
        baseline_report=baseline,
        candidates=[
            {
                "step": 1200,
                "report": step1200,
                "checkpoint_path": "step1200.pt",
                "lineage_manifest_path": "lineage-step1200.json",
            },
            {
                "step": 2400,
                "report": step2400,
                "checkpoint_path": "step2400.pt",
                "lineage_manifest_path": "lineage-step2400.json",
            },
        ],
    )
    assert selection["status"] == "ok"
    assert selection["selected_step"] == 2400
    assert selection["selected_checkpoint_path"] == "step2400.pt"
    assert selection["selected_lineage_path"] == "lineage-step2400.json"
~~~

- [x] **Step 2: Run tests and confirm RED**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_stage0_gate.py
~~~

- [x] **Step 3: Implement metric extraction**

~~~python
BENCHMARKS = ("global", "toolbench_g3", "traject_bench")


def handoff_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    mode = dict((report.get("query_modes") or {}).get("checkpoint_state_query") or {})
    benchmarks = dict(mode.get("benchmarks") or {})
    return {
        "global": dict(mode.get("global") or {}),
        "toolbench_g3": dict(benchmarks.get("toolbench_g3") or {}),
        "traject_bench": dict(benchmarks.get("traject_bench") or {}),
    }


def primary_composite(metrics: dict[str, dict[str, float]]) -> float:
    return sum(float(metrics[name]["next_recall@100"]) for name in BENCHMARKS) / 3.0
~~~

- [x] **Step 4: Implement the exact approved rules**

Smoke status is ok only when:

- status is ok;
- loss is finite;
- mined_hard_negative_rows and pair_count are positive;
- route_scorer is unified_memory;
- prompt version is clstr_causal_state_v1 with max chars 2000 and head_tail_v1;
- trains_encoder_backbone is false;
- optimizer names contain initial_belief_head and unified_retriever;
- optimizer names do not contain encoder.backbone.

Step 1200 status is ok only when:

~~~python
global_100 > baseline_global_100
global_500 > baseline_global_500
toolbench_500 >= baseline_toolbench_500 - 0.01
traject_500 >= baseline_traject_500 - 0.01
(
    (toolbench_100 > baseline_toolbench_100 and toolbench_500 > baseline_toolbench_500)
    or
    (traject_100 > baseline_traject_100 and traject_500 > baseline_traject_500)
)
~~~

Later segments are ok only when:

~~~python
composite_gain >= 0.005 or max(hard_domain_next100_gains) >= 0.01
~~~

and every next_recall@500 regression is no worse than -0.01.

Release floors are:

~~~python
global.next_recall@500 >= 0.90
toolbench_g3.next_recall@500 >= 0.84
traject_bench.next_recall@500 >= 0.70
traject_bench.next_recall@200 >= 0.52
~~~

A miss smaller than 0.02 is borderline. A larger miss is action_required.

- [x] **Step 5: Implement CLI subcommands**

~~~text
smoke --train_report_path --lineage_manifest_path --output_path
promote --baseline_report --current_report --previous_report --target_step --output_path
release --baseline_report --current_report --checkpoint_path --output_path
select --baseline_report \
  --candidate 1200=step1200/report.json=checkpoints/stage0-step1200.pt=lineage-step1200.json \
  --candidate 2400=step2400/report.json=checkpoints/stage0-step2400.pt=lineage-step2400.json \
  --output_path stage0_selection.json
~~~

The select command considers only release-safe checkpoints and chooses the highest primary composite, breaking ties in favor of the smaller step. Its output must include selected_step, selected_checkpoint_path, selected_report_path, and selected_lineage_path.

- [x] **Step 6: Run tests and commit**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_stage0_gate.py
git add clstr/qwen_clstr_stage0_gate.py scripts/audit_qwen06_clstr_stage0_gate.py \
  tests/test_qwen_clstr_stage0_gate.py
git commit -m "feat: gate qwen clstr stage0 continuation"
~~~

---

### Task 4: Add Safe Qwen Stage0 Training and Audit Launchers

**Files:**
- Modify: scripts/sbatch/_clstr_gpu_env.sh
- Create: scripts/sbatch/run_qwen06_clstr_stage0_train.sh
- Create: scripts/sbatch/run_qwen06_clstr_stage0_audit.sh
- Create: tests/test_qwen_clstr_training_launchers.py
- Modify: tests/test_sbatch_scripts.py

- [x] **Step 1: Write launcher RED tests**

Assert that Stage0 launcher:

~~~python
text = Path("scripts/sbatch/run_qwen06_clstr_stage0_train.sh").read_text()
assert "FULL_RUN=${FULL_RUN:-0}" in text
assert "models/Qwen3-Embedding-0.6B" in text
assert "ROUTE_SCORER=unified_memory" in text
assert "BELIEF_TOP_K=64" in text
assert "STATE_QUERY_PROMPT_VERSION=clstr_causal_state_v1" in text
assert "STATE_QUERY_MAX_CHARS=2000" in text
assert "STATE_QUERY_TRUNCATION=head_tail_v1" in text
assert "MINED_HARD_NEGATIVE_LOSS_WEIGHT=0.2" in text
assert "MINED_HARD_NEGATIVE_MARGIN=0.1" in text
assert "MINED_HARD_NEGATIVE_TOP_K=32" in text
assert "EXPLICIT_NEGATIVE_LOSS_WEIGHT=0.0" in text
assert "TRAIN_ENCODER_BACKBONE=0" in text
assert "--train_encoder_backbone" not in text
assert "TARGET_STEP" in text
assert "PREVIOUS_GATE_PATH" in text
~~~

Also assert the audit launcher uses 2048 rows, top-k 20/50/100/200/500, checkpoint_state_query, and the promotion CLI.

- [x] **Step 2: Run tests and confirm RED**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_training_launchers.py
~~~

- [x] **Step 3: Implement the Stage0 launcher**

The launcher must:

- derive PROJECT_ROOT from its own location;
- source scripts/sbatch/_clstr_gpu_env.sh;
- use ASSET_ROOT for model/data;
- default to an 80-step, 8192-skill, 4096-query smoke;
- require FULL_RUN=1 for uncapped training;
- allow TARGET_STEP only in 1200, 2400, 3600, 5000;
- require no resume checkpoint at 1200;
- require the prior explicit checkpoint and an ok PREVIOUS_GATE_PATH after 1200;
- hard-set unified memory, frozen backbone, and mined-hard-negative parameters;
- write stdout.log, train_report.json, a lineage manifest, and smoke_gate.json.

Core environment passed to the generic launcher:

~~~bash
export DATA_ROOT="$ASSET_ROOT/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2"
export MODEL_NAME_OR_PATH="$ASSET_ROOT/models/Qwen3-Embedding-0.6B"
export OUTPUT_DIR
export MAX_STEPS="$TARGET_STEP"
export BATCH_SIZE=128
export GRADIENT_ACCUMULATION_STEPS=2
export MODEL_DIM=1024
export TOP_K=100
export LEARNING_RATE=2.0e-5
export TORCH_DTYPE=bfloat16
export ENCODER_POOLING=last_token
export CROSS_ENCODER_POOLING=last_token
export TOKENIZER_PADDING_SIDE=left
export STATE_QUERY_PROMPT_VERSION=clstr_causal_state_v1
export STATE_QUERY_MAX_CHARS=2000
export STATE_QUERY_TRUNCATION=head_tail_v1
export ROUTE_SCORER=unified_memory
export BELIEF_TOP_K=64
export RETRIEVAL_LOSS_MODE=multi_positive_nll
export SAMPLING_STRATEGY=handoff_balanced
export EXPAND_ALIAS_POSITIVES=1
export TRAIN_SKILL_EMBEDDINGS=0
export TRAIN_SKILL_BIAS=1
export TRAIN_ENCODER_BACKBONE=0
export TRAIN_ENCODER_PROJECTION=1
export TRAIN_SKILL_ADAPTER=1
export TRAIN_RETRIEVAL_SCALE=1
export EXPLICIT_NEGATIVE_LOSS_WEIGHT=0.0
export MINED_HARD_NEGATIVE_LOSS_WEIGHT=0.2
export MINED_HARD_NEGATIVE_MARGIN=0.1
export MINED_HARD_NEGATIVE_TOP_K=32
export INIT_CHECKPOINT_PATH=
export INIT_CHECKPOINT_SKILLS_PATH=
export RESUME_CHECKPOINT_PATH
~~~

Then run:

~~~bash
bash scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  2>&1 | tee "$OUTPUT_DIR/stdout.log"
~~~

Create lineage manifests for step 0 and TARGET_STEP. In smoke mode run the smoke gate immediately and exit nonzero if it is not ok.

- [x] **Step 4: Implement the fixed audit launcher**

The audit command is:

~~~bash
"$PYTHON_BIN" scripts/audit_clstr_stage0_handoff_coverage.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --train_path "$ASSET_ROOT/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/trajectories.jsonl" \
  --skills_path "$ASSET_ROOT/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl" \
  --output_path "$OUTPUT_DIR/report.json" \
  --top_k_values "20,50,100,200,500" \
  --query_modes checkpoint_state_query \
  --max_rows 2048 \
  --batch_size 8 \
  --model_cache_dir "$OUTPUT_DIR/model_cache"
~~~

For target step 1200, generate both step0 and step1200 reports, then call promote with step0 as baseline. For later steps, require BASELINE_REPORT and PREVIOUS_REPORT and call promote.

- [x] **Step 5: Run launcher tests and shell syntax**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_training_launchers.py
bash -n scripts/sbatch/run_qwen06_clstr_stage0_train.sh
bash -n scripts/sbatch/run_qwen06_clstr_stage0_audit.sh
~~~

- [x] **Step 6: Commit**

~~~bash
git add scripts/sbatch/run_qwen06_clstr_stage0_train.sh \
  scripts/sbatch/run_qwen06_clstr_stage0_audit.sh \
  tests/test_qwen_clstr_training_launchers.py
git commit -m "feat: launch frozen qwen clstr stage0 safely"
~~~

---

### Task 5: Add Quality-Gated Qwen Stage1, Stage2, and Stage4 Launchers

**Files:**
- Modify: clstr/qwen_clstr_lineage.py
- Modify: scripts/audit_qwen06_clstr_lineage.py
- Create: scripts/sbatch/run_qwen06_clstr_stage1_train.sh
- Create: scripts/sbatch/run_qwen06_clstr_stage2_train.sh
- Create: scripts/sbatch/run_qwen06_clstr_stage4_train.sh
- Modify: tests/test_qwen_clstr_lineage.py
- Modify: tests/test_qwen_clstr_training_launchers.py

The wrappers use these exact full-run contracts:

~~~text
Stage1 input:
  routing checkpoint = checkpoint_path from stage0_selection.json
  Stage0 lineage = selected_lineage_path from stage0_selection.json
  Stage0 gate = stage0_selection.json with status ok
Stage1 output:
  outputs/qwen06_clstr_postfix/stage1_full
  checkpoints/clstr_stage1_heads-step3000.pt
  stage1_quality_gate.json
  lineage.json

Stage2 input:
  routing checkpoint = selected Stage0 checkpoint
  head checkpoint = stage1_full/checkpoints/clstr_stage1_heads-step3000.pt
  Stage0 and Stage1 lineage manifests
  stage1_quality_gate.json with status ok
Stage2 output:
  outputs/qwen06_clstr_postfix/stage2_full
  checkpoints/clstr_full_base-step10000.pt
  stage2_quality_gate.json
  lineage.json

Stage4 input:
  routing checkpoint = selected Stage0 checkpoint
  head checkpoint = stage2_full/checkpoints/clstr_full_base-step10000.pt
  Stage0 and Stage2 lineage manifests
  stage2_quality_gate.json with status ok
Stage4 output:
  outputs/qwen06_clstr_postfix/stage4_full
  checkpoints/clstr_stage4_act-step3000.pt
  stage4_quality_gate.json
  lineage.json
~~~

- [x] **Step 1: Extend launcher RED tests**

For every downstream launcher assert:

- FULL_RUN defaults to 0;
- smoke defaults to 2 steps and 128 rows;
- full mode requires an ok upstream gate;
- route scorer is unified_memory;
- Stage0 handoff query mode is checkpoint_state_query;
- function-aug-v2 skill and trajectory paths are explicit;
- no backbone-training surface exists;
- lineage validation runs before training;
- lineage creation runs after the quality gate succeeds.

Also assert:

~~~python
assert "STAGE0_TOP_M=500" in stage1
assert "STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query" in stage1
assert "TRANSITION_INVENTORY_MIN_CANDIDATES=64" in stage1
assert "NEXT_SKILL_POOL_MODE=full_pool" in stage2
assert "TRANSITION_INVENTORY_MASK_MODE=explicit_only" in stage2
assert "COUNTERFACTUAL_UTILITY_LOSS_WEIGHT=0.05" in stage2
assert "AUTO_REPLAY_PREFIX_MAX_STEPS=3" in stage2
assert "TRAINABLE_REPLAY_PREFIX=1" in stage2
assert "STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query" in stage2
assert "CANDIDATE_COUNT=64" in stage4
assert "COUNTERFACTUAL_UTILITY_WEIGHT=0.05" in stage4
assert "STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query" in stage4
~~~

- [x] **Step 2: Implement Stage1 wrapper**

Full settings:

~~~text
MAX_STEPS=3000
BATCH_SIZE=16
LEARNING_RATE=1e-4
POLICY_LOSS_WEIGHT=0.6
TRANSITION_LOSS_WEIGHT=0.2
TRANSITION_SKILL_CE_LOSS_WEIGHT=0.6
ROUTE_SCORER=unified_memory
STAGE0_TOP_M=500
STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
TRANSITION_INVENTORY_MIN_CANDIDATES=64
SAMPLING_STRATEGY=balanced_random
~~~

Before training validate the selected Stage0 lineage and release gate. After training run:

~~~bash
"$PYTHON_BIN" scripts/audit_clstr_stage1_heads_quality.py \
  --output_dir "$OUTPUT_DIR" \
  --expected_route_scorer unified_memory \
  --output_path "$OUTPUT_DIR/stage1_quality_gate.json" \
  --fail_on_action_required
~~~

- [x] **Step 3: Implement Stage2 wrapper**

Full settings:

~~~text
MAX_STEPS=10000
BATCH_SIZE=16
LEARNING_RATE=1e-4
ROUTE_SCORER=unified_memory
STAGE0_TOP_M=500
STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
NEXT_SKILL_POOL_MODE=full_pool
TRANSITION_INVENTORY_MASK_MODE=explicit_only
TRANSITION_INVENTORY_MIN_CANDIDATES=64
COUNTERFACTUAL_UTILITY_LOSS_WEIGHT=0.05
COUNTERFACTUAL_GAIN_MARGIN=0.1
COUNTERFACTUAL_SAFETY_TOLERANCE=0.01
COUNTERFACTUAL_WARMUP_FRACTION=0.1
AUTO_REPLAY_PREFIX_MAX_STEPS=3
TRAINABLE_REPLAY_PREFIX=1
~~~

Run the existing Stage2 preflight and handoff gate before training. After training run audit_clstr_stage2_quality.py with expected unified memory and full_pool.

- [x] **Step 4: Implement Stage4 wrapper**

Full settings:

~~~text
MAX_STEPS=3000
BATCH_SIZE=16
LEARNING_RATE=1e-4
ROUTE_SCORER=unified_memory
STAGE0_TOP_M=500
STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
CANDIDATE_COUNT=64
NEXT_SKILL_POOL_MODE=full_pool
COUNTERFACTUAL_UTILITY_WEIGHT=0.05
COUNTERFACTUAL_GAIN_MARGIN=0.1
COUNTERFACTUAL_SAFETY_TOLERANCE=0.01
COUNTERFACTUAL_WARMUP_FRACTION=0.05
AUTO_REPLAY_PREFIX_MAX_STEPS=3
TRAINABLE_REPLAY_PREFIX=1
TRAIN_TRANSITION=1
~~~

Require an ok Stage2 quality gate and valid Stage0 plus Stage2 lineage. Because the Stage4 delta checkpoint intentionally has no standalone Qwen config, create its lineage through the `create-derived` CLI using the validated Stage2 parent as the identity source; never fabricate checkpoint config. After training run audit_clstr_stage4_quality.py with min_steps 3000 and fail_on_action_required.

- [x] **Step 5: Run tests and syntax checks**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_clstr_training_launchers.py
for file in scripts/sbatch/run_qwen06_clstr_stage{1,2,4}_train.sh; do
  bash -n "$file"
done
~~~

- [x] **Step 6: Commit**

~~~bash
git add scripts/sbatch/run_qwen06_clstr_stage1_train.sh \
  scripts/sbatch/run_qwen06_clstr_stage2_train.sh \
  scripts/sbatch/run_qwen06_clstr_stage4_train.sh \
  tests/test_qwen_clstr_training_launchers.py
git commit -m "feat: gate qwen clstr downstream stages"
~~~

---

### Task 6: Expose Reliability Modes Through Full Route Evaluators

**Files:**
- Modify: clstr/toolbench_full_clstr_route_eval.py
- Modify: clstr/global_pool_route_eval.py
- Modify: scripts/run_toolbench_g3_full_clstr_route_eval.py
- Modify: scripts/run_global_pool_clstr_route_eval.py
- Modify: scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh
- Modify: scripts/sbatch/run_global_pool_clstr_route_eval.sh
- Modify: tests/test_toolbench_full_clstr_route_eval.py
- Modify: tests/test_global_pool_route_eval.py

- [x] **Step 1: Write failing pass-through tests**

Add tests that call each outer evaluator with:

~~~python
reliability_mode="fixed_alpha",
fixed_alpha=0.25,
~~~

Capture evaluate_logged_online_stage4_rows calls and assert:

~~~python
assert prior_call["reliability_mode"] == "static"
assert stage4_call["reliability_mode"] == "fixed_alpha"
assert stage4_call["fixed_alpha"] == pytest.approx(0.25)
~~~

- [x] **Step 2: Run tests and confirm RED**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_global_pool_route_eval.py
~~~

- [x] **Step 3: Add evaluator arguments and safe validation**

Add to both evaluator signatures:

~~~python
reliability_mode: str = "dynamic",
fixed_alpha: float = 1.0,
~~~

Validate reliability_mode against:

~~~python
{"static", "dynamic", "fixed_alpha", "heuristic"}
~~~

Do not expose learned. Require fixed_alpha in [0, 1].

Pass static to prior evaluation and the requested mode to Stage4 evaluation and route-record generation.

- [x] **Step 4: Thread CLI and sbatch controls**

Add:

~~~text
--reliability_mode {static,dynamic,fixed_alpha,heuristic}
--fixed_alpha FLOAT
~~~

Shell defaults:

~~~bash
RELIABILITY_MODE=${RELIABILITY_MODE:-dynamic}
FIXED_ALPHA=${FIXED_ALPHA:-1.0}
~~~

- [x] **Step 5: Run tests and commit**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_global_pool_route_eval.py \
  tests/test_current_state_route_eval.py \
  tests/test_memory_utility_gate.py
git add clstr/toolbench_full_clstr_route_eval.py clstr/global_pool_route_eval.py \
  scripts/run_toolbench_g3_full_clstr_route_eval.py \
  scripts/run_global_pool_clstr_route_eval.py \
  scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh \
  scripts/sbatch/run_global_pool_clstr_route_eval.sh \
  tests/test_toolbench_full_clstr_route_eval.py tests/test_global_pool_route_eval.py
git commit -m "feat: expose reliability modes in route evaluation"
~~~

---

### Task 7: Perform Lightweight Pre-Submission Verification

**Files:**
- All files changed in Tasks 1-6.

- [ ] **Step 1: Run focused tests only**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_qwen_clstr_lineage.py \
  tests/test_qwen_clstr_stage0_gate.py \
  tests/test_qwen_clstr_training_launchers.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_global_pool_route_eval.py \
  tests/test_current_state_route_eval.py \
  tests/test_memory_utility_gate.py
~~~

If this focused set is too large for the storage node, submit the exact command as a short CPU Slurm job. Do not retry a killed local process.

- [ ] **Step 2: Run syntax and static checks**

~~~bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/retrieval_warmup.py \
  clstr/qwen_clstr_lineage.py \
  clstr/qwen_clstr_stage0_gate.py \
  scripts/audit_qwen06_clstr_lineage.py \
  scripts/audit_qwen06_clstr_stage0_gate.py
for file in scripts/sbatch/run_qwen06_clstr_*.sh; do
  bash -n "$file"
done
git diff --check
~~~

- [ ] **Step 3: Request code review**

Use the requesting-code-review skill. Fix only verified issues and rerun the affected focused tests.

- [ ] **Step 4: Commit verification/documentation updates**

~~~bash
git add .planning docs/superpowers/specs/2026-07-11-qwen06-clstr-postfix-training-design.md \
  docs/superpowers/plans/2026-07-11-qwen06-clstr-postfix-training.md
git commit -m "docs: finalize qwen clstr training readiness"
~~~

---

### Task 8: Submit and Validate the Mechanical Stage0 Smoke

**Files created by run:**
- outputs/qwen06_clstr_postfix/stage0_smoke/

- [x] **Step 1: Submit smoke**

From WORKTREE:

~~~bash
sbatch scripts/sbatch/run_qwen06_clstr_stage0_train.sh
~~~

- [x] **Step 2: Monitor through Slurm**

Use squeue and sacct. Do not load the model or tail large files on the storage node. Inspect only the final small train_report.json, lineage.json, and smoke_gate.json after completion.

- [x] **Step 3: Enforce the smoke gate**

Required artifacts:

~~~text
stage0_smoke/checkpoints/clstr_unified_retrieval_v2-step0.pt
stage0_smoke/checkpoints/clstr_unified_retrieval_v2-step80.pt
stage0_smoke/lineage-step0.json
stage0_smoke/lineage-step80.json
stage0_smoke/smoke_gate.json
~~~

Proceed only when smoke_gate status is ok.

---

### Task 9: Run Segmented Full Stage0 and Select the Best Safe Checkpoint

- [x] **Step 1: Submit step 1200**

~~~bash
sbatch --export=ALL,FULL_RUN=1,TARGET_STEP=1200,SMOKE_GATE_PATH=$RUN_ROOT/stage0_smoke/smoke_gate.json \
  scripts/sbatch/run_qwen06_clstr_stage0_train.sh
~~~

- [x] **Step 2: Audit step 0 and step 1200**

~~~bash
sbatch --export=ALL,TARGET_STEP=1200 \
  scripts/sbatch/run_qwen06_clstr_stage0_audit.sh
~~~

Proceed only when step1200/promotion_gate.json is ok.

- [x] **Step 3: Resume to 2400 and audit**

~~~bash
sbatch --export=ALL,FULL_RUN=1,TARGET_STEP=2400,RESUME_CHECKPOINT_PATH=$RUN_ROOT/stage0_full/checkpoints/clstr_unified_retrieval_v2-step1200.pt,PREVIOUS_GATE_PATH=$RUN_ROOT/stage0_audits/step1200/promotion_gate.json \
  scripts/sbatch/run_qwen06_clstr_stage0_train.sh
~~~

Then submit the audit with baseline step0 and previous step1200 reports.

- [x] **Step 4: Resume to 3600 and conditionally 5000**

Repeat the same pattern. Submit 5000 only when the step3600 promotion gate is ok.

- [x] **Step 5: Select the checkpoint and produce release gate**

Run the select subcommand over every completed segment. The result must name one checkpoint and have release_status ok. A borderline result stops and returns to the user. An action_required result stops downstream training.

---

### Task 10: Train Fresh Stage1, Stage2, and Stage4

- [ ] **Step 1: Run Stage1 smoke, then full**

Submit default smoke. If its quality and lineage checks pass, submit:

~~~bash
sbatch --export=ALL,FULL_RUN=1,STAGE0_SELECTION_PATH=$RUN_ROOT/stage0_full/stage0_selection.json \
  scripts/sbatch/run_qwen06_clstr_stage1_train.sh
~~~

- [ ] **Step 2: Run Stage2 smoke, then full**

Require Stage1 quality_gate status ok and valid Stage0 plus Stage1 lineage:

~~~bash
sbatch --export=ALL,FULL_RUN=1 \
  scripts/sbatch/run_qwen06_clstr_stage2_train.sh
~~~

- [ ] **Step 3: Run Stage4 smoke, then full**

Require Stage2 quality_gate status ok and valid Stage0 plus Stage2 lineage:

~~~bash
sbatch --export=ALL,FULL_RUN=1 \
  scripts/sbatch/run_qwen06_clstr_stage4_train.sh
~~~

Do not train a learned reliability gate.

---

### Task 11: Evaluate Reliability Modes, Candidate Union, Closed-Loop Control, and Memory Utility

- [ ] **Step 1: Run ToolBench-G3 with route records**

Run static, dynamic, fixed_alpha, and heuristic as separate jobs. Use fixed_alpha=0.5 for the pre-audit fixed control; the oracle audit will select its own fixed alpha from training folds.

- [ ] **Step 2: Run TrajectBench replay-prefix evaluation**

Submit run_trajectbench_full_clstr_eval.sh with:

~~~text
TRAJECTORIES_PATH=.tmp/stage4_sanitized_trajectbench/trajectories_visible_global_full.jsonl
SKILLS_PATH=$DATA/skill_pool.jsonl
ROUTING_CHECKPOINT_PATH=$("$PYTHON" -c 'import json; print(json.load(open("outputs/qwen06_clstr_postfix/stage0_full/stage0_selection.json"))["selected_checkpoint_path"])')
HEAD_CHECKPOINT_PATH=$RUN_ROOT/stage4_full/checkpoints/clstr_stage4_act-step3000.pt
STAGE0_TOP_M=500
CANDIDATE_COUNT=64
ROUTE_SCORER=unified_memory
STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
REPLAY_PREFIX_MAX_STEPS=3
MAX_UPDATES=0
EVAL_SPLIT_MODE=trajectory_prefix
~~~

Resolve ROUTING_CHECKPOINT_PATH from stage0_selection.json before submission and print it in the Slurm preflight summary. Require replay-prefix attachment for every retained row before treating dynamic-versus-static metrics as memory evidence.

- [ ] **Step 3: Run global-pool tau2, ToolSandbox, and BFCL**

For each benchmark pass:

~~~text
STAGE0_TOP_M=500
DYNAMIC_EXTRA_K=64
FINAL_K=64
ROUTE_SCORER=unified_memory
ROUTE_RECORDS_PATH=$RUN_ROOT/eval/$BENCHMARK/$RELIABILITY_MODE/routes.jsonl
ROUTE_RECORD_MANIFEST_PATH=$RUN_ROOT/eval/$BENCHMARK/$RELIABILITY_MODE/route_manifest.json
ROUTE_RECORD_MODEL_DIGEST=$("$PYTHON" -c 'import json; print(json.load(open("outputs/qwen06_clstr_postfix/stage4_full/lineage.json"))["checkpoint"]["sha256"])')
~~~

Before submission, resolve ROUTE_RECORD_MODEL_DIGEST with:

~~~bash
ROUTE_RECORD_MODEL_DIGEST=$("$PYTHON" -c \
  'import json; print(json.load(open("outputs/qwen06_clstr_postfix/stage4_full/lineage.json"))["checkpoint"]["sha256"])')
~~~

- [ ] **Step 4: Run ALFWorld closed-loop evaluation**

First submit a two-episode smoke with run_alfworld_qwen_clstr_executor_gate.sh:

~~~text
CHECKPOINT_PATH=$RUN_ROOT/stage2_full/checkpoints/clstr_full_base-step10000.pt
STAGE4_CHECKPOINT=$RUN_ROOT/stage4_full/checkpoints/clstr_stage4_act-step3000.pt
SKILL_ROWS_PATH_OVERRIDE=$DATA/skill_pool.jsonl
AUX_DATA_ROOT=$DATA
QWEN_MODEL_NAME_OR_PATH=$ASSET_ROOT/models/Qwen3-14B
CLSTR_PRIOR_MODE=unified_memory
RUN_QWEN_ONLY=0
RUN_HYBRID=1
MAX_EPISODES=2
LOOP_GUARD=1
~~~

The smoke passes only when stage4_overlay_loaded is true, clstr_prior_mode is unified_memory, and the recorded CLSTR model config points to Qwen3-Embedding-0.6B. Then run valid_seen with MAX_EPISODES=140 under the same settings.

- [ ] **Step 5: Run the trajectory-disjoint oracle audit**

~~~bash
sbatch --wrap="cd $WORKTREE && $PYTHON scripts/audit_clstr_memory_utility_oracle.py \
  --record_manifest $RUN_ROOT/eval/toolbench/dynamic/route_manifest.json \
  --record_manifest $RUN_ROOT/eval/tau2/dynamic/route_manifest.json \
  --record_manifest $RUN_ROOT/eval/toolsandbox/dynamic/route_manifest.json \
  --record_manifest $RUN_ROOT/eval/bfcl/dynamic/route_manifest.json \
  --output_path $RUN_ROOT/memory_audit/oracle_report.json"
~~~

Use an explicit bash login shell if the cluster defaults sbatch --wrap to sh.

- [ ] **Step 6: Apply the learned-gate stop rule**

Do not train or enable learned mode unless oracle_report.json has status ok and recommendation train_learned_gate. Otherwise retain static, dynamic, fixed_alpha, and heuristic results only.

---

## Completion Criteria

Implementation is ready for training when:

- all focused tests and shell checks pass;
- Stage0 launcher cannot expose backbone training;
- Stage0/1/2/4 restore the same `clstr_causal_state_v1` prompt contract;
- route_scorer is unified_memory in Stage0/1/2/4;
- stable step0 and lineage artifacts are test-covered;
- later segments cannot run without an ok prior gate;
- outer route evaluators expose four safe reliability modes and exclude learned;
- no heavy command has been run on the storage node.

The experimental chain is complete when:

- a release-safe Stage0 checkpoint is selected;
- Stage1, Stage2, and Stage4 quality gates pass;
- held-out reports and route-record manifests exist;
- the trajectory-disjoint memory utility audit is complete;
- the learned gate remains disabled unless explicitly recommended and separately approved.
