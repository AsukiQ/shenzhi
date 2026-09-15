import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.native_rerank import CLSTRNativeResidualRerankHead
from clstr.aux_trajectories import (
    build_aux_trajectory_schema_report,
    import_aux_trajectory_datasets,
    load_aux_training_rows,
    map_action_to_pseudo_skill,
)
from clstr.aux_pretrain import (
    _batch_rows_for_step,
    _build_action_pools,
    _build_frozen_auxiliary_feature_cache,
    _candidate_actions_for_row,
    _encode_text_batches,
    _filter_trajectories_by_split,
    _flatten_step_rows,
    run_aux_trajectory_pretrain,
)
from clstr.data import validate_data_roles
from clstr.action_adapter import UniversalActionAdapter


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_aux_sources(root: Path) -> dict[str, Path]:
    alf = root / "sft_alfworld_trajectory_dataset_v5_cleaned"
    auto = root / "auto-dreamer"
    eto = root / "eto-sft-trajectory"
    write_jsonl(
        alf / "train.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "You are in a kitchen."},
                    {"role": "assistant", "content": "go to fridge"},
                    {"role": "user", "content": "You arrive at the fridge."},
                    {"role": "assistant", "content": "open fridge"},
                ],
                "metadata": {
                    "task_id": "alf-task-1",
                    "task_type": "pick_and_place",
                    "success": True,
                    "reward": 1.0,
                    "num_steps": 2,
                },
            }
        ],
    )
    write_jsonl(
        auto / "data" / "train.jsonl",
        [
            {
                "task_id": "auto-task-1",
                "task_type": "scienceworld",
                "game_file": "scienceworld-1",
                "success": False,
                "total_reward": 0.25,
                "steps": [
                    {
                        "observation": "A thermometer is on the table.",
                        "action": "measure temperature of water",
                        "next_observation": "The water is 20 C.",
                        "reward": 0.25,
                        "done": False,
                    },
                    {
                        "observation": "The water is 20 C.",
                        "action": "mix water with salt",
                        "next_observation": "The salt dissolves.",
                        "reward": 0.0,
                        "done": True,
                    },
                ],
            }
        ],
    )
    (eto / "alfworld").mkdir(parents=True)
    (eto / "alfworld" / "train.json").write_text(
        json.dumps(
            [
                {
                    "id": "eto-task-1",
                    "source": "alfworld",
                    "trajectory": [
                        {"observation": "You see a drawer.", "action": "open drawer"},
                        {"observation": "The drawer is open.", "action": "take apple"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return {
        "sft_alfworld": alf,
        "auto_dreamer": auto,
        "eto_sft": eto,
    }


def make_normalized_aux_pretrain_data(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        root / "pseudo_skills.jsonl",
        [
            {
                "skill_id": "alfworld/go_to_object",
                "name": "go to object",
                "description": "Move to a target object.",
                "source": "auxiliary_pseudo_skill",
                "environment": "alfworld",
                "low_confidence_mapping": False,
            },
            {
                "skill_id": "scienceworld/measure_property",
                "name": "measure property",
                "description": "Measure a science property.",
                "source": "auxiliary_pseudo_skill",
                "environment": "scienceworld",
                "low_confidence_mapping": False,
            },
            {
                "skill_id": "webshop/text_action",
                "name": "text action",
                "description": "Generic text action.",
                "source": "auxiliary_pseudo_skill",
                "environment": "webshop",
                "low_confidence_mapping": True,
            },
        ],
    )
    write_jsonl(
        root / "trajectories.jsonl",
        [
            {
                "dataset": "unit",
                "source_path": "train.jsonl",
                "split": "train",
                "environment": "alfworld",
                "task_id": "train-task",
                "task_type": "pick_and_place",
                "success": True,
                "reward": 1.0,
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "You are in a kitchen.",
                        "action_text": "go to fridge",
                        "next_observation_text": "You are at the fridge.",
                        "done": False,
                        "reward": 0.0,
                        "pseudo_skill_id": "alfworld/go_to_object",
                        "pseudo_skill_low_confidence": False,
                    },
                    {
                        "t": 1,
                        "observation_text": "You are at the fridge.",
                        "action_text": "go to counter",
                        "next_observation_text": "You are at the counter.",
                        "done": True,
                        "reward": 1.0,
                        "pseudo_skill_id": "alfworld/go_to_object",
                        "pseudo_skill_low_confidence": False,
                    },
                ],
            },
            {
                "dataset": "unit",
                "source_path": "valid_seen.jsonl",
                "split": "valid_seen",
                "environment": "scienceworld",
                "task_id": "valid-task",
                "task_type": "temperature",
                "success": None,
                "reward": None,
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "A thermometer is nearby.",
                        "action_text": "measure temperature",
                        "next_observation_text": None,
                        "done": True,
                        "reward": None,
                        "pseudo_skill_id": "scienceworld/measure_property",
                        "pseudo_skill_low_confidence": False,
                    }
                ],
            },
            {
                "dataset": "unit",
                "source_path": "test.jsonl",
                "split": "test",
                "environment": "webshop",
                "task_id": "test-task",
                "task_type": "shopping",
                "success": False,
                "reward": 0.0,
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "A product page is open.",
                        "action_text": "click checkout",
                        "next_observation_text": "Checkout is open.",
                        "done": True,
                        "reward": 0.0,
                        "pseudo_skill_id": "webshop/text_action",
                        "pseudo_skill_low_confidence": True,
                    }
                ],
            },
        ],
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "record_counts": {"trajectories": 3, "steps": 4},
                "pseudo_skill_count": 3,
                "low_confidence_mapping_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def make_tiny_native_routing_init(tmp_path: Path, model_dir: Path, data_root: Path) -> Path:
    skills = load_eval_pool(data_root / "pseudo_skills.jsonl")
    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=str(model_dir),
            d=16,
            d_a=4,
            top_k=4,
            use_cross_encoder=False,
            freeze_backbone=True,
            defer_skill_table_init=False,
        ),
        skills,
    )
    base_checkpoint = tmp_path / "base_clstr.pt"
    torch.save(
        {
            "stage": "skillret_retrieval_warmup",
            "step": 512,
            "config": {
                "base_model_name": str(model_dir),
                "d": 16,
                "d_a": 4,
                "top_k": 4,
                "encoder_pooling": "masked_mean",
                "cross_encoder_pooling": "masked_mean",
                "freeze_backbone": True,
                "defer_skill_table_init": False,
                "skill_text_format": "clstr",
                "skill_table_batch_size": 2,
                "use_cross_encoder": False,
            },
            "model_state_dict": model.state_dict(),
        },
        base_checkpoint,
    )
    reranker = CLSTRNativeResidualRerankHead(16)
    rerank_checkpoint = tmp_path / "native_rerank.pt"
    torch.save(
        {
            "method": "clstr_native_rerank",
            "config": {
                "d": 16,
                "qdoc_adapter_used": False,
                "loss_taxonomy": "L_retr",
                "base_checkpoint_path": str(base_checkpoint),
            },
            "reranker_state_dict": reranker.state_dict(),
        },
        rerank_checkpoint,
    )
    manifest = tmp_path / "routing_init_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ok",
                "base_clstr_checkpoint": str(base_checkpoint),
                "native_rerank_adopted": True,
                "native_rerank_checkpoint": str(rerank_checkpoint),
                "downstream_init_checkpoint": str(rerank_checkpoint),
                "downstream_init_for": "auxiliary_trajectory_pretrain",
                "qdoc_adapter_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_map_action_to_pseudo_skill_uses_environment_specific_templates():
    assert map_action_to_pseudo_skill("alfworld", "go to fridge").pseudo_skill_id == "alfworld/go_to_object"
    assert map_action_to_pseudo_skill("alfworld", "open drawer").pseudo_skill_id == "alfworld/open_object"
    assert map_action_to_pseudo_skill("scienceworld", "measure temperature").pseudo_skill_id == "scienceworld/measure_property"
    generic = map_action_to_pseudo_skill("webshop", "click checkout")
    assert generic.pseudo_skill_id == "webshop/text_action"
    assert generic.low_confidence is True


def test_build_aux_trajectory_schema_report_detects_fields_without_unifying_schema(tmp_path):
    sources = make_aux_sources(tmp_path)

    report = build_aux_trajectory_schema_report(
        dataset_name="agent-eto/eto-sft-trajectory",
        dataset_key="eto_sft",
        source_root=sources["eto_sft"],
    )

    assert report["dataset_key"] == "eto_sft"
    assert report["file_count"] == 1
    assert report["sample_count"] == 1
    assert "trajectory" in report["fields"]
    assert report["format_counts"]["json"] == 1


def test_import_aux_trajectory_datasets_emits_normalized_records_and_manifest(tmp_path):
    sources = make_aux_sources(tmp_path / "sources")
    output_dir = tmp_path / "out"
    report_dir = tmp_path / "reports"

    manifest = import_aux_trajectory_datasets(
        dataset_roots=sources,
        output_dir=output_dir,
        report_dir=report_dir,
    )

    assert manifest["data_role"] == "auxiliary_trajectory_not_skillsbench_clean_router"
    assert manifest["record_counts"]["trajectories"] == 3
    assert manifest["record_counts"]["steps"] == 6
    assert manifest["environment_counts"]["alfworld"] == 2
    assert manifest["environment_counts"]["scienceworld"] == 1
    assert manifest["coverage"]["success"]["present"] == 2
    assert manifest["coverage"]["reward"]["present"] == 2
    assert manifest["missing_fields"]["eto_sft"]["success"] == 1
    assert manifest["missing_fields"]["eto_sft"]["next_observation_text"] == 2
    assert manifest["pseudo_skill_count"] >= 4
    assert manifest["low_confidence_mapping_count"] == 0
    assert (output_dir / "trajectories.jsonl").exists()
    assert (output_dir / "pseudo_skills.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert (report_dir / "sft_alfworld_schema_report.json").exists()
    assert (report_dir / "auto_dreamer_schema_report.json").exists()
    assert (report_dir / "eto_sft_schema_report.json").exists()
    trajectories = [
        json.loads(line)
        for line in (output_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert trajectories[0]["dataset"] == "moroqq/sft_alfworld_trajectory_dataset_v5_cleaned"
    assert trajectories[0]["steps"][0]["pseudo_skill_id"] == "alfworld/go_to_object"
    assert trajectories[1]["steps"][0]["pseudo_skill_id"] == "scienceworld/measure_property"


def test_import_aux_trajectory_datasets_groups_auto_dreamer_step_rows_by_episode(tmp_path):
    auto = tmp_path / "auto-dreamer"
    write_jsonl(
        auto / "alfworld" / "train" / "episodes.jsonl",
        [
            {
                "episode_id": "ep-1",
                "task_id": "alf-task",
                "task_type": "pick_and_place",
                "split": "train",
                "success": True,
                "total_reward": 1.0,
            }
        ],
    )
    write_jsonl(
        auto / "alfworld" / "train" / "steps.jsonl",
        [
            {
                "episode_id": "ep-1",
                "task_id": "alf-task",
                "task_type": "pick_and_place",
                "split": "train",
                "step_idx": 0,
                "observation_raw": "You are in a room.",
                "action_taken": "go to cabinet 1",
                "env_feedback_raw": "You arrive at cabinet 1.",
                "reward": 0.0,
                "done": False,
                "success": True,
            },
            {
                "episode_id": "ep-1",
                "task_id": "alf-task",
                "task_type": "pick_and_place",
                "split": "train",
                "step_idx": 1,
                "observation_raw": "You are at cabinet 1.",
                "action_taken": "open cabinet 1",
                "env_feedback_raw": "The cabinet is open.",
                "reward": 1.0,
                "done": True,
                "success": True,
            },
        ],
    )

    manifest = import_aux_trajectory_datasets(
        dataset_roots={"auto_dreamer": auto},
        output_dir=tmp_path / "out",
        report_dir=tmp_path / "reports",
    )

    trajectories = [
        json.loads(line)
        for line in (tmp_path / "out" / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert manifest["record_counts"]["trajectories"] == 1
    assert manifest["record_counts"]["steps"] == 2
    assert manifest["missing_fields"]["auto_dreamer"] == {}
    assert trajectories[0]["task_id"] == "alf-task"
    assert trajectories[0]["success"] is True
    assert trajectories[0]["reward"] == 1.0
    assert trajectories[0]["steps"][0]["action_text"] == "go to cabinet 1"
    assert trajectories[0]["steps"][0]["next_observation_text"] == "You arrive at cabinet 1."
    assert trajectories[0]["steps"][0]["pseudo_skill_id"] == "alfworld/go_to_object"
    assert trajectories[0]["steps"][1]["done"] is True


def test_import_aux_trajectory_datasets_maps_sciworld_paths_to_scienceworld(tmp_path):
    auto = tmp_path / "auto-dreamer"
    write_jsonl(
        auto / "sciworld" / "train" / "steps.jsonl",
        [
            {
                "episode_id": "science-episode",
                "task_id": "science-task",
                "task_type": "state-of-matter",
                "split": "train",
                "step_idx": 0,
                "observation_raw": "A thermometer is nearby.",
                "action_taken": "measure temperature of water",
                "env_feedback_raw": "The water is 20 C.",
                "reward": 1.0,
                "done": True,
                "success": True,
            }
        ],
    )

    import_aux_trajectory_datasets(
        dataset_roots={"auto_dreamer": auto},
        output_dir=tmp_path / "out",
        report_dir=tmp_path / "reports",
    )

    trajectory = json.loads((tmp_path / "out" / "trajectories.jsonl").read_text(encoding="utf-8"))
    assert trajectory["environment"] == "scienceworld"
    assert trajectory["steps"][0]["pseudo_skill_id"] == "scienceworld/measure_property"


def test_data_roles_support_aux_trajectory_without_clean_router_side_effects(tmp_path):
    aux = tmp_path / "aux"
    aux.mkdir()
    for name in ["trajectories.jsonl", "pseudo_skills.jsonl", "manifest.json"]:
        (aux / name).write_text("{}\n", encoding="utf-8")
    cfg = {
        "skillrouter_eval_root": str(tmp_path / "skillrouter_eval_core"),
        "skillsbench_root": str(tmp_path / "skillsbench"),
        "alfworld_root": str(tmp_path / "alfworld"),
        "leakage_audit_dir": str(tmp_path / "audit"),
        "clean_router_data_root": str(tmp_path / "clean_router"),
        "skillret_root": str(tmp_path / "raw_skillret"),
        "retrieval_warmup_data_root": str(tmp_path / "skillret"),
        "aux_trajectory_root": str(tmp_path / "raw_aux"),
        "aux_trajectory_data_root": str(aux),
        "training_data_source": "aux_trajectory",
    }

    roles = validate_data_roles(cfg, require_training_data=True)

    assert roles["aux_trajectory"]["role"] == "auxiliary-transition-pretraining-source"
    assert roles["aux_trajectory_data"]["role"] == "training-auxiliary-trajectory-data"
    assert roles["training"]["source"] == "aux_trajectory"


def test_aux_trajectory_pretrain_writes_checkpoint_and_skips_missing_targets(tmp_path):
    sources = make_aux_sources(tmp_path / "sources")
    data_root = tmp_path / "aux_data"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "pretrain"
    model_dir = tmp_path / "tiny-model"
    import_aux_trajectory_datasets(sources, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_aux_trajectory_pretrain(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=2,
        model_dim=16,
        top_k=4,
        max_trajectories=3,
    )

    assert report["status"] == "ok"
    assert report["training_data_source"] == "aux_trajectory"
    assert report["steps"] == 1
    assert report["metrics"]["action_imitation_loss"] >= 0.0
    assert report["metrics"]["stop_loss"] >= 0.0
    assert "transition_consistency_proxy" in report["trained_targets"]
    assert "belief_update_proxy" in report["trained_targets"]
    assert report["metrics"]["belief_update_proxy"] >= 0.0
    assert "next_observation_text_missing" in report["skipped_targets"]
    assert Path(report["checkpoint"]).exists()
    assert (output_dir / "train_report.json").exists()


def test_aux_pretrain_filters_splits_and_records_readiness(tmp_path):
    data_root = make_normalized_aux_pretrain_data(tmp_path / "aux_data")
    output_dir = tmp_path / "pretrain"
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_aux_trajectory_pretrain(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=2,
        model_dim=16,
        top_k=4,
        max_trajectories=None,
        include_splits=["train", "valid_seen"],
        exclude_splits=["test"],
    )

    readiness = report["data_readiness"]
    assert report["split_filter"]["include_splits"] == ["train", "valid_seen"]
    assert report["split_filter"]["exclude_splits"] == ["test"]
    assert report["split_filter"]["test_split_excluded_from_training"] is True
    assert readiness["included_split_counts"] == {"train": 1, "valid_seen": 1}
    assert readiness["excluded_split_counts"] == {"test": 1}
    assert readiness["environment_counts"] == {"alfworld": 1, "scienceworld": 1}
    assert readiness["trajectory_count"] == 2
    assert readiness["flattened_step_count"] == 3
    assert readiness["pseudo_skill_count"] == 3
    assert readiness["low_confidence_mapping_count"] == 0
    assert readiness["coverage"]["success"] == {"present": 1, "missing": 1}
    assert readiness["coverage"]["reward"] == {"present": 1, "missing": 1}
    assert readiness["missing_fields"]["next_observation_text"] == {"present": 2, "missing": 1}
    assert readiness["missing_fields"]["step_reward"] == {"present": 2, "missing": 1}
    assert "test" not in readiness["included_split_counts"]


def test_aux_pretrain_max_trajectories_uses_balanced_environment_selection():
    trajectories = [
        {"split": "train", "environment": "alfworld", "task_id": f"alf-{idx}", "steps": []}
        for idx in range(4)
    ] + [
        {"split": "train", "environment": "scienceworld", "task_id": f"sci-{idx}", "steps": []}
        for idx in range(4)
    ] + [
        {"split": "unknown", "environment": "webshop", "task_id": f"web-{idx}", "steps": []}
        for idx in range(4)
    ]

    selected, excluded, report = _filter_trajectories_by_split(
        trajectories,
        include_splits=["train", "unknown"],
        exclude_splits=["test"],
        max_trajectories=3,
    )

    assert [row["environment"] for row in selected] == ["alfworld", "scienceworld", "webshop"]
    assert report["selection_strategy"] == "environment_round_robin"
    assert report["included_environment_counts"] == {"alfworld": 1, "scienceworld": 1, "webshop": 1}
    assert len(excluded) == 9


def test_aux_pretrain_batches_advance_by_batch_stride_instead_of_sliding_one_row():
    rows = [{"row": idx} for idx in range(6)]

    batch1 = _batch_rows_for_step(rows, step_idx=1, batch_size=2)
    batch2 = _batch_rows_for_step(rows, step_idx=2, batch_size=2)
    batch3 = _batch_rows_for_step(rows, step_idx=3, batch_size=2)
    batch4 = _batch_rows_for_step(rows, step_idx=4, batch_size=2)

    assert [row["row"] for row in batch1] == [0, 1]
    assert [row["row"] for row in batch2] == [2, 3]
    assert [row["row"] for row in batch3] == [4, 5]
    assert [row["row"] for row in batch4] == [0, 1]


def test_candidate_actions_stop_scanning_after_enough_negatives():
    class ExplodingPool(list):
        def __iter__(self):
            raise AssertionError("fallback action pool should not be scanned once enough negatives exist")

    row = {
        "action_text": "look",
        "environment": "alfworld",
        "pseudo_skill_id": "navigate",
    }
    pools = {
        "by_env_skill": {("alfworld", "navigate"): ["look", "open door", "take apple", "open door"]},
        "by_env": {"alfworld": ExplodingPool(["unused env action"])},
        "global": ExplodingPool(["unused global action"]),
    }

    candidates = _candidate_actions_for_row(
        row,
        pools,
        negative_k=2,
        candidate_strategy="environment_action_pool",
    )

    assert candidates == ["look", "open door", "take apple"]


def test_frozen_auxiliary_feature_cache_precomputes_training_embeddings(tmp_path):
    data_root = make_normalized_aux_pretrain_data(tmp_path / "aux_data")
    skill_rows, trajectories = load_aux_training_rows(data_root)
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skill_rows)}
    rows = _flatten_step_rows(trajectories[:2], skill_id_to_idx, None)
    for idx, row in enumerate(rows):
        row["row_index"] = idx
    action_pools = _build_action_pools(rows)

    class CountingEncoder:
        def __init__(self):
            self.calls = []

        def encode_observations(self, texts):
            self.calls.append(list(texts))
            values = torch.arange(len(texts) * 4, dtype=torch.float32)
            return values.view(len(texts), 4)

    encoder = CountingEncoder()
    cache = _build_frozen_auxiliary_feature_cache(
        encoder,
        rows,
        action_pools,
        action_negative_k=2,
        candidate_strategy="environment_action_pool",
        encode_batch_size=2,
    )

    assert cache.state_embeddings.shape == (len(rows), 4)
    assert cache.next_observation_embeddings.shape == (len(rows), 4)
    assert cache.action_embeddings.shape[0] == len(cache.action_text_to_index)
    assert cache.row_action_indices.shape == (len(rows),)
    assert cache.row_candidate_indices.shape == (len(rows), 3)
    assert cache.row_candidate_mask.dtype == torch.bool
    assert cache.report()["used"] is True
    assert cache.report()["state_rows"] == len(rows)
    assert encoder.calls


def test_encode_text_batches_backs_off_after_cuda_oom():
    class OOMOnceEncoder:
        def __init__(self):
            self.batch_sizes = []

        def encode_observations(self, texts):
            self.batch_sizes.append(len(texts))
            if len(texts) > 2:
                raise torch.OutOfMemoryError("synthetic oom")
            values = torch.arange(len(texts) * 4, dtype=torch.float32)
            return values.view(len(texts), 4)

    stats = {}
    encoder = OOMOnceEncoder()
    encoded = _encode_text_batches(
        encoder,
        ["a", "b", "c", "d", "e"],
        batch_size=4,
        label="unit",
        stats=stats,
    )

    assert encoded.shape == (5, 4)
    assert 4 in encoder.batch_sizes
    assert max(encoder.batch_sizes[1:]) <= 2
    assert stats["oom_backoff_count"] == 1
    assert stats["effective_batch_sizes"]["unit"] == 2


def test_aux_pretrain_loads_native_routing_init_and_writes_eval_report(tmp_path):
    data_root = make_normalized_aux_pretrain_data(tmp_path / "aux_data")
    output_dir = tmp_path / "pretrain"
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    routing_manifest = make_tiny_native_routing_init(tmp_path, model_dir, data_root)

    report = run_aux_trajectory_pretrain(
        data_root=data_root,
        base_model_name=None,
        output_dir=output_dir,
        max_steps=1,
        batch_size=2,
        model_dim=16,
        top_k=4,
        max_trajectories=None,
        include_splits=["train"],
        exclude_splits=["test"],
        routing_init_manifest=routing_manifest,
        eval_splits=["test"],
    )

    routing = report["routing_init"]
    assert routing["manifest"] == str(routing_manifest)
    assert routing["qdoc_adapter_used"] is False
    assert routing["base_clstr_checkpoint"].endswith("base_clstr.pt")
    assert routing["native_rerank_checkpoint"].endswith("native_rerank.pt")
    assert routing["base_clstr_loaded"] is True
    assert routing["native_rerank_loaded"] is True
    assert routing["native_rerank_used_for_loss"] is False
    assert routing["native_rerank_loss_reason"] == (
        "auxiliary pretrain optimizes L_trans/L_act_proxy/STOP proxy over pseudo-skills"
    )
    assert report["loss_taxonomy"]["action_imitation"] == "L_act_pretraining_proxy"
    assert report["loss_taxonomy"]["transition_consistency_proxy"] == "L_trans"
    assert report["loss_taxonomy"]["belief_update_proxy"] == "L_trans"
    assert report["loss_taxonomy"]["stop_done_prediction"] == "STOP_policy_auxiliary_proxy"
    assert report["loss_taxonomy"]["retrieval_listwise_sidecar"] == "L_retr"
    assert Path(report["design_report"]).exists()
    assert Path(report["eval_report"]).exists()
    eval_report = json.loads(Path(report["eval_report"]).read_text(encoding="utf-8"))
    assert eval_report["split_counts"] == {"test": 1}
    assert eval_report["not_skillsbench_pass_rate"] is True
    assert eval_report["not_closed_loop_harness_success"] is True
    assert eval_report["metrics"]["action_recall_at_1"] >= 0.0
    assert "transition_consistency_proxy" in eval_report["metrics"]
    assert "belief_update_proxy" in eval_report["metrics"]


def test_universal_action_adapter_scores_candidate_action_texts():
    adapter = UniversalActionAdapter(d=8, hidden_dim=12)
    state_embs = torch.randn(2, 8)
    action_embs = torch.randn(2, 4, 8)

    scores = adapter(state_embs, action_embs)
    loss = adapter.ranking_loss(scores)

    assert scores.shape == (2, 4)
    assert loss.item() >= 0.0
    assert adapter.loss_taxonomy == "L_act_pretraining_proxy"


def test_aux_pretrain_gate_freezes_routing_and_uses_universal_action_adapter(tmp_path):
    data_root = make_normalized_aux_pretrain_data(tmp_path / "aux_data")
    output_dir = tmp_path / "gate"
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    routing_manifest = make_tiny_native_routing_init(tmp_path, model_dir, data_root)

    report = run_aux_trajectory_pretrain(
        data_root=data_root,
        base_model_name=None,
        output_dir=output_dir,
        max_steps=1,
        batch_size=2,
        model_dim=16,
        top_k=4,
        max_trajectories=None,
        include_splits=["train", "valid_seen"],
        exclude_splits=["test"],
        routing_init_manifest=routing_manifest,
        eval_splits=["test"],
        freeze_routing_foundation=True,
        use_universal_action_adapter=True,
        action_negative_k=2,
        candidate_strategy="environment_action_pool",
        max_eval_steps=4,
    )

    assert report["paper_role"] == "debug_gate_not_final"
    assert report["not_for_paper_table"] is True
    assert report["not_full_training_init"] is True
    assert report["frozen_routing_foundation"] is True
    assert report["qdoc_adapter_used"] is False
    assert report["routing_init"]["native_rerank_loaded"] is True
    assert report["routing_init"]["native_rerank_used_for_loss"] is False
    assert report["action_loss_source"] == "universal_action_adapter"
    assert report["candidate_strategy"] == "environment_action_pool"
    assert report["action_negative_k"] == 2
    assert "universal_action_ranking_loss" in report["metrics"]
    assert "action_imitation_loss" in report["metrics"]
    assert set(report["trainable_modules"]) == {
        "transition",
        "gate",
        "stop_head",
        "universal_action_adapter",
    }
    assert {
        "encoder.backbone",
        "encoder.proj",
        "skill_table.W",
        "skill_table.E",
        "native_rerank_sidecar",
    }.issubset(set(report["frozen_modules"]))
    checkpoint = torch.load(report["checkpoint"], map_location="cpu")
    assert checkpoint["paper_role"] == "debug_gate_not_final"
    assert checkpoint["not_for_paper_table"] is True
    assert checkpoint["not_full_training_init"] is True
    assert checkpoint["action_loss_source"] == "universal_action_adapter"
    assert checkpoint["config"]["base_model_name"] == str(model_dir)
    assert checkpoint["config"]["freeze_backbone"] is True
    assert "universal_action_adapter_state_dict" in checkpoint
    design_report = json.loads(Path(report["design_report"]).read_text(encoding="utf-8"))
    assert design_report["retention_failure_analysis"]["old_checkpoint_role"] == "debug_only_not_full_init"
    assert design_report["routing_foundation_components"] == [
        "encoder.backbone",
        "encoder.proj",
        "skill_table.W",
        "skill_table.E",
        "native_rerank_sidecar",
    ]
    assert design_report["action_imitation_policy"]["disallowed_old_path"] == "skill_table.logits pseudo-skill CE"
    eval_report = json.loads(Path(report["eval_report"]).read_text(encoding="utf-8"))
    assert eval_report["action_loss_source"] == "universal_action_adapter"
    assert "action_recall_at_1" in eval_report["metrics"]


def test_aux_pretrain_full_can_freeze_routing_without_gate_markers(tmp_path):
    data_root = make_normalized_aux_pretrain_data(tmp_path / "aux_data")
    output_dir = tmp_path / "full"
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    routing_manifest = make_tiny_native_routing_init(tmp_path, model_dir, data_root)

    report = run_aux_trajectory_pretrain(
        data_root=data_root,
        base_model_name=None,
        output_dir=output_dir,
        max_steps=1,
        batch_size=2,
        model_dim=16,
        top_k=4,
        max_trajectories=None,
        include_splits=["train", "valid_seen"],
        exclude_splits=["test"],
        routing_init_manifest=routing_manifest,
        eval_splits=["test"],
        freeze_routing_foundation=True,
        use_universal_action_adapter=True,
        action_negative_k=2,
        candidate_strategy="environment_action_pool",
        run_role="auxiliary_pretrain_full_candidate",
    )

    assert report["paper_role"] == "auxiliary_pretrain_full_candidate"
    assert report["not_for_paper_table"] is False
    assert report["not_full_training_init"] is False
    assert report["frozen_routing_foundation"] is True
    assert report["feature_cache"]["used"] is True
    assert report["feature_cache"]["state_rows"] == report["step_count"]
    assert report["feature_cache"]["candidate_strategy"] == "environment_action_pool"
    checkpoint = torch.load(report["checkpoint"], map_location="cpu")
    assert checkpoint["paper_role"] == "auxiliary_pretrain_full_candidate"
    assert checkpoint["not_for_paper_table"] is False
    assert checkpoint["not_full_training_init"] is False
    assert checkpoint["feature_cache"]["used"] is True


def test_aux_pretrain_next_action_ce_replaces_cosine_transition_alignment(tmp_path):
    data_root = make_normalized_aux_pretrain_data(tmp_path / "aux_data")
    output_dir = tmp_path / "gate_next_action"
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    routing_manifest = make_tiny_native_routing_init(tmp_path, model_dir, data_root)

    report = run_aux_trajectory_pretrain(
        data_root=data_root,
        base_model_name=None,
        output_dir=output_dir,
        max_steps=1,
        batch_size=2,
        model_dim=16,
        top_k=4,
        max_trajectories=None,
        include_splits=["train", "valid_seen"],
        exclude_splits=["test"],
        routing_init_manifest=routing_manifest,
        eval_splits=["test"],
        freeze_routing_foundation=True,
        use_universal_action_adapter=True,
        action_negative_k=2,
        candidate_strategy="environment_action_pool",
        transition_objective="next_action_ce",
        max_eval_steps=4,
    )

    assert report["transition_objective"] == "next_action_ce"
    assert report["cosine_transition_alignment_enabled"] is False
    assert "next_action_prediction_loss" in report["trained_targets"]
    assert "transition_consistency_proxy" not in report["trained_targets"]
    assert "belief_update_proxy" not in report["trained_targets"]
    assert report["loss_taxonomy"]["next_action_prediction"] == "supervised_transition_prior_action_ce"
    assert report["metrics"]["next_action_prediction_loss"] >= 0.0
    assert "transition_consistency_proxy" not in report["metrics"]
    checkpoint = torch.load(report["checkpoint"], map_location="cpu")
    assert checkpoint["transition_objective"] == "next_action_ce"
    assert checkpoint["cosine_transition_alignment_enabled"] is False
    assert checkpoint["loss_taxonomy"]["cosine_transition_alignment"] == "disabled_ablation_only"


def test_build_aux_gate_report_compares_skillret_retention_thresholds(tmp_path):
    from clstr.aux_pretrain import build_aux_pretrain_gate_report

    train_report_path = tmp_path / "train_report.json"
    eval_report_path = tmp_path / "eval_report.json"
    retention_metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "gate_report.json"
    train_report_path.write_text(
        json.dumps(
            {
                "metrics": {"loss": 1.0, "action_imitation_loss": 0.5},
                "frozen_routing_foundation": True,
                "trainable_modules": ["transition", "gate", "stop_head", "universal_action_adapter"],
                "frozen_modules": ["encoder.backbone", "encoder.proj", "skill_table.W", "skill_table.E"],
                "checkpoint": "outputs/aux_trajectory_pretrain_gate/checkpoints/aux_trajectory_pretrain-step512.pt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_report_path.write_text(json.dumps({"metrics": {"action_recall_at_1": 0.25}}) + "\n", encoding="utf-8")
    retention_metrics_path.write_text(
        json.dumps({"NDCG@10": 0.704, "Recall@10": 0.755, "MAP@10": 0.631})
        + "\n",
        encoding="utf-8",
    )

    report = build_aux_pretrain_gate_report(
        train_report_path=train_report_path,
        eval_report_path=eval_report_path,
        retention_metrics_path=retention_metrics_path,
        output_path=output_path,
    )

    assert report["status"] == "pass"
    assert report["allow_full_non_test_auxiliary_training"] is True
    assert report["retention_thresholds"] == {
        "NDCG@10": 0.005,
        "Recall@10": 0.005,
        "MAP@10": 0.005,
    }
    assert report["retention_deltas"]["NDCG@10"] == -0.00184
    assert report["paper_role"] == "gate_debug_not_paper_result"
    assert output_path.exists()


def test_env_adapter_skeleton_exposes_non_trainable_closed_loop_interface():
    from clstr.envs.alfworld_adapter import ALFWorldEnvAdapter
    from clstr.envs.base import EnvStep
    from clstr.envs.webshop_adapter import WebShopEnvAdapter

    alfworld = ALFWorldEnvAdapter(env_id="alf-debug")
    webshop = WebShopEnvAdapter(env_id="webshop-debug")

    for adapter in (alfworld, webshop):
        observation = adapter.reset(task_id="task-1")
        assert isinstance(observation, str)
        assert adapter.observation_text() == observation
        assert adapter.candidate_actions()
        result = adapter.step(adapter.candidate_actions()[0])
        assert isinstance(result, EnvStep)
        assert isinstance(result.observation_text, str)
        assert isinstance(result.valid_actions, list)
        assert result.success in {True, False, None}
        assert adapter.is_trainable is False
