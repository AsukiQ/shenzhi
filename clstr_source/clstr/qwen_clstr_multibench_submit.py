from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clstr.memory_utility_records import canonical_digest
from clstr.frozen_clstr_route_eval import load_final_chain_manifest


QWEN_CLSTR_MULTIBENCH_STAGE_ORDER = (
    "frozen/toolbench_g3",
    "frozen/toolsandbox",
    "frozen/tau2",
    "frozen/alfworld_offline",
    "native/toolbench_g3",
    "native/toolsandbox",
    "native/tau2",
    "closed_loop/alfworld_valid_seen",
    "closed_loop/alfworld_valid_unseen",
)

DEFAULT_QWEN_CLSTR_EVAL_BATCH_PROFILE = {
    "frozen_tau2": 128,
    "native_tau2": 64,
    "native_tau2_stage0": 128,
    "alfworld": 4,
}


@dataclass(frozen=True)
class QwenClstrEvalStage:
    key: str
    job_name: str
    launcher: Path
    exports: Mapping[str, str]
    partition: str
    time_limit: str

    def command(self) -> list[str]:
        export_parts = ["ALL"]
        for key, value in sorted(self.exports.items()):
            text = str(value)
            if any(character in text for character in (",", "\n", "\r")):
                raise ValueError(f"unsafe Slurm export value for {key}")
            export_parts.append(f"{key}={text}")
        return [
            "sbatch",
            "--parsable",
            f"--partition={self.partition}",
            f"--time={self.time_limit}",
            f"--job-name={self.job_name}",
            f"--export={','.join(export_parts)}",
            str(self.launcher),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "job_name": self.job_name,
            "launcher": str(self.launcher),
            "exports": dict(sorted(self.exports.items())),
            "partition": self.partition,
            "time_limit": self.time_limit,
        }


def _path(value: Any, *, label: str, require_file: bool = False, require_dir: bool = False) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    path = Path(text).resolve()
    if require_file and not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if require_dir and not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _load_pinned_manifest(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _path(path, label=label, require_file=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    recorded = str(payload.get("manifest_sha256") or "")
    digest_payload = dict(payload)
    digest_payload.pop("manifest_sha256", None)
    if not recorded or canonical_digest(digest_payload) != recorded:
        raise ValueError(f"{label} self-hash mismatch")
    return payload


def load_accepted_qwen_clstr_smoke_gate(
    path: str | Path,
    *,
    checkpoint_chain_digest: str,
    final_chain_manifest_sha256: str,
    corpus_manifest_sha256_by_stage: Mapping[str, str] | None = None,
    expected_report_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = _path(path, label="accepted smoke gate", require_file=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("accepted smoke gate must contain an object")
    recorded = str(payload.get("report_sha256") or "")
    digest_payload = dict(payload)
    digest_payload.pop("report_sha256", None)
    if not recorded or canonical_digest(digest_payload) != recorded:
        raise ValueError("accepted smoke gate self-hash mismatch")
    if expected_report_sha256 is not None and recorded != str(
        expected_report_sha256
    ):
        raise ValueError("accepted smoke gate SHA-256 mismatch")
    if payload.get("schema_version") != "qwen06_clstr_multibench_report_v1":
        raise ValueError("accepted smoke gate schema mismatch")
    if payload.get("status") != "ok" or payload.get("blockers"):
        raise ValueError("accepted smoke gate is not ok")
    if payload.get("scope") != "smoke":
        raise ValueError("accepted smoke gate scope mismatch")
    if payload.get("stage_order") != list(QWEN_CLSTR_MULTIBENCH_STAGE_ORDER):
        raise ValueError("accepted smoke gate stage order mismatch")
    execution = payload.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("status") != "ok"
        or execution.get("blockers")
    ):
        raise ValueError("accepted smoke gate lacks terminal job evidence")
    jobs = execution.get("jobs")
    if not isinstance(jobs, Mapping) or set(jobs) != set(
        QWEN_CLSTR_MULTIBENCH_STAGE_ORDER
    ):
        raise ValueError("accepted smoke gate terminal job evidence is incomplete")
    for stage in QWEN_CLSTR_MULTIBENCH_STAGE_ORDER:
        row = jobs.get(stage)
        if (
            not isinstance(row, Mapping)
            or not re.fullmatch(r"[0-9]+", str(row.get("job_id") or ""))
            or row.get("state") != "COMPLETED"
            or row.get("exit_code") != "0:0"
        ):
            raise ValueError(
                f"accepted smoke gate terminal job evidence is invalid: {stage}"
            )
    if payload.get("checkpoint_chain_digest") != checkpoint_chain_digest:
        raise ValueError("accepted smoke gate checkpoint chain mismatch")
    if payload.get("final_chain_manifest_sha256") != final_chain_manifest_sha256:
        raise ValueError("accepted smoke gate final-chain manifest mismatch")
    if corpus_manifest_sha256_by_stage is not None and payload.get(
        "corpus_manifest_sha256_by_stage"
    ) != dict(corpus_manifest_sha256_by_stage):
        raise ValueError("accepted smoke gate corpus manifests mismatch")
    return {
        "path": resolved,
        "report_sha256": recorded,
        "report": payload,
    }


def _offline_corpora(value: Any) -> dict[str, dict[str, Path | str]]:
    required_benchmarks = {
        "toolbench_g3",
        "toolsandbox",
        "tau2",
        "alfworld_offline",
    }
    if not isinstance(value, Mapping) or set(value) != required_benchmarks:
        raise ValueError("offline_corpora must define all four frozen benchmarks")
    output: dict[str, dict[str, Path | str]] = {}
    expected_benchmark = {
        "toolbench_g3": "toolbench_g3",
        "toolsandbox": "toolsandbox",
        "tau2": "tau2",
        "alfworld_offline": "alfworld",
    }
    for key in QWEN_CLSTR_MULTIBENCH_STAGE_ORDER[:4]:
        benchmark = key.split("/", 1)[1]
        spec = value[benchmark]
        if not isinstance(spec, Mapping):
            raise ValueError(f"{benchmark} frozen corpus spec must be an object")
        manifest_path = _path(
            spec.get("manifest_path"),
            label=f"{benchmark} manifest",
            require_file=True,
        )
        source_rows_path = _path(
            spec.get("source_rows_path"),
            label=f"{benchmark} source rows",
            require_file=True,
        )
        skills_path = _path(
            spec.get("skills_path"),
            label=f"{benchmark} skills",
            require_file=True,
        )
        manifest = _load_pinned_manifest(manifest_path, label=f"{benchmark} manifest")
        if manifest.get("benchmark") != expected_benchmark[benchmark]:
            raise ValueError(f"{benchmark} manifest benchmark mismatch")
        output[benchmark] = {
            "manifest_path": manifest_path,
            "source_rows_path": source_rows_path,
            "skills_path": skills_path,
            "manifest_sha256": str(manifest["manifest_sha256"]),
        }
    return output


def _alfworld_split_manifests(value: Any) -> dict[str, dict[str, Path | str]]:
    if not isinstance(value, Mapping) or set(value) != {"valid_seen", "valid_unseen"}:
        raise ValueError("alfworld_manifest_by_split must define valid_seen and valid_unseen")
    output: dict[str, dict[str, Path | str]] = {}
    for split in ("valid_seen", "valid_unseen"):
        path = _path(value[split], label=f"ALFWorld {split} manifest", require_file=True)
        manifest = _load_pinned_manifest(path, label=f"ALFWorld {split} manifest")
        if manifest.get("benchmark") != "alfworld" or manifest.get("split") != split:
            raise ValueError(f"ALFWorld {split} manifest identity mismatch")
        output[split] = {
            "manifest_path": path,
            "manifest_sha256": str(manifest["manifest_sha256"]),
        }
    return output


def _native_inputs(value: Any) -> dict[str, dict[str, Path]]:
    if not isinstance(value, Mapping) or set(value) != {
        "toolbench_g3",
        "toolsandbox",
        "tau2",
        "alfworld",
    }:
        raise ValueError("native_inputs must define ToolBench, ToolSandbox, tau2, and ALFWorld")
    toolbench = value["toolbench_g3"]
    toolsandbox = value["toolsandbox"]
    tau2 = value["tau2"]
    alfworld = value["alfworld"]
    for label, spec in (
        ("toolbench_g3", toolbench),
        ("toolsandbox", toolsandbox),
        ("tau2", tau2),
        ("alfworld", alfworld),
    ):
        if not isinstance(spec, Mapping):
            raise ValueError(f"{label} native input spec must be an object")
    return {
        "toolbench_g3": {
            "train_trajectories_path": _path(
                toolbench.get("train_trajectories_path"),
                label="ToolBench train trajectories",
                require_file=True,
            ),
            "eval_trajectories_path": _path(
                toolbench.get("eval_trajectories_path"),
                label="ToolBench eval trajectories",
                require_file=True,
            ),
        },
        "toolsandbox": {
            "scenarios_root": _path(
                toolsandbox.get("scenarios_root"),
                label="ToolSandbox scenarios root",
                require_dir=True,
            ),
            "tools_root": _path(
                toolsandbox.get("tools_root"),
                label="ToolSandbox tools root",
                require_dir=True,
            ),
        },
        "tau2": {
            "data_root": _path(
                tau2.get("data_root"),
                label="tau2 data root",
                require_dir=True,
            ),
        },
        "alfworld": {
            "official_repo": _path(
                alfworld.get("official_repo"),
                label="ALFWorld official repo",
                require_dir=True,
            ),
            "data_dir": _path(
                alfworld.get("data_dir"),
                label="ALFWorld data dir",
                require_dir=True,
            ),
            "qwen_model_name_or_path": _path(
                alfworld.get("qwen_model_name_or_path"),
                label="ALFWorld Qwen3-14B executor",
                require_dir=True,
            ),
        },
    }


def _evaluation_batch_profile(value: Any) -> dict[str, int]:
    if value is None:
        overrides: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        overrides = value
    else:
        raise ValueError("batch_profile must be an object")
    unknown = set(overrides) - set(DEFAULT_QWEN_CLSTR_EVAL_BATCH_PROFILE)
    if unknown:
        raise ValueError(f"batch_profile has unsupported keys: {sorted(unknown)}")
    output = dict(DEFAULT_QWEN_CLSTR_EVAL_BATCH_PROFILE)
    for key, raw in overrides.items():
        if type(raw) is not int or int(raw) <= 0:
            raise ValueError(f"batch_profile.{key} must be a positive integer")
        output[str(key)] = int(raw)
    return output


def _common_exports(
    *,
    project_root: Path,
    output_dir: Path,
    scope: str,
    final_chain_path: Path,
    final_chain: Mapping[str, Any],
    corpus_manifest_path: Path,
    corpus_manifest_sha256: str,
) -> dict[str, str]:
    checkpoints = final_chain["checkpoints"]
    reliability = final_chain["reliability"]
    mode = str(reliability["mode"])
    runtime_mode = "cmc" if mode == "cmc_candidate_gate" else mode
    gate = reliability.get("gate_checkpoint") or {}
    return {
        "PROJECT_ROOT": str(project_root),
        "OUTPUT_DIR": str(output_dir),
        "EVAL_SCOPE": scope,
        "FINAL_CHAIN_MANIFEST_PATH": str(final_chain_path),
        "FINAL_CHAIN_MANIFEST_SHA256": str(final_chain["manifest_sha256"]),
        "CHECKPOINT_CHAIN_DIGEST": str(final_chain["checkpoint_chain_digest"]),
        "STAGE0_CHECKPOINT_PATH": str(checkpoints["stage0"]["path"]),
        "STAGE2_CHECKPOINT_PATH": str(checkpoints["stage2"]["path"]),
        "STAGE4_CHECKPOINT_PATH": str(checkpoints["stage4"]["path"]),
        "TRAINING_SKILLS_PATH": str(final_chain["skill_pool"]["path"]),
        "CORPUS_MANIFEST_PATH": str(corpus_manifest_path),
        "CORPUS_MANIFEST_SHA256": str(corpus_manifest_sha256),
        "ROUTE_SCORER": "unified_memory",
        "RELIABILITY_MODE": runtime_mode,
        "FIXED_ALPHA": str(reliability.get("fixed_alpha", 1.0) if mode == "fixed_alpha" else 1.0),
        "SAFE_MEMORY_RESIDUAL_BOUND": str(
            reliability.get("safe_memory_residual_bound", 2.0)
        ),
        "MEMORY_UTILITY_GATE_CHECKPOINT_PATH": str(gate.get("path") or ""),
        "MEMORY_UTILITY_GATE_CHECKPOINT_SHA256": str(gate.get("sha256") or ""),
        "MEMORY_UTILITY_GATE_AUDIT_SHA256": str(reliability.get("audit_manifest_sha256") or ""),
        "FEATURE_UPDATE_COUNT_CAP": str(
            reliability.get("feature_update_count_cap", 1.0)
        ),
        "FEATURE_CANDIDATE_COUNT_CAP": str(
            reliability.get("feature_candidate_count_cap", 1.0)
        ),
    }


def build_qwen_clstr_multibench_stages(
    config: Mapping[str, Any],
    *,
    scope: str,
    partition: str = "gpu_a800",
) -> list[QwenClstrEvalStage]:
    if not isinstance(config, Mapping):
        raise ValueError("evaluation config must be an object")
    scope = str(scope).strip()
    if scope not in {"smoke", "full"}:
        raise ValueError("scope must be smoke or full")
    project_root = _path(config.get("project_root"), label="project_root", require_dir=True)
    output_root = _path(config.get("output_root"), label="output_root")
    final_chain_path = _path(
        config.get("final_chain_manifest_path"),
        label="final chain manifest",
        require_file=True,
    )
    final_chain = load_final_chain_manifest(final_chain_path)
    batch_profile = _evaluation_batch_profile(config.get("batch_profile"))
    offline = _offline_corpora(config.get("offline_corpora"))
    split_manifests = _alfworld_split_manifests(config.get("alfworld_manifest_by_split"))
    native = _native_inputs(config.get("native_inputs"))
    expected_corpus_manifests = {
        "frozen/toolbench_g3": str(offline["toolbench_g3"]["manifest_sha256"]),
        "frozen/toolsandbox": str(offline["toolsandbox"]["manifest_sha256"]),
        "frozen/tau2": str(offline["tau2"]["manifest_sha256"]),
        "frozen/alfworld_offline": str(offline["alfworld_offline"]["manifest_sha256"]),
        "native/toolbench_g3": str(offline["toolbench_g3"]["manifest_sha256"]),
        "native/toolsandbox": str(offline["toolsandbox"]["manifest_sha256"]),
        "native/tau2": str(offline["tau2"]["manifest_sha256"]),
        "closed_loop/alfworld_valid_seen": str(
            split_manifests["valid_seen"]["manifest_sha256"]
        ),
        "closed_loop/alfworld_valid_unseen": str(
            split_manifests["valid_unseen"]["manifest_sha256"]
        ),
    }
    accepted_smoke_gate = None
    if scope == "full":
        accepted_smoke_gate = load_accepted_qwen_clstr_smoke_gate(
            config.get("accepted_smoke_gate_path"),
            checkpoint_chain_digest=str(final_chain["checkpoint_chain_digest"]),
            final_chain_manifest_sha256=str(final_chain["manifest_sha256"]),
            corpus_manifest_sha256_by_stage=expected_corpus_manifests,
        )
    smoke_gate_exports = (
        {}
        if accepted_smoke_gate is None
        else {
            "ACCEPTED_SMOKE_GATE_PATH": str(accepted_smoke_gate["path"]),
            "ACCEPTED_SMOKE_GATE_SHA256": str(
                accepted_smoke_gate["report_sha256"]
            ),
        }
    )
    frozen_launcher = _path(
        project_root / "scripts/sbatch/run_qwen06_clstr_frozen_route_eval.sh",
        label="frozen route launcher",
        require_file=True,
    )
    native_launcher = _path(
        project_root / "scripts/sbatch/run_qwen06_clstr_native_route_eval.sh",
        label="native route launcher",
        require_file=True,
    )
    alfworld_launcher = _path(
        project_root / "scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh",
        label="ALFWorld executor launcher",
        require_file=True,
    )
    stages: list[QwenClstrEvalStage] = []

    for benchmark in ("toolbench_g3", "toolsandbox", "tau2", "alfworld_offline"):
        corpus = offline[benchmark]
        exports = {
            **_common_exports(
                project_root=project_root,
                output_dir=output_root / scope / "frozen" / benchmark,
                scope=scope,
                final_chain_path=final_chain_path,
                final_chain=final_chain,
                corpus_manifest_path=Path(corpus["manifest_path"]),
                corpus_manifest_sha256=str(corpus["manifest_sha256"]),
            ),
            **smoke_gate_exports,
            "BENCHMARK": "alfworld" if benchmark == "alfworld_offline" else benchmark,
            "BENCHMARK_MANIFEST_PATH": str(corpus["manifest_path"]),
            "SOURCE_ROWS_PATH": str(corpus["source_rows_path"]),
            "FROZEN_SKILLS_PATH": str(corpus["skills_path"]),
            "BATCH_SIZE": str(
                batch_profile["frozen_tau2"] if benchmark == "tau2" else 16
            ),
            "TOP_K": "100",
        }
        if scope == "smoke":
            exports["MAX_EVAL_ROWS"] = "32"
        stages.append(
            QwenClstrEvalStage(
                key=f"frozen/{benchmark}",
                job_name=f"q06_clstr_frozen_{benchmark}"[:64],
                launcher=frozen_launcher,
                exports=exports,
                partition=str(partition),
                time_limit=(
                    "02:00:00"
                    if scope == "smoke"
                    else {
                        "toolbench_g3": "12:00:00",
                        "toolsandbox": "04:00:00",
                        "tau2": "08:00:00",
                        "alfworld_offline": "04:00:00",
                    }[benchmark]
                ),
            )
        )

    for benchmark in ("toolbench_g3", "toolsandbox", "tau2"):
        corpus = offline[benchmark]
        exports = {
            **_common_exports(
                project_root=project_root,
                output_dir=output_root / scope / "native" / benchmark,
                scope=scope,
                final_chain_path=final_chain_path,
                final_chain=final_chain,
                corpus_manifest_path=Path(corpus["manifest_path"]),
                corpus_manifest_sha256=str(corpus["manifest_sha256"]),
            ),
            **smoke_gate_exports,
            "BENCHMARK": benchmark,
            "BATCH_SIZE": str(
                batch_profile["native_tau2"] if benchmark == "tau2" else 8
            ),
            "STAGE0_CANDIDATE_BATCH_SIZE": str(
                batch_profile["native_tau2_stage0"] if benchmark == "tau2" else 16
            ),
            "CANDIDATE_COUNT": "100" if benchmark == "toolbench_g3" else "64",
        }
        if benchmark == "toolbench_g3":
            exports.update({"STATIC_K": "500", "DYNAMIC_EXTRA_K": "64"})
        elif benchmark == "toolsandbox":
            exports.update(
                {
                    "MODEL_SKILL_POOL_MODE": "checkpoint_faithful",
                    "TOOLSANDBOX_REPLAY_MODE": "corrected_causal",
                }
            )
        elif benchmark == "tau2":
            exports["TASK_SPLIT"] = "base"
        exports.update({key.upper(): str(value) for key, value in native[benchmark].items()})
        if scope == "smoke":
            if benchmark == "toolbench_g3":
                exports.update({"MAX_TRAIN_ROWS": "256", "MAX_EVAL_ROWS": "128"})
            elif benchmark == "toolsandbox":
                exports.update({"MAX_SCENARIOS": "20", "MAX_EVAL_ROWS": "128"})
            else:
                exports.update({"MAX_TASKS_PER_DOMAIN": "20", "MAX_EVAL_ROWS": "64"})
        stages.append(
            QwenClstrEvalStage(
                key=f"native/{benchmark}",
                job_name=f"q06_clstr_native_{benchmark}"[:64],
                launcher=native_launcher,
                exports=exports,
                partition=str(partition),
                time_limit=(
                    "02:00:00"
                    if scope == "smoke"
                    else {"toolbench_g3": "12:00:00", "toolsandbox": "04:00:00", "tau2": "08:00:00"}[
                        benchmark
                    ]
                ),
            )
        )

    for split in ("valid_seen", "valid_unseen"):
        manifest = split_manifests[split]
        exports = {
            **_common_exports(
                project_root=project_root,
                output_dir=output_root / scope / "closed_loop" / f"alfworld_{split}",
                scope=scope,
                final_chain_path=final_chain_path,
                final_chain=final_chain,
                corpus_manifest_path=Path(manifest["manifest_path"]),
                corpus_manifest_sha256=str(manifest["manifest_sha256"]),
            ),
            **smoke_gate_exports,
            "OFFICIAL_REPO": str(native["alfworld"]["official_repo"]),
            "DATA_DIR": str(native["alfworld"]["data_dir"]),
            "ALFWORLD_SKILLS_PATH": str(offline["alfworld_offline"]["skills_path"]),
            "ALFWORLD_SKILLS_MANIFEST_PATH": str(
                offline["alfworld_offline"]["manifest_path"]
            ),
            "ALFWORLD_SKILLS_MANIFEST_SHA256": str(
                offline["alfworld_offline"]["manifest_sha256"]
            ),
            "SPLITS": split,
            "MAX_STEPS": "50",
            "BATCH_SIZE": str(batch_profile["alfworld"]),
            "QWEN_MODEL_NAME_OR_PATH": str(
                native["alfworld"]["qwen_model_name_or_path"]
            ),
            "QWEN_WEIGHT": "1.0",
            "CLSTR_WEIGHT": "0.25",
            "CLSTR_PRIOR_MODE": "unified_memory_concrete_action",
            "SCORING_METHOD": "generate",
            "LOOP_GUARD": "1",
            "RUN_QWEN_ONLY": "1",
            "RUN_HYBRID": "1",
            "LOCAL_FILES_ONLY": "1",
            "MEMORY_PROTOCOL": "stateful_post_action_v1",
        }
        if scope == "smoke":
            exports["MAX_EPISODES"] = "5"
        else:
            exports["MAX_EPISODES"] = "0"
        stages.append(
            QwenClstrEvalStage(
                key=f"closed_loop/alfworld_{split}",
                job_name=f"q06_clstr_alfworld_{split}"[:64],
                launcher=alfworld_launcher,
                exports=exports,
                partition=str(partition),
                time_limit="03:00:00" if scope == "smoke" else "1-00:00:00",
            )
        )

    if tuple(stage.key for stage in stages) != QWEN_CLSTR_MULTIBENCH_STAGE_ORDER:
        raise RuntimeError("Qwen CLSTR multibench stage order drifted")
    return stages


def _submission_identity(stages: list[QwenClstrEvalStage]) -> dict[str, Any]:
    if not stages:
        raise ValueError("Qwen CLSTR multibench stage plan must not be empty")
    keys = [stage.key for stage in stages]
    if len(keys) != len(set(keys)):
        raise ValueError("Qwen CLSTR multibench stage keys must be unique")
    chain_digests = {stage.exports.get("CHECKPOINT_CHAIN_DIGEST") for stage in stages}
    final_manifests = {stage.exports.get("FINAL_CHAIN_MANIFEST_SHA256") for stage in stages}
    scopes = {stage.exports.get("EVAL_SCOPE") for stage in stages}
    if len(chain_digests) != 1 or None in chain_digests:
        raise ValueError("all multibench stages must pin one checkpoint-chain digest")
    if len(final_manifests) != 1 or None in final_manifests:
        raise ValueError("all multibench stages must pin one final-chain manifest")
    if len(scopes) != 1 or None in scopes:
        raise ValueError("all multibench stages must use one evaluation scope")
    scope = next(iter(scopes))
    accepted_smoke_gates = {
        str(stage.exports.get("ACCEPTED_SMOKE_GATE_SHA256") or "").strip()
        for stage in stages
    }
    if scope == "full":
        if len(accepted_smoke_gates) != 1 or not re.fullmatch(
            r"[0-9a-f]{64}", next(iter(accepted_smoke_gates), "")
        ):
            raise ValueError("all full multibench stages must pin one accepted smoke gate")
        accepted_smoke_gate_sha256: str | None = next(iter(accepted_smoke_gates))
    else:
        if accepted_smoke_gates != {""}:
            raise ValueError("smoke multibench stages must not pin an accepted smoke gate")
        accepted_smoke_gate_sha256 = None
    corpus_manifests = {
        stage.key: str(stage.exports.get("CORPUS_MANIFEST_SHA256") or "")
        for stage in stages
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in corpus_manifests.values()):
        raise ValueError("every multibench stage must pin a corpus manifest SHA-256")
    return {
        "stage_order": keys,
        "checkpoint_chain_digest": next(iter(chain_digests)),
        "final_chain_manifest_sha256": next(iter(final_manifests)),
        "scope": scope,
        "accepted_smoke_gate_sha256": accepted_smoke_gate_sha256,
        "corpus_manifest_sha256_by_stage": corpus_manifests,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Qwen CLSTR multibench registry: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Qwen CLSTR multibench registry must contain an object")
    return payload


def _job_id(value: str) -> str:
    token = str(value).strip().splitlines()[-1].split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", token):
        raise ValueError(f"invalid parsable Slurm job ID: {value!r}")
    return token


def submit_qwen_clstr_multibench_stage_plan(
    *,
    stages: list[QwenClstrEvalStage],
    registry_path: str | Path,
    submitter: Any,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    identity = _submission_identity(stages)
    input_fingerprint = canonical_digest(identity)
    plan_fingerprint = canonical_digest([stage.to_dict() for stage in stages])
    if dry_run:
        return {
            "status": "dry_run",
            "input_fingerprint": input_fingerprint,
            "plan_fingerprint": plan_fingerprint,
            "stage_order": list(identity["stage_order"]),
            "commands": [stage.command() for stage in stages],
        }

    registry_path = Path(registry_path)
    if registry_path.exists():
        if not resume:
            raise FileExistsError(f"submission registry already exists: {registry_path}")
        registry = _read_registry(registry_path)
        if registry.get("input_fingerprint") != input_fingerprint:
            raise ValueError("Qwen CLSTR multibench input fingerprint mismatch")
        if registry.get("plan_fingerprint") != plan_fingerprint:
            raise ValueError("Qwen CLSTR multibench stage plan mismatch")
    else:
        registry = {
            "schema_version": 1,
            "status": "submitting",
            "input_fingerprint": input_fingerprint,
            "plan_fingerprint": plan_fingerprint,
            "submission_identity": identity,
            "stage_order": list(identity["stage_order"]),
            "stages": {},
            "created_at_unix": time.time(),
        }
        _atomic_json(registry_path, registry)

    stored_stages = registry.get("stages")
    if not isinstance(stored_stages, dict):
        raise ValueError("Qwen CLSTR multibench registry stages must be an object")
    if registry.get("status") == "submitted" and all(
        isinstance(stored_stages.get(stage.key), Mapping)
        and stored_stages[stage.key].get("status") == "submitted"
        and str(stored_stages[stage.key].get("job_id") or "")
        for stage in stages
    ):
        return registry

    for stage in stages:
        previous = stored_stages.get(stage.key)
        if isinstance(previous, Mapping):
            if previous.get("status") == "submitted" and str(previous.get("job_id") or ""):
                continue
            if previous.get("status") == "submitting":
                raise RuntimeError(
                    f"ambiguous prior submission for stage {stage.key}; inspect Slurm before resuming"
                )
        command = stage.command()
        stored_stages[stage.key] = {
            "status": "submitting",
            "stage": stage.to_dict(),
            "command": command,
            "updated_at_unix": time.time(),
        }
        _atomic_json(registry_path, registry)
        job_id = _job_id(submitter(command))
        stored_stages[stage.key] = {
            "status": "submitted",
            "job_id": job_id,
            "stage": stage.to_dict(),
            "command": command,
            "updated_at_unix": time.time(),
        }
        _atomic_json(registry_path, registry)

    registry["status"] = "submitted"
    registry["updated_at_unix"] = time.time()
    _atomic_json(registry_path, registry)
    return registry
