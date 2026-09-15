from __future__ import annotations

import importlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from clstr.envs.base import EnvAdapter, EnvStep


DEFAULT_APPWORLD_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
DEFAULT_APPWORLD_CACHE = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_appworld_paths(
    appworld_root: str | Path | None = None,
    appworld_cache: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(appworld_root or os.environ.get("APPWORLD_ROOT") or DEFAULT_APPWORLD_ROOT)
    cache = Path(appworld_cache or os.environ.get("APPWORLD_CACHE") or DEFAULT_APPWORLD_CACHE)
    os.environ["APPWORLD_ROOT"] = str(root)
    os.environ["APPWORLD_CACHE"] = str(cache)
    os.environ.setdefault("IPYTHONDIR", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython")
    cache.mkdir(parents=True, exist_ok=True)
    Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)
    return root, cache


def load_appworld_task_ids(appworld_root: str | Path, split: str = "train") -> list[str]:
    dataset_path = Path(appworld_root) / "data" / "datasets" / f"{split}.txt"
    if not dataset_path.exists():
        return []
    return [line.strip() for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_appworld_task_id(appworld_root: str | Path, split: str = "train") -> str | None:
    task_ids = load_appworld_task_ids(appworld_root, split=split)
    return task_ids[0] if task_ids else None


def _json_if_exists(path: Path, default: Any) -> Any:
    return _read_json(path) if path.exists() else default


def _api_docs_summary(data_dir: Path) -> dict[str, Any]:
    docs_dir = data_dir / "api_docs" / "standard"
    if not docs_dir.exists():
        return {"available": False, "apps": [], "app_count": 0}
    apps = sorted(path.stem for path in docs_dir.glob("*.json") if path.stem != "api_docs")
    return {"available": bool(apps), "apps": apps, "app_count": len(apps)}


def _world_factory_from_appworld() -> Callable[..., Any]:
    module = importlib.import_module("appworld.environment")
    return getattr(module, "AppWorld")


@dataclass(frozen=True)
class AppWorldTaskContext:
    task_id: str
    instruction: str
    required_apps: list[str]
    allowed_apps: list[str]
    api_docs: dict[str, Any]
    db_available: bool
    ground_truth_available: bool
    verifier_available: bool
    reward_done_available: bool

    def to_report(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "instruction_nonempty": bool(self.instruction.strip()),
            "required_apps": self.required_apps,
            "required_apps_count": len(self.required_apps),
            "allowed_apps": self.allowed_apps,
            "allowed_apps_count": len(self.allowed_apps),
            "api_docs": self.api_docs,
            "api_docs_available": bool(self.api_docs.get("available")),
            "db_available": self.db_available,
            "ground_truth_available": self.ground_truth_available,
            "verifier_available": self.verifier_available,
            "reward_done_available": self.reward_done_available,
        }


class AppWorldEnvAdapter(EnvAdapter):
    """Minimal CLSTR adapter for one AppWorld task.

    This adapter intentionally keeps the first milestone small: it loads task
    metadata, can initialize one official AppWorld runtime, and exposes the
    Python-code execution entry. It does not run a benchmark loop or train.
    """

    def __init__(
        self,
        *,
        appworld_root: str | Path | None = None,
        appworld_cache: str | Path | None = None,
        split: str = "train",
        task_id: str | None = None,
        experiment_name: str = "clstr_appworld_adapter_smoke",
        world_factory: Callable[..., Any] | None = None,
        timeout_seconds: int = 20,
        max_interactions: int = 3,
    ) -> None:
        self.appworld_root, self.appworld_cache = configure_appworld_paths(appworld_root, appworld_cache)
        self.split = split
        self.task_id = task_id or select_appworld_task_id(self.appworld_root, split=split)
        self.experiment_name = experiment_name
        self.world_factory = world_factory
        self.timeout_seconds = timeout_seconds
        self.max_interactions = max_interactions
        self._world: Any | None = None
        self._context: AppWorldTaskContext | None = None
        self._observation = ""

    @property
    def data_dir(self) -> Path:
        return self.appworld_root / "data"

    @property
    def task_dir(self) -> Path:
        if not self.task_id:
            return self.data_dir / "tasks" / "__missing__"
        return self.data_dir / "tasks" / self.task_id

    def load_task_context(self) -> AppWorldTaskContext:
        if not self.task_id:
            raise FileNotFoundError(f"No task id found for split '{self.split}'.")
        specs_path = self.task_dir / "specs.json"
        if not specs_path.exists():
            raise FileNotFoundError(f"AppWorld task specs not found: {specs_path}")
        specs = _read_json(specs_path)
        ground_truth_dir = self.task_dir / "ground_truth"
        required_apps = _json_if_exists(ground_truth_dir / "required_apps.json", [])
        if not isinstance(required_apps, list):
            required_apps = []
        db_dir = self.task_dir / "dbs"
        allowed_apps = sorted(path.stem for path in db_dir.glob("*.jsonl")) if db_dir.exists() else []
        api_docs = _api_docs_summary(self.data_dir)
        self._context = AppWorldTaskContext(
            task_id=self.task_id,
            instruction=str(specs.get("instruction", "")),
            required_apps=[str(item) for item in required_apps],
            allowed_apps=allowed_apps,
            api_docs=api_docs,
            db_available=db_dir.exists() and any(db_dir.glob("*.jsonl")),
            ground_truth_available=ground_truth_dir.exists(),
            verifier_available=(ground_truth_dir / "evaluation.py").exists(),
            reward_done_available=(ground_truth_dir / "evaluation.py").exists(),
        )
        return self._context

    def reset(self, task_id: str | None = None) -> str:
        if task_id is not None:
            self.task_id = task_id
            self._context = None
        context = self.load_task_context()
        factory = self.world_factory or _world_factory_from_appworld()
        self._world = factory(
            self.task_id,
            experiment_name=self.experiment_name,
            max_interactions=self.max_interactions,
            timeout_seconds=self.timeout_seconds,
            load_ground_truth=True,
            ground_truth_mode="minimal",
            show_api_response_schemas=False,
        )
        self._observation = f"instruction: {context.instruction}"
        return self._observation

    def step(self, action_text: str) -> EnvStep:
        if self._world is None:
            raise RuntimeError("AppWorld runtime is not initialized. Call reset() first.")
        output = self._world.execute(action_text)
        done = False
        try:
            done = bool(self._world.task_completed())
        except Exception:
            done = False
        self._observation = str(output)
        return EnvStep(
            observation_text=self._observation,
            reward=None,
            done=done,
            success=None,
            valid_actions=self.candidate_actions(),
        )

    def observation_text(self) -> str:
        if self._observation:
            return self._observation
        context = self._context or self.load_task_context()
        return f"instruction: {context.instruction}"

    def candidate_actions(self) -> list[str]:
        return ["execute_python_code"]

    def execution_entry_available(self) -> bool:
        if self._world is not None:
            return callable(getattr(self._world, "execute", None))
        if self.world_factory is not None:
            return True
        try:
            spec = importlib.util.find_spec("appworld.environment")
        except ModuleNotFoundError:
            return False
        if spec is None:
            return False
        try:
            return callable(getattr(_world_factory_from_appworld(), "execute", None))
        except Exception:
            return False

    def reward_done_available(self) -> bool:
        if self._world is not None:
            return callable(getattr(self._world, "task_completed", None))
        context = self._context or self.load_task_context()
        return context.reward_done_available

    def close(self) -> None:
        if self._world is not None and callable(getattr(self._world, "close", None)):
            self._world.close()


def run_appworld_adapter_smoke(
    *,
    appworld_root: str | Path | None = None,
    appworld_cache: str | Path | None = None,
    output_dir: str | Path = "outputs/appworld_adapter_smoke",
    split: str = "train",
    task_id: str | None = None,
    attempt_reset: bool = True,
    world_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    adapter = AppWorldEnvAdapter(
        appworld_root=appworld_root,
        appworld_cache=appworld_cache,
        split=split,
        task_id=task_id,
        world_factory=world_factory,
    )
    report: dict[str, Any] = {
        "status": "ok",
        "appworld_root": str(adapter.appworld_root),
        "appworld_cache": str(adapter.appworld_cache),
        "split": split,
        "appworld_import_ok": importlib.util.find_spec("appworld") is not None,
        "reset_attempted": attempt_reset,
        "reset_ok": None,
    }
    try:
        context = adapter.load_task_context()
        report.update(context.to_report())
        report["execution_entry_available"] = adapter.execution_entry_available()
        if attempt_reset:
            observation = adapter.reset()
            report["reset_ok"] = True
            report["reset_observation_nonempty"] = bool(observation.strip())
            report["execution_entry_available"] = adapter.execution_entry_available()
            report["reward_done_available"] = adapter.reward_done_available()
    except Exception as exc:
        report["status"] = "blocked" if not report.get("instruction_nonempty") else "partial"
        report["blocker"] = "appworld_adapter_smoke_failed"
        report["error"] = repr(exc)
    finally:
        adapter.close()

    out = Path(output_dir)
    _write_json(out / "report.json", report)
    if report["status"] != "ok":
        _write_json(out / "blocker_report.json", report)
    return report
