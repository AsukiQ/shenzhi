from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clstr.envs.base import EnvAdapter, EnvStep


DEFAULT_SCIENCEWORLD_REPO_PATH = Path(__file__).resolve().parents[2] / "third_party" / "ScienceWorld"


OFFICIAL_SCIENCEWORLD_REFERENCE = {
    "github": "https://github.com/allenai/ScienceWorld",
    "pypi": "https://pypi.org/project/scienceworld/",
    "install": [
        "python -m pip install scienceworld",
        f"git clone https://github.com/allenai/ScienceWorld.git {DEFAULT_SCIENCEWORLD_REPO_PATH} && cd {DEFAULT_SCIENCEWORLD_REPO_PATH} && python -m pip install .",
    ],
    "java": "Java 1.8+ runtime is required by the official ScienceWorld harness.",
    "smoke": "python examples/random_agent.py --task-num=13 --num-episodes=5 --simplifications-preset easy",
}


def _repo_is_scienceworld(repo_path: Path) -> bool:
    return (
        repo_path.exists()
        and (repo_path / "examples" / "random_agent.py").exists()
        and ((repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists())
    )


def build_scienceworld_smoke_report(
    repo_path: str | Path | None = None,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    run_java: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    run_smoke: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    blockers: dict[str, Any] = {}
    package_info: dict[str, Any] = {"available": False}
    java_info: dict[str, Any] = {"available": False}
    repo_info: dict[str, Any] = {"checked": repo_path is not None, "available": repo_path is None}
    smoke_info: dict[str, Any] = {"attempted": False, "success": False}

    try:
        scienceworld = import_module("scienceworld")
        package_info = {
            "available": True,
            "version": getattr(scienceworld, "__version__", None),
        }
    except Exception as exc:  # pragma: no cover - exact import exception depends on local env
        blockers["python_package"] = {
            "message": "Python package 'scienceworld' is not importable.",
            "error": repr(exc),
            "install_command": "python -m pip install scienceworld",
        }

    try:
        java = run_java(["java", "-version"], capture_output=True, text=True, timeout=10)
        java_output = (java.stderr or java.stdout or "").strip()
        if java.returncode == 0:
            java_info = {"available": True, "version_output": java_output}
        else:
            blockers["java_runtime"] = {
                "message": "Java runtime check failed.",
                "returncode": java.returncode,
                "output": java_output,
                "install_command": "sudo apt-get update && sudo apt-get install -y openjdk-21-jdk",
            }
    except Exception as exc:  # pragma: no cover - exact subprocess exception depends on local env
        blockers["java_runtime"] = {
            "message": "Java runtime is not available on PATH.",
            "error": repr(exc),
            "install_command": "sudo apt-get update && sudo apt-get install -y openjdk-21-jdk",
        }

    if repo_path is not None:
        repo = Path(repo_path)
        repo_info = {
            "checked": True,
            "path": str(repo),
            "available": _repo_is_scienceworld(repo),
        }
        if not repo_info["available"]:
            blockers["repo_path"] = {
                "message": "ScienceWorld repo path is missing or does not look like the official repo.",
                "path": str(repo),
                "install_command": f"git clone https://github.com/allenai/ScienceWorld.git {repo}",
            }

    if not blockers and repo_path is not None:
        repo = Path(repo_path)
        smoke_command = [
            sys.executable,
            "examples/random_agent.py",
            "--task-num=13",
            "--num-episodes=1",
            "--simplifications-preset",
            "easy",
        ]
        try:
            smoke = run_smoke(smoke_command, cwd=str(repo), capture_output=True, text=True, timeout=120)
            smoke_output = "\n".join(part for part in [smoke.stdout, smoke.stderr] if part).strip()
            smoke_info = {
                "attempted": True,
                "success": smoke.returncode == 0,
                "command": " ".join(smoke_command),
                "returncode": smoke.returncode,
                "output": smoke_output[-4000:],
            }
            if smoke.returncode != 0:
                blockers["smoke_run"] = {
                    "message": "Official ScienceWorld smoke command failed.",
                    "command": " ".join(smoke_command),
                    "returncode": smoke.returncode,
                    "output": smoke_output[-4000:],
                    "install_command": "python -m pip install scienceworld && sudo apt-get update && sudo apt-get install -y openjdk-21-jdk",
                }
        except Exception as exc:  # pragma: no cover - exact runtime exception depends on local env
            smoke_info = {
                "attempted": True,
                "success": False,
                "error": repr(exc),
            }
            blockers["smoke_run"] = {
                "message": "Official ScienceWorld smoke command could not be executed.",
                "error": repr(exc),
                "install_command": "python -m pip install scienceworld && sudo apt-get update && sudo apt-get install -y openjdk-21-jdk",
            }

    status = "ok" if not blockers else "blocker"
    return {
        "status": status,
        "smoke_success": smoke_info["success"],
        "package": package_info,
        "java": java_info,
        "repo": repo_info,
        "smoke": smoke_info,
        "blockers": blockers,
        "install_commands": sorted(
            {
                item["install_command"]
                for item in blockers.values()
                if isinstance(item, dict) and item.get("install_command")
            }
        ),
        "official_reference": OFFICIAL_SCIENCEWORLD_REFERENCE,
    }


def write_scienceworld_smoke_report(
    output_path: str | Path = "outputs/scienceworld_eval/harness_smoke_report.json",
    repo_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_scienceworld_smoke_report(repo_path=repo_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _extract_valid_actions(info: dict[str, Any] | None) -> list[str]:
    if not isinstance(info, dict):
        return []
    for key in ("valid", "valid_actions", "admissible_actions", "look", "possibleActions"):
        values = info.get(key)
        if isinstance(values, (list, tuple)):
            return [str(item) for item in values]
    return []


def _env_valid_actions(env: Any) -> list[str]:
    for name in ("get_valid_action_object_combinations", "getValidActionObjectCombinations"):
        getter = getattr(env, name, None)
        if callable(getter):
            return [str(item) for item in getter()]
    for name in ("get_valid_action_object_combinations_with_templates", "getValidActionObjectCombinationsWithTemplates"):
        getter = getattr(env, name, None)
        if callable(getter):
            rows = getter()
            actions = []
            for row in rows or []:
                if isinstance(row, dict) and row.get("action"):
                    actions.append(str(row["action"]))
                elif row:
                    actions.append(str(row))
            return actions
    return []


def _extract_success(reward: float | int | None, done: bool, info: dict[str, Any] | None) -> bool | None:
    if not isinstance(info, dict):
        return bool(done and reward and float(reward) > 0.0) if done else None
    for key in ("success", "won", "completed"):
        if key in info:
            return bool(info[key])
    score = info.get("score")
    if score is not None:
        try:
            return bool(done and float(score) >= 1.0)
        except (TypeError, ValueError):
            return None
    return bool(done and reward and float(reward) > 0.0) if done else None


class ScienceWorldEnvAdapter(EnvAdapter):
    """Closed-loop adapter around the official ScienceWorld Python API."""

    def __init__(
        self,
        *,
        env_factory: Callable[..., Any] | None = None,
        task_id: int | str = 13,
        variation: int = 0,
        simplifications: str | list[str] = "easy",
        step_limit: int = 100,
        thread_num: int = 0,
    ):
        self.env_factory = env_factory
        self.task_id = task_id
        self.variation = variation
        self.simplifications = simplifications
        self.step_limit = step_limit
        self.thread_num = thread_num
        self._env: Any | None = None
        self._observation = ""
        self._valid_actions: list[str] = []

    def _make_env(self) -> Any:
        if self.env_factory is None:
            scienceworld = importlib.import_module("scienceworld")
            self.env_factory = scienceworld.ScienceWorldEnv
        try:
            signature = inspect.signature(self.env_factory)
            if "envStepLimit" in signature.parameters:
                return self.env_factory("", None, envStepLimit=self.step_limit)
        except (TypeError, ValueError):
            pass
        try:
            return self.env_factory("", None, self.step_limit)
        except TypeError:
            return self.env_factory("", None, self.step_limit, self.thread_num)

    def _task_name(self) -> str:
        if isinstance(self.task_id, str) and not self.task_id.isdigit():
            return self.task_id
        task_index = int(self.task_id)
        if hasattr(self._env, "get_task_names"):
            task_names = list(self._env.get_task_names())
        else:
            task_names = list(self._env.getTaskNames())
        if 0 <= task_index < len(task_names):
            return str(task_names[task_index])
        one_based = task_index - 1
        if 0 <= one_based < len(task_names):
            return str(task_names[one_based])
        return str(self.task_id)

    def reset(self, task_id: str | None = None) -> str:
        if task_id is not None:
            self.task_id = task_id
        self._env = self._make_env()
        task_name = self._task_name()
        self._env.load(task_name, self.variation, self.simplifications)
        if hasattr(self._env, "resetWithVariation"):
            observation, info = self._env.resetWithVariation(self.variation, self.simplifications)
        else:
            observation, info = self._env.reset()
        self._observation = str(observation)
        self._valid_actions = _extract_valid_actions(info) or _env_valid_actions(self._env)
        return self._observation

    def step(self, action_text: str) -> EnvStep:
        if self._env is None:
            raise RuntimeError("ScienceWorldEnvAdapter.reset() must be called before step().")
        observation, reward, done, info = self._env.step(action_text)
        self._observation = str(observation)
        self._valid_actions = _extract_valid_actions(info) or _env_valid_actions(self._env)
        return EnvStep(
            observation_text=self._observation,
            reward=float(reward) if reward is not None else None,
            done=bool(done),
            success=_extract_success(reward, bool(done), info),
            valid_actions=list(self._valid_actions),
        )

    def observation_text(self) -> str:
        return self._observation

    def state_text(self) -> str:
        return f"observation: {self._observation}"

    def candidate_actions(self) -> list[str]:
        return list(self._valid_actions)

    def admissible_actions(self) -> list[str]:
        return self.candidate_actions()

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "shutdown"):
            self._env.shutdown()
        elif self._env is not None and hasattr(self._env, "close"):
            self._env.close()
