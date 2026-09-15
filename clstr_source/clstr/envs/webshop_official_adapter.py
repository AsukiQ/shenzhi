from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clstr.envs.base import EnvAdapter, EnvStep


DEFAULT_WEBSHOP_REPO_PATH = Path(__file__).resolve().parents[2] / "third_party" / "WebShop"


OFFICIAL_WEBSHOP_REFERENCE = {
    "github": "https://github.com/princeton-nlp/WebShop",
    "setup": [
        f"git clone https://github.com/princeton-nlp/WebShop.git {DEFAULT_WEBSHOP_REPO_PATH}",
        f"cd {DEFAULT_WEBSHOP_REPO_PATH} && ./setup.sh",
        f"cd {DEFAULT_WEBSHOP_REPO_PATH} && ./run_web_agent_text_env.sh",
    ],
    "java": "Java runtime is required by WebShop search/index tooling.",
    "smoke": "bash run_web_agent_text_env.sh",
}


def _repo_is_webshop(repo_path: Path) -> bool:
    return (
        repo_path.exists()
        and (repo_path / "run_web_agent_text_env.sh").exists()
        and ((repo_path / "web_agent_site").exists() or (repo_path / "run_envs").exists())
    )


def _instruction_query(observation: str) -> str:
    text = str(observation or "")
    match = re.search(r"instruction\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        if "[SEP]" in query:
            parts = [part.strip() for part in query.split("[SEP]")]
            query = next((part for part in parts if part and part.lower() != "search"), query)
        query = re.sub(r"\s+", " ", query)
        return query
    return ""


def _actions_from_available(raw: Any, observation: str = "") -> list[str]:
    actions: list[str] = []
    if isinstance(raw, dict):
        if raw.get("has_search_bar"):
            query = _instruction_query(observation) or "query"
            actions.append(f"search[{query}]")
        for value in raw.get("clickables") or raw.get("clickable") or raw.get("buttons") or []:
            if str(value).strip().lower() == "search":
                continue
            actions.append(f"click[{value}]")
        for value in raw.get("valid_actions") or raw.get("admissible_actions") or []:
            actions.append(str(value))
    elif isinstance(raw, (list, tuple)):
        actions.extend(str(item) for item in raw)
    seen: set[str] = set()
    deduped: list[str] = []
    for action in actions:
        if action and action not in seen:
            seen.add(action)
            deduped.append(action)
    return deduped


def build_webshop_smoke_report(
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
    data_info: dict[str, Any] = {"checked": repo_path is not None, "available": repo_path is None}
    smoke_info: dict[str, Any] = {"attempted": False, "success": False}
    install_repo = Path(repo_path) if repo_path is not None else DEFAULT_WEBSHOP_REPO_PATH

    repo_for_import = Path(repo_path) if repo_path is not None else None
    inserted_repo = False
    if repo_for_import is not None and str(repo_for_import) not in sys.path:
        sys.path.insert(0, str(repo_for_import))
        inserted_repo = True
    try:
        module = import_module("web_agent_site")
        package_info = {"available": True, "module": getattr(module, "__name__", "web_agent_site")}
    except Exception as exc:  # noqa: BLE001
        blockers["python_package"] = {
            "message": "WebShop Python package/module 'web_agent_site' is not importable.",
            "error": repr(exc),
            "install_command": f"cd {install_repo} && ./setup.sh",
        }
    finally:
        if inserted_repo:
            try:
                sys.path.remove(str(repo_for_import))
            except ValueError:
                pass

    try:
        java = run_java(["java", "-version"], capture_output=True, text=True, timeout=10)
        output = (java.stderr or java.stdout or "").strip()
        if java.returncode == 0:
            java_info = {"available": True, "version_output": output}
        else:
            blockers["java_runtime"] = {
                "message": "Java runtime check failed.",
                "returncode": java.returncode,
                "output": output,
                "install_command": "sudo apt-get update && sudo apt-get install -y openjdk-21-jdk",
            }
    except Exception as exc:  # noqa: BLE001
        blockers["java_runtime"] = {
            "message": "Java runtime is not available on PATH.",
            "error": repr(exc),
            "install_command": "sudo apt-get update && sudo apt-get install -y openjdk-21-jdk",
        }

    repo: Path | None = None
    if repo_path is not None:
        repo = Path(repo_path)
        repo_info = {"checked": True, "path": str(repo), "available": _repo_is_webshop(repo)}
        if not repo_info["available"]:
            blockers["repo_path"] = {
                "message": "WebShop repo path is missing or does not look like the official repo.",
                "path": str(repo),
                "install_command": f"git clone https://github.com/princeton-nlp/WebShop.git {repo}",
            }
        candidate_data = [
            repo / "data",
            repo / "search_engine" / "indexes",
            repo / "web_agent_site" / "engine" / "indexes",
        ]
        data_available = any(path.exists() for path in candidate_data)
        data_info = {
            "checked": True,
            "available": data_available,
            "candidate_paths": [str(path) for path in candidate_data],
        }
        if not data_available:
            blockers["data_or_index"] = {
                "message": "WebShop data/search index paths are missing.",
                "install_command": f"cd {repo} && ./setup.sh",
            }

    if not blockers and repo is not None:
        smoke_code = (
            "import gym; "
            "import web_agent_site.envs; "
            "from web_agent_site.utils import DEBUG_PROD_SIZE; "
            "env = gym.make('WebAgentTextEnv-v0', observation_mode='text', "
            "num_products=DEBUG_PROD_SIZE, disable_env_checker=True); "
            "obs = env.reset(); "
            "actions = env.get_available_actions(); "
            "assert actions and ('has_search_bar' in actions or 'clickables' in actions); "
            "env.step('search[shoes]'); "
            "env.close(); "
            "print('webshop_text_env_smoke_ok')"
        )
        command = [sys.executable, "-c", smoke_code]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")
        env["PATH"] = str(Path(env["JAVA_HOME"]) / "bin") + os.pathsep + env.get("PATH", "")
        try:
            smoke = run_smoke(command, cwd=str(repo), env=env, capture_output=True, text=True, timeout=120)
            output = "\n".join(part for part in [smoke.stdout, smoke.stderr] if part).strip()
            smoke_info = {
                "attempted": True,
                "success": smoke.returncode == 0,
                "command": " ".join(command),
                "returncode": smoke.returncode,
                "output": output[-4000:],
            }
            if smoke.returncode != 0:
                blockers["smoke_run"] = {
                    "message": "Official WebShop smoke command failed.",
                    "command": " ".join(command),
                    "returncode": smoke.returncode,
                    "output": output[-4000:],
                    "install_command": f"cd {repo} && ./setup.sh",
                }
        except Exception as exc:  # noqa: BLE001
            blockers["smoke_run"] = {
                "message": "Official WebShop smoke command could not be executed.",
                "error": repr(exc),
                "install_command": f"cd {repo} && ./setup.sh",
            }
            smoke_info = {"attempted": True, "success": False, "error": repr(exc)}

    return {
        "status": "ok" if not blockers else "blocker",
        "smoke_success": smoke_info["success"],
        "package": package_info,
        "java": java_info,
        "repo": repo_info,
        "data_or_index": data_info,
        "smoke": smoke_info,
        "blockers": blockers,
        "install_commands": sorted(
            {
                value["install_command"]
                for value in blockers.values()
                if isinstance(value, dict) and value.get("install_command")
            }
        ),
        "leakage_policy": {
            "webshop_test_split_train_then_eval": "forbidden",
            "train_split_for_same_split_eval": "forbidden",
            "eval_only_sources": ["Ricardo-H/ws-step60-webshop-wm-w2r-qwen3-8b-40k", "Ricardo-H/ws-step80-webshop-wm-w2r-qwen3-32b-32k"],
        },
        "official_reference": OFFICIAL_WEBSHOP_REFERENCE,
    }


def write_webshop_smoke_report(
    output_path: str | Path = "outputs/webshop_eval/harness_smoke_report.json",
    repo_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_webshop_smoke_report(repo_path=repo_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


class WebShopOfficialAdapter(EnvAdapter):
    """Adapter for the official WebShop text/simple environment API."""

    def __init__(self, *, env_factory: Callable[..., Any], env_kwargs: dict[str, Any] | None = None):
        self.env_factory = env_factory
        self.env_kwargs = dict(env_kwargs or {})
        self._env: Any | None = None
        self._observation = ""
        self._actions: list[str] = []

    def _available_actions(self) -> Any:
        if self._env is None:
            return []
        if hasattr(self._env, "get_available_actions"):
            return self._env.get_available_actions()
        return getattr(self._env, "actions", [])

    def reset(self, task_id: str | None = None) -> str:
        kwargs = dict(self.env_kwargs)
        self._env = self.env_factory(**kwargs)
        try:
            reset_result = self._env.reset(session=task_id) if task_id is not None else self._env.reset()
        except TypeError:
            reset_result = self._env.reset()
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self._observation = str(observation)
        self._actions = _actions_from_available(self._available_actions(), self._observation)
        return self._observation

    def step(self, action_text: str) -> EnvStep:
        if self._env is None:
            raise RuntimeError("WebShopOfficialAdapter.reset() must be called before step().")
        observation, reward, done, info = self._env.step(action_text)
        self._observation = str(observation)
        raw_actions = info.get("available_actions") if isinstance(info, dict) and "available_actions" in info else self._available_actions()
        self._actions = _actions_from_available(raw_actions, self._observation)
        success = None
        if isinstance(info, dict):
            if "success" in info:
                success = bool(info["success"])
            elif "score" in info:
                try:
                    success = bool(done and float(info["score"]) >= 1.0)
                except (TypeError, ValueError):
                    success = None
        return EnvStep(
            observation_text=self._observation,
            reward=float(reward) if reward is not None else None,
            done=bool(done),
            success=success,
            valid_actions=list(self._actions),
        )

    def observation_text(self) -> str:
        return self._observation

    def state_text(self) -> str:
        return f"observation: {self._observation}"

    def candidate_actions(self) -> list[str]:
        return list(self._actions)
