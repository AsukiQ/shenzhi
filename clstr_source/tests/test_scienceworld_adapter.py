import json
import subprocess
import types
from pathlib import Path

from clstr.envs.scienceworld_adapter import (
    DEFAULT_SCIENCEWORLD_REPO_PATH,
    ScienceWorldEnvAdapter,
    build_scienceworld_smoke_report,
)


class _FakeScienceWorldEnv:
    def __init__(self, *args):
        self.args = args
        self.loaded = None
        self.reset_variation = None

    def getTaskNames(self):
        return ["boil", "melt", "find-non-living-thing"]

    def load(self, task_name, variation, simplifications):
        self.loaded = (task_name, variation, simplifications)

    def resetWithVariation(self, variation, simplifications):
        self.reset_variation = (variation, simplifications)
        return "You are in a lab.", {"valid": ["look around", "open door"]}

    def step(self, action):
        return (
            f"did {action}",
            1.0,
            True,
            {"valid": ["restart"], "score": 1.0},
        )

    def shutdown(self):
        pass


class _OfficialStyleScienceWorldEnv:
    def __init__(self, taskName=None, serverPath=None, envStepLimit=100):  # noqa: N803 - mirrors official API
        self.args = (taskName, serverPath, envStepLimit)
        self.loaded = None
        self.closed = False

    def get_task_names(self):
        return ["boil", "melt", "find-non-living-thing"]

    def load(self, task_name, variation, simplifications):
        self.loaded = (task_name, variation, simplifications)

    def reset(self):
        return "You are in a lab.", {"score": 0.0}

    def get_valid_action_object_combinations(self):
        return ["look around", "open door"]

    def step(self, action):
        return (
            f"did {action}",
            1.0,
            True,
            {"score": 1.0},
        )

    def close(self):
        self.closed = True


def test_build_scienceworld_smoke_report_blocks_when_package_java_or_repo_missing(tmp_path):
    missing_repo = tmp_path / "missing_scienceworld_repo"

    def missing_import(_name):
        raise ModuleNotFoundError("scienceworld")

    def missing_java(*_args, **_kwargs):
        raise FileNotFoundError("java")

    report = build_scienceworld_smoke_report(
        repo_path=missing_repo,
        import_module=missing_import,
        run_java=missing_java,
    )

    assert report["status"] == "blocker"
    assert report["smoke_success"] is False
    assert {"python_package", "java_runtime", "repo_path"} <= set(report["blockers"])
    assert "python -m pip install scienceworld" in report["install_commands"]
    assert any("sudo apt-get install" in command for command in report["install_commands"])
    assert str(missing_repo) in json.dumps(report)


def test_default_scienceworld_repo_path_is_clstr_local():
    repo_root = Path(__file__).resolve().parents[1]

    assert str(DEFAULT_SCIENCEWORLD_REPO_PATH).startswith(str(repo_root))
    assert DEFAULT_SCIENCEWORLD_REPO_PATH.parts[-2:] == ("third_party", "ScienceWorld")


def test_build_scienceworld_smoke_report_does_not_fake_smoke_success(tmp_path):
    repo = tmp_path / "ScienceWorld"
    (repo / "examples").mkdir(parents=True)
    (repo / "examples" / "random_agent.py").write_text("print('smoke')\n", encoding="utf-8")
    (repo / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")

    def fake_import(_name):
        return types.SimpleNamespace(__version__="test")

    def fake_java(*_args, **_kwargs):
        return subprocess.CompletedProcess(["java", "-version"], 0, stderr='openjdk version "21"')

    def failing_smoke(*_args, **_kwargs):
        return subprocess.CompletedProcess(["python", "examples/random_agent.py"], 7, stdout="offline trace", stderr="")

    report = build_scienceworld_smoke_report(
        repo_path=repo,
        import_module=fake_import,
        run_java=fake_java,
        run_smoke=failing_smoke,
    )

    assert report["status"] == "blocker"
    assert report["smoke_success"] is False
    assert report["blockers"]["smoke_run"]["returncode"] == 7
    assert "offline trace" in report["blockers"]["smoke_run"]["output"]


def test_build_scienceworld_smoke_report_runs_random_agent_relative_to_repo_cwd(tmp_path):
    repo = tmp_path / "ScienceWorld"
    (repo / "examples").mkdir(parents=True)
    (repo / "examples" / "random_agent.py").write_text("print('smoke')\n", encoding="utf-8")
    (repo / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    captured = {}

    def fake_import(_name):
        return types.SimpleNamespace(__version__="test")

    def fake_java(*_args, **_kwargs):
        return subprocess.CompletedProcess(["java", "-version"], 0, stderr='openjdk version "21"')

    def fake_smoke(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = build_scienceworld_smoke_report(
        repo_path=repo,
        import_module=fake_import,
        run_java=fake_java,
        run_smoke=fake_smoke,
    )

    assert report["status"] == "ok"
    assert captured["cwd"] == str(repo)
    assert captured["command"][1] == "examples/random_agent.py"


def test_scienceworld_adapter_maps_valid_actions_from_info():
    adapter = ScienceWorldEnvAdapter(
        env_factory=_FakeScienceWorldEnv,
        task_id=2,
        variation=0,
        simplifications="easy",
        step_limit=25,
        thread_num=3,
    )

    observation = adapter.reset()

    assert observation == "You are in a lab."
    assert adapter.candidate_actions() == ["look around", "open door"]
    assert adapter.admissible_actions() == ["look around", "open door"]
    assert adapter.state_text() == "observation: You are in a lab."
    assert adapter._env.loaded == ("find-non-living-thing", 0, "easy")
    assert adapter._env.args == ("", None, 25)

    step = adapter.step("open door")

    assert step.observation_text == "did open door"
    assert step.reward == 1.0
    assert step.done is True
    assert step.success is True
    assert step.valid_actions == ["restart"]
    assert adapter.candidate_actions() == ["restart"]


def test_scienceworld_adapter_supports_official_scienceworld_1_2_api():
    adapter = ScienceWorldEnvAdapter(
        env_factory=_OfficialStyleScienceWorldEnv,
        task_id=2,
        variation=0,
        simplifications="easy",
        step_limit=25,
    )

    observation = adapter.reset()

    assert observation == "You are in a lab."
    assert adapter._env.args == ("", None, 25)
    assert adapter._env.loaded == ("find-non-living-thing", 0, "easy")
    assert adapter.candidate_actions() == ["look around", "open door"]

    step = adapter.step("open door")

    assert step.observation_text == "did open door"
    assert step.reward == 1.0
    assert step.done is True
    assert step.success is True
    assert step.valid_actions == ["look around", "open door"]
