import subprocess
from pathlib import Path

from clstr.envs.webshop_official_adapter import (
    DEFAULT_WEBSHOP_REPO_PATH,
    WebShopOfficialAdapter,
    _actions_from_available,
    build_webshop_smoke_report,
)


class _FakeWebShopEnv:
    def __init__(self):
        self.actions = {"has_search_bar": True, "clickables": ["Buy Now", "Back to Search"]}
        self._obs = "Instruction: red mug"

    def reset(self):
        return self._obs

    def get_available_actions(self):
        return self.actions

    def step(self, action):
        assert action == "search[red mug]"
        return "Results page", 0.25, False, {"available_actions": {"has_search_bar": False, "clickables": ["Buy Now"]}}


class _TupleResetWebShopEnv:
    def __init__(self):
        self.actions = {"has_search_bar": True, "clickables": ["Buy Now"]}
        self.reset_sessions = []

    def reset(self, session=None):
        self.reset_sessions.append(session)
        return "Instruction: blue bowl", {"session": session}

    def get_available_actions(self):
        return self.actions

    def step(self, action):
        return "Done", 1.0, True, {"score": 1.0}


def test_webshop_official_adapter_maps_search_and_click_actions():
    adapter = WebShopOfficialAdapter(env_factory=lambda **kwargs: _FakeWebShopEnv())

    observation = adapter.reset()

    assert "red mug" in observation
    assert adapter.candidate_actions() == ["search[red mug]", "click[Buy Now]", "click[Back to Search]"]
    step = adapter.step("search[red mug]")
    assert step.observation_text == "Results page"
    assert step.reward == 0.25
    assert step.valid_actions == ["click[Buy Now]"]


def test_webshop_official_adapter_handles_tuple_reset_and_passes_session():
    env = _TupleResetWebShopEnv()
    adapter = WebShopOfficialAdapter(env_factory=lambda **kwargs: env)

    observation = adapter.reset(task_id="7")

    assert observation == "Instruction: blue bowl"
    assert env.reset_sessions == ["7"]
    assert adapter.candidate_actions() == ["search[blue bowl]", "click[Buy Now]"]


def test_webshop_actions_strip_sep_instruction_and_filter_search_click():
    observation = (
        "WebShop [SEP] Instruction: [SEP] Find me machine wash men's t-shirts "
        "with color: country [SEP] Search"
    )

    actions = _actions_from_available({"has_search_bar": True, "clickables": ["search"]}, observation)

    assert actions == ["search[Find me machine wash men's t-shirts with color: country]"]


def test_webshop_smoke_report_blocks_without_dependencies_and_records_leakage_policy(tmp_path):
    def missing_import(name):
        raise ModuleNotFoundError(name)

    def missing_java(*args, **kwargs):
        raise FileNotFoundError("java")

    report = build_webshop_smoke_report(
        repo_path=tmp_path / "missing_webshop",
        import_module=missing_import,
        run_java=missing_java,
    )

    assert report["status"] == "blocker"
    assert report["smoke_success"] is False
    assert "python_package" in report["blockers"]
    assert "java_runtime" in report["blockers"]
    assert report["leakage_policy"]["webshop_test_split_train_then_eval"] == "forbidden"


def test_default_webshop_repo_path_is_clstr_local():
    repo_root = Path(__file__).resolve().parents[1]

    assert str(DEFAULT_WEBSHOP_REPO_PATH).startswith(str(repo_root))
    assert DEFAULT_WEBSHOP_REPO_PATH.parts[-2:] == ("third_party", "WebShop")


def test_webshop_smoke_success_requires_official_smoke_exit_zero(tmp_path):
    repo = tmp_path / "webshop"
    (repo / "run_envs").mkdir(parents=True)
    (repo / "web_agent_site" / "envs").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    (repo / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "run_web_agent_text_env.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    class _Module:
        pass

    def fake_import(name):
        return _Module()

    def ok_java(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr='openjdk version "21"')

    def ok_smoke(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr="")

    report = build_webshop_smoke_report(
        repo_path=repo,
        import_module=fake_import,
        run_java=ok_java,
        run_smoke=ok_smoke,
    )

    assert report["status"] == "ok"
    assert report["smoke_success"] is True
    assert report["smoke"]["attempted"] is True


def test_webshop_smoke_runs_official_text_gym_env_with_repo_pythonpath(tmp_path):
    repo = tmp_path / "webshop"
    (repo / "run_envs").mkdir(parents=True)
    (repo / "web_agent_site" / "envs").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    (repo / "search_engine" / "indexes").mkdir(parents=True)
    (repo / "run_web_agent_text_env.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    class _Module:
        pass

    def fake_import(name):
        return _Module()

    def ok_java(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr='openjdk version "21"')

    captured = {}

    def ok_smoke(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="webshop_text_env_smoke_ok", stderr="")

    report = build_webshop_smoke_report(
        repo_path=repo,
        import_module=fake_import,
        run_java=ok_java,
        run_smoke=ok_smoke,
    )

    command_text = " ".join(captured["command"])
    assert "WebAgentTextEnv-v0" in command_text
    assert "disable_env_checker=True" in command_text
    assert str(repo) in captured["kwargs"]["env"]["PYTHONPATH"]
    assert report["status"] == "ok"
    assert report["smoke_success"] is True
