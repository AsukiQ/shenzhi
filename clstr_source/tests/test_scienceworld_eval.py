import json
from pathlib import Path

from clstr.envs.base import EnvStep
from scripts.run_scienceworld_clstr_eval import run_scienceworld_eval_or_blocker


class _FakeAdapter:
    def __init__(self):
        self.actions_seen = []
        self._actions = ["look", "open box"]

    def reset(self, task_id=None):
        return "A closed box is here."

    def state_text(self):
        return "observation: A closed box is here."

    def admissible_actions(self):
        return list(self._actions)

    def step(self, action):
        self.actions_seen.append(action)
        self._actions = ["restart"]
        return EnvStep(
            observation_text="The box is open.",
            reward=1.0,
            done=True,
            success=True,
            valid_actions=["restart"],
        )


class _RecordingController:
    def __init__(self):
        self.calls = []

    def choose_action(self, state_text, admissible_actions):
        self.calls.append(
            {
                "state_text": state_text,
                "admissible_actions": list(admissible_actions),
            }
        )
        return "open box"


def test_run_scienceworld_eval_writes_blocker_outputs_when_harness_unavailable(tmp_path):
    report = run_scienceworld_eval_or_blocker(
        output_dir=tmp_path / "clstr_controller_gate",
        smoke_report={"status": "blocker", "smoke_success": False, "blockers": {"java_runtime": "missing"}},
        adapter_factory=lambda: _FakeAdapter(),
        controller=_RecordingController(),
    )

    metrics_path = tmp_path / "clstr_controller_gate" / "metrics.json"
    blocker_path = tmp_path / "clstr_controller_gate" / "blocker_report.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    blocker = json.loads(blocker_path.read_text(encoding="utf-8"))

    assert report["status"] == "blocked_no_closed_loop_eval_harness"
    assert metrics["status"] == "blocked_no_closed_loop_eval_harness"
    assert metrics["success_rate"] is None
    assert metrics["episodes"] == 0
    assert blocker["status"] == "blocked_no_closed_loop_eval_harness"
    assert "ok" not in {metrics["status"], blocker["status"]}


def test_run_scienceworld_eval_maps_observation_and_valid_actions_to_controller(tmp_path):
    adapter = _FakeAdapter()
    controller = _RecordingController()
    output_dir = tmp_path / "scienceworld_run"
    output_dir.mkdir()
    (output_dir / "blocker_report.json").write_text('{"status":"old_blocker"}\n', encoding="utf-8")

    report = run_scienceworld_eval_or_blocker(
        output_dir=output_dir,
        smoke_report={"status": "ok", "smoke_success": True},
        adapter_factory=lambda: adapter,
        controller=controller,
        max_episodes=1,
        max_steps=3,
    )

    run_rows = [
        json.loads(line)
        for line in (output_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert not (output_dir / "blocker_report.json").exists()
    assert controller.calls == [
        {
            "state_text": "observation: A closed box is here.",
            "admissible_actions": ["look", "open box"],
        }
    ]
    assert adapter.actions_seen == ["open box"]
    assert metrics["success_rate"] == 1.0
    assert run_rows[0]["steps"][0]["admissible_actions"] == ["look", "open box"]
    assert run_rows[0]["steps"][0]["action"] == "open box"


def test_scienceworld_cli_builds_clstr_text_controller(tmp_path, monkeypatch):
    import scripts.run_scienceworld_clstr_eval as module

    captured = {}

    class _Args:
        routing_init_manifest = "outputs/clstr_native_routing_init/manifest.json"
        checkpoint_path = "outputs/clstr_full_base_train/checkpoints/clstr_full_base-step800.pt"
        aux_data_root = "data/clstr_full_base_train"
        controller_mode = "policy_plus_transition_belief_stop_loop_penalty"

    class _Controller:
        def choose_action(self, state_text, admissible_actions):
            return admissible_actions[-1]

    def fake_build_controller(**kwargs):
        captured.update(kwargs)
        return _Controller(), {"loaded": True}

    monkeypatch.setattr(module, "build_clstr_text_action_controller", fake_build_controller, raising=False)

    controller, report = module._build_cli_controller(_Args(), tmp_path / "scienceworld_eval")

    assert controller.choose_action("state", ["look", "open box"]) == "open box"
    assert report == {"loaded": True}
    assert captured["routing_init_manifest"] == "outputs/clstr_native_routing_init/manifest.json"
    assert captured["checkpoint_path"] == "outputs/clstr_full_base_train/checkpoints/clstr_full_base-step800.pt"
    assert captured["aux_data_root"] == "data/clstr_full_base_train"
    assert captured["controller_mode"] == "policy_plus_transition_belief_stop_loop_penalty"
    assert captured["output_dir"] == tmp_path / "scienceworld_eval" / "model_cache"
