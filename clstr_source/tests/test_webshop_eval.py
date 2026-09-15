import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_webshop_clstr_eval import run_webshop_eval_or_blocker


def test_webshop_eval_writes_blocker_when_harness_smoke_not_ok(tmp_path):
    report = run_webshop_eval_or_blocker(
        output_dir=tmp_path / "webshop_eval",
        smoke_report={
            "status": "blocker",
            "smoke_success": False,
            "blockers": {"java_runtime": {"message": "missing"}},
            "leakage_policy": {"webshop_test_split_train_then_eval": "forbidden"},
        },
        max_episodes=1,
        max_steps=1,
    )

    assert report["status"] == "blocked_no_closed_loop_eval_harness"
    assert report["candidate_recall"]["candidate_recall_applicability_reason"] == (
        "environment_admissible_actions"
    )
    metrics = json.loads((tmp_path / "webshop_eval" / "metrics.json").read_text(encoding="utf-8"))
    blocker = json.loads((tmp_path / "webshop_eval" / "blocker_report.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "blocked_no_closed_loop_eval_harness"
    assert metrics["success_rate"] is None
    assert blocker["not_closed_loop_success"] is True
    assert blocker["leakage_policy"]["webshop_test_split_train_then_eval"] == "forbidden"


class _FakeAdapter:
    def __init__(self):
        self._obs = ""
        self._actions = ["search[red mug]"]

    def reset(self, task_id=None):
        self._obs = "Instruction: red mug"
        return self._obs

    def state_text(self):
        return f"observation: {self._obs}"

    def candidate_actions(self):
        return list(self._actions)

    def step(self, action):
        from clstr.envs.base import EnvStep

        assert action == "search[red mug]"
        self._actions = ["click[Buy Now]"]
        return EnvStep("Done", reward=1.0, done=True, success=True, valid_actions=self._actions)

    def close(self):
        pass


class _FakeController:
    def choose_action(self, state_text, admissible_actions):
        assert "observation:" in state_text
        return admissible_actions[0]


def test_webshop_eval_runs_closed_loop_with_controller_when_smoke_ok(tmp_path):
    output_dir = tmp_path / "webshop_eval"
    output_dir.mkdir()
    (output_dir / "blocker_report.json").write_text('{"status":"old_blocker"}\n', encoding="utf-8")

    report = run_webshop_eval_or_blocker(
        output_dir=output_dir,
        smoke_report={"status": "ok", "smoke_success": True, "leakage_policy": {"webshop_test_split_train_then_eval": "forbidden"}},
        adapter_factory=lambda: _FakeAdapter(),
        controller=_FakeController(),
        max_episodes=1,
        max_steps=3,
    )

    assert report["status"] == "ok"
    assert report["candidate_recall"] == {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "environment_admissible_actions",
        "candidate_recall_saturated": False,
        "pool_protocol": "environment_candidates",
        "candidate_source": "environment_admissible_actions",
    }
    assert not (output_dir / "blocker_report.json").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    run_rows = [json.loads(line) for line in (output_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics["success_rate"] == 1.0
    assert run_rows[0]["steps"][0]["admissible_actions"] == ["search[red mug]"]
    assert run_rows[0]["steps"][0]["action"] == "search[red mug]"


def test_webshop_cli_builds_clstr_text_controller(tmp_path, monkeypatch):
    import scripts.run_webshop_clstr_eval as module

    captured = {}

    class _Args:
        method = "clstr"
        routing_init_manifest = "outputs/clstr_native_routing_init/manifest.json"
        checkpoint_path = "outputs/clstr_full_base_train/checkpoints/clstr_full_base-step800.pt"
        stage4_checkpoint_path = "outputs/stage4/checkpoints/clstr_stage4_act-step2000.pt"
        skill_rows_path_override = "data/current/skill_pool.jsonl"
        aux_data_root = "data/clstr_full_base_train"
        controller_mode = "policy_plus_transition_belief_stop_loop_penalty"
        skillrouter_model_name_or_path = ".cache/hf_models/SkillRouter-Embedding-0.6B"
        skillrouter_adapter_checkpoint_path = None
        skillrouter_batch_size = 16
        skillrouter_max_length = 2048

    class _Controller:
        def choose_action(self, state_text, admissible_actions):
            return admissible_actions[-1]

    def fake_build_controller(**kwargs):
        captured.update(kwargs)
        return _Controller(), {"loaded": True}

    monkeypatch.setattr(module, "build_clstr_text_action_controller", fake_build_controller, raising=False)

    controller, report = module._build_cli_controller(_Args(), tmp_path / "webshop_eval")

    assert controller.choose_action("state", ["search[red mug]", "click[Buy Now]"]) == "click[Buy Now]"
    assert report == {"loaded": True}
    assert captured["routing_init_manifest"] == "outputs/clstr_native_routing_init/manifest.json"
    assert captured["checkpoint_path"] == "outputs/clstr_full_base_train/checkpoints/clstr_full_base-step800.pt"
    assert captured["stage4_checkpoint_path"] == "outputs/stage4/checkpoints/clstr_stage4_act-step2000.pt"
    assert captured["skill_rows_path_override"] == "data/current/skill_pool.jsonl"
    assert captured["aux_data_root"] == "data/clstr_full_base_train"
    assert captured["controller_mode"] == "policy_plus_transition_belief_stop_loop_penalty"
    assert captured["output_dir"] == tmp_path / "webshop_eval" / "model_cache"


def test_webshop_cli_can_build_skillrouter_text_controller(tmp_path, monkeypatch):
    import scripts.run_webshop_clstr_eval as module

    captured = {}

    class _Args:
        method = "skillrouter"
        routing_init_manifest = "unused"
        checkpoint_path = None
        stage4_checkpoint_path = None
        skill_rows_path_override = None
        aux_data_root = "unused"
        controller_mode = "policy_plus_transition_belief_stop_loop_penalty"
        skillrouter_model_name_or_path = ".cache/hf_models/SkillRouter-Embedding-0.6B"
        skillrouter_adapter_checkpoint_path = "outputs/unified_skillrouter.pt"
        skillrouter_batch_size = 8
        skillrouter_max_length = 1024

    class _FakeScorer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, state_texts, candidate_texts):
            import torch

            return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    monkeypatch.setattr(module, "WebShopSkillRouterActionScorer", _FakeScorer, raising=False)

    controller, report = module._build_cli_controller(_Args(), tmp_path / "webshop_eval")

    assert controller.choose_action("state", ["search[red mug]", "click[Buy Now]"]) == "click[Buy Now]"
    assert report["controller_class"] == "ClstrTextActionController"
    assert report["method"] == "webshop_skillrouter_controller_gate"
    assert report["skillrouter_model_name_or_path"] == ".cache/hf_models/SkillRouter-Embedding-0.6B"
    assert report["skillrouter_adapter_checkpoint_path"] == "outputs/unified_skillrouter.pt"
    assert captured["model_name_or_path"] == ".cache/hf_models/SkillRouter-Embedding-0.6B"
    assert captured["adapter_checkpoint_path"] == "outputs/unified_skillrouter.pt"
    assert captured["batch_size"] == 8
    assert captured["max_length"] == 1024


def test_webshop_eval_uses_official_adapter_factory_when_smoke_ok(tmp_path, monkeypatch):
    import scripts.run_webshop_clstr_eval as module

    called = {}

    def fake_factory(repo_path):
        called["repo_path"] = repo_path
        return lambda: _FakeAdapter()

    monkeypatch.setattr(module, "_make_official_webshop_adapter_factory", fake_factory, raising=False)

    report = module.run_webshop_eval_or_blocker(
        output_dir=tmp_path / "webshop_eval",
        smoke_report={"status": "ok", "smoke_success": True, "leakage_policy": {"webshop_test_split_train_then_eval": "forbidden"}},
        controller=_FakeController(),
        max_episodes=1,
        max_steps=3,
        repo_path=tmp_path / "official_webshop",
    )

    assert report["status"] == "ok"
    assert called["repo_path"] == tmp_path / "official_webshop"


def test_webshop_official_factory_forces_java17_runtime(tmp_path, monkeypatch):
    import scripts.run_webshop_clstr_eval as module

    envs = types.ModuleType("web_agent_site.envs")
    utils = types.ModuleType("web_agent_site.utils")

    class _FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    envs.WebAgentTextEnv = _FakeEnv
    utils.DEBUG_PROD_SIZE = 7
    monkeypatch.setitem(sys.modules, "web_agent_site.envs", envs)
    monkeypatch.setitem(sys.modules, "web_agent_site.utils", utils)
    monkeypatch.setenv("JAVA_HOME", "/old/java8")
    monkeypatch.setenv("PATH", "/old/java8/bin:/usr/bin")

    factory = module._make_official_webshop_adapter_factory(tmp_path / "WebShop")
    adapter = factory()

    assert adapter.env_factory().kwargs["num_products"] == 7
    java_home = Path(module.os.environ["JAVA_HOME"])
    assert (java_home / "bin" / "java").exists()
    assert module.os.environ["PATH"].split(":")[0] == str(java_home / "bin")
    if (java_home / "lib" / "jvm" / "lib" / "server" / "libjvm.so").exists():
        assert module.os.environ["JVM_PATH"] == str(java_home / "lib" / "jvm" / "lib" / "server" / "libjvm.so")


def test_webshop_unified_memory_scorer_maps_actions_and_uses_replay_prefix(monkeypatch):
    import torch
    import scripts.run_webshop_clstr_eval as module

    captured = {}

    def fake_encode_texts(model, texts, freeze=True):
        del model, freeze
        return torch.ones(len(texts), 4)

    def fake_apply_replay_prefix_beliefs(model, rows, m_obs, skill_id_to_idx, skill_count, device, trainable=False):
        del model, skill_id_to_idx, skill_count, device, trainable
        captured["replay_rows"] = rows
        return m_obs + 1.0, sum(1 for row in rows if row.get("replay_prefix"))

    monkeypatch.setattr(module, "_encode_texts", fake_encode_texts)
    monkeypatch.setattr(module, "_apply_replay_prefix_beliefs", fake_apply_replay_prefix_beliefs)

    class _FakeModel:
        device = torch.device("cpu")
        skills = [
            {"skill_id": "webshop/webshop-search-executor"},
            {"skill_id": "webshop/webshop-click-executor"},
            {"skill_id": "webshop/webshop-finish-executor"},
            {"skill_id": "webshop/webshop-reasoning-planner"},
        ]

        def eval(self):
            return self

        def initial_belief(self, h):
            return torch.zeros_like(h)

        def unified_route_logits(self, h, m_t, candidate_rows):
            del h, m_t
            captured["candidate_rows"] = candidate_rows
            return torch.tensor([[3.0, 4.0, -1.0]], dtype=torch.float32)

    scorer = module.WebShopUnifiedMemoryActionScorer(_FakeModel(), replay_prefix_max_steps=3)
    scores = scorer(
        ["observation: page\nhistory: search[red mug] | click[item]"],
        [["search[blue mug]", "click[Buy Now]", "invalid_action"]],
    )

    assert scores.shape == (1, 3)
    assert captured["candidate_rows"] == [[0, 1, 0]]
    assert captured["replay_rows"][0]["replay_prefix"][0]["skill_id"] == "webshop/webshop-search-executor"
    assert captured["replay_rows"][0]["replay_prefix"][1]["skill_id"] == "webshop/webshop-click-executor"
    assert scorer.last_metadata[0]["route_scorer"] == "unified_memory"
    assert scorer.last_metadata[0]["uses_recurrent_m_t"] is True


def test_webshop_unified_memory_concrete_action_scorer_scores_action_text_with_replayed_mt(monkeypatch):
    import torch
    import scripts.run_webshop_clstr_eval as module

    captured = {}

    def fake_encode_texts(model, texts, freeze=True):
        del model, freeze
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "buy now" in lowered:
                rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
            elif "reviews" in lowered:
                rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
        return torch.stack(rows)

    def fake_apply_replay_prefix_beliefs(model, rows, m_obs, skill_id_to_idx, skill_count, device, trainable=False):
        del model, m_obs, skill_id_to_idx, skill_count, device, trainable
        captured["replay_rows"] = rows
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32), 1

    monkeypatch.setattr(module, "_encode_texts", fake_encode_texts)
    monkeypatch.setattr(module, "_apply_replay_prefix_beliefs", fake_apply_replay_prefix_beliefs)

    class _SkillHead:
        def __call__(self, candidate_embs, belief_embs):
            captured["belief_embs"] = belief_embs.detach().clone()
            return torch.einsum("bcd,bd->bc", candidate_embs.float(), belief_embs.float())

    class _FakeModel:
        device = torch.device("cpu")
        skills = [
            {"skill_id": "webshop/webshop-search-executor"},
            {"skill_id": "webshop/webshop-click-executor"},
            {"skill_id": "webshop/webshop-finish-executor"},
            {"skill_id": "webshop/webshop-reasoning-planner"},
        ]
        skill_head = _SkillHead()

        def eval(self):
            return self

        def initial_belief(self, h):
            return torch.zeros_like(h)

    scorer = module.WebShopUnifiedMemoryConcreteActionScorer(_FakeModel(), replay_prefix_max_steps=3)
    scores = scorer(
        ["observation: product page\nhistory: search[red mug]"],
        [["click[Reviews]", "click[Buy Now]"]],
    )

    assert int(torch.argmax(scores[0]).item()) == 1
    assert torch.allclose(captured["belief_embs"], torch.tensor([[0.0, 1.0]]))
    assert captured["replay_rows"][0]["replay_prefix"][0]["skill_id"] == "webshop/webshop-search-executor"
    assert scorer.last_metadata[0]["policy_family"] == "clstr_webshop_unified_memory_concrete_action_scorer"
    assert scorer.last_metadata[0]["route_scorer"] == "unified_memory_concrete_action"
    assert scorer.last_metadata[0]["uses_recurrent_m_t"] is True


@pytest.mark.parametrize(
    "scorer_name",
    ["WebShopUnifiedMemoryActionScorer", "WebShopUnifiedMemoryConcreteActionScorer"],
)
def test_webshop_unified_memory_scorers_require_initial_belief(scorer_name):
    import torch
    import scripts.run_webshop_clstr_eval as module

    class _MissingInitializerModel:
        device = torch.device("cpu")
        skills = [{"skill_id": "webshop/webshop-search-executor"}]

        def eval(self):
            return self

    scorer_cls = getattr(module, scorer_name)
    with pytest.raises(ValueError, match="unified memory.*model.initial_belief"):
        scorer_cls(_MissingInitializerModel())


def test_webshop_eval_writes_progress_and_mt_trace_from_controller(tmp_path):
    class _TwoStepAdapter:
        def __init__(self):
            self.step_count = 0

        def reset(self, task_id=None):
            del task_id
            return "Instruction: red mug"

        def state_text(self):
            return f"observation: step {self.step_count}"

        def candidate_actions(self):
            return ["search[red mug]"] if self.step_count == 0 else ["click[Buy Now]"]

        def step(self, action):
            from clstr.envs.base import EnvStep

            self.step_count += 1
            return EnvStep(
                f"obs {self.step_count}",
                reward=1.0 if self.step_count == 2 else 0.0,
                done=self.step_count == 2,
                success=self.step_count == 2,
                valid_actions=self.candidate_actions(),
            )

        def close(self):
            pass

    class _Controller:
        def __init__(self):
            self.seen_states = []
            self.reset_count = 0
            self.candidate_scorer = SimpleNamespace(last_metadata=[])
            self.last_decision = None

        def reset(self):
            self.reset_count += 1

        def choose_action(self, state_text, admissible_actions):
            self.seen_states.append(state_text)
            uses_mt = "history: search[red mug]" in state_text
            self.candidate_scorer.last_metadata = [{"uses_recurrent_m_t": uses_mt, "route_scorer": "unified_memory"}]
            self.last_decision = SimpleNamespace(
                chosen_index=0,
                chosen_reason="test",
                component_scores=[],
            )
            return admissible_actions[0]

    controller = _Controller()
    report = run_webshop_eval_or_blocker(
        output_dir=tmp_path / "webshop_eval",
        smoke_report={"status": "ok", "smoke_success": True},
        adapter_factory=lambda: _TwoStepAdapter(),
        controller=controller,
        max_episodes=1,
        max_steps=3,
    )

    assert report["status"] == "ok"
    assert controller.reset_count == 1
    assert "history: <empty>" in controller.seen_states[0]
    assert "history: search[red mug]" in controller.seen_states[1]
    metrics = json.loads((tmp_path / "webshop_eval" / "metrics.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "webshop_eval" / "progress.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (tmp_path / "webshop_eval" / "run.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics["trace_summary"]["clstr_recurrent_mt_steps"] == 1
    assert progress["completed_episodes"] == 1
    assert rows[0]["steps"][1]["policy_metadata"]["uses_recurrent_m_t"] is True
