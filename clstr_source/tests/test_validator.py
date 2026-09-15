import subprocess
import sys
import json
from pathlib import Path

from clstr.validator import (
    CachedLLMValidator,
    build_verified_templates,
    parse_validator_output,
    render_validator_prompt,
)


def test_render_validator_prompt_mentions_llm_based_validator():
    prompt = render_validator_prompt({"query": "find weather", "steps": [], "answer": "done"})
    assert "LLM-based validator" in prompt


def test_parse_validator_output_returns_indices():
    parsed = parse_validator_output('{"essential_steps":[1,3],"reason":"kept only required calls"}')
    assert parsed["essential_steps"] == [1, 3]


def test_build_verified_pairs_script_runs_from_repo_root(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        '{"task_id":"t1","validator_output":"{\\"essential_steps\\":[1,2],\\"reason\\":\\"ok\\"}"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_verified_pairs.py",
            "--input_jsonl",
            str(input_path),
            "--output_jsonl",
            str(output_path),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["task_id"] == "t1"
    assert payload["essential_steps"] == [1, 2]


def test_build_verified_pairs_script_writes_full_templates_when_steps_exist(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "query": "find weather",
                "steps": [
                    {
                        "tool": "weather.lookup",
                        "tool_idx": 0,
                        "result": "first",
                        "raw_topk": [0],
                    },
                    {
                        "tool": "weather.lookup",
                        "tool_idx": 0,
                        "result": "second",
                        "raw_topk": [0],
                    },
                ],
                "validator_output": '{"essential_steps":[1,2],"reason":"ok"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_verified_pairs.py",
            "--input_jsonl",
            str(input_path),
            "--output_jsonl",
            str(output_path),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["step_idx"] == 0
    assert rows[0]["action_at_t"] == 0
    assert rows[0]["candidates_next"] == [0]


def test_build_verified_templates_uses_essential_local_step_idx_for_nonconsecutive_essential_steps():
    templates = build_verified_templates(
        [
            {
                "task_id": "t-gap",
                "query": "find weather",
                "steps": [
                    {"tool": "search", "tool_idx": 4, "result": "noise"},
                    {"tool": "weather.lookup", "tool_idx": 1, "result": "first"},
                    {"tool": "calendar.lookup", "tool_idx": 5, "result": "noise"},
                    {"tool": "weather.lookup", "tool_idx": 2, "result": "second", "raw_topk": [2, 3]},
                ],
                "validator_output": '{"essential_steps":[2,4],"reason":"skip noise"}',
            }
        ],
        topk_recall_fn=lambda _state, _k: [2, 3],
        K=2,
    )

    assert len(templates) == 1
    assert templates[0].step_idx == 0
    assert templates[0].action_at_t == 1
    assert templates[0].a_next_plus == 2


def test_build_verified_templates_marks_essential_replay_prefix_as_target_belief_path():
    templates = build_verified_templates(
        [
            {
                "task_id": "t-gap",
                "query": "find weather",
                "steps": [
                    {"tool": "search", "tool_idx": 4, "result": "noise"},
                    {"tool": "weather.lookup", "tool_idx": 1, "result": "first"},
                    {"tool": "calendar.lookup", "tool_idx": 5, "result": "noise"},
                    {"tool": "weather.lookup", "tool_idx": 2, "result": "second", "raw_topk": [2, 3]},
                    {"tool": "summarize", "tool_idx": 3, "result": "third", "raw_topk": [3, 4]},
                ],
                "validator_output": '{"essential_steps":[2,4,5],"reason":"skip noise"}',
            }
        ],
        topk_recall_fn=lambda _state, _k: [2, 3],
        K=2,
    )

    assert templates[0].state_before.history == []
    assert templates[0].replay_prefix == []
    assert templates[0].replay_prefix_is_essential_subsequence is True

    assert templates[1].step_idx == 1
    assert templates[1].state_before.history == [
        ("weather.lookup", "first"),
    ]
    assert len(templates[1].replay_prefix) == 1
    assert templates[1].replay_prefix[0].x.history == []
    assert templates[1].replay_prefix[0].skill_idx == 1


def test_build_verified_templates_uses_supplied_topk_recall_fn_when_candidates_missing():
    templates = build_verified_templates(
        [
            {
                "task_id": "t-topk",
                "query": "find weather",
                "steps": [
                    {"tool": "weather.lookup", "tool_idx": 1, "result": "first"},
                    {"tool": "weather.lookup", "tool_idx": 4, "result": "second"},
                ],
                "validator_output": '{"essential_steps":[1,2],"reason":"ok"}',
            }
        ],
        topk_recall_fn=lambda _state, _k: [9, 8, 7],
        K=3,
    )

    assert templates[0].was_in_raw_topk is False
    assert templates[0].candidates_next == [9, 8, 4]


def test_build_verified_templates_does_not_count_injected_candidates_as_raw_topk():
    templates = build_verified_templates(
        [
            {
                "task_id": "t-injected",
                "query": "find weather",
                "steps": [
                    {"tool": "weather.lookup", "tool_idx": 1, "result": "first"},
                    {
                        "tool": "weather.lookup",
                        "tool_idx": 4,
                        "result": "second",
                        "raw_topk": [9, 8, 7],
                        "candidates_next": [9, 8, 4],
                    },
                ],
                "validator_output": '{"essential_steps":[1,2],"reason":"ok"}',
            }
        ],
        topk_recall_fn=lambda _state, _k: [9, 8, 7],
        K=3,
    )

    assert templates[0].was_in_raw_topk is False
    assert templates[0].candidates_next == [9, 8, 4]


def test_build_verified_pairs_script_errors_when_topk_missing_without_model(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "t-missing",
                "query": "find weather",
                "steps": [
                    {"tool": "weather.lookup", "tool_idx": 0, "result": "first"},
                    {"tool": "weather.lookup", "tool_idx": 1, "result": "second"},
                ],
                "validator_output": '{"essential_steps":[1,2],"reason":"ok"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_verified_pairs.py",
            "--input_jsonl",
            str(input_path),
            "--output_jsonl",
            str(output_path),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing raw_topk/raw_candidates" in result.stderr
    assert "t-missing" in result.stderr


def test_cached_llm_validator_reuses_cached_outputs(tmp_path: Path):
    calls = []

    def client(prompt: str) -> str:
        calls.append(prompt)
        return '{"essential_steps":[1],"reason":"cached"}'

    validator = CachedLLMValidator(cache_path=tmp_path / "validator_cache.jsonl", client=client)
    traj = {"task_id": "t1", "query": "find weather", "steps": [], "answer": "done"}

    first = validator.validate(traj)
    second = validator.validate(traj)

    assert first["essential_steps"] == [1]
    assert second["reason"] == "cached"
    assert len(calls) == 1
