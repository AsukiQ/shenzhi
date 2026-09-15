from __future__ import annotations

from pathlib import Path

import pytest

from clstr.tau2_tool_result_replay import replay_tau2_golden_tool_results


def test_tau2_replay_rejects_nontrain_split(tmp_path: Path) -> None:
    (tmp_path / "data" / "tau2").mkdir(parents=True)
    with pytest.raises(ValueError, match="restricted to official Tau2 train"):
        replay_tau2_golden_tool_results(tmp_path, task_split="test")
