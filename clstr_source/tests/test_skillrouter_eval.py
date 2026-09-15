import subprocess
import sys
from pathlib import Path

from scripts.run_skillrouter_eval_only import build_skillrouter_eval_command


def test_build_skillrouter_eval_command_targets_clstr_outputs_and_readonly_skillrouter():
    command = build_skillrouter_eval_command(
        skillrouter_repo=Path("/root/autodl-tmp/skillrouter"),
        data_root=Path("/root/autodl-tmp/clstr/data/skillrouter_eval_core"),
        encoder_model=Path("/root/autodl-tmp/clstr/.cache/hf_models/SkillRouter-Embedding-0.6B"),
        reranker_model=Path("/root/autodl-tmp/clstr/.cache/hf_models/SkillRouter-Reranker-0.6B"),
        output_dir=Path("/root/autodl-tmp/clstr/outputs/skillrouter_baseline/open_model_eval"),
        tiers=["easy", "hard"],
        task_mode="core",
        retrieval_top_k=20,
    )

    assert command[:3] == [sys.executable, "-m", "src.run_open_model_eval"]
    assert "--data_root" in command
    assert "/root/autodl-tmp/clstr/data/skillrouter_eval_core" in command
    assert "--output_dir" in command
    assert "/root/autodl-tmp/clstr/outputs/skillrouter_baseline/open_model_eval" in command
    assert "/root/autodl-tmp/skillrouter/outputs" not in " ".join(command)


def test_skillrouter_eval_only_cli_dry_run_does_not_execute_models(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_skillrouter_eval_only.py",
            "--skillrouter_repo",
            "/root/autodl-tmp/skillrouter",
            "--data_root",
            str(tmp_path / "skillrouter_eval_core"),
            "--encoder_model",
            str(tmp_path / "SkillRouter-Embedding-0.6B"),
            "--reranker_model",
            str(tmp_path / "SkillRouter-Reranker-0.6B"),
            "--output_dir",
            str(tmp_path / "outputs" / "open_model_eval"),
            "--dry_run",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    assert "PYTHONPATH=/root/autodl-tmp/skillrouter" in result.stdout
    assert "--output_dir" in result.stdout
    assert str(tmp_path / "outputs" / "open_model_eval") in result.stdout
