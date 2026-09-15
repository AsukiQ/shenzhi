import os
import stat
import subprocess
from pathlib import Path


def run_command(args, cwd, env=None):
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_package_importable():
    import clstr

    assert clstr.__version__ == "0.1.0"
    assert "get_version" in clstr.__all__


def test_bootstrap_script_exists():
    script = Path("scripts/bootstrap_skillrouter.sh")
    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert script.stat().st_mode & stat.S_IXUSR


def test_bootstrap_script_clones_local_repo(tmp_path):
    source_repo = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_repo.mkdir()

    run_command(["git", "init"], cwd=source_repo)
    run_command(["git", "config", "user.name", "Test User"], cwd=source_repo)
    run_command(["git", "config", "user.email", "test@example.com"], cwd=source_repo)
    (source_repo / "README.md").write_text("upstream\n", encoding="utf-8")
    run_command(["git", "add", "README.md"], cwd=source_repo)
    run_command(["git", "commit", "-m", "init"], cwd=source_repo)

    env = os.environ.copy()
    env["SKILLROUTER_REPO_URL"] = str(source_repo)
    env["SKILLROUTER_TARGET_DIR"] = str(target_dir)

    result = subprocess.run(
        ["bash", "scripts/bootstrap_skillrouter.sh"],
        cwd=Path.cwd(),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (target_dir / ".git").exists()
    assert result.stdout.strip()


def test_bootstrap_script_clones_into_existing_empty_dir(tmp_path):
    source_repo = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_repo.mkdir()
    target_dir.mkdir()

    run_command(["git", "init"], cwd=source_repo)
    run_command(["git", "config", "user.name", "Test User"], cwd=source_repo)
    run_command(["git", "config", "user.email", "test@example.com"], cwd=source_repo)
    (source_repo / "README.md").write_text("upstream\n", encoding="utf-8")
    run_command(["git", "add", "README.md"], cwd=source_repo)
    run_command(["git", "commit", "-m", "init"], cwd=source_repo)

    env = os.environ.copy()
    env["SKILLROUTER_REPO_URL"] = str(source_repo)
    env["SKILLROUTER_TARGET_DIR"] = str(target_dir)

    result = subprocess.run(
        ["bash", "scripts/bootstrap_skillrouter.sh"],
        cwd=Path.cwd(),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (target_dir / ".git").exists()
    assert result.stdout.strip()


def test_bootstrap_script_rejects_existing_non_git_dir(tmp_path):
    source_repo = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_repo.mkdir()
    target_dir.mkdir()
    (target_dir / "keep.txt").write_text("keep\n", encoding="utf-8")

    run_command(["git", "init"], cwd=source_repo)
    run_command(["git", "config", "user.name", "Test User"], cwd=source_repo)
    run_command(["git", "config", "user.email", "test@example.com"], cwd=source_repo)
    (source_repo / "README.md").write_text("upstream\n", encoding="utf-8")
    run_command(["git", "add", "README.md"], cwd=source_repo)
    run_command(["git", "commit", "-m", "init"], cwd=source_repo)

    env = os.environ.copy()
    env["SKILLROUTER_REPO_URL"] = str(source_repo)
    env["SKILLROUTER_TARGET_DIR"] = str(target_dir)

    result = subprocess.run(
        ["bash", "scripts/bootstrap_skillrouter.sh"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert target_dir.exists()
    assert (target_dir / "keep.txt").exists()
