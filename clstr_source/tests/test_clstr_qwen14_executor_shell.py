import json
import subprocess
from pathlib import Path


def test_provider_credential_loader_is_explicit_and_silent(tmp_path: Path):
    credential_file = tmp_path / "provider.json"
    credential_file.write_text(
        json.dumps(
            [
                {
                    "api_key": "secret-test-key",
                    "api_base": "https://provider.invalid/v1",
                }
            ]
        ),
        encoding="utf-8",
    )
    script = Path("scripts/sbatch/_clstr_qwen14_executor.sh").resolve()
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; CLSTR_PROVIDER_CREDENTIAL_FILE="$2"; '
            "load_clstr_provider_credentials; "
            '[[ "$OPENAI_API_KEY" == "secret-test-key" ]]; '
            '[[ "$OPENAI_BASE_URL" == "https://provider.invalid/v1" ]]',
            "bash",
            str(script),
            str(credential_file),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert "secret-test-key" not in proc.stderr


def test_provider_credential_loader_fails_closed_on_empty_file(tmp_path: Path):
    credential_file = tmp_path / "provider.json"
    credential_file.write_text("[{}]\n", encoding="utf-8")
    script = Path("scripts/sbatch/_clstr_qwen14_executor.sh").resolve()
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; load_clstr_provider_credentials "$2"',
            "bash",
            str(script),
            str(credential_file),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "usable key/base pair" in proc.stderr


def test_alfworld_launcher_supports_separate_seen_and_unseen_protocols():
    script = Path("scripts/sbatch/run_clstr_vnext_alfworld_eval.sh").read_text(
        encoding="utf-8"
    )

    assert "SPLIT=${SPLIT:-valid_seen}" in script
    assert '"${SPLIT}" != "valid_seen"' in script
    assert '"${SPLIT}" != "valid_unseen"' in script
    assert "MAX_EPISODES=${MAX_EPISODES:-140}" in script
    assert "MAX_EPISODES=${MAX_EPISODES:-134}" in script
    assert '--split "${SPLIT}"' in script
    assert "ALFWorld smoke split differs from full split" in script
    assert "CLSTR_EXECUTOR_INTERFACE=${CLSTR_EXECUTOR_INTERFACE:-exact_prior}" in script
    assert "SCORING_METHOD=${SCORING_METHOD:-generate}" in script
    assert "LOOP_GUARD=${LOOP_GUARD:-1}" in script
    assert "LOOP_GUARD must be 0 or 1" in script
    assert 'LOOP_GUARD_FLAG=--no-loop_guard' in script
    assert "abstract_grounded requires SCORING_METHOD=likelihood" not in script
    assert "ALFWorld grounded smoke appended runtime pseudo skills" in script
    assert '"${CLSTR_EXECUTOR_INTERFACE}" != "guided_exact_prior"' in script
    assert '"${CLSTR_EXECUTOR_INTERFACE}" != "guided_static_exact_prior"' in script
    assert '"${CLSTR_EXECUTOR_INTERFACE}" != "legacy_guided_exact_prior"' in script
    assert "ALFWorld guided exact-prior smoke omitted score fusion" in script
    assert "ALFWorld guided exact-prior smoke omitted skill guidance" in script
    assert "ALFWorld guided static exact-prior smoke did not force static grounding" in script
    assert "ALFWorld legacy guided exact-prior smoke omitted score fusion" in script
    assert 'accepted_schemas.add("alfworld_mapped_abstract_skill_v1")' in script


def test_executor_uses_runtime_python_environment_tools():
    script = Path("scripts/sbatch/_clstr_qwen14_executor.sh").read_text(
        encoding="utf-8"
    )

    assert 'export PATH="$(dirname "${python_bin}"):${PATH}"' in script


def test_toolbench_launcher_uses_dependency_complete_runtime():
    script = Path("scripts/sbatch/run_clstr_vnext_toolbench_official.sh").read_text(
        encoding="utf-8"
    )

    assert "TOOLBENCH_PYTHON_BIN" in script
    assert "envs/reasoning_trap/bin/python" in script
    # The readiness preflight names the entrypoint before runtime setup.  The
    # final occurrence is the actual execution command whose interpreter order
    # this contract protects.
    assert script.index("TOOLBENCH_PYTHON_BIN") < script.rindex(
        "scripts/run_stabletoolbench_mirrorapi_server.py"
    )


def test_toolsandbox_and_tau2_launchers_use_dependency_complete_runtimes():
    contracts = (
        (
            "scripts/sbatch/run_clstr_vnext_toolsandbox_success.sh",
            "TOOLSANDBOX_PYTHON_BIN",
            "scripts/run_clstr_vnext_toolsandbox_success.py",
        ),
        (
            "scripts/sbatch/run_clstr_vnext_tau2_success.sh",
            "TAU2_PYTHON_BIN",
            "scripts/run_clstr_vnext_tau2_success.py",
        ),
    )
    for path, variable, entrypoint in contracts:
        script = Path(path).read_text(encoding="utf-8")
        assert variable in script
        assert "envs/reasoning_trap/bin/python" in script
        # The entrypoint is also mentioned by the dependency-light AST
        # readiness preflight. Compare against its actual execution mention.
        assert script.index(variable) < script.rindex(entrypoint)

    toolsandbox = Path(
        "scripts/sbatch/run_clstr_vnext_toolsandbox_success.sh"
    ).read_text(encoding="utf-8")
    assert (
        "CLSTR_INTERVENTION_MODE=${CLSTR_INTERVENTION_MODE:-ranked_guidance}"
        in toolsandbox
    )
    assert "GUIDANCE_TOP_K=${GUIDANCE_TOP_K:-2}" in toolsandbox
    assert "EXECUTOR_MAX_TOKENS=${EXECUTOR_MAX_TOKENS:-1024}" in toolsandbox
    assert "EXECUTOR_ENABLE_THINKING=${EXECUTOR_ENABLE_THINKING:-0}" in toolsandbox
    assert "EXECUTOR_RANDOM_SEED=${EXECUTOR_RANDOM_SEED:-0}" in toolsandbox
    assert "USER_TEMPERATURE=${USER_TEMPERATURE:-0.0}" in toolsandbox
    assert "USER_MAX_TOKENS=${USER_MAX_TOKENS:-1024}" in toolsandbox
    assert '--intervention_mode "${CLSTR_INTERVENTION_MODE}"' in toolsandbox
    assert '--guidance_top_k "${GUIDANCE_TOP_K}"' in toolsandbox
    assert '--executor_max_tokens "${EXECUTOR_MAX_TOKENS}"' in toolsandbox
    assert "args+=(--no-executor_enable_thinking)" in toolsandbox
    assert '--executor_random_seed "${EXECUTOR_RANDOM_SEED}"' in toolsandbox
    assert '--user_temperature "${USER_TEMPERATURE}"' in toolsandbox
    assert '--user_max_tokens "${USER_MAX_TOKENS}"' in toolsandbox
    assert '[[ "${CLSTR_INTERVENTION_MODE}" != "ranked_guidance" ]]' in toolsandbox
    assert '[[ "${EXECUTOR_ENABLE_THINKING}" != "0" ]]' in toolsandbox
    executor = Path("scripts/sbatch/_clstr_qwen14_executor.sh").read_text(
        encoding="utf-8"
    )
    assert '--random-seed "${EXECUTOR_RANDOM_SEED:-0}"' in executor


def test_controlled_tau2_e3_launcher_locks_the_scientific_contract():
    path = Path("scripts/sbatch/run_clstr_matched_history_tau2_e3.sh")
    script = path.read_text(encoding="utf-8")
    assert "METHOD must be static, transformer, or lstr" in script
    assert "clstr_vnext_stage2-step0.pt" in script
    assert "pilot_v2_seed23/transformer/best.pt" in script
    assert "full_seed31/lstr/best.pt" in script
    assert 'TOP_K=${TOP_K:-8}' in script
    assert 'MAX_STEPS=${MAX_STEPS:-100}' in script
    assert 'NUM_TRIALS=${NUM_TRIALS:-1}' in script
    assert 'EXECUTOR_TEMPERATURE=${EXECUTOR_TEMPERATURE:-0.0}' in script
    assert "20/40/40 official Tau2 split" in script
    assert "reasoning_trap/bin/python" in script
    assert "scripts/run_clstr_matched_history_tau2_e3.py" in script
    assert 'CLSTR_PROXY_JUMP_HOST=${CLSTR_PROXY_JUMP_HOST:-ln01}' in script
    assert "CLSTR_PROVIDER_CREDENTIAL_FILE=" in script
    assert '[[ "${RUN_EVAL}" == "1" && ! -r "${CLSTR_PROVIDER_CREDENTIAL_FILE}" ]]' in script
    assert "export CLSTR_PROVIDER_CREDENTIAL_FILE" in script


def test_controlled_selector_smoke_never_calls_an_llm_api():
    launcher = Path(
        "scripts/sbatch/run_clstr_matched_history_selector_smoke_e3.sh"
    ).read_text(encoding="utf-8")
    runner = Path("scripts/smoke_clstr_matched_history_online_e3.py").read_text(
        encoding="utf-8"
    )
    assert "clstr_vnext_stage2-step0.pt" in launcher
    assert "smoke_clstr_matched_history_online_e3.py" in launcher
    assert 'for method in ("static", "transformer", "lstr")' in runner
    assert "zero-history rankings differ across arms" in runner
    assert "history-bearing Static supports differ across arms" in runner
    assert '"no_executor_or_user_model_called": True' in runner
    assert "run_domain" not in runner
    assert "start_clstr_provider_proxy_tunnel" not in launcher


def test_provider_credentials_are_not_passed_in_child_command_lines():
    toolbench = Path(
        "scripts/sbatch/run_clstr_vnext_toolbench_official.sh"
    ).read_text(encoding="utf-8")
    toolsandbox = Path(
        "scripts/sbatch/run_clstr_vnext_toolsandbox_success.sh"
    ).read_text(encoding="utf-8")
    tau2 = Path("scripts/sbatch/run_clstr_vnext_tau2_success.sh").read_text(
        encoding="utf-8"
    )
    mirror_server = Path(
        "scripts/run_stabletoolbench_mirrorapi_server.py"
    ).read_text(encoding="utf-8")

    assert "MIRROR_SERVER_API_KEY" in toolbench
    assert '--api_key "${MIRROR_API_KEY' not in toolbench
    assert 'mirrorapi_runtime/config_mirrorapi.yml"' in toolbench
    assert '--user_api_key "${USER_API_KEY}"' not in toolsandbox
    assert '--user_api_key "${USER_API_KEY}"' not in tau2
    assert 'os.environ.get("MIRROR_SERVER_API_KEY"' in mirror_server
