"""Phase 1 Bug 5: MultiStepStopPolicy must consume CLSTR's STOP head logit when
provided. Explicit env success still wins. CLSTRMultiStepController.select(...)
must expose the stop_head_logit through MultiStepSelection.diagnostics so the
multi-step runtime can pass it to stop_policy.decide(...)."""

from __future__ import annotations

import pytest

from clstr.multistep_stop import MultiStepStopPolicy, StopDecision


def test_default_policy_does_not_stop_without_signal():
    decision = MultiStepStopPolicy().decide(done=False, success=False)
    assert decision.should_stop is False


def test_stop_head_logit_above_threshold_triggers_stop():
    policy = MultiStepStopPolicy(use_stop_head=True, stop_head_threshold=0.0)
    decision = policy.decide(done=False, success=False, stop_head_logit=1.5)
    assert decision.should_stop is True
    assert decision.reason == "stop_head"


def test_stop_head_logit_below_threshold_does_not_stop():
    policy = MultiStepStopPolicy(use_stop_head=True, stop_head_threshold=0.0)
    decision = policy.decide(done=False, success=False, stop_head_logit=-1.5)
    assert decision.should_stop is False


def test_stop_head_does_not_override_explicit_success():
    policy = MultiStepStopPolicy(use_stop_head=True, stop_head_threshold=10.0)
    decision = policy.decide(done=True, success=True, stop_head_logit=-99.0)
    assert decision.should_stop is True
    assert decision.reason == "done_success"


def test_disabled_stop_head_ignores_logit():
    policy = MultiStepStopPolicy(use_stop_head=False)
    decision = policy.decide(done=False, success=False, stop_head_logit=10.0)
    assert decision.should_stop is False


def test_legacy_decide_call_without_stop_logit_still_works():
    """Ensure backwards compatibility — existing call sites pass only done/success."""
    decision = MultiStepStopPolicy().decide(done=False, success=True)
    assert decision.should_stop is True
    assert decision.reason == "success"


def test_stop_decision_has_optional_reason_field():
    decision = StopDecision(should_stop=False)
    assert decision.reason is None


def test_default_policy_use_stop_head_is_off_by_default():
    """Regression: stop_head must be off-by-default. An untrained stop_head with
    threshold 0.0 will fire on any positive logit, prematurely stopping tasks."""
    policy = MultiStepStopPolicy()
    assert policy.use_stop_head is False, (
        "stop_head must be off-by-default to avoid an untrained head causing early termination"
    )
    # Even a high positive logit must not stop when use_stop_head=False (the default)
    decision = policy.decide(done=False, success=False, stop_head_logit=99.0)
    assert decision.should_stop is False


def test_multistep_eval_cli_defaults_use_stop_head_off():
    from pathlib import Path
    text = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/run_appworld_multistep_executor_eval.py").read_text()
    assert "--use_stop_head" in text
    # default must be False
    assert "default=False" in text and "use_stop_head" in text, (
        "argparse default for use_stop_head must be False (off)"
    )


def test_multistep_eval_sbatch_defaults_use_stop_head_zero():
    from pathlib import Path
    text = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/sbatch/run_appworld_multistep_executor_eval.sh").read_text()
    assert "USE_STOP_HEAD=${USE_STOP_HEAD:-0}" in text or 'USE_STOP_HEAD="${USE_STOP_HEAD:-0}"' in text
