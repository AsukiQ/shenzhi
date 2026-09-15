from clstr.multistep_stop import MultiStepStopPolicy


def test_multistep_stop_policy_treats_done_as_terminal_even_without_success():
    policy = MultiStepStopPolicy()

    decision = policy.decide(done=True, success=False)

    assert decision.should_stop is True
    assert decision.reason == "done"


def test_multistep_stop_policy_can_leave_nonterminal_failures_open():
    policy = MultiStepStopPolicy()

    decision = policy.decide(done=False, success=False)

    assert decision.should_stop is False
    assert decision.reason is None


def test_multistep_stop_policy_can_be_configured_for_nonterminal_done():
    policy = MultiStepStopPolicy(stop_on_done=False, stop_on_success=True)

    done_only = policy.decide(done=True, success=False)
    success = policy.decide(done=False, success=True)

    assert done_only.should_stop is False
    assert success.should_stop is True
    assert success.reason == "success"
