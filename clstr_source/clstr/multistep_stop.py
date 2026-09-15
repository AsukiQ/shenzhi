from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str | None = None


@dataclass(frozen=True)
class MultiStepStopPolicy:
    """Benchmark-agnostic stopping policy for side-effecting multi-step loops.

    Combines three sources of stop signal in priority order:
      1. Explicit env done+success (most authoritative).
      2. Explicit env done or success alone (configurable).
      3. CLSTR STOP head logit crossing a threshold (when use_stop_head=True
         and a logit is supplied).
    """

    stop_on_done: bool = True
    stop_on_success: bool = True
    use_stop_head: bool = False
    stop_head_threshold: float = 0.0

    def decide(
        self,
        *,
        done: bool,
        success: bool,
        stop_head_logit: float | None = None,
    ) -> StopDecision:
        if bool(done) and bool(success):
            return StopDecision(True, "done_success")
        if bool(done) and self.stop_on_done:
            return StopDecision(True, "done")
        if bool(success) and self.stop_on_success:
            return StopDecision(True, "success")
        if (
            self.use_stop_head
            and stop_head_logit is not None
            and float(stop_head_logit) > float(self.stop_head_threshold)
        ):
            return StopDecision(True, "stop_head")
        return StopDecision(False, None)
