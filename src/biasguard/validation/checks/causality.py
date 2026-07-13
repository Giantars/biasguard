"""Causality: does truncating history change past decisions? (lookahead / repaint)"""

from __future__ import annotations

from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class LookaheadCheck(IntegrityCheck):
    """Re-runs on ``data[:T]`` for several T; decisions within the first T bars
    must be byte-identical to the full run. Any difference means the strategy (or
    engine) used information it could not have had live. This is a **gate**: a
    failure caps the whole Integrity Score.
    """

    key = "lookahead"
    name = "Lookahead / repaint (truncation test)"
    category = "causality"
    is_gate = True
    weight = 2.0

    def run(self, ctx: IntegrityContext) -> CheckResult:
        data = ctx.spec.data
        n = len(data)
        # The truncation test slices positionally (data.iloc[:t]) but compares
        # signals by timestamp; those agree only for a strictly-increasing index.
        # A non-monotonic / duplicated index is a *data* defect (the OHLC check
        # flags it) — not a lookahead leak, so skip rather than falsely FAIL.
        if not (data.index.is_monotonic_increasing and data.index.is_unique):
            return self.skip("data index is not strictly increasing; see the OHLC data check")
        baseline_signals = ctx.baseline.signals
        if not baseline_signals:
            return self.skip("strategy emitted no signals to test")

        cuts = sorted({int(n * f) for f in (0.25, 0.5, 0.75)})
        cuts = [t for t in cuts if 2 <= t < n]
        if not cuts:
            return self.skip("not enough bars to truncate")

        for t in cuts:
            out = ctx.rerun(data=data.iloc[:t])
            cutoff = data.index[t - 1]
            expected = tuple(s for s in baseline_signals if s.timestamp <= cutoff)
            if out.signals != expected:
                return self.result(
                    Status.FAIL,
                    0.0,
                    f"decisions change when history is truncated at bar {t} — lookahead/repaint",
                    detail=(
                        "Re-running on data[:T] produced different past decisions than the full "
                        "run. Live trading cannot see the future the backtest used. This is the "
                        "single most expensive class of backtest lie."
                    ),
                    diverged_at=t,
                    n_expected=len(expected),
                    n_got=len(out.signals),
                )

        return self.result(
            Status.PASS,
            1.0,
            f"decisions byte-identical under truncation at bars {cuts} — causal",
            cuts_tested=cuts,
            n_signals=len(baseline_signals),
        )


__all__ = ["LookaheadCheck"]
