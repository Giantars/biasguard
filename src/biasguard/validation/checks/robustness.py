"""Robustness: how fast does the edge die as execution gets more realistic?

This module also demonstrates the perturbation seam that future robustness
analyses (price-noise, execution-delay, parameter sweeps) plug into: each just
calls ``ctx.rerun(<perturbation>)`` — no engine change required.
"""

from __future__ import annotations

from biasguard.execution.costs import TickSlippage
from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class SlippageSensitivityCheck(IntegrityCheck):
    """Re-runs with escalating slippage and finds the break-even level in ticks."""

    key = "slippage_sensitivity"
    name = "Slippage sensitivity"
    category = "robustness"
    weight = 1.0
    levels_ticks: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)

    def run(self, ctx: IntegrityContext) -> CheckResult:
        if not ctx.baseline.trades:
            return self.skip("no trades to stress")
        nets = {k: ctx.rerun(slippage=TickSlippage(k)).net_pnl for k in self.levels_ticks}
        return self._verdict(nets)

    def _verdict(self, nets: dict[float, float]) -> CheckResult:
        net_by_level = {str(k): v for k, v in nets.items()}
        # If P&L is identical at every slippage level, slippage was never
        # actually applied (maker-only limit fills). This sweep cannot assess
        # such a strategy — do not certify it as robust; defer to the
        # fill-realism trade-through test.
        if len(set(nets.values())) == 1:
            return self.skip(
                "slippage never applied (maker-only fills); see the fill-realism trade-through test",
                nets=net_by_level,
            )
        if nets[0.0] <= 0:
            return self.skip("no positive edge even at zero slippage", nets=net_by_level)

        breakeven = next((k for k in self.levels_ticks if nets[k] <= 0), None)
        if breakeven is None:
            return self.result(
                Status.PASS,
                1.0,
                f"edge survives {self.levels_ticks[-1]} ticks of slippage",
                nets=net_by_level,
            )
        if breakeven <= 1.0:
            return self.result(
                Status.FAIL,
                breakeven / 2.0 * 0.5,
                f"edge disappears at {breakeven} tick(s) of slippage",
                detail=(
                    "A tick of slippage is routine; an edge that dies inside one tick is a fill "
                    "artifact, not alpha."
                ),
                breakeven_ticks=breakeven,
                nets=net_by_level,
            )
        if breakeven <= 2.0:
            return self.result(
                Status.WARN,
                0.6,
                f"edge dies at {breakeven} ticks of slippage — thin margin",
                breakeven_ticks=breakeven,
                nets=net_by_level,
            )
        return self.result(
            Status.PASS,
            0.85,
            f"edge tolerates up to {breakeven} ticks of slippage",
            breakeven_ticks=breakeven,
            nets=net_by_level,
        )


__all__ = ["SlippageSensitivityCheck"]
