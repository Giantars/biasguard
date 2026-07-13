"""Transaction-cost integrity: are costs real, and is the edge > 2x round-trip?"""

from __future__ import annotations

import numpy as np

from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class CostIntegrityCheck(IntegrityCheck):
    key = "costs"
    name = "Transaction costs"
    category = "execution"
    weight = 1.5

    def run(self, ctx: IntegrityContext) -> CheckResult:
        fills = ctx.baseline.fills
        if not fills:
            return self.skip("no fills to assess costs")

        mult = ctx.spec.instrument.multiplier
        total_comm = sum(f.commission for f in fills)
        total_slip = sum(f.slippage * f.quantity * mult for f in fills)  # price units -> $

        if ctx.spec.zero_cost() or (total_comm == 0.0 and total_slip == 0.0):
            return self.result(
                Status.FAIL,
                0.0,
                "zero commission AND slippage — costs are not modelled",
                detail=(
                    "A zero-cost backtest is the maximally optimistic case; real fills are "
                    "never free. Many high-frequency edges flip negative once costs apply."
                ),
                total_commission=total_comm,
                total_slippage_cost=total_slip,
            )

        cost_per_fill = (total_comm + total_slip) / len(fills)
        round_trip = 2.0 * cost_per_fill

        trade_pnls = [t.pnl for t in ctx.baseline.trades]
        if not trade_pnls or round_trip <= 0:
            return self.result(
                Status.PASS, 0.9, "costs modelled; not enough trades to compare edge vs cost"
            )

        median_gross = float(np.median(np.abs(trade_pnls)))
        ratio = median_gross / round_trip
        if ratio < 2.0:
            return self.result(
                Status.WARN,
                ratio / 2.0 * 0.7,
                f"median trade is only {ratio:.1f}x round-trip cost (sub-2x is fragile)",
                detail="An edge that is only a small multiple of costs is fragile to real execution.",
                median_gross_pnl=median_gross,
                round_trip_cost=round_trip,
                ratio=ratio,
            )
        return self.result(
            Status.PASS,
            1.0,
            f"median trade is {ratio:.1f}x round-trip cost; costs modelled",
            median_gross_pnl=median_gross,
            round_trip_cost=round_trip,
            ratio=ratio,
        )


__all__ = ["CostIntegrityCheck"]
