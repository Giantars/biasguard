"""Statistical integrity: regime concentration and in-/out-of-sample consistency."""

from __future__ import annotations

import math
from typing import Any

from biasguard.analytics.metrics import per_year_breakdown, split_is_oos
from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class RegimeConcentrationCheck(IntegrityCheck):
    """Flags when most of the net P&L comes from a single calendar year — an
    all-green window can hide a bad regime that simply is not in the sample."""

    key = "regime"
    name = "Regime concentration"
    category = "statistics"
    weight = 1.0

    def run(self, ctx: IntegrityContext) -> CheckResult:
        trades = ctx.baseline.trades
        if len(trades) < 3:
            return self.skip("fewer than 3 trades")
        frame = per_year_breakdown(ctx.baseline.equity, trades)
        year_net = frame["net_pnl"]
        n_years = len(year_net)
        if n_years < 2:
            return self.skip("only one year of data — cannot assess regime concentration")

        total = float(year_net.sum())
        if total <= 0:
            return self.result(
                Status.WARN,
                0.4,
                "not net profitable across years",
                n_years=n_years,
                total_net=total,
            )

        best = float(year_net.max())
        frac = best / total
        even = 1.0 / n_years
        score = (1.0 - frac) / (1.0 - even) if even < 1.0 else 1.0
        if frac > 0.7:
            return self.result(
                Status.WARN,
                score,
                f"{frac:.0%} of net P&L comes from a single year — regime-dependent",
                detail=(
                    "A tiny drawdown in an all-green window is a floor, not a ceiling: the bad "
                    "regime is simply absent from the sample."
                ),
                best_year_fraction=frac,
                n_years=n_years,
            )
        return self.result(
            Status.PASS,
            score,
            f"P&L spread across {n_years} years (top year {frac:.0%})",
            best_year_fraction=frac,
            n_years=n_years,
        )


class OutOfSampleCheck(IntegrityCheck):
    """Splits the run in time and compares the later period to the earlier one.

    Note: this is a *time split*, not a substitute for a true out-of-sample test
    with a pre-registered cut date — but a strategy whose edge evaporates in the
    later period is a classic overfitting/regime signature worth flagging.
    """

    key = "out_of_sample"
    name = "In-sample vs. out-of-sample consistency"
    category = "statistics"
    weight = 1.5

    def run(self, ctx: IntegrityContext) -> CheckResult:
        equity = ctx.baseline.equity
        trades = ctx.baseline.trades
        if len(trades) < 6 or len(equity) < 4:
            return self.skip("too few trades/bars for an IS/OOS split")

        cut = ctx.oos_cut if ctx.oos_cut is not None else equity.index[int(len(equity) * 0.7)]
        m_is, m_oos = split_is_oos(
            equity, trades, cut, initial_capital=ctx.baseline.initial_capital
        )
        if m_is.n_trades == 0 or m_oos.n_trades == 0:
            return self.skip("split leaves one side with no trades")

        detail_split = f"cut at {cut}; IS trades={m_is.n_trades}, OOS trades={m_oos.n_trades}"
        if m_is.total_pnl > 0 and m_oos.total_pnl <= 0:
            return self.result(
                Status.FAIL,
                0.1,
                "profitable in-sample, unprofitable out-of-sample",
                detail=(
                    "Performance does not persist into the later period — a classic overfitting "
                    "or regime signature. (This is a time split, not a pre-registered OOS.)"
                ),
                is_net=m_is.total_pnl,
                oos_net=m_oos.total_pnl,
                is_trades=m_is.n_trades,
                oos_trades=m_oos.n_trades,
            )

        # Retention is only meaningful when there is an in-sample edge to persist;
        # otherwise a negative/negative ratio would falsely read as "persisted".
        if m_is.expectancy <= 0 or not math.isfinite(m_is.expectancy):
            return self.skip("no in-sample edge to test out-of-sample persistence of")

        retention = m_oos.expectancy / m_is.expectancy
        common: dict[str, Any] = {
            "is_net": m_is.total_pnl,
            "oos_net": m_oos.total_pnl,
            "expectancy_retention": retention,
        }
        if math.isfinite(retention) and retention < 0.3:
            return self.result(
                Status.WARN,
                max(0.0, retention),
                f"OOS expectancy is only {retention:.0%} of in-sample ({detail_split})",
                **common,
            )
        if math.isfinite(retention):
            return self.result(
                Status.PASS,
                min(1.0, retention),
                f"performance persists out-of-sample (expectancy retention {retention:.0%})",
                **common,
            )
        return self.result(Status.PASS, 0.8, "performance persists out-of-sample", **common)


__all__ = ["OutOfSampleCheck", "RegimeConcentrationCheck"]
