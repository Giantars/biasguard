"""Monte Carlo robustness as an integrity check.

Resamples the trade ledger with a block bootstrap (preserving streaks) and asks:
under realistic reorderings, how often does the strategy still make money, and
how bad do the drawdowns get? A historical profit that survives resampling only
a minority of the time is not a reliable result — so this check contributes
``P(profit)`` (blended with ``1 - P(breach)`` when account limits are set) to the
Integrity Score.
"""

from __future__ import annotations

from typing import Any

from biasguard.montecarlo.account import AccountConfig
from biasguard.montecarlo.simulation import MonteCarloSimulator
from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class MonteCarloCheck(IntegrityCheck):
    key = "monte_carlo"
    name = "Monte Carlo robustness"
    category = "statistics"
    weight = 1.5
    #: Paths for the check (block bootstrap; deterministic given the seed).
    n_paths = 2000
    #: Minimum trades before a resampling distribution is meaningful.
    min_trades = 10

    def applicable(self, ctx: IntegrityContext) -> bool:
        return len(ctx.baseline.trades) >= self.min_trades

    def run(self, ctx: IntegrityContext) -> CheckResult:
        trades = ctx.baseline.trades
        if len(trades) < self.min_trades:
            return self.skip(f"fewer than {self.min_trades} trades")
        if ctx.baseline.net_pnl <= 0:
            return self.skip("strategy is not net profitable; nothing to stress")

        configured = ctx.config.get("account")
        account = (
            configured
            if isinstance(configured, AccountConfig)
            else AccountConfig(starting_balance=ctx.baseline.initial_capital)
        )

        sim = MonteCarloSimulator(n_paths=self.n_paths, seed=ctx.seed)
        result = sim.run(trades, account=account)

        p_profit = result.prob_profit
        p_breach = result.prob_breach
        fp = result.final_pnl_percentiles
        metrics: dict[str, Any] = {
            "prob_profit": p_profit,
            "final_p5": fp["p5"],
            "final_p50": fp["p50"],
            "final_p95": fp["p95"],
            "worst_case_drawdown": result.worst_case_drawdown,
            "bootstrap": result.bootstrap_name,
        }
        # Only report a breach probability when limits are actually configured,
        # so it never shows a spurious "P(breach)=0%" for a limitless account
        # (keeps this in step with MonteCarloResult, which omits it likewise).
        if account.has_limits:
            metrics["prob_breach"] = p_breach

        score = p_profit * (1.0 - p_breach) if account.has_limits else p_profit
        base_summary = (
            f"P(profit)={p_profit:.0%} under {result.bootstrap_name} resampling; "
            f"final p5/p95 = ${fp['p5']:,.0f}/${fp['p95']:,.0f}, worst DD ${result.worst_case_drawdown:,.0f}"
        )
        if account.has_limits:
            base_summary += f"; P(breach)={p_breach:.0%}"

        if p_profit < 0.5:
            return self.result(
                Status.FAIL,
                score,
                f"only {p_profit:.0%} of resampled paths stay profitable — result is not robust",
                detail=(
                    "A block bootstrap that preserves win/loss streaks finds the historical "
                    "profit reverses in most reorderings. The edge is likely luck of sequence."
                ),
                **metrics,
            )
        if p_profit < 0.75 or (account.has_limits and p_breach > 0.30):
            return self.result(
                Status.WARN,
                score,
                base_summary,
                detail="The result is positive but fragile under resampling / account limits.",
                **metrics,
            )
        return self.result(Status.PASS, score, base_summary, **metrics)


__all__ = ["MonteCarloCheck"]
