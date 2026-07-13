"""Example 6 - Monte Carlo uncertainty & prop-firm risk analysis.

Run:  python examples/06_monte_carlo.py

Answers "how uncertain is this result, and how bad could realistic outcomes be?"
Shows: (1) the block-bootstrap distribution (P(profit), p5/p50/p95, worst DD),
(2) why block bootstrap matters - IID shuffling destroys streaks and *under*-
estimates drawdown, (3) prop-firm risk limits (P(hit trailing drawdown)), and
(4) Monte Carlo feeding the Integrity Score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from biasguard.events import SignalEvent
from biasguard.execution import NQ, PerContractCommission, TickSlippage
from biasguard.montecarlo import (
    AccountConfig,
    IIDBootstrap,
    MonteCarloSimulator,
    recent_regime_mask,
)
from biasguard.strategy import Strategy, StrategyContext
from biasguard.validation import BacktestSpec, assess_integrity


def data(n: int = 720) -> pd.DataFrame:
    idx = pd.date_range("2022-01-03 08:30", periods=n, freq="30min", tz="America/Chicago")
    # Regime blocks: mostly up-trending, every third 120-bar block trends down.
    # Trend-following wins in up-blocks and loses in clusters at the turns -> streaks.
    t = np.arange(n)
    drift = np.where((t // 120) % 3 == 2, -1.1, 0.7)
    close = 15000.0 + np.cumsum(drift) + np.sin(t / 2.0) * 4.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": 500.0,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


class PeriodicHold(Strategy):
    """Educational: always take an 8-bar long, then re-enter. On the regime data
    this wins in up-blocks and loses in clusters during down-blocks (streaks)."""

    hold_bars = 8

    def __init__(self) -> None:
        self._entry: int | None = None

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.position == 0 and self._entry is None:
            self._entry = ctx.index
            return (ctx.long(),)
        if self._entry is not None and ctx.index - self._entry >= self.hold_bars:
            self._entry = None
            return (ctx.exit(),)
        return ()


def main() -> None:
    spec = BacktestSpec(
        data=data(),
        strategy_factory=PeriodicHold,
        instrument=NQ,
        commission=PerContractCommission(1.90),
        slippage=TickSlippage(1.0),
    )
    trades = spec.run().trades
    print(f"Historical: {len(trades)} trades, net ${sum(t.net_pnl for t in trades):,.0f}\n")

    account = AccountConfig(
        starting_balance=100_000.0, trailing_drawdown_limit=3_000.0, max_loss_limit=5_000.0
    )
    sim = MonteCarloSimulator(n_paths=10_000, seed=1)

    print("=" * 72)
    print("1) BLOCK-BOOTSTRAP DISTRIBUTION (with prop-firm limits)")
    print("=" * 72)
    print(sim.run(trades, account=account).summary())
    print()

    print("=" * 72)
    print("2) WHY BLOCK, NOT IID  (IID destroys streaks -> understates drawdown)")
    print("=" * 72)
    block = sim.run(trades)
    iid = MonteCarloSimulator(bootstrap=IIDBootstrap(), n_paths=10_000, seed=1).run(trades)
    print(
        f"  worst-case drawdown (p95):  block ${block.worst_case_drawdown:,.0f}  "
        f"vs  IID ${iid.worst_case_drawdown:,.0f}"
    )
    print()

    print("=" * 72)
    print("3) REGIME-CONDITIONED (resample only the recent half)")
    print("=" * 72)
    recent = sim.run(trades, regime_mask=recent_regime_mask(len(trades), 0.5))
    print(
        f"  P(profit): full history {block.prob_profit:.0%}  vs  recent regime {recent.prob_profit:.0%}"
    )
    print()

    print("=" * 72)
    print("4) MONTE CARLO IN THE INTEGRITY SCORE")
    print("=" * 72)
    report = assess_integrity(spec, config={"account": account})
    mc = report.get("monte_carlo")
    if mc is not None:
        print(f"  [{mc.status}] {mc.name}: {mc.summary}")
    print(f"  Overall integrity: {report.score:.0f}/100 ({report.grade})")


if __name__ == "__main__":
    main()
