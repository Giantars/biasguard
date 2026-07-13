"""Example 5 - the Backtest Integrity Framework.

Run:  python examples/05_integrity_framework.py

Answers "how much should you trust these results?" with a 0-100 Integrity Score.
Shows: (1) a causal strategy scoring well, (2) a strategy with a planted
lookahead leak getting its score capped by the gate, and (3) plugging in a
custom check without touching the engine.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from biasguard.events import SignalEvent
from biasguard.execution import NQ, PerContractCommission, TickSlippage
from biasguard.strategy import Strategy, StrategyContext
from biasguard.validation import (
    BacktestSpec,
    CheckResult,
    IntegrityCheck,
    IntegrityContext,
    Status,
    assess_integrity,
    register_check,
)
from biasguard.validation.registry import DEFAULT_REGISTRY


def data(n: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2023-01-03 08:30", periods=n, freq="15min", tz="America/Chicago")
    close = 15000.0 + np.cumsum(np.sin(np.arange(n) / 8.0)) * 4.0 + np.arange(n) * 0.4
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


class Momentum(Strategy):
    """Causal: long after 3 rising closes, flat otherwise."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        c = ctx.view.closes
        if len(c) < 4:
            return ()
        rising = bool(np.all(np.diff(c[-4:]) > 0))
        if rising and ctx.position == 0:
            return (ctx.long(),)
        if not rising and ctx.position != 0:
            return (ctx.exit(),)
        return ()


class LeakyMomentum(Strategy):
    """PLANTED LOOKAHEAD: peeks one bar into the future via the view's .base."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        base = closes.base
        i = ctx.index
        if base is not None and i + 1 < len(base) and base[i + 1] > ctx.bar.close:
            return (ctx.long(),) if ctx.position == 0 else ()
        return (ctx.exit(),) if ctx.position != 0 else ()


class MinimumTradesCheck(IntegrityCheck):
    """A custom plugin: warns when the sample is too small to trust."""

    key = "min_trades"
    name = "Sample size"
    category = "statistics"
    weight = 0.5

    def run(self, ctx: IntegrityContext) -> CheckResult:
        n = len(ctx.baseline.trades)
        if n >= 30:
            return self.result(Status.PASS, 1.0, f"{n} trades - adequate sample")
        return self.result(
            Status.WARN, n / 30.0, f"only {n} trades - small sample; stats are noisy", n_trades=n
        )


def make_spec(strategy: type[Strategy]) -> BacktestSpec:
    return BacktestSpec(
        data=data(),
        strategy_factory=strategy,
        instrument=NQ,
        commission=PerContractCommission(1.90),
        slippage=TickSlippage(1.0),
    )


def main() -> None:
    # Plug in a custom check (no engine change).
    if DEFAULT_REGISTRY.get("min_trades") is None:
        register_check(MinimumTradesCheck())

    for label, strat in (
        ("Causal momentum", Momentum),
        ("Leaky momentum (planted lookahead)", LeakyMomentum),
    ):
        report = assess_integrity(make_spec(strat))
        print("=" * 72)
        print(label)
        print("=" * 72)
        print(report.summary())
        print()


if __name__ == "__main__":
    main()
