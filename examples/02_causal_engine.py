"""Example 2 — the causal engine and the truncation test.

Run:  python examples/02_causal_engine.py

Shows the event loop running a simple strategy, then demonstrates the mechanism
that makes biasguard trustworthy: re-running on truncated history (``data[:T]``)
must reproduce the earlier decisions exactly. A causal strategy passes; a
strategy that peeks one bar ahead is caught — a preview of the Phase 7
``LookaheadDetector``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from biasguard.engine import run_backtest
from biasguard.events import SignalEvent
from biasguard.strategy import Strategy, StrategyContext


def sample_data(n: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="1min", tz="America/Chicago")
    # Deterministic sawtooth around a drift so signals are non-trivial (no RNG).
    close = 15000.0 + np.arange(n) * 0.1 + (np.arange(n) % 11 - 5) * 3.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx
    )
    df.index.name = "timestamp"
    return df


class UptickStrategy(Strategy):
    """Causal: goes long when the last close exceeds the prior close."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        view = ctx.view
        if len(view) < 2:
            return ()
        return (ctx.long(),) if view.closes[-1] > view.closes[-2] else ()


class PeekingStrategy(Strategy):
    """LEAKY: reaches through the numpy view's ``.base`` to read the *next* bar.

    This is exactly the entry-bar lookahead from the brief, and it is precisely
    what the truncation test is built to catch.
    """

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        base = closes.base  # the engine's full column for THIS run
        i = ctx.index
        if base is not None and i + 1 < len(base) and base[i + 1] > ctx.bar.close:
            return (ctx.long(),)
        return ()


def first_divergence(data: pd.DataFrame, factory, cuts: Sequence[int]) -> int | None:
    """Return the first T at which the truncated run's decisions differ, else None."""
    full = run_backtest(data, factory, symbol="NQ")
    for t in cuts:
        partial = run_backtest(data, factory, symbol="NQ", upto=t)
        cutoff = data.index[t - 1]
        expected = tuple(s for s in full.signals if s.timestamp <= cutoff)
        if partial.signals != expected:
            return t
    return None


def main() -> None:
    data = sample_data()
    cuts = (40, 80, 120, 160)

    result = run_backtest(data, UptickStrategy, symbol="NQ")
    print(f"UptickStrategy produced {len(result.signals)} signals over {result.n_bars} bars.\n")

    print("Truncation test (decisions on data[:T] must match the full run):")
    for name, factory in (
        ("UptickStrategy (causal)", UptickStrategy),
        ("PeekingStrategy (leaky)", PeekingStrategy),
    ):
        diverged = first_divergence(data, factory, cuts)
        verdict = "PASS — causal" if diverged is None else f"LEAK DETECTED at T={diverged}"
        print(f"  {name:28s} -> {verdict}")


if __name__ == "__main__":
    main()
