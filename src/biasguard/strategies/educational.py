"""Educational example strategies — teaching artifacts, not trading systems.

Every class here is deliberately simple and carries the same disclaimer:

    Educational example — not intended as a profitable trading strategy.

Two are honest, causal templates (moving-average crossover, RSI mean reversion);
four are *intentionally flawed* to demonstrate what the Backtest Integrity
Framework catches:

* :class:`LookaheadStrategy`      -> FAIL the lookahead gate (reads a future bar);
* :class:`OverfitMeanReversion`   -> WARN (P&L concentrated in one regime);
* :class:`ZeroSlippageScalper`    -> WARN (a thin edge slippage erases);
* :class:`FreeLunchChurn`         -> FAIL on costs when run with zero fees.

The point is to show *how BiasGuard is used* and *how it evaluates behavior*, not
to publish edges. Data generators live here too so the examples are reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from biasguard.events import SignalEvent
from biasguard.indicators import rsi, sma
from biasguard.strategy.base import Strategy, StrategyContext

EDUCATIONAL_LABEL = "Educational example — not intended as a profitable trading strategy."


class EducationalStrategy(Strategy):
    """Base class stamping the educational disclaimer on every example."""

    #: Captured by ``build_manifest`` and surfaced in the HTML report header and
    #: the AI audit export, so a generated artifact is never mistaken for an edge.
    label: str = EDUCATIONAL_LABEL


# --------------------------------------------------------------------------- #
# Honest, causal templates
# --------------------------------------------------------------------------- #
class MovingAverageCrossover(EducationalStrategy):
    """Long-only fast/slow SMA crossover; flat when the fast SMA is below the slow.

    Educational example — not intended as a profitable trading strategy. Causal:
    both SMAs are read from the completed-bars view at ``[-1]``.
    """

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("periods must be >= 1")
        if fast >= slow:
            raise ValueError("fast period must be shorter than slow period")
        self.fast = fast
        self.slow = slow

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        if len(closes) < self.slow:
            return ()
        fast = sma(closes, self.fast)[-1]
        slow = sma(closes, self.slow)[-1]
        if fast > slow and ctx.position <= 0:
            return (ctx.long(),)
        if fast < slow and ctx.position > 0:
            return (ctx.exit(),)
        return ()


class RsiMeanReversion(EducationalStrategy):
    """Long-only RSI mean reversion: buy oversold, exit as RSI recovers to neutral.

    Educational example — not intended as a profitable trading strategy. Causal:
    RSI is read from the completed-bars view at ``[-1]``.
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, exit_level: float = 55.0) -> None:
        if not 0.0 < oversold < exit_level < 100.0:
            raise ValueError("require 0 < oversold < exit_level < 100")
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        if len(closes) < self.period + 1:
            return ()
        value = rsi(closes, self.period)[-1]
        if math.isnan(value):
            return ()
        if value < self.oversold and ctx.position <= 0:
            return (ctx.long(),)
        if value > self.exit_level and ctx.position > 0:
            return (ctx.exit(),)
        return ()


# --------------------------------------------------------------------------- #
# Intentionally flawed examples
# --------------------------------------------------------------------------- #
class LookaheadStrategy(EducationalStrategy):
    """PLANTED LOOKAHEAD: peeks at the next bar's close via the numpy view's
    ``.base``, then "predicts" it. Expected verdict: FAIL the lookahead gate.

    Educational example — not intended as a profitable trading strategy. This is
    the archetype of an accidental leak (global normalization, a shifted column,
    a vectorized indicator computed over all data): the truncation test re-runs on
    ``data[:T]`` and the decisions change, so it is caught.
    """

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        backing = closes.base  # the full column, including the future
        i = ctx.index
        if backing is None or i + 1 >= len(backing):
            # Cannot peek (the last bar of a truncated run) -> flatten. This is
            # the asymmetry the truncation gate detects: the full run peeked and
            # decided differently at this same bar, so the decisions diverge.
            return (ctx.exit(),) if ctx.position > 0 else ()
        up_next = float(backing[i + 1]) > ctx.bar.close  # tomorrow's close
        if up_next and ctx.position <= 0:
            return (ctx.long(),)
        if not up_next and ctx.position > 0:
            return (ctx.exit(),)
        return ()


class OverfitMeanReversion(RsiMeanReversion):
    """The same causal RSI reversion as :class:`RsiMeanReversion`, but with
    thresholds *curve-fit* to one high-volatility regime. Its edge does not
    generalize: run across a regime change it earns almost everything in the
    calm-fitting window and little afterward. Expected verdict: WARN (regime
    concentration / weak out-of-sample retention).

    Educational example — not intended as a profitable trading strategy. Causal
    and honest about *timing* — the flaw is generalization, which is exactly what
    the statistics checks are for.
    """

    def __init__(self, period: int = 12, oversold: float = 33.0, exit_level: float = 53.0) -> None:
        super().__init__(period=period, oversold=oversold, exit_level=exit_level)


class ZeroSlippageScalper(EducationalStrategy):
    """Buys after a down-tick and exits one bar later, harvesting a 1-2 tick
    mean-reversion bounce. Expected verdict: WARN (slippage sensitivity) — the
    edge exists at zero slippage and evaporates within a tick or two.

    Educational example — not intended as a profitable trading strategy.
    """

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        if ctx.position != 0:
            return (ctx.exit(),)
        if len(closes) >= 2 and closes[-1] < closes[-2]:
            return (ctx.long(),)
        return ()


class FreeLunchChurn(EducationalStrategy):
    """Very high turnover: long when flat, exit the next bar. Round-trips every two
    bars, so trading costs dominate. Run under a zero-cost profile it looks fine;
    that is the point. Expected verdict: FAIL (transaction costs) under zero fees.

    Educational example — not intended as a profitable trading strategy.
    """

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.position == 0:
            return (ctx.long(),)
        return (ctx.exit(),)


# --------------------------------------------------------------------------- #
# Deterministic synthetic data generators (no RNG state is shared)
# --------------------------------------------------------------------------- #
def _frame(close: np.ndarray, start: str, freq: str, tz: str = "America/Chicago") -> pd.DataFrame:
    """Assemble a valid OHLCV frame around a close series (open = prior close)."""
    n = close.size
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.75
    low = np.minimum(open_, close) - 0.75
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def trending_data(n: int = 400, start_price: float = 15000.0) -> pd.DataFrame:
    """A persistent uptrend with mild noise — friendly to trend following."""
    t = np.arange(n, dtype="float64")
    close = start_price + t * 1.5 + np.sin(t / 9.0) * 12.0
    return _frame(close, "2023-01-02 08:30", "30min")


def swinging_data(n: int = 500, start_price: float = 15000.0) -> pd.DataFrame:
    """Alternating up/down trends (slow swings) so a crossover round-trips both
    ways — long entries and exits, not a single buy-and-hold."""
    t = np.arange(n, dtype="float64")
    close = start_price + np.sin(t / 28.0) * 130.0 + np.sin(t / 6.0) * 10.0
    return _frame(close, "2023-01-02 08:30", "30min")


def mean_reverting_data(n: int = 400, start_price: float = 15000.0) -> pd.DataFrame:
    """A range-bound oscillation around a level — friendly to mean reversion."""
    t = np.arange(n, dtype="float64")
    close = start_price + np.sin(t / 6.0) * 30.0 + np.cos(t / 2.3) * 8.0
    return _frame(close, "2023-01-02 08:30", "30min")


def regime_break_data(n: int = 520, start_price: float = 15000.0) -> pd.DataFrame:
    """Two calendar years: strong mean reversion in year 1, weak in year 2.

    A reversion strategy earns most of its P&L in the first regime and little in
    the second — a textbook overfitting / regime-concentration signature.
    """
    t = np.arange(n, dtype="float64")
    half = n // 2
    amplitude = np.where(t < half, 55.0, 10.0)
    close = start_price + np.sin(t / 5.0) * amplitude + np.cos(t / 2.1) * 3.0
    return _frame(close, "2023-01-02", "1D", tz="UTC")


def choppy_data(n: int = 500, start_price: float = 15000.0) -> pd.DataFrame:
    """A tight bar-to-bar zigzag (~2-tick amplitude) — a thin scalping 'edge' that
    is real at zero slippage but dies within a tick or two of it."""
    t = np.arange(n, dtype="float64")
    close = start_price + 0.5 * ((-1.0) ** t) + np.sin(t / 60.0) * 0.25
    return _frame(close, "2023-01-02 08:30", "15min")


__all__ = [
    "EDUCATIONAL_LABEL",
    "EducationalStrategy",
    "FreeLunchChurn",
    "LookaheadStrategy",
    "MovingAverageCrossover",
    "OverfitMeanReversion",
    "RsiMeanReversion",
    "ZeroSlippageScalper",
    "choppy_data",
    "mean_reverting_data",
    "regime_break_data",
    "swinging_data",
    "trending_data",
]
