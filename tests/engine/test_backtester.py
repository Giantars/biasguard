"""Tests for the Backtester event loop.

The two headline tests here are the causal guarantees:

* ``TestTruncationDeterminism`` — running on ``data[:T]`` reproduces every
  decision made within the first ``T`` bars byte-for-byte.
* ``TestNextBarExecution`` — an order created from bar ``i``'s signal never
  fills on bar ``i``; it fills on bar ``i+1``, and an order created on the final
  bar never fills at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from tests.conftest import make_ohlcv

from biasguard.engine import Backtester, DataHandler, run_backtest
from biasguard.engine.data_handler import MarketView
from biasguard.events import FillEvent, OrderEvent, SignalEvent
from biasguard.strategy import NoOpStrategy, Strategy, StrategyContext
from biasguard.types import Direction, OrderSide, OrderType

# --------------------------------------------------------------------------- #
# Test strategies and execution doubles (kept out of src on purpose)
# --------------------------------------------------------------------------- #


class UptickStrategy(Strategy):
    """Signal LONG whenever the last close is above the prior close.

    Depends only on backward-looking data, so it must be truncation-stable.
    """

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        view = ctx.view
        if len(view) < 2:
            return ()
        if view.closes[-1] > view.closes[-2]:
            return (ctx.long(),)
        return ()


class AlwaysLongStrategy(Strategy):
    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        return (ctx.long(),)


class UnitPortfolio:
    """A minimal Portfolio double: one market order per signal, 1 contract."""

    def __init__(self) -> None:
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def on_signal(self, signal: SignalEvent, view: MarketView) -> Sequence[OrderEvent]:
        if signal.direction is Direction.LONG:
            return (
                OrderEvent(signal.timestamp, signal.symbol, OrderSide.BUY, 1.0, OrderType.MARKET),
            )
        if signal.direction is Direction.SHORT:
            return (
                OrderEvent(signal.timestamp, signal.symbol, OrderSide.SELL, 1.0, OrderType.MARKET),
            )
        return ()

    def on_fill(self, fill: FillEvent) -> None:
        self._position += int(fill.signed_quantity)

    def mark_to_market(self, bar: object) -> None:
        return None


class RecordingBroker:
    """Fills each resting order at the next bar's open and records timing."""

    def __init__(self) -> None:
        self._open: list[OrderEvent] = []
        self.seen_bars: list[pd.Timestamp] = []

    def place(self, order: OrderEvent) -> None:
        self._open.append(order)

    def process_bar(self, bar: object) -> Sequence[FillEvent]:
        self.seen_bars.append(bar.timestamp)  # type: ignore[attr-defined]
        fills = [
            FillEvent(bar.timestamp, o.symbol, o.side, o.quantity, bar.open)  # type: ignore[attr-defined]
            for o in self._open
        ]
        self._open = []
        return fills


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestBasicRun:
    def test_noop_runs_end_to_end(self, clean_df: pd.DataFrame) -> None:
        result = Backtester(DataHandler(clean_df), NoOpStrategy(), symbol="NQ").run()
        assert result.n_bars == len(clean_df)
        assert result.signals == ()
        assert result.orders == ()
        assert result.fills == ()

    def test_signals_are_logged(self, clean_df: pd.DataFrame) -> None:
        result = Backtester(DataHandler(clean_df), UptickStrategy(), symbol="NQ").run()
        # clean_df is a monotonic uptrend, so every bar after the first is an uptick.
        assert len(result.signals) == len(clean_df) - 1
        assert all(s.direction is Direction.LONG for s in result.signals)


class TestTruncationDeterminism:
    def test_signals_identical_under_truncation(self) -> None:
        data = make_ohlcv(n=200, tz="America/Chicago")
        # Give the trend some structure so signals are non-trivial.
        data["close"] = data["close"] + (pd.Series(range(200), index=data.index) % 7 - 3) * 2.0
        data["high"] = data[["open", "close"]].max(axis=1) + 0.5
        data["low"] = data[["open", "close"]].min(axis=1) - 0.5

        def factory() -> Strategy:
            return UptickStrategy()

        full = run_backtest(data, factory, symbol="NQ")
        for t in (50, 100, 150):
            partial = run_backtest(data, factory, symbol="NQ", upto=t)
            cutoff = data.index[t - 1]
            expected = tuple(s for s in full.signals if s.timestamp <= cutoff)
            assert partial.signals == expected, f"decisions diverged at T={t}"


class TestNextBarExecution:
    def test_order_fills_on_following_bar_not_its_own(self, clean_df: pd.DataFrame) -> None:
        broker = RecordingBroker()
        result = Backtester(
            DataHandler(clean_df),
            AlwaysLongStrategy(),
            symbol="NQ",
            portfolio=UnitPortfolio(),
            broker=broker,
        ).run()

        n = len(clean_df)
        # A signal is created on every bar; the order from the final bar has no
        # following bar to fill against, so exactly n-1 fills occur.
        assert len(result.orders) == n
        assert len(result.fills) == n - 1

        ts = list(clean_df.index)
        # Each fill happens on the bar strictly after the order that caused it.
        for k, fill in enumerate(result.fills):
            order_bar_ts = ts[k]  # order k was created on bar k
            assert fill.timestamp == ts[k + 1]
            assert fill.timestamp > order_bar_ts
            # And the fill price is the following bar's open (market fill).
            assert fill.price == clean_df["open"].iloc[k + 1]

    def test_no_fill_happens_on_the_creating_bar(self, clean_df: pd.DataFrame) -> None:
        broker = RecordingBroker()
        Backtester(
            DataHandler(clean_df),
            AlwaysLongStrategy(),
            symbol="NQ",
            portfolio=UnitPortfolio(),
            broker=broker,
        ).run()
        # The broker's first process_bar (bar 0) saw no resting orders, so it
        # produced no fill stamped at bar 0.
        assert broker.seen_bars[0] == clean_df.index[0]


class TestNullPortfolio:
    def test_defaults_are_flat_and_inert(self, clean_df: pd.DataFrame) -> None:
        from biasguard.engine import NullPortfolio

        portfolio = NullPortfolio()
        view = DataHandler(clean_df).view(0)
        sig = SignalEvent(clean_df.index[0], "NQ", Direction.LONG)
        fill = FillEvent(clean_df.index[0], "NQ", OrderSide.BUY, 1.0, 100.0)
        assert portfolio.position == 0
        assert portfolio.on_signal(sig, view) == ()
        assert portfolio.on_fill(fill) is None


class TestRunBacktestEntrypoint:
    def test_upto_limits_bars(self, clean_df: pd.DataFrame) -> None:
        result = run_backtest(clean_df, NoOpStrategy, symbol="NQ", upto=10)
        assert result.n_bars == 10

    def test_upto_none_runs_all(self, clean_df: pd.DataFrame) -> None:
        result = run_backtest(clean_df, NoOpStrategy, symbol="NQ")
        assert result.n_bars == len(clean_df)
