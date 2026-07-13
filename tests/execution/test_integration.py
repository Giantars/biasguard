"""End-to-end: strategy -> portfolio -> broker -> equity curve, through the engine.

Proves the execution core composes with the Phase 2 loop while preserving
next-bar causality and reconciling accounting to the cent.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from tests.conftest import make_ohlcv

from biasguard.engine import Backtester, DataHandler
from biasguard.events import SignalEvent
from biasguard.execution.broker import SimulatedBroker
from biasguard.execution.costs import FixedSlippage, PerContractCommission
from biasguard.execution.instrument import NQ
from biasguard.execution.portfolio import FixedSizer, Portfolio
from biasguard.strategy import Strategy, StrategyContext


class EntryExitStrategy(Strategy):
    def __init__(self, entry_bar: int, exit_bar: int) -> None:
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.index == self.entry_bar:
            return (ctx.long(),)
        if ctx.index == self.exit_bar:
            return (ctx.exit(),)
        return ()


def test_round_trip_through_engine() -> None:
    data = make_ohlcv(n=20)  # deterministic uptrend, open[i] == close[i-1]
    portfolio = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
    broker = SimulatedBroker(
        NQ, commission=PerContractCommission(1.90), slippage=FixedSlippage(0.0)
    )
    result = Backtester(
        DataHandler(data),
        EntryExitStrategy(entry_bar=2, exit_bar=10),
        portfolio=portfolio,
        broker=broker,
    ).run()

    opens = data["open"].to_numpy()

    # Entry signal on bar 2 fills at bar 3's open; exit signal on bar 10 at bar 11's open.
    assert len(portfolio.trades) == 1
    trade = portfolio.trades[0]
    assert trade.entry_price == pytest.approx(opens[3])
    assert trade.exit_price == pytest.approx(opens[11])
    assert trade.pnl == pytest.approx((opens[11] - opens[3]) * 20.0)
    assert trade.commission == pytest.approx(3.80)

    # Flat by the end: equity reconciles to initial + net trade P&L.
    assert portfolio.position == 0.0
    assert portfolio.equity == pytest.approx(100_000.0 + trade.net_pnl)

    # The equity curve has one point per bar, exposed via the result.
    assert result.portfolio is portfolio
    equity = portfolio.equity_series()
    assert len(equity) == len(data)
    assert equity.iloc[-1] == pytest.approx(100_000.0 + trade.net_pnl)


def test_no_fill_leaks_onto_signal_bar() -> None:
    # The entry order must not fill on the bar that produced the signal.
    data = make_ohlcv(n=8)
    portfolio = Portfolio(NQ, sizer=FixedSizer(1))
    broker = SimulatedBroker(NQ, commission=PerContractCommission(1.0), slippage=FixedSlippage(0.0))
    result = Backtester(
        DataHandler(data),
        EntryExitStrategy(entry_bar=2, exit_bar=99),
        portfolio=portfolio,
        broker=broker,
    ).run()
    (fill,) = result.fills
    assert fill.timestamp == data.index[3]  # bar 3, not bar 2
