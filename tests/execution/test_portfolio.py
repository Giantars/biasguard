"""Tests for the Portfolio: sizing, accounting, trade ledger, reconciliation."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from biasguard.engine import DataHandler
from biasguard.events import FillEvent, SignalEvent
from biasguard.execution.instrument import NQ
from biasguard.execution.portfolio import FixedSizer, Portfolio
from biasguard.types import Direction, OrderSide, OrderType

TS = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")


def a_view() -> object:
    return DataHandler(make_ohlcv(n=10)).view(5)


def buy(qty: float, price: float, commission: float = 0.0) -> FillEvent:
    return FillEvent(TS, "NQ", OrderSide.BUY, qty, price, commission=commission)


def sell(qty: float, price: float, commission: float = 0.0) -> FillEvent:
    return FillEvent(TS, "NQ", OrderSide.SELL, qty, price, commission=commission)


class TestSizing:
    def test_long_from_flat(self) -> None:
        pf = Portfolio(NQ, sizer=FixedSizer(1))
        (order,) = pf.on_signal(SignalEvent(TS, "NQ", Direction.LONG), a_view())  # type: ignore[arg-type]
        assert order.side is OrderSide.BUY and order.quantity == 1.0
        assert order.order_type is OrderType.MARKET

    def test_no_order_when_already_at_target(self) -> None:
        pf = Portfolio(NQ, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 100.0))
        assert pf.on_signal(SignalEvent(TS, "NQ", Direction.LONG), a_view()) == ()  # type: ignore[arg-type]

    def test_flat_closes_position(self) -> None:
        pf = Portfolio(NQ, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 100.0))
        (order,) = pf.on_signal(SignalEvent(TS, "NQ", Direction.FLAT), a_view())  # type: ignore[arg-type]
        assert order.side is OrderSide.SELL and order.quantity == 1.0

    def test_reversal_sizes_delta(self) -> None:
        pf = Portfolio(NQ, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 100.0))  # long 1
        (order,) = pf.on_signal(SignalEvent(TS, "NQ", Direction.SHORT), a_view())  # type: ignore[arg-type]
        assert order.side is OrderSide.SELL and order.quantity == 2.0  # -1 target from +1


class TestAccountingReconciles:
    def test_simple_round_trip(self) -> None:
        pf = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 15000.0, commission=1.90))
        pf.on_fill(sell(1.0, 15020.0, commission=1.90))

        assert len(pf.trades) == 1
        trade = pf.trades[0]
        assert trade.pnl == pytest.approx(400.0)  # (15020-15000) * 1 * 20
        assert trade.commission == pytest.approx(3.80)  # entry + exit allocated
        assert trade.net_pnl == pytest.approx(396.20)
        assert pf.position == 0.0
        assert pf.equity == pytest.approx(100_000.0 + 396.20)

    def test_scale_in_partial_close_allocation(self) -> None:
        pf = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(2))
        pf.on_fill(buy(2.0, 100.0, commission=4.0))  # long 2, open_comm 4
        pf.on_fill(sell(1.0, 110.0, commission=2.0))  # close 1
        pf.on_fill(sell(1.0, 120.0, commission=2.0))  # close 1, now flat

        assert [pytest.approx(t.pnl) for t in pf.trades] == [200.0, 400.0]
        # Every dollar of commission is allocated across the two trades.
        assert sum(t.commission for t in pf.trades) == pytest.approx(pf.total_commission)
        assert pf.total_commission == pytest.approx(8.0)
        # Flat: equity == initial + sum of net trade P&L.
        assert pf.equity == pytest.approx(100_000.0 + sum(t.net_pnl for t in pf.trades))

    def test_unrealized_reflected_in_equity_while_open(self) -> None:
        pf = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 15000.0, commission=2.0))
        pf.mark_to_market(_bar(15010.0))
        assert pf.equity == pytest.approx(100_000.0 - 2.0 + 200.0)  # unrealized (15010-15000)*20


class TestReversalAccounting:
    def test_reversal_records_trade_and_flips_position(self) -> None:
        pf = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 100.0, commission=2.0))  # long 1
        pf.on_fill(sell(3.0, 110.0, commission=6.0))  # close 1, open short 2

        assert pf.position == -2.0
        assert pf.avg_price == 110.0  # leftover short opened at fill price
        assert pf.realized_pnl == pytest.approx(200.0)  # only the closed 1 realizes
        trade = pf.trades[0]
        assert trade.direction is Direction.LONG and trade.quantity == 1.0
        assert trade.commission == pytest.approx(4.0)  # entry 2 + exit-share 2

    def test_reversal_round_trip_reconciles(self) -> None:
        pf = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
        pf.on_fill(buy(1.0, 100.0, commission=2.0))
        pf.on_fill(sell(3.0, 110.0, commission=6.0))  # -> short 2
        pf.on_fill(buy(2.0, 105.0, commission=4.0))  # -> flat

        assert pf.position == 0.0
        assert sum(t.commission for t in pf.trades) == pytest.approx(pf.total_commission)
        assert pf.equity == pytest.approx(100_000.0 + sum(t.net_pnl for t in pf.trades))


class TestSizerGuards:
    def test_fixed_sizer_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            FixedSizer(0)


class TestEquityCurve:
    def test_equity_series_records_each_mark(self) -> None:
        pf = Portfolio(NQ, initial_capital=100_000.0)
        for px in (15000.0, 15005.0, 15010.0):
            pf.mark_to_market(_bar(px))
        series = pf.equity_series()
        assert len(series) == 3
        assert series.iloc[0] == pytest.approx(100_000.0)  # flat -> equity unchanged


def _bar(close: float) -> object:
    from biasguard.data.schema import Bar

    return Bar(TS, close, close, close, close, 100.0)
