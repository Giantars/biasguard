"""Tests for Position accounting, Order lifecycle, and Trade records."""

from __future__ import annotations

import pandas as pd
import pytest

from biasguard.events import OrderEvent
from biasguard.execution.orders import Order, OrderStatus, Position, Trade
from biasguard.types import Direction, OrderSide, OrderType

MULT = 20.0  # NQ
TS = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")


class TestPositionOpenAndAdd:
    def test_open_long(self) -> None:
        p = Position()
        r = p.apply(2.0, 100.0, MULT)
        assert p.quantity == 2.0 and p.avg_price == 100.0
        assert r.realized_pnl == 0.0 and r.closed_quantity == 0.0

    def test_add_to_long_averages_price(self) -> None:
        p = Position(quantity=2.0, avg_price=100.0)
        p.apply(1.0, 106.0, MULT)
        assert p.quantity == 3.0
        assert p.avg_price == pytest.approx(102.0)  # (2*100 + 1*106) / 3


class TestPositionReduceCloseReverse:
    def test_reduce_long_realizes_pnl(self) -> None:
        p = Position(quantity=2.0, avg_price=100.0)
        r = p.apply(-1.0, 110.0, MULT)
        assert r.realized_pnl == pytest.approx(200.0)  # (110-100) * 1 * 20
        assert r.closed_quantity == 1.0
        assert p.quantity == 1.0 and p.avg_price == 100.0  # avg unchanged on partial close

    def test_close_long_flat(self) -> None:
        p = Position(quantity=2.0, avg_price=100.0)
        r = p.apply(-2.0, 110.0, MULT)
        assert r.realized_pnl == pytest.approx(400.0)
        assert p.is_flat and p.avg_price == 0.0

    def test_reverse_long_to_short(self) -> None:
        p = Position(quantity=2.0, avg_price=100.0)
        r = p.apply(-3.0, 110.0, MULT)
        assert r.closed_quantity == 2.0
        assert r.realized_pnl == pytest.approx(400.0)  # only the closed 2 realize
        assert p.quantity == -1.0
        assert p.avg_price == 110.0  # leftover opens fresh at the fill price

    def test_cover_short_realizes_pnl(self) -> None:
        p = Position(quantity=-2.0, avg_price=100.0)
        r = p.apply(1.0, 95.0, MULT)
        assert r.realized_pnl == pytest.approx(100.0)  # short profit: (100-95)*1*20
        assert p.quantity == -1.0


class TestUnrealized:
    def test_long_unrealized(self) -> None:
        p = Position(quantity=2.0, avg_price=100.0)
        assert p.unrealized_pnl(105.0, MULT) == pytest.approx(200.0)

    def test_short_unrealized(self) -> None:
        p = Position(quantity=-2.0, avg_price=100.0)
        assert p.unrealized_pnl(95.0, MULT) == pytest.approx(200.0)

    def test_flat_unrealized_is_zero(self) -> None:
        assert Position().unrealized_pnl(100.0, MULT) == 0.0

    def test_direction(self) -> None:
        assert Position(1.0, 1.0).direction is Direction.LONG
        assert Position(-1.0, 1.0).direction is Direction.SHORT
        assert Position().direction is Direction.FLAT


class TestOrderLifecycle:
    def _order(self, qty: float = 3.0) -> Order:
        ev = OrderEvent(TS, "NQ", OrderSide.BUY, qty, OrderType.MARKET)
        return Order(event=ev, id=1)

    def test_partial_then_full_fill(self) -> None:
        o = self._order(3.0)
        o.record_fill(1.0)
        assert o.status is OrderStatus.PARTIALLY_FILLED and o.remaining == 2.0
        o.record_fill(2.0)
        assert o.status is OrderStatus.FILLED and o.remaining == 0.0
        assert not o.is_active

    def test_overfill_raises(self) -> None:
        o = self._order(1.0)
        with pytest.raises(ValueError):
            o.record_fill(2.0)

    def test_cancel(self) -> None:
        o = self._order()
        o.cancel()
        assert o.status is OrderStatus.CANCELLED and not o.is_active

    def test_cancel_after_fill_is_noop(self) -> None:
        o = self._order(1.0)
        o.record_fill(1.0)
        o.cancel()
        assert o.status is OrderStatus.FILLED  # already terminal


class TestTrade:
    def test_net_pnl_and_win(self) -> None:
        t = Trade(
            "NQ", Direction.LONG, 1.0, 100.0, 110.0, TS, TS + pd.Timedelta("5min"), 200.0, 3.8
        )
        assert t.net_pnl == pytest.approx(196.2)
        assert t.is_win
        assert t.duration == pd.Timedelta("5min")

    def test_loss(self) -> None:
        t = Trade("NQ", Direction.SHORT, 1.0, 100.0, 105.0, TS, TS, -100.0, 3.8)
        assert not t.is_win
