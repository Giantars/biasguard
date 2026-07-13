"""Tests for fill-model geometry — the realism knob."""

from __future__ import annotations

import pandas as pd
import pytest

from biasguard.data.schema import Bar
from biasguard.events import OrderEvent
from biasguard.execution.fill_models import FillRequest, TouchFill, TradeThroughFill
from biasguard.types import OrderSide, OrderType

TS = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")
TICK = 0.25  # NQ tick


def bar(o: float, h: float, low: float, c: float) -> Bar:
    return Bar(TS, o, h, low, c, 100.0)


def order(
    side: OrderSide,
    otype: OrderType,
    *,
    limit: float | None = None,
    stop: float | None = None,
    qty: float = 1.0,
) -> OrderEvent:
    return OrderEvent(TS, "NQ", side, qty, otype, limit_price=limit, stop_price=stop)


def req(o: OrderEvent, b: Bar) -> FillRequest:
    return FillRequest(o, b, o.quantity, TICK)


class TestMarket:
    def test_market_fills_at_open_as_taker(self) -> None:
        d = TouchFill().fill(req(order(OrderSide.BUY, OrderType.MARKET), bar(100, 101, 99, 100.5)))
        assert d.filled and d.price == 100.0 and d.is_taker


class TestStop:
    def test_buy_stop_fills_at_stop(self) -> None:
        d = TouchFill().fill(
            req(order(OrderSide.BUY, OrderType.STOP, stop=102), bar(100, 105, 99, 104))
        )
        assert d.filled and d.price == 102.0 and d.is_taker

    def test_buy_stop_gap_fills_worse_at_open(self) -> None:
        d = TouchFill().fill(
            req(order(OrderSide.BUY, OrderType.STOP, stop=102), bar(103, 105, 102.5, 104))
        )
        assert d.price == 103.0  # gapped above the stop -> fill at the open

    def test_buy_stop_not_triggered(self) -> None:
        d = TouchFill().fill(
            req(order(OrderSide.BUY, OrderType.STOP, stop=106), bar(100, 105, 99, 104))
        )
        assert not d.filled

    def test_sell_stop_fills_at_stop(self) -> None:
        d = TradeThroughFill().fill(
            req(order(OrderSide.SELL, OrderType.STOP, stop=97), bar(100, 101, 95, 96))
        )
        assert d.filled and d.price == 97.0


class TestLimitTouchVsThrough:
    def test_buy_limit_touch_fills(self) -> None:
        d = TouchFill().fill(
            req(order(OrderSide.BUY, OrderType.LIMIT, limit=100), bar(100.5, 101, 100.0, 100.5))
        )
        assert d.filled and d.price == 100.0 and not d.is_taker

    def test_buy_limit_touch_does_not_fill_through_model(self) -> None:
        # low == limit is a touch, not a trade-through.
        d = TradeThroughFill().fill(
            req(order(OrderSide.BUY, OrderType.LIMIT, limit=100), bar(100.5, 101, 100.0, 100.5))
        )
        assert not d.filled

    def test_buy_limit_through_fills(self) -> None:
        d = TradeThroughFill().fill(
            req(order(OrderSide.BUY, OrderType.LIMIT, limit=100), bar(100.5, 100.6, 99.75, 100.0))
        )
        assert d.filled and d.price == 100.0  # min(limit, open)

    def test_buy_limit_gap_down_gives_price_improvement(self) -> None:
        d = TouchFill().fill(
            req(order(OrderSide.BUY, OrderType.LIMIT, limit=100), bar(98, 99, 97, 98.5))
        )
        assert d.price == 98.0  # opened below the limit -> filled at the better open

    def test_sell_limit_touch_vs_through(self) -> None:
        b = bar(99.5, 100.0, 99.0, 99.5)  # high == limit (touch only)
        assert TouchFill().fill(req(order(OrderSide.SELL, OrderType.LIMIT, limit=100), b)).filled
        assert (
            not TradeThroughFill()
            .fill(req(order(OrderSide.SELL, OrderType.LIMIT, limit=100), b))
            .filled
        )

    def test_sell_limit_gap_up_price_improvement(self) -> None:
        d = TouchFill().fill(
            req(order(OrderSide.SELL, OrderType.LIMIT, limit=100), bar(102, 103, 101, 102.5))
        )
        assert d.price == 102.0  # max(limit, open)


def test_min_ticks_through_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TradeThroughFill(min_ticks_through=0.0)
