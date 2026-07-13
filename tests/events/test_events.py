"""Tests for the event value objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from biasguard.data.schema import Bar
from biasguard.events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from biasguard.types import Direction, OrderSide, OrderType


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")


class TestMarketEvent:
    def test_construction(self) -> None:
        bar = Bar(_ts(), 100.0, 101.0, 99.0, 100.5, 10.0)
        ev = MarketEvent(timestamp=_ts(), index=0, bar=bar)
        assert ev.index == 0
        assert ev.bar.close == 100.5

    def test_is_frozen(self) -> None:
        bar = Bar(_ts(), 100.0, 101.0, 99.0, 100.5)
        ev = MarketEvent(timestamp=_ts(), index=0, bar=bar)
        with pytest.raises(FrozenInstanceError):
            ev.index = 1  # type: ignore[misc]


class TestSignalEvent:
    def test_value_equality(self) -> None:
        # Frozen dataclasses compare by value — the property truncation relies on.
        a = SignalEvent(_ts(), "NQ", Direction.LONG, 1.0)
        b = SignalEvent(_ts(), "NQ", Direction.LONG, 1.0)
        assert a == b

    def test_direction_matters_for_equality(self) -> None:
        a = SignalEvent(_ts(), "NQ", Direction.LONG)
        b = SignalEvent(_ts(), "NQ", Direction.SHORT)
        assert a != b


class TestOrderEvent:
    def test_market_order(self) -> None:
        o = OrderEvent(_ts(), "NQ", OrderSide.BUY, 1.0, OrderType.MARKET)
        assert o.order_type is OrderType.MARKET

    def test_limit_requires_price(self) -> None:
        with pytest.raises(ValueError, match="limit_price"):
            OrderEvent(_ts(), "NQ", OrderSide.BUY, 1.0, OrderType.LIMIT)

    def test_stop_requires_price(self) -> None:
        with pytest.raises(ValueError, match="stop_price"):
            OrderEvent(_ts(), "NQ", OrderSide.SELL, 1.0, OrderType.STOP)

    def test_quantity_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            OrderEvent(_ts(), "NQ", OrderSide.BUY, 0.0, OrderType.MARKET)


class TestFillEvent:
    def test_construction_and_signed_qty(self) -> None:
        f = FillEvent(_ts(), "NQ", OrderSide.SELL, 2.0, 15000.0, commission=3.8)
        assert f.price == 15000.0
        assert f.signed_quantity == -2.0  # SELL is negative

    def test_buy_signed_qty(self) -> None:
        f = FillEvent(_ts(), "NQ", OrderSide.BUY, 3.0, 15000.0)
        assert f.signed_quantity == 3.0
