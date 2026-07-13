"""Tests for the SimulatedBroker: matching, OCO, same-bar policy, costs."""

from __future__ import annotations

import pandas as pd
import pytest

from biasguard.data.schema import Bar
from biasguard.events import OrderEvent
from biasguard.execution.broker import SameBarPolicy, SimulatedBroker
from biasguard.execution.costs import FixedSlippage, PerContractCommission
from biasguard.execution.fill_models import TouchFill
from biasguard.execution.instrument import NQ
from biasguard.execution.orders import OrderStatus
from biasguard.types import OrderSide, OrderType

TS = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")


def bar(o: float, h: float, low: float, c: float) -> Bar:
    return Bar(TS, o, h, low, c, 100.0)


def make_broker(**kw: object) -> SimulatedBroker:
    kw.setdefault("fill_model", TouchFill())
    kw.setdefault("commission", PerContractCommission(1.0))
    kw.setdefault("slippage", FixedSlippage(0.25))
    return SimulatedBroker(NQ, **kw)  # type: ignore[arg-type]


def mkt(side: OrderSide, qty: float = 1.0, group: int | None = None) -> OrderEvent:
    return OrderEvent(TS, "NQ", side, qty, OrderType.MARKET, group_id=group)


class TestCostsAndTakerMaker:
    def test_market_fill_pays_slippage_and_commission(self) -> None:
        broker = make_broker()
        wo = broker.place(mkt(OrderSide.BUY))
        (fill,) = broker.process_bar(bar(100, 101, 99, 100.5))
        assert fill.price == pytest.approx(100.25)  # 100 open + 0.25 buy slippage
        assert fill.slippage == pytest.approx(0.25)
        assert fill.commission == pytest.approx(1.0)
        assert fill.order_id == wo.id
        assert wo.status is OrderStatus.FILLED

    def test_limit_fill_is_maker_no_slippage(self) -> None:
        broker = make_broker()
        broker.place(OrderEvent(TS, "NQ", OrderSide.BUY, 1.0, OrderType.LIMIT, limit_price=100))
        (fill,) = broker.process_bar(bar(100.5, 101, 99.5, 100.5))
        assert fill.price == pytest.approx(100.0)  # min(limit, open); no slippage
        assert fill.slippage == 0.0
        assert fill.commission == pytest.approx(1.0)

    def test_stop_fill_is_taker(self) -> None:
        broker = make_broker()
        broker.place(OrderEvent(TS, "NQ", OrderSide.SELL, 1.0, OrderType.STOP, stop_price=97))
        (fill,) = broker.process_bar(bar(100, 101, 95, 96))
        assert fill.price == pytest.approx(96.75)  # min(stop, open)=97, minus 0.25 sell slippage


class TestRestingAndCancel:
    def test_untouched_limit_stays_open(self) -> None:
        broker = make_broker()
        wo = broker.place(OrderEvent(TS, "NQ", OrderSide.BUY, 1.0, OrderType.LIMIT, limit_price=96))
        assert broker.process_bar(bar(100, 101, 99, 100)) == []
        assert wo.is_active and wo in broker.open_orders

    def test_cancel_prevents_fill(self) -> None:
        broker = make_broker()
        wo = broker.place(mkt(OrderSide.BUY))
        broker.cancel(wo.id)
        assert broker.process_bar(bar(100, 101, 99, 100)) == []
        assert wo.status is OrderStatus.CANCELLED

    def test_ids_are_sequential(self) -> None:
        broker = make_broker()
        a = broker.place(mkt(OrderSide.BUY))
        b = broker.place(mkt(OrderSide.SELL))
        assert (a.id, b.id) == (1, 2)

    def test_get_order(self) -> None:
        broker = make_broker()
        wo = broker.place(mkt(OrderSide.BUY))
        assert broker.get_order(wo.id) is wo
        assert broker.get_order(999) is None

    def test_independent_orders_both_fill(self) -> None:
        broker = make_broker()
        broker.place(mkt(OrderSide.BUY))
        broker.place(mkt(OrderSide.BUY))
        assert len(broker.process_bar(bar(100, 101, 99, 100))) == 2


class TestOCO:
    def _bracket(self, broker: SimulatedBroker) -> tuple[object, object]:
        # A long position's protective exits: stop below, target above, same group.
        stop = broker.place(
            OrderEvent(TS, "NQ", OrderSide.SELL, 1.0, OrderType.STOP, stop_price=90, group_id=7)
        )
        target = broker.place(
            OrderEvent(TS, "NQ", OrderSide.SELL, 1.0, OrderType.LIMIT, limit_price=110, group_id=7)
        )
        return stop, target

    def test_one_fill_cancels_sibling(self) -> None:
        broker = make_broker(slippage=FixedSlippage(0.0))
        stop, target = self._bracket(broker)
        # Bar hits only the target (high>=110, low>90).
        fills = broker.process_bar(bar(100, 111, 95, 100))
        assert len(fills) == 1
        assert fills[0].order_id == target.id  # type: ignore[attr-defined]
        assert target.status is OrderStatus.FILLED  # type: ignore[attr-defined]
        assert stop.status is OrderStatus.CANCELLED  # type: ignore[attr-defined]

    def test_same_bar_stop_first_is_default(self) -> None:
        broker = make_broker(slippage=FixedSlippage(0.0))
        stop, target = self._bracket(broker)
        # Bar spans BOTH the stop and the target.
        fills = broker.process_bar(bar(100, 111, 89, 100))
        assert len(fills) == 1
        assert (
            fills[0].order_id == stop.id
        )  # pessimistic: the stop filled  # type: ignore[attr-defined]
        assert stop.status is OrderStatus.FILLED  # type: ignore[attr-defined]
        assert target.status is OrderStatus.CANCELLED  # type: ignore[attr-defined]

    def test_same_bar_limit_first_is_opt_in(self) -> None:
        broker = make_broker(slippage=FixedSlippage(0.0), same_bar_policy=SameBarPolicy.LIMIT_FIRST)
        stop, target = self._bracket(broker)
        fills = broker.process_bar(bar(100, 111, 89, 100))
        assert fills[0].order_id == target.id  # type: ignore[attr-defined]
        assert target.status is OrderStatus.FILLED  # type: ignore[attr-defined]
        assert stop.status is OrderStatus.CANCELLED  # type: ignore[attr-defined]

    def test_same_type_group_falls_back_to_lowest_id(self) -> None:
        # Two limits in one group, no stop: the policy's preferred type is absent,
        # so the lowest-id order fills and the sibling cancels (deterministic).
        broker = make_broker(slippage=FixedSlippage(0.0))
        first = broker.place(
            OrderEvent(TS, "NQ", OrderSide.SELL, 1.0, OrderType.LIMIT, limit_price=110, group_id=3)
        )
        second = broker.place(
            OrderEvent(TS, "NQ", OrderSide.SELL, 1.0, OrderType.LIMIT, limit_price=111, group_id=3)
        )
        fills = broker.process_bar(bar(100, 112, 99, 100))  # touches both
        assert len(fills) == 1
        assert fills[0].order_id == first.id
        assert second.status is OrderStatus.CANCELLED
