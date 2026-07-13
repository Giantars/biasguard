"""The simulated broker: order book, matching, OCO, and cost application.

Deterministic by construction: orders are assigned sequential ids and always
processed in id order, so a run is reproducible to the cent. The broker matches
resting orders against each incoming bar via the pluggable
:class:`~biasguard.execution.fill_models.FillModel`, resolves same-bar OCO
conflicts with a pessimistic-by-default policy, then applies slippage (takers
only) and commission.

Causality note: the engine loop calls :meth:`process_bar` *before* the strategy
decides, so an order placed on bar ``i`` is only ever matched from bar ``i+1``.
The broker itself never looks beyond the bar it is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from biasguard.data.schema import Bar
from biasguard.events import FillEvent, OrderEvent
from biasguard.execution.costs import (
    CommissionModel,
    NoSlippage,
    SlippageModel,
    ZeroCommission,
)
from biasguard.execution.fill_models import (
    FillDecision,
    FillModel,
    FillRequest,
    TradeThroughFill,
)
from biasguard.execution.instrument import Instrument
from biasguard.execution.orders import Order, OrderStatus, OrderType


class SameBarPolicy(Enum):
    """How to resolve an OCO group when several members are fillable on one bar.

    ``STOP_FIRST`` (default) is pessimistic: assume the protective stop filled
    before the target. Optimistic resolution must be chosen explicitly.
    """

    STOP_FIRST = "stop_first"
    LIMIT_FIRST = "limit_first"

    def __str__(self) -> str:
        return self.name


class SimulatedBroker:
    """A bar-driven order-matching broker with pluggable fills and costs."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        fill_model: FillModel | None = None,
        commission: CommissionModel | None = None,
        slippage: SlippageModel | None = None,
        same_bar_policy: SameBarPolicy = SameBarPolicy.STOP_FIRST,
    ) -> None:
        self.instrument = instrument
        # Conservative default fill model; costs default to zero but the
        # validation module errors on a zero-cost backtest.
        self.fill_model = fill_model if fill_model is not None else TradeThroughFill()
        self.commission = commission if commission is not None else ZeroCommission()
        self.slippage = slippage if slippage is not None else NoSlippage()
        self.same_bar_policy = same_bar_policy
        self._orders: dict[int, Order] = {}
        self._next_id = 1

    # -- order management --------------------------------------------------- #
    def place(self, order: OrderEvent) -> Order:
        """Register a resting order and return its working :class:`Order`."""
        oid = self._next_id
        self._next_id += 1
        working = Order(event=order, id=oid)
        self._orders[oid] = working
        return working

    def cancel(self, order_id: int) -> None:
        order = self._orders.get(order_id)
        if order is not None:
            order.cancel()

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_active]

    def get_order(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    # -- matching ----------------------------------------------------------- #
    def process_bar(self, bar: Bar) -> Sequence[FillEvent]:
        """Match resting orders against ``bar`` and return the resulting fills."""
        candidates: list[tuple[Order, FillDecision]] = []
        for oid in sorted(self._orders):
            order = self._orders[oid]
            if not order.is_active:
                continue
            decision = self.fill_model.fill(
                FillRequest(order.event, bar, order.remaining, self.instrument.tick_size)
            )
            if decision.filled:
                candidates.append((order, decision))

        fills: list[FillEvent] = []
        for order, decision in self._resolve_same_bar(candidates):
            if not order.is_active:  # a sibling in this bar may have cancelled it
                continue
            fills.append(self._execute(order, decision, bar))
        return fills

    def _execute(self, order: Order, decision: FillDecision, bar: Bar) -> FillEvent:
        reference = decision.price
        exec_price = reference
        slippage = 0.0
        if decision.is_taker:
            exec_price = self.slippage.apply(reference, order.side, self.instrument)
            slippage = abs(exec_price - reference)
        quantity = min(decision.quantity, order.remaining)
        commission = self.commission.calculate(quantity, exec_price, self.instrument)
        order.record_fill(quantity)
        if order.status is OrderStatus.FILLED:
            self._cancel_group_siblings(order)
        return FillEvent(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=exec_price,
            commission=commission,
            slippage=slippage,
            order_id=order.id,
        )

    def _resolve_same_bar(
        self, candidates: list[tuple[Order, FillDecision]]
    ) -> list[tuple[Order, FillDecision]]:
        """Keep at most one fillable order per OCO group, per the same-bar policy."""
        by_group: dict[int, list[tuple[Order, FillDecision]]] = {}
        result: list[tuple[Order, FillDecision]] = []
        for order, decision in candidates:
            gid = order.group_id
            if gid is None:
                result.append((order, decision))
            else:
                by_group.setdefault(gid, []).append((order, decision))

        for group in by_group.values():
            if len(group) == 1:
                result.append(group[0])
            else:
                result.append(self._pick_same_bar(group))
        result.sort(key=lambda pair: pair[0].id)
        return result

    def _pick_same_bar(self, group: list[tuple[Order, FillDecision]]) -> tuple[Order, FillDecision]:
        prefer = (
            OrderType.STOP if self.same_bar_policy is SameBarPolicy.STOP_FIRST else OrderType.LIMIT
        )
        for order, decision in sorted(group, key=lambda pair: pair[0].id):
            if order.order_type is prefer:
                return order, decision
        return min(group, key=lambda pair: pair[0].id)

    def _cancel_group_siblings(self, filled: Order) -> None:
        gid = filled.group_id
        if gid is None:
            return
        for order in self._orders.values():
            if order.id != filled.id and order.group_id == gid and order.is_active:
                order.cancel()


__all__ = ["SameBarPolicy", "SimulatedBroker"]
