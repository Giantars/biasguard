"""Order lifecycle, position accounting, and the trade record.

* :class:`Order` is the broker's *mutable working copy* of an immutable
  :class:`~biasguard.events.OrderEvent` — it tracks status and fills over time.
* :class:`Position` is signed-quantity + average-price accounting with correct
  realized P&L on open / add / reduce / close / reverse.
* :class:`Trade` is a closed round-trip record for analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from biasguard.events import OrderEvent
from biasguard.types import Direction, OrderSide, OrderType

_QTY_EPS = 1e-9  # tolerance for treating a residual quantity as zero


class OrderStatus(Enum):
    """Lifecycle state of a working order."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.name


@dataclass(slots=True)
class Order:
    """A broker-side working order wrapping an immutable :class:`OrderEvent`."""

    event: OrderEvent
    id: int
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0

    @property
    def symbol(self) -> str:
        return self.event.symbol

    @property
    def side(self) -> OrderSide:
        return self.event.side

    @property
    def order_type(self) -> OrderType:
        return self.event.order_type

    @property
    def group_id(self) -> int | None:
        return self.event.group_id

    @property
    def remaining(self) -> float:
        return self.event.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)

    def record_fill(self, quantity: float) -> None:
        """Apply a (partial) fill and advance the status."""
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if quantity > self.remaining + _QTY_EPS:
            raise ValueError(
                f"fill quantity {quantity} exceeds remaining {self.remaining} for order {self.id}"
            )
        self.filled_quantity += quantity
        if self.remaining <= _QTY_EPS:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        if self.is_active:
            self.status = OrderStatus.CANCELLED


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Outcome of applying a signed fill to a :class:`Position`."""

    realized_pnl: float
    closed_quantity: float
    entry_price: float  # the average price of the closed portion (0.0 if none)


@dataclass(slots=True)
class Position:
    """Signed-quantity position with average-price P&L accounting."""

    quantity: float = 0.0
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < _QTY_EPS

    @property
    def direction(self) -> Direction:
        if self.quantity > _QTY_EPS:
            return Direction.LONG
        if self.quantity < -_QTY_EPS:
            return Direction.SHORT
        return Direction.FLAT

    def unrealized_pnl(self, price: float, multiplier: float) -> float:
        """Mark-to-market P&L at ``price`` (signed quantity handles direction)."""
        if self.is_flat:
            return 0.0
        return (price - self.avg_price) * self.quantity * multiplier

    def apply(self, signed_qty: float, price: float, multiplier: float) -> ApplyResult:
        """Apply a signed fill (+buy / -sell); return realized P&L and closed size."""
        old = self.quantity
        pre_avg = self.avg_price

        opening = abs(old) < _QTY_EPS or (old > 0) == (signed_qty > 0)
        if opening:
            new_qty = old + signed_qty
            if abs(old) < _QTY_EPS:
                self.avg_price = price
            else:
                self.avg_price = (abs(old) * pre_avg + abs(signed_qty) * price) / (
                    abs(old) + abs(signed_qty)
                )
            self.quantity = new_qty
            return ApplyResult(realized_pnl=0.0, closed_quantity=0.0, entry_price=pre_avg)

        # Opposite sign: reduce / close / reverse.
        closed = min(abs(signed_qty), abs(old))
        dir_old = 1.0 if old > 0 else -1.0
        realized = (price - pre_avg) * closed * multiplier * dir_old
        new_qty = old + signed_qty
        self.quantity = new_qty
        if abs(new_qty) < _QTY_EPS:
            self.quantity = 0.0
            self.avg_price = 0.0
        elif (new_qty > 0) == (old > 0):
            self.avg_price = pre_avg  # partial close, same side
        else:
            self.avg_price = price  # reversed: leftover opens fresh at fill price
        return ApplyResult(realized_pnl=realized, closed_quantity=closed, entry_price=pre_avg)


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed (round-trip) trade record for analytics."""

    symbol: str
    direction: Direction  # side of the position that was closed
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float  # gross realized $ (price difference x quantity x multiplier)
    commission: float  # allocated entry + exit commission

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission

    @property
    def duration(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


__all__ = [
    "ApplyResult",
    "Order",
    "OrderStatus",
    "Position",
    "Trade",
]
