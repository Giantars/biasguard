"""The four event types that flow through the engine.

``MarketEvent -> SignalEvent -> OrderEvent -> FillEvent``

All events are **frozen** dataclasses. Immutability is not decoration: it means
an event compares by *value*, which is what makes the truncation/lookahead test
possible — two runs that make the same decisions produce byte-identical event
streams that ``==`` can verify.

Each event carries the timestamp of *when it was decided/created*, never a
future time. A strategy cannot stamp a signal with a timestamp it has not yet
reached, because the timestamp comes from the current bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from biasguard.data.schema import Bar
from biasguard.types import Direction, OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A completed bar has become available to the engine.

    Emitted by the :class:`~biasguard.engine.data_handler.DataHandler` for bar
    ``index``. The strategy may only see data up to and including this bar.
    """

    timestamp: pd.Timestamp
    index: int
    bar: Bar


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A strategy's directional intent, decided on a bar's close.

    ``timestamp`` is the decision time (the current bar's close). Sizing and
    order construction happen downstream in the portfolio, so a signal carries
    intent (``direction`` + ``strength``), not a concrete order.
    """

    timestamp: pd.Timestamp
    symbol: str
    direction: Direction
    strength: float = 1.0


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A concrete instruction to the broker, created from a signal.

    Created on bar ``i``'s close; it may not fill before bar ``i+1`` (the broker
    only matches it on the *next* ``process_bar``).
    """

    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: int | None = None
    group_id: int | None = None  # OCO group: when one member fills, siblings cancel

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for a LIMIT order")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop_price is required for a STOP order")


@dataclass(frozen=True, slots=True)
class FillEvent:
    """The broker's report that an order executed.

    ``timestamp`` is the fill time (the bar the order matched against, never
    earlier than the bar after the order was created). ``commission`` and
    ``slippage`` are recorded per fill — costs are first-class, not an
    afterthought.
    """

    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    order_id: int | None = None

    @property
    def signed_quantity(self) -> float:
        """Quantity signed by side: +qty for BUY, -qty for SELL."""
        return self.side.sign * self.quantity


__all__ = ["FillEvent", "MarketEvent", "OrderEvent", "SignalEvent"]
