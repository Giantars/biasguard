"""Fill models — how a resting order matches against a bar.

This is the single most under-modelled area in most backtesters and the one the
brief flags hardest. The fill model is a first-class, swappable object so the
validator can re-run a strategy under ``TouchFill`` vs ``TradeThroughFill`` and
report how much of the P&L is real alpha vs. fill mechanics.

Semantics (all fills happen against the *next* bar — causality lives in the
engine loop, not here):

* **MARKET** — fills at ``bar.open`` (liquidity-taking → slippage applies).
* **STOP** — triggers when the bar reaches the stop, fills at the stop or worse
  on a gap (liquidity-taking → slippage applies). Reliable once triggered.
* **LIMIT** — the model-dependent part:
  - :class:`TouchFill` fills if the bar merely *touches* the limit (optimistic);
  - :class:`TradeThroughFill` fills only if the bar trades *through* the limit
    by at least ``min_ticks_through`` ticks (conservative; the default),
    modelling the fact that a touch does not guarantee a queued passive fill.

  Limit fills are liquidity-*providing*: ``is_taker=False`` so the broker does
  not add slippage on top (the queue risk is already priced in by the model).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from biasguard.data.schema import Bar
from biasguard.events import OrderEvent
from biasguard.types import OrderSide, OrderType

_NAN = float("nan")


@dataclass(frozen=True, slots=True)
class FillRequest:
    """A resting order presented to a fill model against one bar."""

    order: OrderEvent
    bar: Bar
    remaining_qty: float
    tick_size: float = 0.0


@dataclass(frozen=True, slots=True)
class FillDecision:
    """A fill model's verdict for one order against one bar."""

    filled: bool
    price: float = _NAN
    quantity: float = 0.0
    is_taker: bool = True
    reason: str = ""

    @classmethod
    def no_fill(cls) -> FillDecision:
        return cls(filled=False)


class FillModel(ABC):
    """Base class: shared MARKET/STOP logic, abstract LIMIT logic.

    Subclasses implement :meth:`_limit_fill` only.
    """

    def fill(self, req: FillRequest) -> FillDecision:
        order_type = req.order.order_type
        if order_type is OrderType.MARKET:
            return self._market_fill(req)
        if order_type is OrderType.STOP:
            return self._stop_fill(req)
        return self._limit_fill(req)

    def _market_fill(self, req: FillRequest) -> FillDecision:
        return FillDecision(
            filled=True,
            price=req.bar.open,
            quantity=req.remaining_qty,
            is_taker=True,
            reason="market",
        )

    def _stop_fill(self, req: FillRequest) -> FillDecision:
        order, bar = req.order, req.bar
        stop = order.stop_price
        assert stop is not None  # guaranteed by OrderEvent validation
        if order.side is OrderSide.BUY:
            if bar.high >= stop:
                price = max(stop, bar.open)  # gap up through the stop fills worse
                return FillDecision(True, price, req.remaining_qty, True, "stop")
        else:  # SELL stop
            if bar.low <= stop:
                price = min(stop, bar.open)
                return FillDecision(True, price, req.remaining_qty, True, "stop")
        return FillDecision.no_fill()

    @abstractmethod
    def _limit_fill(self, req: FillRequest) -> FillDecision:
        raise NotImplementedError


class TouchFill(FillModel):
    """Optimistic: a limit fills if the bar touches its price."""

    def _limit_fill(self, req: FillRequest) -> FillDecision:
        order, bar = req.order, req.bar
        limit = order.limit_price
        assert limit is not None
        if order.side is OrderSide.BUY:
            if bar.low <= limit:
                return FillDecision(
                    True, min(limit, bar.open), req.remaining_qty, False, "limit_touch"
                )
        else:  # SELL limit
            if bar.high >= limit:
                return FillDecision(
                    True, max(limit, bar.open), req.remaining_qty, False, "limit_touch"
                )
        return FillDecision.no_fill()


class TradeThroughFill(FillModel):
    """Conservative (default): a limit fills only if price trades *through* it.

    Requires the bar to exceed the limit by ``min_ticks_through`` ticks (using
    the instrument tick size supplied on the :class:`FillRequest`). A pure touch
    does not fill — you may have been behind the queue.
    """

    def __init__(self, min_ticks_through: float = 1.0) -> None:
        if min_ticks_through <= 0:
            raise ValueError("min_ticks_through must be positive")
        self.min_ticks_through = min_ticks_through

    def _limit_fill(self, req: FillRequest) -> FillDecision:
        order, bar = req.order, req.bar
        limit = order.limit_price
        assert limit is not None
        through = req.tick_size * self.min_ticks_through
        if order.side is OrderSide.BUY:
            if bar.low <= limit - through:
                return FillDecision(
                    True, min(limit, bar.open), req.remaining_qty, False, "limit_through"
                )
        else:  # SELL limit
            if bar.high >= limit + through:
                return FillDecision(
                    True, max(limit, bar.open), req.remaining_qty, False, "limit_through"
                )
        return FillDecision.no_fill()


__all__ = [
    "FillDecision",
    "FillModel",
    "FillRequest",
    "TouchFill",
    "TradeThroughFill",
]
