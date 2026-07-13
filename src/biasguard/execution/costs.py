"""Commission and slippage models — mandatory, first-class, swappable.

Costs are not an afterthought in biasguard: every fill carries its commission
and slippage, and the Phase 5 validator *errors* if a backtest runs with both
set to zero. These interfaces are the extension points; ship your own by
subclassing the ABCs.

Slippage is **adverse by construction**: a BUY executes at a *higher* price, a
SELL at a *lower* one. It applies only to liquidity-*taking* fills (market,
stop); a resting limit that fills at its price is not slipped again (that would
double-count — the queue risk is already modelled by the fill model).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from biasguard.execution.instrument import Instrument
from biasguard.types import OrderSide


class CommissionModel(ABC):
    """Maps a fill (quantity, price, instrument) to a dollar commission (>= 0)."""

    @abstractmethod
    def calculate(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Return the commission in account currency for this fill."""
        raise NotImplementedError


class ZeroCommission(CommissionModel):
    """No commission. Convenient for isolating fill geometry in tests; a real
    backtest that uses this will be flagged by the validation module."""

    def calculate(self, quantity: float, price: float, instrument: Instrument) -> float:
        return 0.0


class PerContractCommission(CommissionModel):
    """A flat charge per contract, per side — the futures norm."""

    def __init__(self, per_contract: float) -> None:
        if per_contract < 0:
            raise ValueError("per_contract commission must be non-negative")
        self.per_contract = per_contract

    def calculate(self, quantity: float, price: float, instrument: Instrument) -> float:
        return abs(quantity) * self.per_contract


class PercentCommission(CommissionModel):
    """A percentage of traded notional (``price * multiplier * quantity``)."""

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("pct commission must be non-negative")
        self.pct = pct

    def calculate(self, quantity: float, price: float, instrument: Instrument) -> float:
        notional = abs(quantity) * price * instrument.multiplier
        return notional * self.pct


class SlippageModel(ABC):
    """Maps a reference fill price + side to an adverse executed price."""

    @abstractmethod
    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        """Return the executed price after adverse slippage."""
        raise NotImplementedError


class NoSlippage(SlippageModel):
    """Fill at the reference price. The maximally optimistic case."""

    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        return price


class FixedSlippage(SlippageModel):
    """A fixed price amount, applied against the trade's direction."""

    def __init__(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("slippage amount must be non-negative")
        self.amount = amount

    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        return price + self.amount * side.sign


class TickSlippage(SlippageModel):
    """Slippage measured in ticks of the instrument."""

    def __init__(self, ticks: float) -> None:
        if ticks < 0:
            raise ValueError("slippage ticks must be non-negative")
        self.ticks = ticks

    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        return price + self.ticks * instrument.tick_size * side.sign


class PercentSlippage(SlippageModel):
    """Slippage as a fraction of price (e.g. 0.0001 = 1 bp)."""

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("slippage pct must be non-negative")
        self.pct = pct

    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        return price + price * self.pct * side.sign


__all__ = [
    "CommissionModel",
    "FixedSlippage",
    "NoSlippage",
    "PerContractCommission",
    "PercentCommission",
    "PercentSlippage",
    "SlippageModel",
    "TickSlippage",
    "ZeroCommission",
]
