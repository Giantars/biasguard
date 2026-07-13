"""Shared enums and type aliases used across biasguard.

Kept deliberately small: only symbols needed by more than one subpackage live
here. Domain-specific enums (order types, sides) are introduced in the phase
that first needs them.
"""

from __future__ import annotations

from enum import Enum


class Severity(Enum):
    """Severity of a data-quality or validation finding.

    Ordered from least to most serious so findings can be compared and sorted::

        Severity.INFO < Severity.WARNING < Severity.ERROR
    """

    INFO = 10
    WARNING = 20
    ERROR = 30

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value <= other.value

    def __str__(self) -> str:
        return self.name


class Direction(Enum):
    """The directional intent of a :class:`~biasguard.events.SignalEvent`."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    def __str__(self) -> str:
        return self.name


class OrderType(Enum):
    """Order execution type.

    ``MARKET`` and ``STOP`` fill *through* the level (reliable); ``LIMIT`` is
    queue-dependent (optimistic). This distinction drives the fill-realism
    checks in the validation module.
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

    def __str__(self) -> str:
        return self.name


class OrderSide(Enum):
    """Side of an order or fill. The value doubles as a position sign."""

    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL — convenient for signed position math."""
        return int(self.value)

    def __str__(self) -> str:
        return self.name


__all__ = ["Direction", "OrderSide", "OrderType", "Severity"]
