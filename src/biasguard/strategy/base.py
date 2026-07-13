"""The strategy interface.

A strategy is a **pure function of its causal context and its own accumulated
state**. It receives a :class:`StrategyContext` (a read-only view of the past
plus its current position) and returns zero or more
:class:`~biasguard.events.SignalEvent` objects. It is given no handle to the
future, to the broker, or to mutable engine state — which is what makes the
"never expose the future" guarantee hold and the truncation test meaningful.

``on_tick`` is intentionally absent for now; the interface is designed so it can
be added later without breaking ``on_bar`` users.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from biasguard.events import SignalEvent
from biasguard.types import Direction

if TYPE_CHECKING:  # import only for typing to avoid an engine <-> strategy import cycle
    from biasguard.data.schema import Bar
    from biasguard.engine.data_handler import MarketView


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy is allowed to know on a given bar.

    Deliberately minimal: a causal :class:`MarketView`, the instrument symbol,
    and the current signed position. The signal factories stamp the decision
    time from the view, so a strategy cannot emit a signal dated in the future.
    """

    view: MarketView
    symbol: str
    position: float = 0.0

    @property
    def bar(self) -> Bar:
        """The current (most recent completed) bar."""
        return self.view.current

    @property
    def index(self) -> int:
        return self.view.index

    @property
    def timestamp(self) -> pd.Timestamp:
        return self.view.timestamp

    def signal(self, direction: Direction, strength: float = 1.0) -> SignalEvent:
        """Build a signal stamped with the current bar's decision time."""
        return SignalEvent(
            timestamp=self.timestamp,
            symbol=self.symbol,
            direction=direction,
            strength=strength,
        )

    def long(self, strength: float = 1.0) -> SignalEvent:
        return self.signal(Direction.LONG, strength)

    def short(self, strength: float = 1.0) -> SignalEvent:
        return self.signal(Direction.SHORT, strength)

    def exit(self) -> SignalEvent:
        """Signal intent to flatten the position."""
        return self.signal(Direction.FLAT, 1.0)


class Strategy(ABC):
    """Base class for user strategies.

    Subclass and implement :meth:`on_bar`. Keep all state on the instance;
    never read data other than through the provided context, or the truncation
    test (and live trading) will diverge from the backtest.
    """

    def params(self) -> dict[str, object]:
        """Return this strategy's configuration for the replay fingerprint.

        The default introspects public, JSON-serializable instance attributes,
        following the convention that *configuration is public and mutable state
        is underscore-prefixed*. Override for precise control. Compute the replay
        manifest **before** running so no post-run state can leak in.
        """
        out: dict[str, object] = {}
        for key, value in vars(self).items():
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
        return dict(sorted(out.items()))

    def on_start(self) -> None:  # noqa: B027  (intentional optional no-op hook)
        """Optional hook called once before the first bar."""

    @abstractmethod
    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        """Handle one completed bar; return zero or more signals."""
        raise NotImplementedError

    def on_finish(self) -> None:  # noqa: B027  (intentional optional no-op hook)
        """Optional hook called once after the last bar."""


class NoOpStrategy(Strategy):
    """A strategy that never trades — the trivial baseline and a smoke test."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        return ()


__all__ = ["NoOpStrategy", "Strategy", "StrategyContext"]
