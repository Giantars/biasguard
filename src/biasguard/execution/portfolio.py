"""The portfolio: sizing, position tracking, equity, and the trade ledger.

Implements the engine's ``Portfolio`` protocol structurally (it does not import
the engine, avoiding an import cycle — ``MarketView`` is a typing-only import).

Accounting is P&L-based, which is the natural model for futures:

    equity = initial_capital + realized_pnl + unrealized_pnl - total_commission

Slippage is already inside each fill price (so it reduces P&L exactly once);
commission is tracked separately and **allocated per trade** across the entry
and exit legs, so per-trade net P&L is honest and the ledger reconciles with the
equity curve to the cent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from biasguard.data.schema import Bar
from biasguard.events import FillEvent, OrderEvent, SignalEvent
from biasguard.execution.instrument import Instrument
from biasguard.execution.orders import Position, Trade
from biasguard.types import Direction, OrderSide, OrderType

if TYPE_CHECKING:  # typing only — avoids an execution -> engine import cycle
    from biasguard.engine.data_handler import MarketView

_QTY_EPS = 1e-9


class Sizer(ABC):
    """Decides the target position *magnitude* (contracts) for a signal."""

    @abstractmethod
    def size(self, signal: SignalEvent, view: MarketView, portfolio: Portfolio) -> float:
        raise NotImplementedError


class FixedSizer(Sizer):
    """Always target a fixed number of contracts."""

    def __init__(self, contracts: float = 1.0) -> None:
        if contracts <= 0:
            raise ValueError("contracts must be positive")
        self.contracts = contracts

    def size(self, signal: SignalEvent, view: MarketView, portfolio: Portfolio) -> float:
        return self.contracts


class Portfolio:
    """Turns signals into market orders and tracks the resulting P&L."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        initial_capital: float = 100_000.0,
        sizer: Sizer | None = None,
    ) -> None:
        self.instrument = instrument
        self.initial_capital = initial_capital
        self.sizer: Sizer = sizer if sizer is not None else FixedSizer(1.0)

        self._position = Position()
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self._open_commission = 0.0  # commission attributable to the open position
        self._opened_at: pd.Timestamp | None = None
        self._trades: list[Trade] = []
        self._last_close = float("nan")
        self._equity_times: list[pd.Timestamp] = []
        self._equity_values: list[float] = []

    # -- read-only state ---------------------------------------------------- #
    @property
    def position(self) -> float:
        return self._position.quantity

    @property
    def avg_price(self) -> float:
        return self._position.avg_price

    @property
    def trades(self) -> tuple[Trade, ...]:
        return tuple(self._trades)

    def unrealized_pnl(self, price: float | None = None) -> float:
        px = self._last_close if price is None else price
        if pd.isna(px):
            return 0.0
        return self._position.unrealized_pnl(px, self.instrument.multiplier)

    @property
    def equity(self) -> float:
        return (
            self.initial_capital + self.realized_pnl - self.total_commission + self.unrealized_pnl()
        )

    def equity_series(self) -> pd.Series:
        """The recorded mark-to-market equity curve indexed by timestamp."""
        return pd.Series(
            self._equity_values,
            index=pd.DatetimeIndex(self._equity_times, name="timestamp"),
            name="equity",
            dtype="float64",
        )

    # -- engine hooks ------------------------------------------------------- #
    def on_signal(self, signal: SignalEvent, view: MarketView) -> Sequence[OrderEvent]:
        """Translate a directional signal into a market order toward the target."""
        if signal.direction is Direction.FLAT:
            target = 0.0
        else:
            magnitude = self.sizer.size(signal, view, self)
            target = magnitude if signal.direction is Direction.LONG else -magnitude

        delta = target - self._position.quantity
        if abs(delta) < _QTY_EPS:
            return ()
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        return (
            OrderEvent(
                timestamp=signal.timestamp,
                symbol=self.instrument.symbol,
                side=side,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
            ),
        )

    def on_fill(self, fill: FillEvent) -> None:
        """Update position, commission, realized P&L, and the trade ledger."""
        entry_time = self._opened_at
        old_qty = self._position.quantity
        was_opening = abs(old_qty) < _QTY_EPS or (old_qty > 0) == (fill.signed_quantity > 0)

        result = self._position.apply(fill.signed_quantity, fill.price, self.instrument.multiplier)
        self.realized_pnl += result.realized_pnl
        self.total_commission += fill.commission

        if was_opening:
            self._open_commission += fill.commission
            if abs(old_qty) < _QTY_EPS:
                self._opened_at = fill.timestamp
            return

        # A close (and possibly a reversal): allocate commission to the trade.
        closed = result.closed_quantity
        entry_comm = self._open_commission * (closed / abs(old_qty))
        self._open_commission -= entry_comm

        reversed_ = abs(fill.signed_quantity) > abs(old_qty) + _QTY_EPS
        if reversed_:
            exit_comm = fill.commission * (abs(old_qty) / abs(fill.signed_quantity))
            self._open_commission += fill.commission - exit_comm
            self._opened_at = fill.timestamp
        else:
            exit_comm = fill.commission
            if self._position.is_flat:
                self._opened_at = None

        self._trades.append(
            Trade(
                symbol=fill.symbol,
                direction=Direction.LONG if old_qty > 0 else Direction.SHORT,
                quantity=closed,
                entry_price=result.entry_price,
                exit_price=fill.price,
                entry_time=entry_time if entry_time is not None else fill.timestamp,
                exit_time=fill.timestamp,
                pnl=result.realized_pnl,
                commission=entry_comm + exit_comm,
            )
        )

    def mark_to_market(self, bar: Bar) -> None:
        """Record the equity point at this bar's close."""
        self._last_close = bar.close
        self._equity_times.append(bar.timestamp)
        self._equity_values.append(self.equity)


__all__ = ["FixedSizer", "Portfolio", "Sizer"]
