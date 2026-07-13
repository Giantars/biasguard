"""The event loop.

Correctness over speed: the loop is a plain, readable, deterministic pass over
bars with a fixed step order that makes the causal contract structural.

For each bar ``i`` the order is always:

1. **fill-first** — the broker matches orders resting from earlier bars against
   bar ``i`` (an order placed on bar ``i-1`` can fill now);
2. **decide** — the strategy sees a causal view of bars ``[0..i]`` and emits
   signals;
3. **size** — the portfolio turns signals into orders;
4. **rest** — orders are placed but not matched until step 1 of bar ``i+1``.

The :class:`Portfolio` and :class:`Broker` seams are Protocols so Phase 4/5 can
drop in real position tracking, costs, and fill models without touching the
loop. In this phase the default portfolio is a no-op, so a bare
``Backtester(handler, strategy)`` is a pure *decision* engine (signals only),
which is exactly what the truncation test needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from biasguard.data.schema import Bar
from biasguard.engine.data_handler import DataHandler, MarketView
from biasguard.events import FillEvent, OrderEvent, SignalEvent
from biasguard.strategy.base import Strategy, StrategyContext


@runtime_checkable
class Portfolio(Protocol):
    """Turns signals into orders and tracks the resulting position.

    The concrete implementation lives in :mod:`biasguard.execution`; the engine
    depends only on this minimal protocol so it never imports the execution
    layer (no cycle). ``mark_to_market`` is called once per bar so a portfolio
    can record its equity curve.
    """

    @property
    def position(self) -> float: ...

    def on_signal(self, signal: SignalEvent, view: MarketView) -> Sequence[OrderEvent]: ...

    def on_fill(self, fill: FillEvent) -> None: ...

    def mark_to_market(self, bar: Bar) -> None: ...


@runtime_checkable
class Broker(Protocol):
    """Holds resting orders and matches them against incoming bars.

    ``place`` may return a working-order handle (the engine ignores it), so the
    return type is left open for concrete brokers that expose one.
    """

    def place(self, order: OrderEvent) -> object: ...

    def process_bar(self, bar: Bar) -> Sequence[FillEvent]: ...


class NullPortfolio:
    """Default portfolio: never sizes an order, always flat.

    Makes the bare engine a pure decision recorder — the truncation/lookahead
    tests run against this so they measure *decisions*, not execution.
    """

    @property
    def position(self) -> float:
        return 0.0

    def on_signal(self, signal: SignalEvent, view: MarketView) -> Sequence[OrderEvent]:
        return ()

    def on_fill(self, fill: FillEvent) -> None:
        return None

    def mark_to_market(self, bar: Bar) -> None:
        return None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The record of a run.

    Phase 2 captures the event streams; later phases extend this with the equity
    curve, trade ledger, and analytics.
    """

    n_bars: int
    signals: tuple[SignalEvent, ...] = ()
    orders: tuple[OrderEvent, ...] = ()
    fills: tuple[FillEvent, ...] = ()
    portfolio: Portfolio | None = None


class Backtester:
    """Runs one strategy over one data handler, deterministically."""

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        *,
        symbol: str = "",
        portfolio: Portfolio | None = None,
        broker: Broker | None = None,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.symbol = symbol or data_handler.symbol
        self.portfolio: Portfolio = portfolio if portfolio is not None else NullPortfolio()
        self.broker = broker

    def run(self) -> BacktestResult:
        signals: list[SignalEvent] = []
        orders: list[OrderEvent] = []
        fills: list[FillEvent] = []

        self.strategy.on_start()
        for market_event in self.data_handler:
            i = market_event.index

            # (1) FILL-FIRST — resting orders match against THIS bar.
            if self.broker is not None:
                for fill in self.broker.process_bar(market_event.bar):
                    self.portfolio.on_fill(fill)
                    fills.append(fill)

            # (2) DECIDE — strategy sees only bars [0..i].
            view = self.data_handler.view(i)
            ctx = StrategyContext(view=view, symbol=self.symbol, position=self.portfolio.position)
            for signal in self.strategy.on_bar(ctx) or ():
                signals.append(signal)

                # (3) SIZE — signal -> order(s).
                for order in self.portfolio.on_signal(signal, view):
                    orders.append(order)

                    # (4) REST — order waits for the next bar to be matched.
                    if self.broker is not None:
                        self.broker.place(order)

            # (5) MARK — record equity at this bar's close (position is settled
            #     for the bar; new orders rest until the next one).
            self.portfolio.mark_to_market(market_event.bar)

        self.strategy.on_finish()
        return BacktestResult(
            n_bars=len(self.data_handler),
            signals=tuple(signals),
            orders=tuple(orders),
            fills=tuple(fills),
            portfolio=self.portfolio,
        )


def run_backtest(
    data: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    *,
    symbol: str = "",
    portfolio_factory: Callable[[], Portfolio] | None = None,
    broker_factory: Callable[[], Broker] | None = None,
    upto: int | None = None,
    validate: bool = True,
) -> BacktestResult:
    """Functional entrypoint — and the truncation harness.

    Takes **factories** (not instances) so each call builds fresh, un-warmed
    state. That is what makes the truncation test valid: ``run_backtest(data,
    F)`` and ``run_backtest(data, F, upto=T)`` share no leaked state, so any
    difference in their first ``T`` decisions is real future-dependence, not a
    stale accumulator.

    Parameters
    ----------
    data:
        Canonical OHLCV frame.
    strategy_factory:
        Zero-argument callable returning a fresh :class:`Strategy`.
    upto:
        If given, run only on ``data.iloc[:upto]``. The core of the lookahead
        test: decisions on the first ``upto`` bars must match the full run.
    """
    sliced = data.iloc[:upto] if upto is not None else data
    handler = DataHandler(sliced, symbol=symbol, validate=validate)
    portfolio = portfolio_factory() if portfolio_factory is not None else None
    broker = broker_factory() if broker_factory is not None else None
    engine = Backtester(
        handler,
        strategy_factory(),
        symbol=symbol,
        portfolio=portfolio,
        broker=broker,
    )
    return engine.run()


__all__ = [
    "BacktestResult",
    "Backtester",
    "Broker",
    "NullPortfolio",
    "Portfolio",
    "run_backtest",
]
