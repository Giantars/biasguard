"""Example 3 — the execution core: fill realism, costs, and an equity curve.

Run:  python examples/03_execution_core.py

Three things the execution engine is built to make honest:

1. Fill realism — the same resting limit fills under the optimistic model but
   NOT under the conservative one (a mere touch is not a queued fill).
2. Costs — commission + slippage are first-class and change the bottom line.
3. Accounting — a trade ledger and a mark-to-market equity curve that reconcile.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from biasguard.data.schema import Bar
from biasguard.engine import Backtester, DataHandler
from biasguard.events import OrderEvent, SignalEvent
from biasguard.execution import (
    NQ,
    FixedSizer,
    PerContractCommission,
    Portfolio,
    SimulatedBroker,
    TickSlippage,
    TouchFill,
    TradeThroughFill,
)
from biasguard.strategy import Strategy, StrategyContext
from biasguard.types import OrderSide, OrderType

TS = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")


def fill_realism_demo() -> None:
    print("=" * 70)
    print("1) FILL REALISM — a touch is not a fill")
    print("=" * 70)
    # A resting BUY limit at 15000. The bar dips exactly TO 15000 but no lower.
    bar = Bar(TS, open=15001, high=15002, low=15000.0, close=15001, volume=500)
    order = OrderEvent(TS, "NQ", OrderSide.BUY, 1.0, OrderType.LIMIT, limit_price=15000.0)

    for name, model in (
        ("TouchFill (optimistic)", TouchFill()),
        ("TradeThroughFill (default)", TradeThroughFill()),
    ):
        broker = SimulatedBroker(NQ, fill_model=model)
        broker.place(order)
        fills = broker.process_bar(bar)
        outcome = f"FILLED @ {fills[0].price}" if fills else "NO FILL (you were behind the queue)"
        print(f"  {name:28s} -> {outcome}")
    print()


class EntryExitStrategy(Strategy):
    def __init__(self, entry: int, exit_: int) -> None:
        self.entry, self.exit_ = entry, exit_

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.index == self.entry:
            return (ctx.long(),)
        if ctx.index == self.exit_:
            return (ctx.exit(),)
        return ()


def accounting_demo() -> None:
    print("=" * 70)
    print("2) COSTS & ACCOUNTING — a market round-trip on a rising tape")
    print("=" * 70)
    idx = pd.date_range("2024-01-02 08:30", periods=20, freq="1min", tz="America/Chicago")
    close = pd.Series(range(20), index=idx) * 1.0 + 15000.0
    data = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 500.0,
        }
    )
    data.index.name = "timestamp"

    for label, commission, slippage in (
        ("zero cost (flattering)", PerContractCommission(0.0), TickSlippage(0.0)),
        ("realistic NQ cost", PerContractCommission(1.90), TickSlippage(1.0)),
    ):
        portfolio = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
        broker = SimulatedBroker(NQ, commission=commission, slippage=slippage)
        Backtester(
            DataHandler(data), EntryExitStrategy(2, 12), portfolio=portfolio, broker=broker
        ).run()
        trade = portfolio.trades[0]
        print(
            f"  {label:24s} -> gross ${trade.pnl:7.2f} | costs ${trade.commission:5.2f} "
            f"| net ${trade.net_pnl:7.2f} | final equity ${portfolio.equity:,.2f}"
        )
    print()
    print(f"  Equity curve points: {len(portfolio.equity_series())} (one per bar)")


if __name__ == "__main__":
    fill_realism_demo()
    accounting_demo()
