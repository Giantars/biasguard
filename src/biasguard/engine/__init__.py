"""The event-driven engine: causal data feed and the backtest loop."""

from __future__ import annotations

from biasguard.engine.backtester import (
    Backtester,
    BacktestResult,
    Broker,
    NullPortfolio,
    Portfolio,
    run_backtest,
)

# data_handler must import first: backtester depends on it.
from biasguard.engine.data_handler import DataHandler, MarketView

__all__ = [
    "BacktestResult",
    "Backtester",
    "Broker",
    "DataHandler",
    "MarketView",
    "NullPortfolio",
    "Portfolio",
    "run_backtest",
]
