"""Monte Carlo: block-bootstrap uncertainty analysis over a trade ledger.

from biasguard.montecarlo import MonteCarloSimulator, AccountConfig

sim = MonteCarloSimulator(n_paths=10_000, seed=1)
result = sim.run(portfolio.trades,
                 account=AccountConfig(starting_balance=100_000, trailing_drawdown_limit=3_000))
print(result.summary())
"""

from __future__ import annotations

from biasguard.montecarlo.account import (
    AccountConfig,
    PathOutcome,
    evaluate_path,
    evaluate_paths,
    path_equity,
)
from biasguard.montecarlo.bootstrap import (
    Bootstrap,
    CircularBlockBootstrap,
    IIDBootstrap,
    StationaryBootstrap,
)
from biasguard.montecarlo.result import MonteCarloResult, percentiles
from biasguard.montecarlo.simulation import (
    MonteCarloSimulator,
    infer_trades_per_day,
    recent_regime_mask,
    trade_pnls,
)

__all__ = [
    "AccountConfig",
    "Bootstrap",
    "CircularBlockBootstrap",
    "IIDBootstrap",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "PathOutcome",
    "StationaryBootstrap",
    "evaluate_path",
    "evaluate_paths",
    "infer_trades_per_day",
    "path_equity",
    "percentiles",
    "recent_regime_mask",
    "trade_pnls",
]
