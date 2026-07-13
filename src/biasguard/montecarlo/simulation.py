"""The Monte Carlo simulator: resample -> paths -> distributions.

Deterministic given a seed. Supports regime-conditioned resampling by passing a
boolean mask (or a helper mask) that restricts the *pool* of trades sampled from
while still generating full-length paths — i.e. "what if the whole track record
had come from this regime?".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from biasguard.execution.orders import Trade
from biasguard.montecarlo.account import AccountConfig, evaluate_paths
from biasguard.montecarlo.bootstrap import Bootstrap, StationaryBootstrap
from biasguard.montecarlo.result import MonteCarloResult


def trade_pnls(trades: Sequence[Trade]) -> np.ndarray:
    """Net P&L per trade as a float array."""
    return np.array([t.net_pnl for t in trades], dtype="float64")


def recent_regime_mask(n: int, fraction: float = 0.5) -> np.ndarray:
    """A boolean mask selecting the most recent ``fraction`` of ``n`` trades."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    mask = np.zeros(n, dtype=bool)
    start = int(n * (1.0 - fraction))
    mask[start:] = True
    return mask


def infer_trades_per_day(trades: Sequence[Trade]) -> int | None:
    """Average trades per calendar day from exit timestamps (for daily-loss chunking)."""
    if not trades:
        return None
    days = {t.exit_time.date() for t in trades}
    if not days:
        return None
    return max(1, round(len(trades) / len(days)))


class MonteCarloSimulator:
    """Runs a seeded block-bootstrap Monte Carlo over a trade ledger."""

    def __init__(
        self,
        *,
        bootstrap: Bootstrap | None = None,
        n_paths: int = 10_000,
        seed: int = 12345,
        sample_curves: int = 100,
    ) -> None:
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        self.bootstrap = bootstrap if bootstrap is not None else StationaryBootstrap()
        self.n_paths = n_paths
        self.seed = seed
        self.sample_curves = sample_curves

    def run(
        self,
        trades: Sequence[Trade],
        *,
        account: AccountConfig | None = None,
        regime_mask: np.ndarray | None = None,
    ) -> MonteCarloResult:
        account = account if account is not None else AccountConfig()
        pnl = trade_pnls(trades)
        n = pnl.size
        if n < 2:
            raise ValueError("Monte Carlo needs at least 2 trades")

        pool = pnl if regime_mask is None else pnl[np.asarray(regime_mask, dtype=bool)]
        if pool.size == 0:
            raise ValueError("regime_mask selects no trades")

        tpd = account.trades_per_day
        if account.daily_loss_limit is not None and tpd is None:
            tpd = infer_trades_per_day(trades)

        rng = np.random.default_rng(self.seed)
        paths = np.empty((self.n_paths, n), dtype="float64")
        for i in range(self.n_paths):
            paths[i] = self.bootstrap.resample(pool, rng, size=n)

        start = account.starting_balance
        equity = np.empty((self.n_paths, n + 1), dtype="float64")
        equity[:, 0] = start
        np.cumsum(paths, axis=1, out=equity[:, 1:])
        equity[:, 1:] += start

        ev = evaluate_paths(equity, paths, account, trades_per_day=tpd)
        final = ev["final_pnl"]
        max_dd = ev["max_drawdown"]
        any_breach = ev["any_breach"]

        by_limit = {
            key[len("breach_") :]: float(arr.mean())
            for key, arr in ev.items()
            if key.startswith("breach_")
        }
        bands_arr = np.percentile(equity, [5, 50, 95], axis=0)
        bands = {"p5": bands_arr[0], "p50": bands_arr[1], "p95": bands_arr[2]}
        samples = equity[: min(self.n_paths, self.sample_curves)].copy()

        return MonteCarloResult(
            n_paths=self.n_paths,
            n_trades=n,
            bootstrap_name=self.bootstrap.name,
            account=account,
            final_pnl=final,
            max_drawdown=max_dd,
            prob_profit=float((final > 0).mean()),
            prob_breach=float(any_breach.mean()),
            prob_reach_target=float(ev["reached_target"].mean()),
            prob_breach_by_limit=by_limit,
            equity_bands=bands,
            equity_samples=samples,
        )


__all__ = [
    "MonteCarloSimulator",
    "infer_trades_per_day",
    "recent_regime_mask",
    "trade_pnls",
]
