"""Analytics: performance metrics and the deterministic replay fingerprint."""

from __future__ import annotations

from biasguard.analytics.fingerprint import (
    MANIFEST_VERSION,
    ReplayManifest,
    build_manifest,
    describe,
    hash_market_data,
)
from biasguard.analytics.metrics import (
    PerformanceMetrics,
    annual_volatility,
    cagr,
    compute_metrics,
    drawdown_series,
    expectancy,
    infer_periods_per_year,
    max_drawdown,
    max_drawdown_dollars,
    max_drawdown_duration,
    per_year_breakdown,
    periodic_returns,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    split_is_oos,
    win_rate,
)

__all__ = [
    "MANIFEST_VERSION",
    "PerformanceMetrics",
    "ReplayManifest",
    "annual_volatility",
    "build_manifest",
    "cagr",
    "compute_metrics",
    "describe",
    "drawdown_series",
    "expectancy",
    "hash_market_data",
    "infer_periods_per_year",
    "max_drawdown",
    "max_drawdown_dollars",
    "max_drawdown_duration",
    "per_year_breakdown",
    "periodic_returns",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "split_is_oos",
    "win_rate",
]
