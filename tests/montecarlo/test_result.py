"""Tests for MonteCarloResult formatting and the percentiles helper."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from biasguard.execution.orders import Trade
from biasguard.montecarlo import AccountConfig, MonteCarloSimulator, percentiles
from biasguard.types import Direction

TS = pd.Timestamp("2022-01-03 09:30", tz="UTC")


def _trades(pnls: list[float]) -> tuple[Trade, ...]:
    return tuple(
        Trade("NQ", Direction.LONG, 1.0, 100.0, 100.0, TS, TS, float(p), 0.0) for p in pnls
    )


def test_percentiles_empty_is_nan() -> None:
    result = percentiles(np.empty(0))
    assert all(math.isnan(v) for v in result.values())


def test_percentiles_values() -> None:
    result = percentiles(np.arange(101, dtype="float64"))
    assert result["p50"] == 50.0
    assert result["min"] == 0.0 and result["max"] == 100.0


def test_summary_and_dict_and_html() -> None:
    trades = _trades([10, -5, 20, -3, 8, -12, 15, -4] * 2)
    result = MonteCarloSimulator(n_paths=200, seed=1).run(
        trades, account=AccountConfig(starting_balance=1000, trailing_drawdown_limit=100)
    )
    text = result.summary()
    assert "P(profit)" in text and "P(breach any limit)" in text
    d = result.to_dict()
    assert 0.0 <= d["prob_profit"] <= 1.0
    assert "final_pnl" in d and "max_drawdown" in d
    assert "Monte Carlo" in result.to_html()


def test_equity_bands_shape() -> None:
    trades = _trades([10.0, -5.0, 8.0, -3.0, 6.0] * 3)
    result = MonteCarloSimulator(n_paths=100, seed=1, sample_curves=20).run(trades)
    n_steps = result.n_trades + 1
    for band in ("p5", "p50", "p95"):
        assert result.equity_bands[band].shape == (n_steps,)
    assert result.equity_samples.shape == (20, n_steps)
    # Bands must be ordered p5 <= p50 <= p95 at every step.
    assert np.all(result.equity_bands["p5"] <= result.equity_bands["p95"] + 1e-9)
