"""Tests for the Monte Carlo simulator: determinism, distributions, regimes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from biasguard.execution.orders import Trade
from biasguard.montecarlo import (
    AccountConfig,
    MonteCarloSimulator,
    infer_trades_per_day,
    recent_regime_mask,
)
from biasguard.types import Direction

TS = pd.Timestamp("2022-01-03 09:30", tz="UTC")


def trades_from(pnls: Sequence[float], *, per_day: int = 1) -> tuple[Trade, ...]:
    out = []
    for i, p in enumerate(pnls):
        day = i // per_day
        t = TS + pd.Timedelta(days=day, minutes=i)
        out.append(
            Trade(
                "NQ",
                Direction.LONG,
                1.0,
                100.0,
                100.0 + p,
                t,
                t + pd.Timedelta(minutes=1),
                float(p),
                0.0,
            )
        )
    return tuple(out)


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        trades = trades_from([10, -5, 20, -3, 8, -12, 15] * 3)
        a = MonteCarloSimulator(n_paths=500, seed=42).run(trades)
        b = MonteCarloSimulator(n_paths=500, seed=42).run(trades)
        np.testing.assert_array_equal(a.final_pnl, b.final_pnl)


class TestDistributions:
    def test_all_wins_certain_profit(self) -> None:
        result = MonteCarloSimulator(n_paths=300, seed=1).run(trades_from([10.0] * 12))
        assert result.prob_profit == 1.0

    def test_all_losses_never_profit(self) -> None:
        result = MonteCarloSimulator(n_paths=300, seed=1).run(trades_from([-10.0] * 12))
        assert result.prob_profit == 0.0

    def test_percentiles_ordered(self) -> None:
        trades = trades_from([10, -5, 20, -3, 8, -12, 15, -4, 9, -6] * 2)
        result = MonteCarloSimulator(n_paths=1000, seed=3).run(trades)
        fp = result.final_pnl_percentiles
        assert fp["p5"] <= fp["p50"] <= fp["p95"]
        dd = result.max_drawdown_percentiles
        assert dd["p50"] <= dd["p95"] <= dd["max"]
        assert result.worst_case_drawdown == pytest.approx(dd["p95"])

    def test_fragile_result_has_lower_prob_profit(self) -> None:
        robust = MonteCarloSimulator(n_paths=1000, seed=5).run(trades_from([5.0] * 20))
        fragile = MonteCarloSimulator(n_paths=1000, seed=5).run(
            trades_from([-10.0] * 18 + [100.0, 100.0])  # net +20, but needs both wins
        )
        assert fragile.prob_profit < robust.prob_profit
        assert fragile.prob_profit < 0.75


class TestRiskLimits:
    def test_tight_trailing_limit_breaches_often(self) -> None:
        trades = trades_from([10, -30, 15, -25, 12, -28, 9, -20, 11, -22] * 2)
        result = MonteCarloSimulator(n_paths=500, seed=2).run(
            trades, account=AccountConfig(starting_balance=1000, trailing_drawdown_limit=5.0)
        )
        assert result.prob_breach > 0.5
        assert "trailing_dd" in result.prob_breach_by_limit


class TestRegimeAndEdges:
    def test_regime_mask_restricts_pool(self) -> None:
        # Recent half is all losses -> regime-conditioned run should rarely profit.
        trades = trades_from([10.0] * 10 + [-10.0] * 10)
        mask = recent_regime_mask(len(trades), 0.5)
        result = MonteCarloSimulator(n_paths=300, seed=1).run(trades, regime_mask=mask)
        assert result.prob_profit < 0.05

    def test_too_few_trades_raises(self) -> None:
        with pytest.raises(ValueError):
            MonteCarloSimulator(n_paths=10).run(trades_from([10.0]))

    def test_empty_regime_mask_raises(self) -> None:
        trades = trades_from([10.0, -5.0, 8.0])
        with pytest.raises(ValueError):
            MonteCarloSimulator(n_paths=10).run(trades, regime_mask=np.zeros(3, dtype=bool))

    def test_infer_trades_per_day(self) -> None:
        assert infer_trades_per_day(trades_from([1.0] * 6, per_day=3)) == 3
