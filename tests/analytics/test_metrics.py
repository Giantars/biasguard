"""Tests for performance metrics — hand-computed values and edge cases."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from biasguard.analytics.metrics import (
    cagr,
    compute_metrics,
    max_drawdown,
    max_drawdown_dollars,
    max_drawdown_duration,
    per_year_breakdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    split_is_oos,
    win_rate,
)
from biasguard.execution.orders import Trade
from biasguard.types import Direction


def equity(values: list[float], start: str = "2021-01-01", freq: str = "D") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def trade(net: float, year: int = 2021) -> Trade:
    ts = pd.Timestamp(f"{year}-06-01", tz="UTC")
    return Trade(
        "NQ", Direction.LONG, 1.0, 100.0, 100.0 + net, ts - pd.Timedelta("1h"), ts, net, 0.0
    )


class TestSharpe:
    def test_known_value(self) -> None:
        # returns [0.01,0.02,0.03] -> mean 0.02, std(ddof=1) 0.01 -> 2.0 * sqrt(252)
        assert sharpe_ratio([0.01, 0.02, 0.03], 252) == pytest.approx(2.0 * math.sqrt(252))

    def test_zero_variance_positive_is_inf(self) -> None:
        # A perfectly compounding curve has zero dispersion and positive mean:
        # its Sharpe is +inf, consistent with Sortino (not a misleading 0.0).
        assert sharpe_ratio([0.01, 0.01, 0.01], 252) == float("inf")

    def test_zero_variance_negative_is_neg_inf(self) -> None:
        assert sharpe_ratio([-0.01, -0.01, -0.01], 252) == float("-inf")

    def test_zero_variance_zero_mean_is_zero(self) -> None:
        assert sharpe_ratio([0.0, 0.0, 0.0], 252) == 0.0

    def test_too_few_is_nan(self) -> None:
        assert math.isnan(sharpe_ratio([0.01], 252))


class TestSortino:
    def test_positive_and_finite(self) -> None:
        s = sortino_ratio([0.02, -0.01, 0.03], 252)
        assert math.isfinite(s) and s > 0

    def test_no_downside_is_inf(self) -> None:
        assert sortino_ratio([0.01, 0.02], 252) == float("inf")


class TestDrawdown:
    def test_max_drawdown_fraction(self) -> None:
        assert max_drawdown(equity([100, 110, 90, 120])) == pytest.approx(-20 / 110)

    def test_max_drawdown_dollars(self) -> None:
        assert max_drawdown_dollars(equity([100, 110, 90, 120])) == pytest.approx(-20.0)

    def test_duration(self) -> None:
        assert max_drawdown_duration(equity([100, 110, 90, 120])) == pd.Timedelta("1D")

    def test_flat_curve_no_drawdown(self) -> None:
        assert max_drawdown(equity([100, 100, 100])) == 0.0


class TestCagr:
    def test_roughly_ten_percent(self) -> None:
        eq = pd.Series(
            [100.0, 110.0],
            index=pd.DatetimeIndex(["2021-01-01", "2022-01-01"]).tz_localize("UTC"),
        )
        assert cagr(eq, 100.0) == pytest.approx(0.10, abs=0.001)

    def test_degenerate_returns_nan(self) -> None:
        assert math.isnan(cagr(equity([100.0]), 100.0))


class TestTradeStats:
    def test_profit_factor(self) -> None:
        assert profit_factor([trade(100), trade(-50), trade(200), trade(-50)]) == pytest.approx(3.0)

    def test_profit_factor_no_losses_is_inf(self) -> None:
        assert profit_factor([trade(100), trade(50)]) == float("inf")

    def test_profit_factor_no_trades_is_nan(self) -> None:
        assert math.isnan(profit_factor([]))

    def test_win_rate(self) -> None:
        assert win_rate([trade(100), trade(-50), trade(200)]) == pytest.approx(2 / 3)


class TestComputeMetrics:
    def test_populates_and_is_consistent(self) -> None:
        eq = equity([100_000, 100_500, 100_200, 101_000])
        trades = [trade(500), trade(-300), trade(800)]
        m = compute_metrics(eq, trades, initial_capital=100_000.0, periods_per_year=252)
        assert m.n_trades == 3
        assert m.final_equity == pytest.approx(101_000)
        assert m.total_return == pytest.approx(0.01)
        assert m.total_pnl == pytest.approx(1000.0)
        assert m.max_drawdown < 0

    def test_empty_does_not_crash(self) -> None:
        m = compute_metrics(pd.Series(dtype=float), [], initial_capital=100_000.0)
        assert m.n_trades == 0
        assert math.isnan(m.win_rate)

    def test_risk_free_hurdle_applies_to_both_ratios(self) -> None:
        eq = equity([100_000, 100_500, 100_200, 101_000, 100_800, 101_500])
        base = compute_metrics(eq, [], initial_capital=100_000.0, periods_per_year=252)
        hurdled = compute_metrics(
            eq, [], initial_capital=100_000.0, periods_per_year=252, risk_free_rate=0.5
        )
        # A large risk-free hurdle must move BOTH ratios (Sortino no longer ignores it).
        assert base.sharpe != hurdled.sharpe
        assert base.sortino != hurdled.sortino


class TestBreakdowns:
    def test_per_year(self) -> None:
        trades = [trade(100, 2021), trade(-40, 2021), trade(250, 2022)]
        eq = equity([100_000, 100_100, 100_060], start="2021-06-01")
        frame = per_year_breakdown(eq, trades)
        assert frame.loc[2021, "net_pnl"] == pytest.approx(60.0)
        assert int(frame.loc[2021, "n_trades"]) == 2
        assert frame.loc[2022, "net_pnl"] == pytest.approx(250.0)

    def test_is_oos_split(self) -> None:
        eq = equity([100_000, 100_500, 100_200, 101_000, 101_500], start="2021-01-01")
        trades = [trade(500, 2021), trade(800, 2021)]
        cut = eq.index[2]
        m_is, m_oos = split_is_oos(eq, trades, cut, initial_capital=100_000.0, periods_per_year=252)
        assert m_is.end <= cut
        assert m_oos.start > cut


def test_sharpe_matches_numpy_reference() -> None:
    rng = np.array([0.005, -0.002, 0.01, 0.0, 0.003, -0.004])
    expected = rng.mean() / rng.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(rng, 252) == pytest.approx(expected)
