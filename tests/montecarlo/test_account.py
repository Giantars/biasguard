"""Tests for the account/risk layer and path evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from biasguard.montecarlo.account import (
    AccountConfig,
    evaluate_path,
    evaluate_paths,
    path_equity,
)


class TestAccountConfig:
    def test_validates_positive_limits(self) -> None:
        with pytest.raises(ValueError):
            AccountConfig(trailing_drawdown_limit=-100)
        with pytest.raises(ValueError):
            AccountConfig(trades_per_day=0)

    def test_has_limits(self) -> None:
        assert not AccountConfig().has_limits
        assert AccountConfig(trailing_drawdown_limit=500).has_limits


class TestPathEquity:
    def test_includes_start(self) -> None:
        eq = path_equity(np.array([10.0, -5.0, 20.0]), 100.0)
        np.testing.assert_array_equal(eq, [100.0, 110.0, 105.0, 125.0])


class TestEvaluatePath:
    def test_no_breach_within_limits(self) -> None:
        pnl = np.array([10.0, 10.0, 10.0])
        out = evaluate_path(pnl, AccountConfig(starting_balance=1000, trailing_drawdown_limit=100))
        assert not out.breached
        assert out.final_pnl == pytest.approx(30.0)
        assert out.max_drawdown == 0.0

    def test_trailing_drawdown_breach(self) -> None:
        # +100 then -250 => drawdown 250 from the peak.
        pnl = np.array([100.0, -250.0])
        out = evaluate_path(pnl, AccountConfig(starting_balance=1000, trailing_drawdown_limit=200))
        assert out.breached and "trailing_dd" in out.breaches
        assert out.max_drawdown == pytest.approx(250.0)

    def test_max_loss_breach(self) -> None:
        pnl = np.array([-600.0])
        out = evaluate_path(pnl, AccountConfig(starting_balance=1000, max_loss_limit=500))
        assert "max_loss" in out.breaches

    def test_daily_loss_breach(self) -> None:
        # trades_per_day=2 => day1 = -30-30 = -60 breaches a 50 daily limit.
        pnl = np.array([-30.0, -30.0, 100.0, 100.0])
        out = evaluate_path(
            pnl,
            AccountConfig(starting_balance=1000, daily_loss_limit=50, trades_per_day=2),
        )
        assert "daily_loss" in out.breaches

    def test_daily_loss_is_intraday_not_end_of_day(self) -> None:
        # Dips to -2200 intraday, recovers to -200 net: must still breach a 2000
        # intraday limit (end-of-day net would have missed it).
        pnl = np.array([-1200.0, -1000.0, 2000.0])
        out = evaluate_path(pnl, AccountConfig(daily_loss_limit=2000, trades_per_day=3))
        assert "daily_loss" in out.breaches

    def test_daily_limit_without_trades_per_day_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_path(np.array([100.0, -2000.0, 100.0]), AccountConfig(daily_loss_limit=1000))

    def test_profit_target_reached(self) -> None:
        pnl = np.array([300.0])
        out = evaluate_path(pnl, AccountConfig(starting_balance=1000, profit_target=250))
        assert out.reached_target


class TestEvaluatePathsVectorized:
    def test_matches_scalar(self) -> None:
        account = AccountConfig(starting_balance=1000, trailing_drawdown_limit=150)
        paths = np.array([[100.0, -200.0, 10.0], [10.0, 10.0, 10.0]])
        equity = np.vstack([path_equity(p, 1000.0) for p in paths])
        vec = evaluate_paths(equity, paths, account)
        for i in range(2):
            scalar = evaluate_path(paths[i], account)
            assert vec["max_drawdown"][i] == pytest.approx(scalar.max_drawdown)
            assert vec["final_pnl"][i] == pytest.approx(scalar.final_pnl)
            assert bool(vec["breach_trailing_dd"][i]) == ("trailing_dd" in scalar.breaches)
