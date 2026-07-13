"""Behavior + power-validation tests for the built-in integrity checks.

Power validation: each detector is shown catching a *planted* flaw. A green
causality suite means nothing unless it can go red on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tests.known_bad.strategies import AlwaysLong, Churn, LeakyPeek, UptickCausal

from biasguard.engine import DataHandler
from biasguard.events import SignalEvent
from biasguard.execution.costs import (
    NoSlippage,
    PerContractCommission,
    TickSlippage,
    ZeroCommission,
)
from biasguard.execution.instrument import NQ
from biasguard.execution.orders import Trade
from biasguard.strategy import Strategy, StrategyContext
from biasguard.types import Direction
from biasguard.validation import BacktestSpec, IntegrityContext, RunOutput, Status, assess_integrity
from biasguard.validation.checks import (
    CostIntegrityCheck,
    FillRealismCheck,
    LookaheadCheck,
    MonteCarloCheck,
    OutOfSampleCheck,
    SlippageSensitivityCheck,
)
from biasguard.validation.checks.fills import _RandomDirectionNull


def trend_data(n: int = 200, slope: float = 0.1, start: str = "2022-01-03 08:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz="America/Chicago")
    close = 15000.0 + slope * np.arange(n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.5,
            "low": np.minimum(open_, close) - 0.5,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def spec(
    strategy_cls: type[Strategy],
    *,
    data: pd.DataFrame | None = None,
    commission: object = PerContractCommission(1.90),
    slippage: object = TickSlippage(1.0),
) -> BacktestSpec:
    return BacktestSpec(
        data=data if data is not None else trend_data(),
        strategy_factory=strategy_cls,
        instrument=NQ,
        commission=commission,  # type: ignore[arg-type]
        slippage=slippage,  # type: ignore[arg-type]
    )


def run_check(check: object, s: BacktestSpec) -> object:
    ctx = IntegrityContext.build(s)
    return check.run(ctx)  # type: ignore[attr-defined]


class TestLookaheadPowerValidation:
    def test_catches_planted_leak(self) -> None:
        result = run_check(LookaheadCheck(), spec(LeakyPeek))
        assert result.status is Status.FAIL
        assert result.is_gate

    def test_passes_causal_strategy(self) -> None:
        result = run_check(LookaheadCheck(), spec(UptickCausal))
        assert result.status is Status.PASS

    def test_gate_failure_craters_score(self) -> None:
        from biasguard.validation import assess_integrity

        report = assess_integrity(spec(LeakyPeek))
        assert report.get("lookahead").status is Status.FAIL
        assert report.score <= 25.0  # gate cap


class TestCostPowerValidation:
    def test_zero_cost_fails(self) -> None:
        result = run_check(
            CostIntegrityCheck(),
            spec(UptickCausal, commission=ZeroCommission(), slippage=NoSlippage()),
        )
        assert result.status is Status.FAIL

    def test_real_costs_do_not_fail(self) -> None:
        result = run_check(CostIntegrityCheck(), spec(UptickCausal))
        assert result.status in (Status.PASS, Status.WARN, Status.SKIP)


class TestSlippagePowerValidation:
    def test_subcost_edge_dies(self) -> None:
        # Churn on a tiny drift: profitable at zero slippage, dead by ~half a tick.
        s = spec(Churn, data=trend_data(n=200, slope=0.1), commission=ZeroCommission())
        result = run_check(SlippageSensitivityCheck(), s)
        assert result.status is Status.FAIL
        assert result.metrics["breakeven_ticks"] <= 1.0


class TestFillRealism:
    def test_directional_edge_reports_alpha(self) -> None:
        # AlwaysLong on a strong uptrend is genuine directional exposure.
        result = run_check(FillRealismCheck(), spec(AlwaysLong, data=trend_data(n=120, slope=0.5)))
        assert result.status in (Status.PASS, Status.WARN, Status.SKIP)
        if result.status is not Status.SKIP:
            assert "alpha" in result.metrics
            assert "null_mean" in result.metrics

    def test_deterministic_given_seed(self) -> None:
        s = spec(AlwaysLong, data=trend_data(n=120, slope=0.5))
        a = run_check(FillRealismCheck(), s)
        b = run_check(FillRealismCheck(), s)
        assert a.metrics.get("null_mean") == b.metrics.get("null_mean")


class TestReviewRegressions:
    """Regressions for the Phase 5 adversarial review findings."""

    def test_lookahead_skips_non_monotonic_index(self) -> None:
        # A duplicate/unsorted index is a data defect, not lookahead: skip, don't FAIL.
        d = trend_data(n=20)
        idx = list(d.index)
        idx[10] = idx[9]  # duplicate timestamp
        d = d.set_axis(pd.DatetimeIndex(idx), axis=0)
        result = run_check(LookaheadCheck(), spec(UptickCausal, data=d))
        assert result.status is Status.SKIP

    def test_null_replays_all_signals_at_a_bar(self) -> None:
        # A bar that emitted [exit, entry] must be replayed as two signals, not one.
        d = trend_data(n=5)
        dh = DataHandler(d, validate=False)
        ts = d.index[2]
        sigs = (SignalEvent(ts, "NQ", Direction.FLAT), SignalEvent(ts, "NQ", Direction.LONG))
        null = _RandomDirectionNull(sigs, seed=1)
        ctx = StrategyContext(view=dh.view(2), symbol="NQ", position=1.0)
        out = null.on_bar(ctx)
        assert len(out) == 2
        assert out[0].direction is Direction.FLAT  # exit preserved verbatim
        assert out[1].direction in (Direction.LONG, Direction.SHORT)  # entry randomized

    def test_oos_skips_when_no_in_sample_edge(self) -> None:
        # Losing in-sample: negative/negative retention must not read as a PASS.
        down = trend_data(n=200, slope=-0.1)
        result = run_check(OutOfSampleCheck(), spec(Churn, data=down, commission=ZeroCommission()))
        assert result.status is Status.SKIP

    def test_slippage_skips_when_no_effect(self) -> None:
        # Maker-only fills => identical net at every level => cannot certify robust.
        check = SlippageSensitivityCheck()
        nets = dict.fromkeys(check.levels_ticks, 500.0)
        assert check._verdict(nets).status is Status.SKIP


def _ctx_with_trades(pnls: list[float], seed: int = 12345) -> IntegrityContext:
    ts = pd.Timestamp("2022-01-03 09:30", tz="UTC")
    trades = tuple(
        Trade(
            "NQ",
            Direction.LONG,
            1.0,
            100.0,
            100.0,
            ts + pd.Timedelta(minutes=i),
            ts + pd.Timedelta(minutes=i + 1),
            float(p),
            0.0,
        )
        for i, p in enumerate(pnls)
    )
    data = trend_data(n=5)
    baseline = RunOutput(
        data=data,
        signals=(),
        fills=(),
        trades=trades,
        equity=pd.Series([100_000.0], index=[ts]),
        net_pnl=float(sum(pnls)),
        initial_capital=100_000.0,
    )
    return IntegrityContext(spec=spec(UptickCausal, data=data), baseline=baseline, seed=seed)


class TestMonteCarloCheck:
    def test_skips_when_not_profitable(self) -> None:
        assert MonteCarloCheck().run(_ctx_with_trades([-10.0] * 20)).status is Status.SKIP

    def test_skips_too_few_trades(self) -> None:
        assert MonteCarloCheck().run(_ctx_with_trades([10.0] * 5)).status is Status.SKIP

    def test_robust_strategy_passes(self) -> None:
        result = MonteCarloCheck().run(_ctx_with_trades([5.0] * 30))
        assert result.status is Status.PASS
        assert result.metrics["prob_profit"] > 0.9

    def test_fragile_result_is_flagged(self) -> None:
        # Net +20 but only if BOTH big wins land — fragile under resampling.
        result = MonteCarloCheck().run(_ctx_with_trades([-10.0] * 18 + [100.0, 100.0]))
        assert result.status in (Status.WARN, Status.FAIL)
        assert result.metrics["prob_profit"] < 0.75

    def test_wired_into_assess_integrity(self) -> None:
        report = assess_integrity(
            spec(
                Churn,
                data=trend_data(n=200, slope=0.5),
                commission=ZeroCommission(),
                slippage=NoSlippage(),
            )
        )
        assert report.get("monte_carlo") is not None


class TestEndToEnd:
    def test_causal_strategy_scores_reasonably(self) -> None:
        from biasguard.validation import assess_integrity

        report = assess_integrity(spec(AlwaysLong, data=trend_data(n=150, slope=0.5)))
        # A causal, cost-bearing, directional strategy should not be graded "F".
        lookahead = report.get("lookahead")
        assert lookahead is not None and lookahead.status != Status.FAIL
        assert report.summary()  # renders
