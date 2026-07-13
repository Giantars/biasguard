"""Tests for the replay fingerprint — determinism and sensitivity."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from tests.conftest import make_ohlcv

from biasguard.analytics.fingerprint import build_manifest, describe, hash_market_data
from biasguard.events import SignalEvent
from biasguard.execution.broker import SimulatedBroker
from biasguard.execution.costs import FixedSlippage, PerContractCommission
from biasguard.execution.instrument import MNQ, NQ
from biasguard.execution.portfolio import FixedSizer, Portfolio
from biasguard.strategy import Strategy, StrategyContext


class ParamStrategy(Strategy):
    def __init__(self, fast: int = 10, slow: int = 20) -> None:
        self.fast = fast
        self.slow = slow
        self._state = 0  # underscore: excluded from params()

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        return ()


def a_broker(commission: float = 1.90, instrument: object = NQ) -> SimulatedBroker:
    return SimulatedBroker(
        instrument,  # type: ignore[arg-type]
        commission=PerContractCommission(commission),
        slippage=FixedSlippage(0.25),
    )


def a_portfolio(instrument: object = NQ) -> Portfolio:
    return Portfolio(instrument, sizer=FixedSizer(1))  # type: ignore[arg-type]


def a_manifest(
    *,
    data: pd.DataFrame | None = None,
    strategy: Strategy | None = None,
    commission: float = 1.90,
    instrument: object = NQ,
) -> object:
    data = data if data is not None else make_ohlcv(n=30)
    strategy = strategy if strategy is not None else ParamStrategy()
    return build_manifest(
        data, strategy, broker=a_broker(commission, instrument), portfolio=a_portfolio(instrument)
    )


class TestDeterminism:
    def test_same_inputs_same_fingerprint(self) -> None:
        assert a_manifest().fingerprint == a_manifest().fingerprint  # type: ignore[attr-defined]

    def test_params_key_order_irrelevant(self) -> None:
        data = make_ohlcv(n=20)
        strat = ParamStrategy(10, 20)
        m1 = build_manifest(data, strat, strategy_params={"fast": 10, "slow": 20})
        m2 = build_manifest(data, strat, strategy_params={"slow": 20, "fast": 10})
        assert m1.fingerprint == m2.fingerprint

    def test_numerically_equal_configs_match(self) -> None:
        # FixedSizer(1) and FixedSizer(1.0) produce identical backtests, so the
        # fingerprint must not distinguish int 1 from float 1.0.
        m_int = build_manifest(
            make_ohlcv(n=20), ParamStrategy(), portfolio=Portfolio(NQ, sizer=FixedSizer(1))
        )
        m_float = build_manifest(
            make_ohlcv(n=20), ParamStrategy(), portfolio=Portfolio(NQ, sizer=FixedSizer(1.0))
        )
        assert m_int.fingerprint == m_float.fingerprint

    def test_underscore_state_excluded(self) -> None:
        s1 = ParamStrategy(10, 20)
        s2 = ParamStrategy(10, 20)
        s2._state = 999  # mutable state must not affect the fingerprint
        assert a_manifest(strategy=s1).fingerprint == a_manifest(strategy=s2).fingerprint  # type: ignore[attr-defined]


class TestSensitivity:
    def test_strategy_param_change(self) -> None:
        base = a_manifest(strategy=ParamStrategy(10, 20))
        changed = a_manifest(strategy=ParamStrategy(11, 20))
        assert base.fingerprint != changed.fingerprint  # type: ignore[attr-defined]

    def test_data_change(self) -> None:
        d1 = make_ohlcv(n=30)
        d2 = d1.copy()
        d2.iloc[5, d2.columns.get_loc("close")] += 0.25
        assert a_manifest(data=d1).fingerprint != a_manifest(data=d2).fingerprint  # type: ignore[attr-defined]

    def test_commission_change(self) -> None:
        assert a_manifest(commission=1.90).fingerprint != a_manifest(commission=2.00).fingerprint  # type: ignore[attr-defined]

    def test_instrument_change(self) -> None:
        assert a_manifest(instrument=NQ).fingerprint != a_manifest(instrument=MNQ).fingerprint  # type: ignore[attr-defined]


class TestMarketDataHash:
    def test_deterministic(self) -> None:
        d = make_ohlcv(n=30)
        assert hash_market_data(d, symbol="NQ") == hash_market_data(d, symbol="NQ")

    def test_symbol_matters(self) -> None:
        d = make_ohlcv(n=30)
        assert hash_market_data(d, symbol="NQ") != hash_market_data(d, symbol="ES")

    def test_value_change_matters(self) -> None:
        d1 = make_ohlcv(n=30)
        d2 = d1.copy()
        d2.iloc[0, d2.columns.get_loc("open")] += 0.25
        assert hash_market_data(d1) != hash_market_data(d2)


class TestReviewRegressions:
    """Regression tests for issues found by the Phase 4 adversarial review."""

    def test_extra_column_value_matters(self) -> None:
        # A strategy can trade off open_interest via view.as_frame(); its content
        # must be hashed so different data cannot collide.
        d1 = make_ohlcv(n=10)
        d1["open_interest"] = 100.0
        d2 = d1.copy()
        d2["open_interest"] = 5000.0
        assert hash_market_data(d1) != hash_market_data(d2)

    def test_set_param_order_independent(self) -> None:
        data = make_ohlcv(n=10)
        m1 = build_manifest(data, ParamStrategy(), strategy_params={"universe": {"NQ", "ES", "GC"}})
        m2 = build_manifest(data, ParamStrategy(), strategy_params={"universe": {"GC", "NQ", "ES"}})
        assert m1.fingerprint == m2.fingerprint  # set ordering is hash-seed dependent

    def test_dict_int_vs_str_key_distinct(self) -> None:
        data = make_ohlcv(n=10)
        m_int = build_manifest(data, ParamStrategy(), strategy_params={"levels": {1: 0.5}})
        m_str = build_manifest(data, ParamStrategy(), strategy_params={"levels": {"1": 0.5}})
        assert m_int.fingerprint != m_str.fingerprint

    def test_describe_captures_callable_attr(self) -> None:
        import math

        class Model:
            def __init__(self, fn: object) -> None:
                self.fn = fn

        # Two configs differing only in a callable field must describe differently.
        assert describe(Model(math.floor)) != describe(Model(math.ceil))

    def test_negative_zero_hashes_same(self) -> None:
        d1 = make_ohlcv(n=5)
        d2 = d1.copy()
        d1.iloc[0, d1.columns.get_loc("volume")] = 0.0
        d2.iloc[0, d2.columns.get_loc("volume")] = -0.0
        assert hash_market_data(d1) == hash_market_data(d2)

    def test_timezone_representation_stable(self) -> None:
        # Same instants, different tz label -> same fingerprint (data hash and
        # the start/end span are both normalized to UTC).
        data = make_ohlcv(n=10, tz="America/Chicago")
        data_utc = data.copy()
        data_utc.index = pd.DatetimeIndex(data.index).tz_convert("UTC")
        assert (
            build_manifest(data, ParamStrategy()).fingerprint
            == build_manifest(data_utc, ParamStrategy()).fingerprint
        )

    def test_unfingerprintable_object_raises(self) -> None:
        import pytest

        class Slotted:
            __slots__ = ()  # no __dict__, no slots to introspect

        with pytest.raises(TypeError):
            build_manifest(make_ohlcv(n=10), ParamStrategy(), strategy_params={"x": Slotted()})


class TestManifestSerialization:
    def test_short_form(self) -> None:
        m = a_manifest()
        assert m.short.startswith("bg1:")  # type: ignore[attr-defined]
        assert len(m.short) == len("bg1:") + 16  # type: ignore[attr-defined]

    def test_to_dict_roundtrip(self) -> None:
        m = a_manifest()
        d = m.to_dict()  # type: ignore[attr-defined]
        assert d["fingerprint"] == m.fingerprint  # type: ignore[attr-defined]
        assert "fingerprint" not in m.payload()  # type: ignore[attr-defined]
        assert d["framework_version"]

    def test_to_json_is_valid(self) -> None:
        import json

        parsed = json.loads(a_manifest().to_json())  # type: ignore[attr-defined]
        assert parsed["manifest_version"] == "bg-manifest-2"
