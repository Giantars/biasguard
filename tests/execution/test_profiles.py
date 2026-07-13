"""Tests for execution profiles — presets, realism probing, registry, from_profile."""

from __future__ import annotations

import pytest
from tests.conftest import make_ohlcv
from tests.known_bad.strategies import Churn

from biasguard.execution.costs import (
    CommissionModel,
    NoSlippage,
    PerContractCommission,
    SlippageModel,
    TickSlippage,
    ZeroCommission,
)
from biasguard.execution.fill_models import (
    FillDecision,
    FillModel,
    FillRequest,
    TouchFill,
    TradeThroughFill,
)
from biasguard.execution.instrument import NQ, Instrument
from biasguard.execution.profiles import (
    PROFILES,
    PROP_FIRM_SIM,
    REAL_MARKET,
    ExecutionProfile,
    assess_realism,
    custom_profile,
    get_profile,
    register_profile,
)
from biasguard.types import OrderSide
from biasguard.validation import BacktestSpec


class TestPresets:
    def test_real_market_is_realistic(self) -> None:
        assert REAL_MARKET.is_realistic
        assert REAL_MARKET.realism.warnings == ()

    def test_prop_firm_is_flagged_simulated(self) -> None:
        realism = PROP_FIRM_SIM.realism
        assert not realism.is_realistic
        # Zero slippage AND optimistic fills should both be surfaced.
        joined = " ".join(realism.warnings).lower()
        assert "slippage" in joined
        assert "optimistic" in joined or "touch" in joined
        assert "SIMULATED" in realism.banner

    def test_presets_registered(self) -> None:
        assert get_profile("Real Market") is REAL_MARKET
        assert get_profile("Prop Firm Simulation") is PROP_FIRM_SIM


class TestRealismProbe:
    def test_zero_commission_flagged(self) -> None:
        r = assess_realism(ZeroCommission(), TickSlippage(1.0), TradeThroughFill())
        assert not r.is_realistic
        assert any("commission" in w.lower() for w in r.warnings)

    def test_custom_zero_slippage_model_cannot_bypass(self) -> None:
        # A user's own model that happens to be a no-op must still be flagged:
        # TickSlippage(0.0) moves no price, so the probe catches it.
        r = assess_realism(PerContractCommission(2.0), TickSlippage(0.0), TradeThroughFill())
        assert not r.is_realistic
        assert any("slippage" in w.lower() for w in r.warnings)

    def test_touch_fill_flagged_optimistic(self) -> None:
        r = assess_realism(PerContractCommission(2.0), TickSlippage(1.0), TouchFill())
        assert not r.is_realistic
        assert any("touch" in w.lower() or "optimistic" in w.lower() for w in r.warnings)

    def test_fully_realistic_has_no_warnings(self) -> None:
        r = assess_realism(PerContractCommission(2.0), TickSlippage(1.0), TradeThroughFill())
        assert r.is_realistic
        assert r.banner.startswith("Realistic")


class _CustomTouchFill(FillModel):
    """Optimistic custom fill that fills on a touch but is NOT a TouchFill subclass.

    The old isinstance(TouchFill) check would have missed this; the behavioral
    probe must catch it.
    """

    def _limit_fill(self, req: FillRequest) -> FillDecision:
        order, bar = req.order, req.bar
        limit = order.limit_price
        assert limit is not None
        if order.side is OrderSide.BUY and bar.low <= limit:
            return FillDecision(True, min(limit, bar.open), req.remaining_qty, False, "custom")
        if order.side is OrderSide.SELL and bar.high >= limit:
            return FillDecision(True, max(limit, bar.open), req.remaining_qty, False, "custom")
        return FillDecision.no_fill()


class _FavorableSlippage(SlippageModel):
    """Fills BUY below and SELL above the reference — 'free money', optimistic."""

    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        return price - 0.25 * side.sign


class _OneSidedSlippage(SlippageModel):
    """Adverse on BUY, zero on SELL."""

    def apply(self, price: float, side: OrderSide, instrument: Instrument) -> float:
        return price + (0.25 if side is OrderSide.BUY else 0.0) * side.sign


class _FreeCommission(CommissionModel):
    """A custom model that charges nothing."""

    def calculate(self, quantity: float, price: float, instrument: Instrument) -> float:
        return 0.0


class TestCustomModelCannotBypass:
    """The 'realism cannot be bypassed by a user's own model' guarantee, tested
    against genuine custom subclasses (not just zeroed built-ins)."""

    def test_custom_touch_filling_model_flagged(self) -> None:
        r = assess_realism(PerContractCommission(2.0), TickSlippage(1.0), _CustomTouchFill())
        assert not r.is_realistic
        assert any("optimistic" in w.lower() or "touch" in w.lower() for w in r.warnings)

    def test_trade_through_near_zero_flagged(self) -> None:
        # TradeThroughFill(0.0001) fills on essentially any touch -> optimistic.
        r = assess_realism(PerContractCommission(2.0), TickSlippage(1.0), TradeThroughFill(0.0001))
        assert not r.is_realistic

    def test_favorable_slippage_flagged(self) -> None:
        r = assess_realism(PerContractCommission(2.0), _FavorableSlippage(), TradeThroughFill())
        assert not r.is_realistic
        assert any("favorable" in w.lower() for w in r.warnings)

    def test_one_sided_slippage_flagged(self) -> None:
        r = assess_realism(PerContractCommission(2.0), _OneSidedSlippage(), TradeThroughFill())
        assert not r.is_realistic
        assert any("slippage" in w.lower() for w in r.warnings)

    def test_custom_zero_commission_flagged(self) -> None:
        r = assess_realism(_FreeCommission(), TickSlippage(1.0), TradeThroughFill())
        assert not r.is_realistic
        assert any("commission" in w.lower() for w in r.warnings)


class TestDescribe:
    def test_describe_is_serializable_and_complete(self) -> None:
        d = REAL_MARKET.describe()
        assert d["name"] == "Real Market"
        assert d["is_realistic"] is True
        assert d["commission"]["type"] == "PerContractCommission"
        assert d["commission"]["params"]["per_contract"] == 2.0
        assert d["slippage"]["type"] == "TickSlippage"
        assert d["fill_model"]["type"] == "TradeThroughFill"
        assert isinstance(d["assumptions"], list)


class TestRegistry:
    def test_register_and_get(self) -> None:
        p = custom_profile("Test Zero", commission=ZeroCommission(), slippage=NoSlippage())
        register_profile(p)
        assert get_profile("Test Zero") is p
        assert "Test Zero" in PROFILES
        # Clean up so the module-level registry is not polluted for other tests.
        PROFILES.pop("Test Zero", None)

    def test_duplicate_without_replace_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_profile(REAL_MARKET, replace=False)

    def test_custom_profile_defaults_conservative_fill(self) -> None:
        p = custom_profile("C", commission=PerContractCommission(1.0), slippage=TickSlippage(1.0))
        assert isinstance(p.fill_model_factory(), TradeThroughFill)
        assert p.is_realistic


class TestFromProfile:
    def test_spec_uses_profile_models(self) -> None:
        spec = BacktestSpec.from_profile(
            data=make_ohlcv(n=40),
            strategy_factory=Churn,
            instrument=NQ,
            profile=REAL_MARKET,
        )
        assert spec.commission is REAL_MARKET.commission
        assert spec.slippage is REAL_MARKET.slippage
        assert spec.fill_model_factory is REAL_MARKET.fill_model_factory
        out = spec.run()
        assert len(out.trades) > 0  # Churn round-trips constantly

    def test_prop_firm_and_real_market_differ_in_pnl(self) -> None:
        # Same strategy, different execution assumptions -> different results,
        # which is the whole point of profiles being swappable.
        data = make_ohlcv(n=60)
        real = BacktestSpec.from_profile(
            data=data, strategy_factory=Churn, instrument=NQ, profile=REAL_MARKET
        ).run()
        prop = BacktestSpec.from_profile(
            data=data, strategy_factory=Churn, instrument=NQ, profile=PROP_FIRM_SIM
        ).run()
        assert real.net_pnl != prop.net_pnl


def test_profile_is_frozen() -> None:
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError or AttributeError
        REAL_MARKET.name = "mutated"  # type: ignore[misc]


def test_execution_profile_constructible_directly() -> None:
    p = ExecutionProfile(
        name="Direct",
        description="built directly",
        commission=PerContractCommission(1.0),
        slippage=TickSlippage(1.0),
    )
    assert p.is_realistic
