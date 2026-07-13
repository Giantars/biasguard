"""Tests for commission and slippage models."""

from __future__ import annotations

import pytest

from biasguard.execution.costs import (
    FixedSlippage,
    NoSlippage,
    PercentCommission,
    PercentSlippage,
    PerContractCommission,
    TickSlippage,
    ZeroCommission,
)
from biasguard.execution.instrument import NQ
from biasguard.types import OrderSide


class TestCommission:
    def test_zero(self) -> None:
        assert ZeroCommission().calculate(3.0, 15000.0, NQ) == 0.0

    def test_per_contract(self) -> None:
        model = PerContractCommission(1.90)
        assert model.calculate(2.0, 15000.0, NQ) == pytest.approx(3.80)

    def test_per_contract_uses_absolute_quantity(self) -> None:
        model = PerContractCommission(1.90)
        assert model.calculate(-2.0, 15000.0, NQ) == pytest.approx(3.80)

    def test_percent(self) -> None:
        model = PercentCommission(0.0001)
        # notional = 1 * 15000 * 20 = 300_000; commission = 30.0
        assert model.calculate(1.0, 15000.0, NQ) == pytest.approx(30.0)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError):
            PerContractCommission(-1.0)


class TestSlippage:
    def test_no_slippage(self) -> None:
        assert NoSlippage().apply(100.0, OrderSide.BUY, NQ) == 100.0

    def test_fixed_is_adverse(self) -> None:
        model = FixedSlippage(0.5)
        assert model.apply(100.0, OrderSide.BUY, NQ) == pytest.approx(100.5)  # buy pays more
        assert model.apply(100.0, OrderSide.SELL, NQ) == pytest.approx(99.5)  # sell gets less

    def test_tick(self) -> None:
        model = TickSlippage(2.0)  # 2 ticks * 0.25 = 0.5
        assert model.apply(100.0, OrderSide.BUY, NQ) == pytest.approx(100.5)
        assert model.apply(100.0, OrderSide.SELL, NQ) == pytest.approx(99.5)

    def test_percent(self) -> None:
        model = PercentSlippage(0.001)
        assert model.apply(100.0, OrderSide.BUY, NQ) == pytest.approx(100.1)
        assert model.apply(100.0, OrderSide.SELL, NQ) == pytest.approx(99.9)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            FixedSlippage(-0.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PercentCommission(-0.1),
        lambda: TickSlippage(-1.0),
        lambda: PercentSlippage(-0.1),
    ],
)
def test_all_negative_rates_rejected(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]
