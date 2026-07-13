"""The reproducible-run substrate that integrity checks analyze and perturb.

* :class:`BacktestSpec` is the recipe for a run — data + strategy factory +
  instrument + cost/fill models — with one method, :meth:`~BacktestSpec.run`,
  that re-executes with optional perturbations (truncated data, a different fill
  model, more slippage, a different strategy factory).
* :class:`RunOutput` bundles a run's signals/fills/trades/equity.
* :class:`IntegrityContext` pairs the spec with the baseline run and a seed; it
  is what every check receives.

The single ``run(...)`` seam is what lets robustness analyses (slippage, price
noise, execution delay, parameter sweeps) be added as plugins with no engine
change: they just call ``spec.run(<perturbation>)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from biasguard.analytics.metrics import PerformanceMetrics, compute_metrics
from biasguard.engine import Backtester, DataHandler
from biasguard.events import FillEvent, SignalEvent
from biasguard.execution.broker import SameBarPolicy, SimulatedBroker
from biasguard.execution.costs import (
    CommissionModel,
    NoSlippage,
    SlippageModel,
    ZeroCommission,
)
from biasguard.execution.fill_models import FillModel, TradeThroughFill
from biasguard.execution.instrument import Instrument
from biasguard.execution.orders import Trade
from biasguard.execution.portfolio import FixedSizer, Portfolio, Sizer
from biasguard.execution.profiles import ExecutionProfile
from biasguard.strategy.base import Strategy


def _default_sizer() -> Sizer:
    return FixedSizer(1)


@dataclass(frozen=True, slots=True)
class RunOutput:
    """The observable outcome of a single backtest run."""

    data: pd.DataFrame
    signals: tuple[SignalEvent, ...]
    fills: tuple[FillEvent, ...]
    trades: tuple[Trade, ...]
    equity: pd.Series
    net_pnl: float
    initial_capital: float

    def metrics(self, periods_per_year: float | None = None) -> PerformanceMetrics:
        return compute_metrics(
            self.equity,
            self.trades,
            initial_capital=self.initial_capital,
            periods_per_year=periods_per_year,
        )


@dataclass(frozen=True)
class BacktestSpec:
    """A reproducible recipe an integrity check can re-run with perturbations."""

    data: pd.DataFrame
    strategy_factory: Callable[[], Strategy]
    instrument: Instrument
    commission: CommissionModel
    slippage: SlippageModel
    fill_model_factory: Callable[[], FillModel] = TradeThroughFill
    sizer_factory: Callable[[], Sizer] = _default_sizer
    same_bar_policy: SameBarPolicy = SameBarPolicy.STOP_FIRST
    initial_capital: float = 100_000.0
    symbol: str = ""

    @classmethod
    def from_profile(
        cls,
        *,
        data: pd.DataFrame,
        strategy_factory: Callable[[], Strategy],
        instrument: Instrument,
        profile: ExecutionProfile,
        sizer_factory: Callable[[], Sizer] = _default_sizer,
        same_bar_policy: SameBarPolicy = SameBarPolicy.STOP_FIRST,
        initial_capital: float = 100_000.0,
        symbol: str = "",
    ) -> BacktestSpec:
        """Build a spec whose execution models come from an :class:`ExecutionProfile`."""
        return cls(
            data=data,
            strategy_factory=strategy_factory,
            instrument=instrument,
            commission=profile.commission,
            slippage=profile.slippage,
            fill_model_factory=profile.fill_model_factory,
            sizer_factory=sizer_factory,
            same_bar_policy=same_bar_policy,
            initial_capital=initial_capital,
            symbol=symbol,
        )

    def run(
        self,
        *,
        data: pd.DataFrame | None = None,
        strategy_factory: Callable[[], Strategy] | None = None,
        fill_model: FillModel | None = None,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ) -> RunOutput:
        """Execute the backtest, overriding any component passed explicitly."""
        d = self.data if data is None else data
        strategy = (strategy_factory or self.strategy_factory)()
        broker = SimulatedBroker(
            self.instrument,
            fill_model=fill_model if fill_model is not None else self.fill_model_factory(),
            commission=commission if commission is not None else self.commission,
            slippage=slippage if slippage is not None else self.slippage,
            same_bar_policy=self.same_bar_policy,
        )
        portfolio = Portfolio(
            self.instrument, initial_capital=self.initial_capital, sizer=self.sizer_factory()
        )
        result = Backtester(
            DataHandler(d, symbol=self.symbol, validate=False),
            strategy,
            portfolio=portfolio,
            broker=broker,
        ).run()
        return RunOutput(
            data=d,
            signals=result.signals,
            fills=result.fills,
            trades=portfolio.trades,
            equity=portfolio.equity_series(),
            net_pnl=float(portfolio.equity - portfolio.initial_capital),
            initial_capital=portfolio.initial_capital,
        )

    def zero_cost(self) -> bool:
        """True when both cost models are the trivial zero models."""
        return isinstance(self.commission, ZeroCommission) and isinstance(self.slippage, NoSlippage)


@dataclass(frozen=True)
class IntegrityContext:
    """What every :class:`~biasguard.validation.base.IntegrityCheck` receives."""

    spec: BacktestSpec
    baseline: RunOutput
    seed: int = 12345
    oos_cut: pd.Timestamp | None = None
    config: dict[str, object] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        spec: BacktestSpec,
        *,
        seed: int = 12345,
        oos_cut: pd.Timestamp | None = None,
        config: dict[str, object] | None = None,
    ) -> IntegrityContext:
        return cls(spec=spec, baseline=spec.run(), seed=seed, oos_cut=oos_cut, config=config or {})

    def rerun(self, **overrides: object) -> RunOutput:
        """Re-run the baseline recipe with perturbations (see :meth:`BacktestSpec.run`)."""
        return self.spec.run(**overrides)  # type: ignore[arg-type]


__all__ = ["BacktestSpec", "IntegrityContext", "RunOutput"]
