"""The educational example catalog: strategy + data + execution profile + the
integrity verdict it is meant to demonstrate.

Each :class:`ExampleCase` is a self-contained teaching unit. The examples script
(``examples/08_educational_strategies.py``) runs every case end-to-end and the
test-suite asserts the watched check produces the documented status, so the
catalog is the single source of truth for "what BiasGuard should say about this".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from biasguard.execution.costs import NoSlippage, ZeroCommission
from biasguard.execution.instrument import NQ, Instrument
from biasguard.execution.profiles import REAL_MARKET, ExecutionProfile, custom_profile
from biasguard.strategies.educational import (
    FreeLunchChurn,
    LookaheadStrategy,
    MovingAverageCrossover,
    OverfitMeanReversion,
    RsiMeanReversion,
    ZeroSlippageScalper,
    choppy_data,
    mean_reverting_data,
    regime_break_data,
    swinging_data,
    trending_data,
)
from biasguard.strategy.base import Strategy

#: A deliberately unrealistic profile: free trading. Used to show the cost check
#: firing (and it is itself flagged as a simulated environment).
ZERO_COST_PROFILE = custom_profile(
    "Zero Cost (unrealistic)",
    commission=ZeroCommission(),
    slippage=NoSlippage(),
    description="Zero commission and zero slippage — deliberately unrealistic, for teaching.",
    assumptions=("No commission charged.", "No slippage applied."),
)


@dataclass(frozen=True)
class ExampleCase:
    """One educational strategy paired with the verdict it should produce."""

    key: str
    title: str
    make_strategy: Callable[[], Strategy]
    make_data: Callable[[], pd.DataFrame]
    #: The check key whose status this case is designed to demonstrate.
    watch: str
    #: The status that check is expected to return ("PASS" / "WARN" / "FAIL").
    expected_status: str
    #: Plain-English explanation of why it passes or fails.
    explanation: str
    profile: ExecutionProfile = REAL_MARKET
    instrument: Instrument = field(default=NQ)


CATALOG: tuple[ExampleCase, ...] = (
    ExampleCase(
        key="ma_crossover",
        title="Moving Average Crossover (honest)",
        make_strategy=MovingAverageCrossover,
        make_data=swinging_data,
        watch="lookahead",
        expected_status="PASS",
        explanation=(
            "A causal trend-follower: both SMAs are read from the completed-bars view. "
            "The truncation gate passes (decisions are byte-identical on data[:T]), and "
            "it is costed with a realistic profile, so BiasGuard trusts the mechanics. "
            "Whether it is *profitable* is a separate question from whether it is *honest*."
        ),
    ),
    ExampleCase(
        key="rsi_reversion",
        title="RSI Mean Reversion (honest)",
        make_strategy=RsiMeanReversion,
        make_data=mean_reverting_data,
        watch="lookahead",
        expected_status="PASS",
        explanation=(
            "A causal mean-reverter: RSI is computed only from past closes. It passes "
            "the lookahead gate and is honestly costed. Again, causal and honest — not a "
            "claim of edge."
        ),
    ),
    ExampleCase(
        key="lookahead",
        title="Lookahead Bias (broken)",
        make_strategy=LookaheadStrategy,
        make_data=mean_reverting_data,
        watch="lookahead",
        expected_status="FAIL",
        explanation=(
            "Reaches past the causal view into the backing array to read tomorrow's close. "
            "The truncation test re-runs on data[:T] and the past decisions change, so the "
            "lookahead GATE fails and caps the whole Integrity Score. This is the single "
            "most expensive class of backtest lie."
        ),
    ),
    ExampleCase(
        key="overfit",
        title="Overfit Parameters (fragile)",
        make_strategy=OverfitMeanReversion,
        make_data=regime_break_data,
        watch="regime",
        expected_status="WARN",
        explanation=(
            "Causal, but nearly all of the P&L comes from the first (strong) regime; the "
            "second regime contributes little. The regime-concentration check WARNs: a big "
            "number from one window is not a durable edge."
        ),
    ),
    ExampleCase(
        key="zero_slippage",
        title="Depends on Zero Slippage (fragile)",
        make_strategy=ZeroSlippageScalper,
        make_data=choppy_data,
        watch="slippage_sensitivity",
        expected_status="WARN",
        explanation=(
            "A thin 1-2 tick scalp edge that exists at zero slippage and decays fast as "
            "slippage rises. The slippage-sensitivity check WARNs (or FAILs if it dies "
            "inside a tick): an edge smaller than routine slippage is a fill artifact."
        ),
    ),
    ExampleCase(
        key="free_lunch",
        title="Unrealistic Transaction Costs (broken)",
        make_strategy=FreeLunchChurn,
        make_data=trending_data,
        watch="costs",
        expected_status="FAIL",
        profile=ZERO_COST_PROFILE,
        explanation=(
            "A high-turnover strategy run with ZERO commission and slippage. The cost "
            "check FAILs outright: a zero-cost backtest is the maximally optimistic case, "
            "and most high-frequency 'edges' flip negative once real fills are charged."
        ),
    ),
)


def get_case(key: str) -> ExampleCase:
    """Look up a case by key (raises ``KeyError`` if unknown)."""
    for case in CATALOG:
        if case.key == key:
            return case
    raise KeyError(key)


__all__ = ["CATALOG", "ZERO_COST_PROFILE", "ExampleCase", "get_case"]
