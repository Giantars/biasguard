"""Educational example strategies + a catalog pairing each with its integrity verdict.

These are teaching artifacts, not trading systems — every one is labelled
"Educational example — not intended as a profitable trading strategy." Two are
honest causal templates (moving-average crossover, RSI mean reversion); four are
intentionally flawed to demonstrate what the Integrity Framework catches.

    from biasguard.strategies import CATALOG
    for case in CATALOG:
        ...  # run case.make_strategy() on case.make_data() under case.profile
"""

from __future__ import annotations

from biasguard.strategies.catalog import (
    CATALOG,
    ZERO_COST_PROFILE,
    ExampleCase,
    get_case,
)
from biasguard.strategies.educational import (
    EDUCATIONAL_LABEL,
    EducationalStrategy,
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

__all__ = [
    "CATALOG",
    "EDUCATIONAL_LABEL",
    "ZERO_COST_PROFILE",
    "EducationalStrategy",
    "ExampleCase",
    "FreeLunchChurn",
    "LookaheadStrategy",
    "MovingAverageCrossover",
    "OverfitMeanReversion",
    "RsiMeanReversion",
    "ZeroSlippageScalper",
    "choppy_data",
    "get_case",
    "mean_reverting_data",
    "regime_break_data",
    "swinging_data",
    "trending_data",
]
