"""Strategy interface and (from Phase 3) educational example strategies."""

from __future__ import annotations

from biasguard.strategy.base import (
    NoOpStrategy,
    Strategy,
    StrategyContext,
)

__all__ = ["NoOpStrategy", "Strategy", "StrategyContext"]
