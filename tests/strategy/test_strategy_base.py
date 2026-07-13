"""Tests for the Strategy ABC and StrategyContext."""

from __future__ import annotations

import pandas as pd

from biasguard.engine import DataHandler
from biasguard.strategy import NoOpStrategy, Strategy, StrategyContext
from biasguard.types import Direction


class TestStrategyContext:
    def _ctx(self, clean_df: pd.DataFrame, i: int = 5, position: int = 0) -> StrategyContext:
        dh = DataHandler(clean_df)
        return StrategyContext(view=dh.view(i), symbol="NQ", position=position)

    def test_convenience_accessors(self, clean_df: pd.DataFrame) -> None:
        ctx = self._ctx(clean_df, i=5)
        assert ctx.index == 5
        assert ctx.timestamp == clean_df.index[5]
        assert ctx.bar.close == clean_df["close"].iloc[5]

    def test_signal_factories_stamp_decision_time(self, clean_df: pd.DataFrame) -> None:
        ctx = self._ctx(clean_df, i=7)
        long_sig = ctx.long()
        assert long_sig.direction is Direction.LONG
        assert long_sig.symbol == "NQ"
        # Decision time is the current bar's close — a strategy cannot fake a future stamp.
        assert long_sig.timestamp == clean_df.index[7]

    def test_short_and_exit(self, clean_df: pd.DataFrame) -> None:
        ctx = self._ctx(clean_df)
        assert ctx.short().direction is Direction.SHORT
        assert ctx.exit().direction is Direction.FLAT

    def test_position_is_exposed(self, clean_df: pd.DataFrame) -> None:
        ctx = self._ctx(clean_df, position=3)
        assert ctx.position == 3


class TestNoOpStrategy:
    def test_emits_nothing(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        strat = NoOpStrategy()
        ctx = StrategyContext(view=dh.view(0), symbol="NQ", position=0)
        assert list(strat.on_bar(ctx)) == []


class TestSubclassing:
    def test_must_implement_on_bar(self) -> None:
        # Strategy is abstract; on_bar must be provided.
        assert getattr(Strategy.on_bar, "__isabstractmethod__", False)

    def test_lifecycle_hooks_are_optional(self, clean_df: pd.DataFrame) -> None:
        class Minimal(Strategy):
            def on_bar(self, ctx: StrategyContext) -> tuple[()]:
                return ()

        strat = Minimal()
        strat.on_start()  # default no-op hooks should not raise
        strat.on_finish()
