"""Strategies with known, deliberate flaws for power-validation."""

from __future__ import annotations

from collections.abc import Sequence

from biasguard.events import SignalEvent
from biasguard.strategy import Strategy, StrategyContext


class UptickCausal(Strategy):
    """Honest baseline: long after an uptick, flat after a downtick."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        view = ctx.view
        if len(view) < 2:
            return ()
        if view.closes[-1] > view.closes[-2]:
            return (ctx.long(),) if ctx.position == 0 else ()
        return (ctx.exit(),) if ctx.position != 0 else ()


class LeakyPeek(Strategy):
    """PLANTED LOOKAHEAD: reaches through the numpy view's ``.base`` to read the
    *next* bar's close. The lookahead detector must catch this."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes
        base = closes.base
        i = ctx.index
        if base is not None and i + 1 < len(base) and base[i + 1] > ctx.bar.close:
            return (ctx.long(),) if ctx.position == 0 else ()
        return (ctx.exit(),) if ctx.position != 0 else ()


class AlwaysLong(Strategy):
    """Enter long once and hold."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        return (ctx.long(),) if ctx.position == 0 else ()


class Churn(Strategy):
    """High-turnover: enter long when flat, exit on the next bar. Round-trips
    every two bars, so costs/slippage dominate a small edge."""

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.position == 0:
            return (ctx.long(),)
        return (ctx.exit(),)
