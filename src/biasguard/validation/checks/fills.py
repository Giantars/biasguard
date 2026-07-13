"""Fill realism: is the P&L real alpha, or direction-free bracket mechanics?

The signature integrity check. It runs the brief's three killer tests:

* **zero-slip** — how much of the P&L only exists without slippage;
* **trade-through** — does the edge survive requiring price to trade *through* a
  resting limit (vs. merely touch it);
* **random-direction null** — replay the same decision *timing* with a coin-flip
  side. If a random-side version earns as much, the P&L is bracket mechanics,
  not signal. Reported as ``alpha = net - null_mean``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from biasguard.events import SignalEvent
from biasguard.execution.costs import NoSlippage
from biasguard.execution.fill_models import TouchFill, TradeThroughFill
from biasguard.strategy.base import Strategy, StrategyContext
from biasguard.types import Direction
from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class _RandomDirectionNull(Strategy):
    """Replays the baseline's decision timing with a coin-flip entry side.

    Entries (LONG/SHORT signals) become a random LONG/SHORT; exits (FLAT) stay
    exits. Same timing, random direction — so any P&L that remains is direction-
    free bracket mechanics, not directional alpha.
    """

    def __init__(self, signals: Sequence[SignalEvent], seed: int) -> None:
        # Key by timestamp -> *list* of directions so a bar that emitted several
        # signals (e.g. a reversal's [exit, entry]) is replayed with the same
        # count and order, not collapsed to the last one.
        by_ts: dict[object, list[Direction]] = defaultdict(list)
        for s in signals:
            by_ts[s.timestamp].append(s.direction)
        self._by_ts = dict(by_ts)
        self._rng = np.random.default_rng(seed)

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        directions = self._by_ts.get(ctx.timestamp)
        if not directions:
            return ()
        out: list[SignalEvent] = []
        for direction in directions:
            if direction is Direction.FLAT:
                out.append(ctx.exit())
            else:
                out.append(ctx.long() if self._rng.random() < 0.5 else ctx.short())
        return tuple(out)


class FillRealismCheck(IntegrityCheck):
    key = "fill_realism"
    name = "Fill realism (alpha vs. mechanics)"
    category = "execution"
    weight = 2.0
    #: Number of random-direction null runs (seeded, so reproducible).
    n_null_runs = 50

    def run(self, ctx: IntegrityContext) -> CheckResult:
        base = ctx.baseline
        if not base.trades:
            return self.skip("no trades to assess fill realism")
        net = base.net_pnl
        if net <= 0:
            return self.skip("strategy is not net profitable; fill-realism attribution is moot")

        touch_net = ctx.rerun(fill_model=TouchFill()).net_pnl
        through_net = ctx.rerun(fill_model=TradeThroughFill()).net_pnl
        retention = (through_net / touch_net) if touch_net != 0.0 else float("nan")
        zero_slip_net = ctx.rerun(slippage=NoSlippage()).net_pnl

        nulls = np.array(
            [
                ctx.rerun(
                    strategy_factory=lambda i=i: _RandomDirectionNull(base.signals, ctx.seed + i)
                ).net_pnl
                for i in range(self.n_null_runs)
            ],
            dtype=float,
        )
        null_mean = float(nulls.mean())
        alpha = net - null_mean
        p_null_beats = float(np.mean(nulls >= net))

        metrics: dict[str, Any] = {
            "net_pnl": net,
            "null_mean": null_mean,
            "alpha": alpha,
            "p_null_beats": p_null_beats,
            "touch_net": touch_net,
            "through_net": through_net,
            "trade_through_retention": retention,
            "zero_slip_net": zero_slip_net,
        }

        if alpha <= 0:
            return self.result(
                Status.FAIL,
                0.0,
                "P&L is indistinguishable from a random-side version — bracket mechanics, not alpha",
                detail=(
                    "A coin-flip-direction replay of the same trade timing earns as much as the "
                    "strategy, so the profit is direction-free execution mechanics, not signal."
                ),
                **metrics,
            )
        if p_null_beats > 0.32:
            return self.result(
                Status.WARN,
                0.4,
                f"a random-side version beats this {p_null_beats:.0%} of the time — weak edge",
                **metrics,
            )
        if np.isfinite(retention) and retention < 0.5:
            return self.result(
                Status.WARN,
                0.5,
                f"P&L depends on optimistic limit fills (only {retention:.0%} survives trade-through)",
                detail="Requiring price to trade through the limit (not merely touch it) removes most of the edge.",
                **metrics,
            )

        alpha_fraction = alpha / net
        score = 0.5 * alpha_fraction + 0.5 * (1.0 - p_null_beats)
        return self.result(
            Status.PASS,
            score,
            f"${alpha:,.0f} of alpha beyond mechanics; null beats it {p_null_beats:.0%} of the time",
            **metrics,
        )


__all__ = ["FillRealismCheck"]
