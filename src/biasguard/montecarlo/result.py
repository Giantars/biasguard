"""The Monte Carlo result: distributions, percentiles, and breach probabilities."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from biasguard.montecarlo.account import AccountConfig


def percentiles(values: np.ndarray) -> dict[str, float]:
    """p5 / p50 / p95 plus mean / std / min / max of a 1-D array."""
    if values.size == 0:
        return dict.fromkeys(("p5", "p50", "p95", "mean", "std", "min", "max"), float("nan"))
    p5, p50, p95 = (float(x) for x in np.percentile(values, [5, 50, 95]))
    return {
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


@dataclass(frozen=True)
class MonteCarloResult:
    """Distributions and risk metrics from a Monte Carlo run."""

    n_paths: int
    n_trades: int
    bootstrap_name: str
    account: AccountConfig
    final_pnl: np.ndarray  # per-path net P&L
    max_drawdown: np.ndarray  # per-path max drawdown ($, >= 0)
    prob_profit: float
    prob_breach: float
    prob_reach_target: float
    prob_breach_by_limit: dict[str, float]
    equity_bands: dict[str, np.ndarray]  # p5 / p50 / p95 equity per step
    equity_samples: np.ndarray  # a small set of raw paths for plotting
    _extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience distributions --------------------------------------- #
    @property
    def final_pnl_percentiles(self) -> dict[str, float]:
        return percentiles(self.final_pnl)

    @property
    def max_drawdown_percentiles(self) -> dict[str, float]:
        return percentiles(self.max_drawdown)

    @property
    def worst_case_drawdown(self) -> float:
        """The 95th-percentile max drawdown — a realistic worst case."""
        return self.max_drawdown_percentiles["p95"]

    def summary(self) -> str:
        fp = self.final_pnl_percentiles
        dd = self.max_drawdown_percentiles
        lines = [
            f"Monte Carlo ({self.bootstrap_name}, {self.n_paths} paths x {self.n_trades} trades):",
            f"  P(profit) = {self.prob_profit:.1%}",
            f"  final P&L  p5/p50/p95 = ${fp['p5']:,.0f} / ${fp['p50']:,.0f} / ${fp['p95']:,.0f}",
            f"  max drawdown p50/p95/worst = ${dd['p50']:,.0f} / ${dd['p95']:,.0f} / ${dd['max']:,.0f}",
        ]
        if self.account.has_limits:
            lines.append(f"  P(breach any limit) = {self.prob_breach:.1%}")
            for limit, prob in sorted(self.prob_breach_by_limit.items()):
                lines.append(f"    P({limit}) = {prob:.1%}")
        if self.account.profit_target is not None:
            lines.append(f"  P(reach target) = {self.prob_reach_target:.1%}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_paths": self.n_paths,
            "n_trades": self.n_trades,
            "bootstrap": self.bootstrap_name,
            "prob_profit": self.prob_profit,
            "prob_breach": self.prob_breach,
            "prob_reach_target": self.prob_reach_target,
            "prob_breach_by_limit": self.prob_breach_by_limit,
            "final_pnl": self.final_pnl_percentiles,
            "max_drawdown": self.max_drawdown_percentiles,
            "worst_case_drawdown": self.worst_case_drawdown,
        }

    def to_html(self) -> str:
        fp = self.final_pnl_percentiles
        dd = self.max_drawdown_percentiles
        rows = [
            ("P(profit)", f"{self.prob_profit:.1%}"),
            (
                "Final P&L (p5 / p50 / p95)",
                f"${fp['p5']:,.0f} / ${fp['p50']:,.0f} / ${fp['p95']:,.0f}",
            ),
            (
                "Max drawdown (p50 / p95 / worst)",
                f"${dd['p50']:,.0f} / ${dd['p95']:,.0f} / ${dd['max']:,.0f}",
            ),
        ]
        if self.account.has_limits:
            rows.append(("P(breach any limit)", f"{self.prob_breach:.1%}"))
        body = "".join(
            f"<tr><td>{_html.escape(k)}</td><td style='text-align:right'>{_html.escape(v)}</td></tr>"
            for k, v in rows
        )
        return (
            f"<div style='font-weight:600;margin-bottom:6px'>Monte Carlo "
            f"({_html.escape(self.bootstrap_name)}, {self.n_paths} paths)</div>"
            f"<table style='width:100%;border-collapse:collapse'>{body}</table>"
        )


__all__ = ["MonteCarloResult", "percentiles"]
