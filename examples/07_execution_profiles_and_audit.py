"""Example 7 - Execution profiles & AI audit export.

Run:  python examples/07_execution_profiles_and_audit.py

Shows (Part 1) how one execution *profile* swaps the whole cost/slippage/fill
environment in a line, how an optimistic profile is flagged as a SIMULATED
environment, and (Part 2) how the integrity assessment is exported as
paste-ready context for an external LLM (audit_report.md / .json /
ai_debug_prompt.txt).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from biasguard.analytics.fingerprint import build_manifest
from biasguard.audit import build_audit
from biasguard.events import SignalEvent
from biasguard.execution import NQ, PROP_FIRM_SIM, REAL_MARKET
from biasguard.montecarlo import AccountConfig, MonteCarloSimulator
from biasguard.strategy import Strategy, StrategyContext
from biasguard.validation import BacktestSpec, assess_integrity


def data(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2022-01-03 08:30", periods=n, freq="30min", tz="America/Chicago")
    t = np.arange(n)
    close = 15000.0 + np.cumsum(np.where((t // 120) % 3 == 2, -1.1, 0.7)) + np.sin(t / 2.0) * 4.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": 500.0,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


class PeriodicHold(Strategy):
    """Educational, not an edge: take an 8-bar long, then re-enter."""

    hold_bars = 8

    def __init__(self) -> None:
        self._entry: int | None = None

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.position == 0 and self._entry is None:
            self._entry = ctx.index
            return (ctx.long(),)
        if self._entry is not None and ctx.index - self._entry >= self.hold_bars:
            self._entry = None
            return (ctx.exit(),)
        return ()


def main() -> None:
    df = data()

    print("=" * 72)
    print("1) SAME STRATEGY, TWO EXECUTION PROFILES")
    print("=" * 72)
    for profile in (REAL_MARKET, PROP_FIRM_SIM):
        spec = BacktestSpec.from_profile(
            data=df, strategy_factory=PeriodicHold, instrument=NQ, profile=profile
        )
        out = spec.run()
        verdict = "realistic" if profile.is_realistic else "SIMULATED (optimistic)"
        print(f"  {profile.name:<22} net ${out.net_pnl:>10,.0f}   [{verdict}]")
        for warning in profile.realism.warnings:
            print(f"      - {warning}")
    print()

    print("=" * 72)
    print("2) AI AUDIT EXPORT (paste-ready LLM debugging context)")
    print("=" * 72)
    account = AccountConfig(starting_balance=100_000.0, trailing_drawdown_limit=3_000.0)
    spec = BacktestSpec.from_profile(
        data=df, strategy_factory=PeriodicHold, instrument=NQ, profile=REAL_MARKET
    )
    out = spec.run()
    integrity = assess_integrity(spec, config={"account": account})
    manifest = build_manifest(df, PeriodicHold())
    mc = MonteCarloSimulator(n_paths=2000, seed=1).run(out.trades, account=account)

    audit = build_audit(
        integrity,
        manifest=manifest,
        profile=REAL_MARKET,
        metrics=out.metrics(),
        monte_carlo=mc,
    )

    print(f"  Integrity: {integrity.score:.0f}/100 ({integrity.grade})")
    print(f"  Investigation targets: {len(audit.targets())}")
    print()
    print("  --- ai_debug_prompt.txt (first 24 lines) ---")
    for line in audit.to_ai_prompt().splitlines()[:24]:
        print(f"  {line}")
    print("  ... (truncated)")
    print()
    print(
        "  audit.write('out/') would emit: audit_report.md, audit_report.json, ai_debug_prompt.txt"
    )


if __name__ == "__main__":
    main()
