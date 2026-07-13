"""Example 4 - a complete HTML report + the deterministic replay fingerprint.

Run:  python examples/04_report_and_fingerprint.py

Builds a *full* self-contained report on multi-year data with enough trades to
populate every section: metrics, the trust verdict, the equity curve, drawdown,
the monthly-returns heatmap (across years), the trade distribution, the Monte
Carlo fan chart, and the per-year breakdown - plus a Download/Copy AI-prompt
button. It also shows the replay fingerprint is deterministic and input-sensitive.

The report is written to examples/output/showcase/report.html - open it in a
browser to see what a complete biasguard report looks like.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biasguard.analytics import build_manifest, compute_metrics
from biasguard.audit import build_audit
from biasguard.execution import NQ, REAL_MARKET
from biasguard.montecarlo import AccountConfig, MonteCarloSimulator
from biasguard.report import build_html_report
from biasguard.strategies import RsiMeanReversion
from biasguard.validation import BacktestSpec, assess_integrity

OUTPUT_DIR = Path(__file__).parent / "output" / "showcase"


def make_strategy() -> RsiMeanReversion:
    """A short-period RSI reverter, tuned to trade often on the sample data."""
    return RsiMeanReversion(period=7, oversold=42.0, exit_level=52.0)


def rich_data(n: int = 900, start_price: float = 15000.0) -> pd.DataFrame:
    """~2.5 years of daily bars: multi-scale mean reversion + a mild drift, so an
    RSI reverter trades often across several years (populates the heatmap)."""
    t = np.arange(n, dtype="float64")
    close = (
        start_price
        + np.sin(t / 2.0) * 95.0  # fast oscillation -> frequent RSI extremes
        + np.sin(t / 19.0) * 150.0  # slower swing
        + np.cos(t / 5.3) * 45.0
        + t * 0.2  # mild uptrend
    )
    idx = pd.date_range("2021-01-04", periods=n, freq="1D", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 3.0,
            "low": np.minimum(open_, close) - 3.0,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = rich_data()

    spec = BacktestSpec.from_profile(
        data=data, strategy_factory=make_strategy, instrument=NQ, profile=REAL_MARKET
    )
    run = spec.run()
    metrics = compute_metrics(run.equity, run.trades, initial_capital=run.initial_capital)
    report = assess_integrity(spec)
    manifest = build_manifest(data, make_strategy())
    mc = MonteCarloSimulator(n_paths=5000, seed=1).run(
        run.trades,
        account=AccountConfig(starting_balance=100_000.0, trailing_drawdown_limit=5_000.0),
    )
    audit = build_audit(
        report, manifest=manifest, profile=REAL_MARKET, metrics=metrics, monte_carlo=mc
    )

    print("=" * 72)
    print("FULL REPORT SHOWCASE  (RSI mean reversion - educational, not an edge)")
    print("=" * 72)
    print(
        f"  {metrics.n_trades} trades over {data.index[0].date()}..{data.index[-1].date()}  "
        f"net ${metrics.total_pnl:,.0f}  Sharpe {metrics.sharpe:.2f}  maxDD {metrics.max_drawdown:.1%}"
    )
    print(f"  Integrity: {report.score:.0f}/100 ({report.grade})   fingerprint {manifest.short}")
    print(
        f"  Monte Carlo: P(profit)={mc.prob_profit:.0%}, worst-case DD ${mc.worst_case_drawdown:,.0f}"
    )

    # Fingerprint determinism + sensitivity.
    same = build_manifest(data, make_strategy())
    changed = build_manifest(data, RsiMeanReversion(period=20))
    print(f"  same config -> same fingerprint:   {manifest.fingerprint == same.fingerprint}")
    print(f"  period 14 vs 20 -> different:       {manifest.fingerprint != changed.fingerprint}")

    out = OUTPUT_DIR / "report.html"
    build_html_report(
        metrics=metrics,
        equity=run.equity,
        trades=run.trades,
        manifest=manifest,
        validation_html=report.to_html(),
        profile=REAL_MARKET,
        ai_prompt=audit.to_ai_prompt(),
        monte_carlo=mc,
        title="biasguard - full report showcase (RSI mean reversion, educational)",
        path=out,
    )
    print(f"\n  Wrote {out}")
    print("  Open it in a browser to see every section populated.")


if __name__ == "__main__":
    main()
