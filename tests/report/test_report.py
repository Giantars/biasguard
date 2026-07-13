"""Tests for the HTML report — self-contained, deterministic, has the fingerprint."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from tests.conftest import make_ohlcv

from biasguard.analytics.fingerprint import build_manifest
from biasguard.analytics.metrics import compute_metrics
from biasguard.engine import Backtester, DataHandler
from biasguard.events import SignalEvent
from biasguard.execution.broker import SimulatedBroker
from biasguard.execution.costs import FixedSlippage, PerContractCommission
from biasguard.execution.instrument import NQ
from biasguard.execution.portfolio import FixedSizer, Portfolio
from biasguard.execution.profiles import PROP_FIRM_SIM, REAL_MARKET
from biasguard.report import build_html_report
from biasguard.strategy import Strategy, StrategyContext


class _Strat(Strategy):
    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        if ctx.index == 3:
            return (ctx.long(),)
        if ctx.index == 20:
            return (ctx.exit(),)
        return ()


def _run() -> tuple[object, Portfolio, object, pd.DataFrame]:
    data = make_ohlcv(n=30)
    pf = Portfolio(NQ, initial_capital=100_000.0, sizer=FixedSizer(1))
    broker = SimulatedBroker(
        NQ, commission=PerContractCommission(1.90), slippage=FixedSlippage(0.25)
    )
    Backtester(DataHandler(data), _Strat(), portfolio=pf, broker=broker).run()
    metrics = compute_metrics(pf.equity_series(), pf.trades, initial_capital=pf.initial_capital)
    manifest = build_manifest(data, _Strat(), broker=broker, portfolio=pf)
    return metrics, pf, manifest, data


def test_report_contains_fingerprint_and_sections() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    assert manifest.short in html  # type: ignore[attr-defined]
    assert manifest.fingerprint in html  # type: ignore[attr-defined]
    assert "Key metrics" in html
    assert "Equity curve" in html
    assert "Validation verdict" in html
    assert "bg-equity" in html and "bg-drawdown" in html


def test_report_is_self_contained() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    # Plotly is inlined, not pulled from a CDN: there must be no external script src.
    # (plotly.js embeds a default topojson URL string for geo maps we never render;
    # that is inert text inside the inlined bundle, not a network dependency.)
    assert "<script src" not in html.lower()
    assert "Plotly.newPlot" in html


def test_plotly_library_loads_before_figures() -> None:
    # The inlined plotly.js must appear BEFORE the first figure's newPlot() call,
    # or every chart throws "Plotly is not defined" and renders blank.
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    lib = html.find("/*plotly-lib*/")
    first_newplot = html.find("Plotly.newPlot")
    assert lib != -1 and first_newplot != -1
    assert lib < first_newplot


def test_report_is_deterministic() -> None:
    metrics, pf, manifest, _ = _run()
    a = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    b = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    assert a == b


def test_report_writes_file_and_manifest(tmp_path: Path) -> None:
    metrics, pf, manifest, _ = _run()
    out = tmp_path / "report.html"
    build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest, path=out
    )
    assert out.exists()
    companion = out.with_suffix(".manifest.json")
    assert companion.exists()
    assert manifest.fingerprint in companion.read_text(encoding="utf-8")  # type: ignore[attr-defined]


def test_report_without_manifest() -> None:
    metrics, pf, _, _ = _run()
    html = build_html_report(metrics=metrics, equity=pf.equity_series(), trades=pf.trades)
    assert "n/a" in html  # fingerprint slot shows n/a


def test_report_shows_realistic_profile_without_banner() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        profile=REAL_MARKET,
    )
    assert "Execution profile" in html
    assert "Real Market" in html
    # A realistic profile shows no "simulated environment" banner.
    assert "Simulated execution environment" not in html


def test_report_flags_simulated_profile() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        profile=PROP_FIRM_SIM,
    )
    assert "Prop Firm Simulation" in html
    assert "Simulated execution environment" in html


def test_report_embeds_ai_prompt_button() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        ai_prompt="PASTE ME INTO CLAUDE",
    )
    assert "Debug with an AI assistant" in html
    assert "Download prompt" in html
    assert "BG_AI_PROMPT" in html and "bgDownloadPrompt" in html
    assert "PASTE ME INTO CLAUDE" in html


def test_report_without_ai_prompt_has_no_button() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    assert "bgDownloadPrompt" not in html
    assert "Debug with an AI assistant" not in html


def test_ai_prompt_cannot_break_out_of_script() -> None:
    # A prompt containing markup must not be able to inject a live <script> or tag.
    metrics, pf, manifest, _ = _run()
    evil = "</script><script>alert(1)</script> & <img src=x onerror=alert(2)>"
    html = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        ai_prompt=evil,
    )
    # No raw breakout tag from the prompt survives.
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    # The characters are preserved but escaped as inert JS-string content.
    assert "\\u003c/script\\u003e" in html


def _monte_carlo_result() -> object:
    from biasguard.execution import NQ, REAL_MARKET
    from biasguard.montecarlo import AccountConfig, MonteCarloSimulator
    from biasguard.strategies import RsiMeanReversion, mean_reverting_data
    from biasguard.validation import BacktestSpec

    spec = BacktestSpec.from_profile(
        data=mean_reverting_data(),
        strategy_factory=RsiMeanReversion,
        instrument=NQ,
        profile=REAL_MARKET,
    )
    trades = spec.run().trades
    account = AccountConfig(starting_balance=100_000.0, trailing_drawdown_limit=3_000.0)
    return MonteCarloSimulator(n_paths=200, seed=1).run(trades, account=account)


def test_report_includes_monte_carlo_section() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        monte_carlo=_monte_carlo_result(),  # type: ignore[arg-type]
    )
    assert "Monte Carlo (resampled outcomes)" in html
    assert "bg-montecarlo" in html  # the fan-chart div
    assert "P(profit)" in html  # the summary table


def test_report_without_monte_carlo_has_no_section() -> None:
    metrics, pf, manifest, _ = _run()
    html = build_html_report(
        metrics=metrics, equity=pf.equity_series(), trades=pf.trades, manifest=manifest
    )
    assert "bg-montecarlo" not in html


def test_ai_prompt_report_is_self_contained_and_deterministic() -> None:
    metrics, pf, manifest, _ = _run()
    prompt = 'line one\nline "two"\tend'
    a = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        ai_prompt=prompt,
    )
    b = build_html_report(
        metrics=metrics,
        equity=pf.equity_series(),
        trades=pf.trades,
        manifest=manifest,
        ai_prompt=prompt,
    )
    assert a == b
    assert "<script src" not in a.lower()  # still no external scripts
