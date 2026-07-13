"""Self-contained Plotly HTML report.

The report is a single HTML file with plotly.js inlined once (no CDN, works
offline) and **fixed div ids**, so — like the rest of biasguard — the same
inputs render byte-for-byte identically. The replay fingerprint is shown in the
header and the full manifest is embedded and exported alongside the report.
"""

from __future__ import annotations

import html as _html
import json
import math
from calendar import month_abbr
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, select_autoescape
from plotly.offline import get_plotlyjs

from biasguard.analytics.fingerprint import ReplayManifest
from biasguard.analytics.metrics import (
    PerformanceMetrics,
    drawdown_series,
    per_year_breakdown,
)
from biasguard.execution.orders import Trade
from biasguard.execution.profiles import ExecutionProfile
from biasguard.montecarlo.result import MonteCarloResult

_TEMPLATE_TEXT = (
    files("biasguard.report").joinpath("templates/report.html").read_text(encoding="utf-8")
)
_ENV = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
_LINE = "#2b6cb0"
_RED = "#c02535"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return (
        "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.2f}%"
    )


def _ratio(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "∞"
    return f"{x:.2f}"


def _money(x: float) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"${x:,.2f}"


def _sign_cls(x: float) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or x == 0:
        return ""
    return "pos" if x > 0 else "neg"


def _metric_cards(m: PerformanceMetrics) -> list[dict[str, str]]:
    return [
        {"label": "Total return", "value": _pct(m.total_return), "cls": _sign_cls(m.total_return)},
        {"label": "CAGR", "value": _pct(m.cagr), "cls": _sign_cls(m.cagr)},
        {"label": "Sharpe", "value": _ratio(m.sharpe), "cls": _sign_cls(m.sharpe)},
        {"label": "Sortino", "value": _ratio(m.sortino), "cls": _sign_cls(m.sortino)},
        {
            "label": "Max drawdown",
            "value": _pct(m.max_drawdown),
            "cls": "neg" if m.max_drawdown < 0 else "",
        },
        {"label": "Profit factor", "value": _ratio(m.profit_factor), "cls": ""},
        {"label": "Win rate", "value": _pct(m.win_rate), "cls": ""},
        {"label": "Expectancy", "value": _money(m.expectancy), "cls": _sign_cls(m.expectancy)},
        {"label": "Trades", "value": str(m.n_trades), "cls": ""},
        {"label": "Net P&L", "value": _money(m.total_pnl), "cls": _sign_cls(m.total_pnl)},
        {"label": "Final equity", "value": _money(m.final_equity), "cls": ""},
        {"label": "Commission", "value": _money(m.total_commission), "cls": ""},
    ]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _layout(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "margin": {"l": 55, "r": 20, "t": 24, "b": 40},
        "height": 320,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#7a828e"},
        "xaxis": {"gridcolor": "rgba(128,128,128,0.15)"},
        "yaxis": {"gridcolor": "rgba(128,128,128,0.15)"},
        "showlegend": False,
    }
    base.update(kw)
    return base


def _to_div(fig: go.Figure, div_id: str) -> str:
    return str(
        fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config={"displayModeBar": False, "responsive": True},
        )
    )


def _placeholder() -> str:
    return '<div class="sub">Not enough data to plot.</div>'


def _equity_fig(equity: pd.Series) -> str:
    if len(equity) < 2:
        return _placeholder()
    fig = go.Figure(
        go.Scatter(
            x=list(equity.index),
            y=list(equity.to_numpy()),
            mode="lines",
            line={"color": _LINE, "width": 1.5},
        )
    )
    fig.update_layout(
        **_layout(yaxis={"title": "Equity ($)", "gridcolor": "rgba(128,128,128,0.15)"})
    )
    return _to_div(fig, "bg-equity")


def _drawdown_fig(equity: pd.Series) -> str:
    if len(equity) < 2:
        return _placeholder()
    dd = drawdown_series(equity) * 100.0
    fig = go.Figure(
        go.Scatter(
            x=list(dd.index),
            y=list(dd.to_numpy()),
            fill="tozeroy",
            mode="lines",
            line={"color": _RED, "width": 1},
        )
    )
    fig.update_layout(
        **_layout(yaxis={"title": "Drawdown (%)", "gridcolor": "rgba(128,128,128,0.15)"})
    )
    return _to_div(fig, "bg-drawdown")


def _monthly_fig(equity: pd.Series) -> str:
    if len(equity) < 2:
        return _placeholder()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100.0
    if monthly.empty:
        return _placeholder()
    idx = pd.DatetimeIndex(monthly.index)
    frame = pd.DataFrame({"year": idx.year, "month": idx.month, "ret": monthly.to_numpy()})
    pivot = frame.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[month_abbr[m] for m in range(1, 13)],
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar={"title": "%"},
        )
    )
    fig.update_layout(**_layout(height=max(160, 40 + 26 * len(pivot.index))))
    return _to_div(fig, "bg-monthly")


def _trades_fig(trades: Sequence[Trade]) -> str:
    if not trades:
        return _placeholder()
    fig = go.Figure(go.Histogram(x=[t.net_pnl for t in trades], marker_color=_LINE, nbinsx=40))
    fig.update_layout(
        **_layout(
            xaxis={"title": "Trade net P&L ($)", "gridcolor": "rgba(128,128,128,0.15)"},
            yaxis={"title": "Count", "gridcolor": "rgba(128,128,128,0.15)"},
        )
    )
    return _to_div(fig, "bg-trades")


def _js_string_literal(text: str) -> str:
    """A safe JavaScript string literal for embedding text inside ``<script>``.

    JSON-encodes the text (handling quotes, backslashes, and control chars) and
    then escapes the characters that could otherwise break out of a ``<script>``
    element (``<``, ``>``, ``&``), so a prompt containing ``</script>`` or HTML
    cannot inject markup.
    """
    encoded = json.dumps(text)
    return encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _profile_banner(profile: ExecutionProfile | None) -> str:
    """A prominent banner shown only when the profile uses optimistic assumptions."""
    if profile is None or profile.is_realistic:
        return ""
    warns = "".join(f"<li>{_html.escape(w)}</li>" for w in profile.realism.warnings)
    return (
        '<div style="background:#fff3cd;border:1px solid #e0b400;border-left:4px solid #e0b400;'
        'color:#5c4600;border-radius:8px;padding:12px 14px;margin-top:12px">'
        f"<strong>⚠ Simulated execution environment — {_html.escape(profile.name)}.</strong> "
        "Results represent a simulated environment because optimistic execution assumptions "
        f'are enabled:<ul style="margin:6px 0 0 18px">{warns}</ul></div>'
    )


def _profile_html(profile: ExecutionProfile | None) -> str:
    """The 'Execution profile' section body (name, description, assumptions)."""
    if profile is None:
        return "<span class='sub'>No execution profile recorded for this run.</span>"
    badge_color = "#0a7d3c" if profile.is_realistic else "#b8860b"
    badge = "Realistic" if profile.is_realistic else "Simulated / optimistic"
    notes = "".join(f"<li>{_html.escape(n)}</li>" for n in profile.assumptions)
    return (
        f'<div style="font-size:1.05rem;font-weight:600">{_html.escape(profile.name)} '
        f'<span style="color:{badge_color};font-size:0.8rem;font-weight:600">&middot; {badge}</span></div>'
        f'<div class="sub" style="margin:4px 0 8px">{_html.escape(profile.description)}</div>'
        f'<ul style="margin:0 0 0 18px;padding:0">{notes}</ul>'
    )


_MC_BAND = "rgba(43,108,176,0.20)"
_MC_SAMPLE = "rgba(120,130,150,0.16)"


def _montecarlo_fig(mc: MonteCarloResult) -> str:
    """A resampled-equity fan chart: the p5-p95 band, sample paths, and the median."""
    bands = mc.equity_bands
    p50 = bands.get("p50")
    if p50 is None or len(p50) < 2:
        return _placeholder()
    x = list(range(len(p50)))
    fig = go.Figure()
    # p5-p95 band (upper trace first, lower fills up to it).
    fig.add_trace(
        go.Scatter(x=x, y=list(bands["p95"]), mode="lines", line={"width": 0}, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=list(bands["p5"]),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_MC_BAND,
            hoverinfo="skip",
        )
    )
    for i in range(min(len(mc.equity_samples), 25)):  # a handful of raw paths
        fig.add_trace(
            go.Scatter(
                x=x,
                y=list(mc.equity_samples[i]),
                mode="lines",
                line={"color": _MC_SAMPLE, "width": 0.6},
                hoverinfo="skip",
            )
        )
    fig.add_trace(go.Scatter(x=x, y=list(p50), mode="lines", line={"color": _LINE, "width": 1.6}))
    fig.update_layout(
        **_layout(
            xaxis={"title": "Trade #", "gridcolor": "rgba(128,128,128,0.15)"},
            yaxis={"title": "Equity ($)", "gridcolor": "rgba(128,128,128,0.15)"},
        )
    )
    return _to_div(fig, "bg-montecarlo")


def _per_year_rows(equity: pd.Series, trades: Sequence[Trade]) -> list[dict[str, str]]:
    frame = per_year_breakdown(equity, trades)
    rows: list[dict[str, str]] = []
    for year, row in frame.iterrows():
        rows.append(
            {
                "year": str(int(cast(int, year))),
                "pnl": _money(float(row.get("net_pnl", float("nan")))),
                "n": str(int(row.get("n_trades", 0))),
                "ret": (
                    _pct(float(row["return_pct"]))
                    if "return_pct" in row and pd.notna(row.get("return_pct"))
                    else "—"
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_html_report(
    *,
    metrics: PerformanceMetrics,
    equity: pd.Series,
    trades: Sequence[Trade],
    manifest: ReplayManifest | None = None,
    title: str = "biasguard backtest report",
    validation_html: str | None = None,
    profile: ExecutionProfile | None = None,
    disclaimer: str | None = None,
    ai_prompt: str | None = None,
    monte_carlo: MonteCarloResult | None = None,
    path: str | Path | None = None,
) -> str:
    """Render the HTML report; optionally write it (and the manifest) to ``path``.

    A companion ``<path>.manifest.json`` is written next to the report when both
    ``path`` and ``manifest`` are provided. Pass ``ai_prompt`` (typically
    ``build_audit(...).to_ai_prompt()``) to embed a "Download / Copy AI debug
    prompt" control in the report — the text is embedded safely and downloaded
    client-side, so the report stays a single offline file. Pass ``monte_carlo``
    (a :class:`~biasguard.montecarlo.MonteCarloResult`) to add a resampled-equity
    fan chart and the distribution summary.
    """
    from biasguard import __version__

    subtitle_parts = []
    if metrics.start is not None and metrics.end is not None:
        subtitle_parts.append(f"{metrics.start.date()} → {metrics.end.date()}")
    subtitle_parts.append(f"{metrics.n_trades} trades")
    if manifest is not None and manifest.data.get("symbol"):
        subtitle_parts.insert(0, str(manifest.data["symbol"]))

    default_validation = (
        "<span class='sub'>No validation report attached. Run the validation module "
        "(Phase 5) to add a trust verdict here.</span>"
    )

    disclaimer_text = disclaimer
    if disclaimer_text is None and manifest is not None:
        disclaimer_text = manifest.strategy_label

    template = _ENV.from_string(_TEMPLATE_TEXT)
    rendered = template.render(
        title=title,
        subtitle=" · ".join(subtitle_parts),
        disclaimer=disclaimer_text or "",
        framework_version=__version__,
        fingerprint_short=manifest.short if manifest is not None else "n/a",
        fingerprint_full=manifest.fingerprint if manifest is not None else "n/a",
        manifest_json=manifest.to_json() if manifest is not None else "{}",
        metric_cards=_metric_cards(metrics),
        validation_html=validation_html or default_validation,
        profile_banner=_profile_banner(profile),
        profile_html=_profile_html(profile),
        has_ai_prompt=bool(ai_prompt),
        ai_prompt_js=_js_string_literal(ai_prompt) if ai_prompt else '""',
        has_montecarlo=monte_carlo is not None,
        montecarlo_html=monte_carlo.to_html() if monte_carlo is not None else "",
        montecarlo_fig=_montecarlo_fig(monte_carlo) if monte_carlo is not None else "",
        equity_fig=_equity_fig(equity),
        drawdown_fig=_drawdown_fig(equity),
        monthly_fig=_monthly_fig(equity),
        trades_fig=_trades_fig(trades),
        per_year_rows=_per_year_rows(equity, trades),
        plotly_js=get_plotlyjs(),
    )

    if path is not None:
        out = Path(path)
        out.write_text(rendered, encoding="utf-8")
        if manifest is not None:
            out.with_suffix(".manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return rendered


__all__ = ["build_html_report"]
