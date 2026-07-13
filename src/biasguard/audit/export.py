"""AI audit export — turn an integrity assessment into paste-ready LLM context.

:class:`AuditExport` gathers a run's deterministic context (replay fingerprint,
execution profile, integrity report, key metrics, Monte Carlo summary) and emits
three artifacts:

* ``audit_report.json`` — the full structured record;
* ``audit_report.md`` — a human-readable audit;
* ``ai_debug_prompt.txt`` — a ready-to-paste prompt for Claude / ChatGPT / any LLM.

Everything is deterministic: no timestamps, no randomness, so the same run
produces byte-identical files (matching the replay-fingerprint philosophy). The
export **describes and asks** — it never proposes a fix; that is the assistant's
job once the user pastes the prompt.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from biasguard.analytics.fingerprint import ReplayManifest
from biasguard.analytics.metrics import PerformanceMetrics
from biasguard.audit.targets import investigation_targets
from biasguard.execution.profiles import ExecutionProfile
from biasguard.montecarlo.result import MonteCarloResult
from biasguard.validation.report import IntegrityReport, Status

MARKDOWN_FILENAME = "audit_report.md"
JSON_FILENAME = "audit_report.json"
PROMPT_FILENAME = "ai_debug_prompt.txt"


# --------------------------------------------------------------------------- #
# Formatting (plain-text, deterministic)
# --------------------------------------------------------------------------- #
def _money(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"${x:,.2f}"


def _pct(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def _ratio(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:.2f}"


def _prob(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x * 100:.0f}%"


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value into strict, deterministic JSON.

    ``CheckResult.metrics`` is an untyped ``dict[str, Any]`` populated through the
    public plugin API, so a custom check can put *any* value there. This mirrors
    the fingerprint's canonicalizer: non-finite floats become strings, numpy
    scalars/arrays collapse to primitives, timestamps/sets/Decimals get a
    deterministic representation, and anything else falls back to ``str`` — so
    ``json.dumps(..., allow_nan=False)`` can never raise.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (set, frozenset)):
        # Sort canonicalized elements so set order (hash-seed dependent) never
        # changes the output.
        return sorted(
            (_json_safe(v) for v in value),
            key=lambda x: json.dumps(x, sort_keys=True, default=str),
        )
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, (pd.Timestamp, _dt.date)):  # datetime is a subclass of date
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, Decimal):
        return _json_safe(float(value))
    return str(value)


# --------------------------------------------------------------------------- #
# Untrusted-text neutralization (prompt structure & markdown tables)
# --------------------------------------------------------------------------- #
_DELIMITER_RE = re.compile(r"={3,}")


def _one_line(text: Any) -> str:
    """Collapse newlines so untrusted text can never span or forge new lines."""
    return " ".join(str(text).splitlines()).strip()


def _prompt_safe(text: Any) -> str:
    """Neutralize untrusted text for the AI prompt.

    Collapses newlines and defangs the ``=== SECTION ===`` delimiter (``===`` ->
    ``==``) so a strategy name, profile field, or check summary cannot forge its
    own prompt section (e.g. a fake ``=== YOUR TASK ===`` block) and hijack the
    instructions. The prompt's section structure stays framework-controlled.
    """
    return _DELIMITER_RE.sub("==", _one_line(text))


def _md_cell(text: Any) -> str:
    """Escape a value for a markdown table cell: single line, ``|`` escaped."""
    return _one_line(text).replace("|", "\\|")


@dataclass(frozen=True)
class AuditExport:
    """The deterministic audit context for a single backtest run."""

    integrity: IntegrityReport
    title: str = "biasguard strategy audit"
    manifest: ReplayManifest | None = None
    profile: ExecutionProfile | None = None
    metrics: PerformanceMetrics | None = None
    monte_carlo: MonteCarloResult | None = None

    # -- derived views ---------------------------------------------------- #
    @property
    def fingerprint(self) -> str:
        return self.manifest.short if self.manifest is not None else "n/a"

    def targets(self) -> tuple[str, ...]:
        """Deterministic investigation actions for the flagged checks."""
        realistic = self.profile.is_realistic if self.profile is not None else None
        return investigation_targets(self.integrity, profile_realistic=realistic)

    def warnings(self) -> tuple[str, ...]:
        """Consolidated caveats: profile realism + every non-passing check."""
        out: list[str] = []
        if self.profile is not None:
            out.extend(self.profile.realism.warnings)
        for result in self.integrity.results:
            if result.status in (Status.FAIL, Status.WARN):
                out.append(f"[{result.status}] {result.name}: {result.summary}")
        return tuple(out)

    def _mc_view(self) -> dict[str, Any] | None:
        """Normalize a Monte Carlo summary from an explicit result or the report."""
        if self.monte_carlo is not None:
            mc = self.monte_carlo
            fp = mc.final_pnl_percentiles
            view = {
                "source": "monte_carlo_result",
                "bootstrap": mc.bootstrap_name,
                "n_paths": mc.n_paths,
                "prob_profit": mc.prob_profit,
                "final_p5": fp["p5"],
                "final_p50": fp["p50"],
                "final_p95": fp["p95"],
                "worst_case_drawdown": mc.worst_case_drawdown,
            }
            if mc.account.has_limits:
                view["prob_breach"] = mc.prob_breach
            return view
        check = self.integrity.get("monte_carlo")
        if check is None or check.status is Status.SKIP or not check.metrics:
            return None
        m = check.metrics
        view = {
            "source": "integrity_check",
            "bootstrap": m.get("bootstrap", "block bootstrap"),
            "prob_profit": m.get("prob_profit"),
            "final_p5": m.get("final_p5"),
            "final_p50": m.get("final_p50"),
            "final_p95": m.get("final_p95"),
            "worst_case_drawdown": m.get("worst_case_drawdown"),
        }
        if "prob_breach" in m:
            view["prob_breach"] = m["prob_breach"]
        return view

    def _metric_rows(self) -> list[tuple[str, str]]:
        m = self.metrics
        if m is None:
            return []
        return [
            ("Net P&L", _money(m.total_pnl)),
            ("Total return", _pct(m.total_return)),
            ("CAGR", _pct(m.cagr)),
            ("Sharpe", _ratio(m.sharpe)),
            ("Sortino", _ratio(m.sortino)),
            ("Max drawdown", _pct(m.max_drawdown)),
            ("Profit factor", _ratio(m.profit_factor)),
            ("Win rate", _pct(m.win_rate)),
            ("Expectancy", _money(m.expectancy)),
            ("Trades", str(m.n_trades)),
            ("Final equity", _money(m.final_equity)),
            ("Commission paid", _money(m.total_commission)),
        ]

    # -- structured output ------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """The full structured audit record (JSON-safe)."""
        payload: dict[str, Any] = {
            "title": self.title,
            "framework_version": (
                self.manifest.framework_version if self.manifest is not None else None
            ),
            "fingerprint": self.fingerprint,
            "replay_manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "execution_profile": self.profile.describe() if self.profile is not None else None,
            "metrics": dict(self._metric_rows()),
            "integrity": self.integrity.to_dict(),
            "monte_carlo": self._mc_view(),
            "warnings": list(self.warnings()),
            "investigation_targets": list(self.targets()),
        }
        return cast(dict[str, Any], _json_safe(payload))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=True, allow_nan=False
        )

    # -- markdown --------------------------------------------------------- #
    def to_markdown(self) -> str:
        r = self.integrity
        score_txt = "—" if math.isnan(r.score) else f"{r.score:.0f}"
        lines: list[str] = [f"# {self.title}", ""]
        lines.append(f"**Replay fingerprint:** `{self.fingerprint}`  ")
        if self.manifest is not None:
            strat = _one_line(self.manifest.strategy.get("identity", "unknown"))
            lines.append(f"**Strategy:** `{strat}`  ")
            if self.manifest.strategy_label:
                lines.append(f"**Disclaimer:** {_one_line(self.manifest.strategy_label)}  ")
        lines.append(
            f"**Integrity score:** {score_txt}/100 ({r.grade} — {r.grade_label}) &middot; "
            f"trustworthy: {'yes' if r.trustworthy else 'no'}"
        )
        lines.append("")

        if self.profile is not None:
            p = self.profile
            lines.append("## Execution profile")
            lines.append(f"**{_one_line(p.name)}** — {_one_line(p.description)}")
            lines.append("")
            lines.append(f"> {_one_line(p.realism.banner)}")
            lines.append("")
            for note in p.assumptions:
                lines.append(f"- {_one_line(note)}")
            lines.append("")

        rows = self._metric_rows()
        if rows:
            lines.append("## Key metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for label, value in rows:
                lines.append(f"| {_md_cell(label)} | {_md_cell(value)} |")
            lines.append("")

        lines.append("## Validation results")
        lines.append("")
        lines.append("| Status | Check | Summary |")
        lines.append("| --- | --- | --- |")
        for result in r.results:
            lines.append(
                f"| {result.status} | {_md_cell(result.name)} | {_md_cell(result.summary)} |"
            )
        lines.append("")

        mc = self._mc_view()
        if mc is not None:
            lines.append("## Monte Carlo")
            lines.append("")
            for text in _mc_lines(mc):
                lines.append(f"- {text}")
            lines.append("")

        warnings = self.warnings()
        if warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {_one_line(warning)}")
            lines.append("")

        targets = self.targets()
        if targets:
            lines.append("## Recommended investigation targets")
            lines.append("")
            for i, target in enumerate(targets, 1):
                lines.append(f"{i}. {_one_line(target)}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # -- AI prompt -------------------------------------------------------- #
    def to_ai_prompt(self) -> str:
        r = self.integrity
        score_txt = "—" if math.isnan(r.score) else f"{r.score:.0f}"
        lines: list[str] = [
            "You are a skeptical quantitative backtesting auditor. A user ran a trading "
            "strategy through biasguard, a bias-resistant backtesting framework whose job "
            "is to catch the mistakes that make backtests lie (lookahead, optimistic fills, "
            "missing costs, overfitting, sequence luck). Below is the deterministic audit "
            "context for one run. Do NOT assume the strategy has a real edge.",
            "",
            "=== RUN IDENTITY ===",
            f"Replay fingerprint: {self.fingerprint}",
        ]
        if self.manifest is not None:
            lines.append(f"Framework: biasguard v{self.manifest.framework_version}")
            lines.append(
                f"Strategy: {_prompt_safe(self.manifest.strategy.get('identity', 'unknown'))}"
            )
            if self.manifest.strategy_label:
                lines.append(f"Disclaimer: {_prompt_safe(self.manifest.strategy_label)}")
            params = self.manifest.strategy.get("params", {})
            if params:
                lines.append(f"Strategy params: {_prompt_safe(json.dumps(params, sort_keys=True))}")
            data_meta = self.manifest.data
            span = f"{data_meta.get('start')} -> {data_meta.get('end')}"
            lines.append(
                f"Data: {_prompt_safe(data_meta.get('symbol') or 'n/a')}, "
                f"{data_meta.get('n_bars')} bars, {_prompt_safe(span)}"
            )

        lines.append("")
        lines.append("=== EXECUTION PROFILE ===")
        if self.profile is not None:
            p = self.profile
            lines.append(f"Profile: {_prompt_safe(p.name)} — {_prompt_safe(p.realism.banner)}")
            lines.append(f"Description: {_prompt_safe(p.description)}")
            for note in p.assumptions:
                lines.append(f"  - {_prompt_safe(note)}")
        else:
            lines.append("No execution profile recorded.")

        rows = self._metric_rows()
        if rows:
            lines.append("")
            lines.append("=== KEY METRICS ===")
            for label, value in rows:
                lines.append(f"{label}: {value}")

        lines.append("")
        lines.append("=== INTEGRITY VERDICT ===")
        lines.append(
            f"Score {score_txt}/100 ({r.grade} — {r.grade_label}); "
            f"trustworthy: {'yes' if r.trustworthy else 'no'}; "
            f"gates passed: {'yes' if r.gates_passed else 'no'}"
        )
        lines.append("")
        lines.append("=== VALIDATION RESULTS (per check) ===")
        for result in r.results:
            gate = " [GATE]" if result.is_gate else ""
            lines.append(
                f"[{result.status}]{gate} {_prompt_safe(result.name)}: {_prompt_safe(result.summary)}"
            )
            if result.detail:
                lines.append(f"    detail: {_prompt_safe(result.detail)}")

        mc = self._mc_view()
        if mc is not None:
            lines.append("")
            lines.append("=== MONTE CARLO (block-bootstrap resampling) ===")
            for text in _mc_lines(mc):
                lines.append(text)

        warnings = self.warnings()
        if warnings:
            lines.append("")
            lines.append("=== WARNINGS ===")
            for warning in warnings:
                lines.append(f"- {_prompt_safe(warning)}")

        targets = self.targets()
        if targets:
            lines.append("")
            lines.append("=== RECOMMENDED INVESTIGATION TARGETS ===")
            for i, target in enumerate(targets, 1):
                lines.append(f"{i}. {_prompt_safe(target)}")

        lines.append("")
        lines.append("=== YOUR TASK ===")
        lines.append(
            "Using only the context above, help the user judge how much to trust this "
            "backtest. Specifically:\n"
            "1. Which findings most threaten the result, and why?\n"
            "2. What are the most likely root causes in the strategy logic or its "
            "configuration (be concrete about what code or setting to inspect)?\n"
            "3. What diagnostic steps should the user run next to confirm or rule out each "
            "issue?\n"
            "Be specific and skeptical. Do not claim the strategy is profitable in live "
            "trading. If the evidence is insufficient to judge, say so and state what is "
            "missing."
        )
        return "\n".join(lines).rstrip() + "\n"

    # -- file output ------------------------------------------------------ #
    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write the three artifacts into ``directory`` (created if needed)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out / JSON_FILENAME,
            "markdown": out / MARKDOWN_FILENAME,
            "prompt": out / PROMPT_FILENAME,
        }
        paths["json"].write_text(self.to_json(), encoding="utf-8")
        paths["markdown"].write_text(self.to_markdown(), encoding="utf-8")
        paths["prompt"].write_text(self.to_ai_prompt(), encoding="utf-8")
        return paths


def _mc_lines(mc: dict[str, Any]) -> list[str]:
    """Human-readable Monte Carlo lines from a normalized view."""
    lines = [
        f"Bootstrap: {mc.get('bootstrap', 'n/a')}"
        + (f" ({mc['n_paths']} paths)" if mc.get("n_paths") else ""),
        f"P(profit) under resampling: {_prob(mc.get('prob_profit'))}",
        f"Final P&L p5 / p50 / p95: {_money(mc.get('final_p5'))} / "
        f"{_money(mc.get('final_p50'))} / {_money(mc.get('final_p95'))}",
        f"Worst-case drawdown (p95): {_money(mc.get('worst_case_drawdown'))}",
    ]
    if "prob_breach" in mc:
        lines.append(f"P(breach account limit): {_prob(mc.get('prob_breach'))}")
    return lines


def build_audit(
    integrity: IntegrityReport,
    *,
    title: str = "biasguard strategy audit",
    manifest: ReplayManifest | None = None,
    profile: ExecutionProfile | None = None,
    metrics: PerformanceMetrics | None = None,
    monte_carlo: MonteCarloResult | None = None,
) -> AuditExport:
    """Assemble an :class:`AuditExport` from the pieces of a run."""
    return AuditExport(
        integrity=integrity,
        title=title,
        manifest=manifest,
        profile=profile,
        metrics=metrics,
        monte_carlo=monte_carlo,
    )


__all__ = [
    "JSON_FILENAME",
    "MARKDOWN_FILENAME",
    "PROMPT_FILENAME",
    "AuditExport",
    "build_audit",
]
