"""Integrity result types and the aggregate Integrity Score.

A backtest ships with a *trust verdict*, not just a Sharpe. Each check emits a
:class:`CheckResult` with a status **and** a score in [0, 1]; the
:class:`IntegrityReport` aggregates those into a single 0-100 Integrity Score
plus a plain-English grade that answers "how much should you trust this?"
"""

from __future__ import annotations

import html as _html
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: A failed *gate* check (e.g. lookahead) caps the whole score here, however
#: good everything else looks — a causality leak means "do not trust".
GATE_FAIL_CAP = 25.0
#: A gate that did not actually run (SKIP/crash) caps the score here: the
#: decisive causality control was never verified, so we cannot certify high
#: integrity — "couldn't check" is not the same as "passed".
GATE_SKIP_CAP = 60.0


class Status(Enum):
    """The verdict of a single integrity check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"  # not applicable / insufficient data

    @property
    def rank(self) -> int:
        return {"SKIP": 0, "PASS": 1, "WARN": 2, "FAIL": 3}[self.value]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One integrity check's structured verdict."""

    key: str
    name: str
    category: str
    status: Status
    score: float  # 0 (untrustworthy) .. 1 (fully trustworthy on this axis)
    summary: str
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    is_gate: bool = False
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "score": self.score,
            "summary": self.summary,
            "detail": self.detail,
            "metrics": self.metrics,
            "is_gate": self.is_gate,
            "weight": self.weight,
        }


_GRADE_BANDS: tuple[tuple[float, str, str], ...] = (
    (90.0, "A", "High integrity"),
    (75.0, "B", "Trustworthy"),
    (60.0, "C", "Moderate — review the flags"),
    (40.0, "D", "Low — significant concerns"),
    (0.0, "F", "Do not trust"),
)


def grade_for(score: float) -> tuple[str, str]:
    """Map a 0-100 score to a (letter, label) grade."""
    if math.isnan(score):
        return ("N/A", "not assessed")
    for threshold, letter, label in _GRADE_BANDS:
        if score >= threshold:
            return (letter, label)
    return ("F", "Do not trust")


def aggregate_score(results: tuple[CheckResult, ...]) -> float:
    """Weighted-mean score of non-skipped checks, capped by gate outcomes.

    A gate that FAILs caps the score at :data:`GATE_FAIL_CAP`. A gate that did
    not run at all (SKIP or crashed-to-SKIP) caps it at :data:`GATE_SKIP_CAP` —
    the decisive control was never verified, so the run cannot be certified as
    high-integrity even if every other check passed.
    """
    scored = [r for r in results if r.status is not Status.SKIP]
    total_w = sum(r.weight for r in scored)
    if not scored or total_w <= 0:
        return float("nan")
    base = sum(r.score * r.weight for r in scored) / total_w * 100.0
    gates = [r for r in results if r.is_gate]  # includes SKIPped gates
    if any(g.status is Status.FAIL for g in gates):
        base = min(base, GATE_FAIL_CAP)
    elif any(g.status is not Status.PASS for g in gates):
        base = min(base, GATE_SKIP_CAP)
    return base


@dataclass(frozen=True)
class IntegrityReport:
    """The aggregate trust verdict for a backtest."""

    results: tuple[CheckResult, ...]
    score: float
    grade: str
    grade_label: str

    @classmethod
    def build(cls, results: tuple[CheckResult, ...]) -> IntegrityReport:
        score = aggregate_score(results)
        letter, label = grade_for(score)
        ordered = tuple(sorted(results, key=lambda r: (-r.status.rank, -r.weight, r.key)))
        return cls(results=ordered, score=score, grade=letter, grade_label=label)

    def get(self, key: str) -> CheckResult | None:
        return next((r for r in self.results if r.key == key), None)

    def by_status(self, status: Status) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status is status)

    @property
    def fails(self) -> tuple[CheckResult, ...]:
        return self.by_status(Status.FAIL)

    @property
    def warns(self) -> tuple[CheckResult, ...]:
        return self.by_status(Status.WARN)

    @property
    def gates_passed(self) -> bool:
        """True when every gate check actually PASSed (vacuously true if none)."""
        return all(r.status is Status.PASS for r in self.results if r.is_gate)

    @property
    def trustworthy(self) -> bool:
        """No FAILs, all gates passed, and a passing grade (C or better)."""
        return (
            not self.fails
            and self.gates_passed
            and not math.isnan(self.score)
            and self.score >= 60.0
        )

    def summary(self) -> str:
        score_txt = "—" if math.isnan(self.score) else f"{self.score:.0f}"
        head = (
            f"Integrity score: {score_txt}/100 ({self.grade} — {self.grade_label}). "
            f"{len(self.fails)} fail, {len(self.warns)} warn, "
            f"{len(self.by_status(Status.PASS))} pass, {len(self.by_status(Status.SKIP))} skip."
        )
        lines = [head]
        for r in self.results:
            lines.append(f"  [{r.status}] {r.name}: {r.summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "grade_label": self.grade_label,
            "trustworthy": self.trustworthy,
            "results": [r.to_dict() for r in self.results],
        }

    def to_html(self) -> str:
        """A compact HTML fragment for the report's validation-verdict slot."""
        colors = {
            Status.PASS: "#0a7d3c",
            Status.WARN: "#b8860b",
            Status.FAIL: "#c02535",
            Status.SKIP: "#8a919c",
        }
        score_txt = "—" if math.isnan(self.score) else f"{self.score:.0f}/100"
        rows = []
        for r in self.results:
            c = colors[r.status]
            rows.append(
                f'<tr><td><span style="color:{c};font-weight:600">{_html.escape(str(r.status))}</span></td>'
                f"<td>{_html.escape(r.name)}</td><td>{_html.escape(r.summary)}</td></tr>"
            )
        return (
            f'<div style="font-size:1.1rem;font-weight:600;margin-bottom:6px">'
            f"Integrity {score_txt} &middot; {_html.escape(self.grade)} — {_html.escape(self.grade_label)}</div>"
            f'<table style="width:100%;border-collapse:collapse">{"".join(rows)}</table>'
        )


__all__ = [
    "GATE_FAIL_CAP",
    "GATE_SKIP_CAP",
    "CheckResult",
    "IntegrityReport",
    "Status",
    "aggregate_score",
    "grade_for",
]
