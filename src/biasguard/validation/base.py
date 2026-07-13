"""The IntegrityCheck plugin interface.

Subclass, set the class attributes, implement :meth:`run`, and register the
instance with :func:`biasguard.validation.registry.register_check`. The engine
never needs to change to gain a new check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class IntegrityCheck(ABC):
    """Base class for a pluggable backtest-integrity check."""

    #: Unique identifier (kebab/snake case), e.g. ``"lookahead"``.
    key: str = "unnamed"
    #: Human-readable name shown in reports.
    name: str = "Unnamed check"
    #: One of ``"data" | "causality" | "execution" | "statistics" | "robustness"``.
    category: str = "other"
    #: A failing gate check caps the whole Integrity Score (see ``GATE_FAIL_CAP``).
    is_gate: bool = False
    #: Relative weight in the aggregate score.
    weight: float = 1.0

    def applicable(self, ctx: IntegrityContext) -> bool:
        """Whether this check can run given the available inputs (default: yes)."""
        return True

    @abstractmethod
    def run(self, ctx: IntegrityContext) -> CheckResult:
        """Analyze the run and return a structured verdict."""
        raise NotImplementedError

    # -- helpers for subclasses ------------------------------------------- #
    def result(
        self,
        status: Status,
        score: float,
        summary: str,
        *,
        detail: str = "",
        **metrics: Any,
    ) -> CheckResult:
        return CheckResult(
            key=self.key,
            name=self.name,
            category=self.category,
            status=status,
            score=max(0.0, min(1.0, score)),
            summary=summary,
            detail=detail,
            metrics=dict(metrics),
            is_gate=self.is_gate,
            weight=self.weight,
        )

    def skip(self, summary: str, **metrics: Any) -> CheckResult:
        return self.result(Status.SKIP, 0.0, summary, **metrics)


__all__ = ["IntegrityCheck"]
