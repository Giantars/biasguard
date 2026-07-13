"""The integrity-check registry and the top-level ``assess_integrity`` driver.

The registry is the plugin surface: register a check once and every call to
:func:`assess_integrity` runs it. Each check is sandboxed in ``try/except`` so a
single buggy plugin degrades to a SKIP rather than crashing the assessment.
"""

from __future__ import annotations

import pandas as pd

from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import BacktestSpec, IntegrityContext
from biasguard.validation.report import CheckResult, IntegrityReport, Status


class IntegrityRegistry:
    """An ordered collection of integrity checks keyed by ``check.key``."""

    def __init__(self) -> None:
        self._checks: dict[str, IntegrityCheck] = {}

    def register(self, check: IntegrityCheck, *, replace: bool = True) -> IntegrityCheck:
        if check.key in self._checks and not replace:
            raise ValueError(f"a check with key {check.key!r} is already registered")
        self._checks[check.key] = check
        return check

    def unregister(self, key: str) -> None:
        self._checks.pop(key, None)

    def get(self, key: str) -> IntegrityCheck | None:
        return self._checks.get(key)

    def checks(self) -> list[IntegrityCheck]:
        return list(self._checks.values())

    def copy(self) -> IntegrityRegistry:
        clone = IntegrityRegistry()
        clone._checks = dict(self._checks)
        return clone


#: The registry the built-in checks register into and ``assess_integrity`` uses.
DEFAULT_REGISTRY = IntegrityRegistry()


def register_check(
    check: IntegrityCheck, *, registry: IntegrityRegistry | None = None
) -> IntegrityCheck:
    """Register a check (into ``DEFAULT_REGISTRY`` unless another is given)."""
    return (registry or DEFAULT_REGISTRY).register(check)


def assess_integrity(
    spec: BacktestSpec,
    *,
    registry: IntegrityRegistry | None = None,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    seed: int = 12345,
    oos_cut: pd.Timestamp | None = None,
    config: dict[str, object] | None = None,
) -> IntegrityReport:
    """Run every applicable registered check and aggregate the Integrity Score.

    Parameters
    ----------
    spec:
        The reproducible backtest recipe to assess.
    include / exclude:
        Optional sets of check keys to restrict / drop.
    seed:
        Seed for stochastic checks (e.g. the random-direction null), so the
        whole assessment is reproducible.
    oos_cut:
        Explicit in-/out-of-sample cut timestamp (defaults to a time-based split).
    """
    reg = registry or DEFAULT_REGISTRY
    ctx = IntegrityContext.build(spec, seed=seed, oos_cut=oos_cut, config=config)

    results: list[CheckResult] = []
    for check in reg.checks():
        if include is not None and check.key not in include:
            continue
        if exclude is not None and check.key in exclude:
            continue
        try:
            if not check.applicable(ctx):
                results.append(check.skip("not applicable to this run"))
                continue
            results.append(check.run(ctx))
        except Exception as exc:  # a buggy plugin must not sink the whole report
            results.append(
                CheckResult(
                    key=check.key,
                    name=check.name,
                    category=check.category,
                    status=Status.SKIP,
                    score=0.0,
                    summary="check raised an error and was skipped",
                    detail=f"{type(exc).__name__}: {exc}",
                    is_gate=check.is_gate,
                    weight=check.weight,
                )
            )
    return IntegrityReport.build(tuple(results))


__all__ = [
    "DEFAULT_REGISTRY",
    "IntegrityRegistry",
    "assess_integrity",
    "register_check",
]
