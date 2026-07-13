"""The Backtest Integrity Framework — biasguard's headline differentiator.

Instead of only saying "your strategy made money", it answers *"how much should
you trust these results?"* — a 0-100 Integrity Score aggregated from a pluggable
set of integrity checks (lookahead, fill realism, costs, regime concentration,
slippage sensitivity, in-/out-of-sample, ...).

Usage::

    from biasguard.validation import BacktestSpec, assess_integrity

    spec = BacktestSpec(data=df, strategy_factory=MyStrategy, instrument=NQ,
                        commission=PerContractCommission(1.90), slippage=TickSlippage(1))
    report = assess_integrity(spec)
    print(report.summary())

Add your own check without touching the engine::

    from biasguard.validation import IntegrityCheck, register_check
    class MyCheck(IntegrityCheck): ...
    register_check(MyCheck())
"""

from __future__ import annotations

# Importing the checks subpackage registers the built-ins into DEFAULT_REGISTRY.
from biasguard.validation import checks as checks
from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import BacktestSpec, IntegrityContext, RunOutput
from biasguard.validation.registry import (
    DEFAULT_REGISTRY,
    IntegrityRegistry,
    assess_integrity,
    register_check,
)
from biasguard.validation.report import (
    GATE_FAIL_CAP,
    CheckResult,
    IntegrityReport,
    Status,
    aggregate_score,
    grade_for,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "GATE_FAIL_CAP",
    "BacktestSpec",
    "CheckResult",
    "IntegrityCheck",
    "IntegrityContext",
    "IntegrityRegistry",
    "IntegrityReport",
    "RunOutput",
    "Status",
    "aggregate_score",
    "assess_integrity",
    "checks",
    "grade_for",
    "register_check",
]
