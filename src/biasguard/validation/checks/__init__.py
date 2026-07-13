"""Built-in integrity checks, and their registration into the default registry."""

from __future__ import annotations

from biasguard.validation.checks.causality import LookaheadCheck
from biasguard.validation.checks.costs import CostIntegrityCheck
from biasguard.validation.checks.data import OHLCIntegrityCheck
from biasguard.validation.checks.fills import FillRealismCheck
from biasguard.validation.checks.montecarlo import MonteCarloCheck
from biasguard.validation.checks.robustness import SlippageSensitivityCheck
from biasguard.validation.checks.statistics import (
    OutOfSampleCheck,
    RegimeConcentrationCheck,
)
from biasguard.validation.registry import IntegrityRegistry, register_check

#: The built-in checks, in a sensible report order (data -> causality -> ...).
BUILTIN_CHECKS = (
    OHLCIntegrityCheck,
    LookaheadCheck,
    CostIntegrityCheck,
    FillRealismCheck,
    SlippageSensitivityCheck,
    RegimeConcentrationCheck,
    OutOfSampleCheck,
    MonteCarloCheck,
)


def register_builtins(registry: IntegrityRegistry | None = None) -> None:
    """Register (or re-register) all built-in checks."""
    for check_cls in BUILTIN_CHECKS:
        register_check(check_cls(), registry=registry)


register_builtins()  # populate DEFAULT_REGISTRY on import


__all__ = [
    "BUILTIN_CHECKS",
    "CostIntegrityCheck",
    "FillRealismCheck",
    "LookaheadCheck",
    "MonteCarloCheck",
    "OHLCIntegrityCheck",
    "OutOfSampleCheck",
    "RegimeConcentrationCheck",
    "SlippageSensitivityCheck",
    "register_builtins",
]
