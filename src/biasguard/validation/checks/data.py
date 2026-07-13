"""Data integrity: is the input even valid?"""

from __future__ import annotations

from biasguard.data.validation import validate_ohlcv
from biasguard.validation.base import IntegrityCheck
from biasguard.validation.context import IntegrityContext
from biasguard.validation.report import CheckResult, Status


class OHLCIntegrityCheck(IntegrityCheck):
    key = "ohlc"
    name = "OHLC & timezone integrity"
    category = "data"
    weight = 1.0

    def run(self, ctx: IntegrityContext) -> CheckResult:
        report = validate_ohlcv(ctx.spec.data)
        n_err = len(report.errors)
        n_warn = len(report.warnings)
        if n_err:
            return self.result(
                Status.FAIL,
                0.0,
                f"{n_err} data error(s) — results are not trustworthy",
                detail=report.summary(),
                errors=n_err,
                warnings=n_warn,
            )
        if n_warn:
            return self.result(
                Status.WARN,
                0.8,
                f"{n_warn} data warning(s) (e.g. gaps / missing volume)",
                detail=report.summary(),
                warnings=n_warn,
            )
        return self.result(Status.PASS, 1.0, "data passes OHLC and timezone validation")


__all__ = ["OHLCIntegrityCheck"]
