"""Deterministic 'what to investigate next' targets from an integrity report.

Each failing or warning integrity check maps to a concrete diagnostic action —
not a fix. The mapping is a plain dict keyed by ``check.key`` so it is fully
deterministic and easy to extend when a new check is added. This is what makes
the AI prompt actionable: it tells the assistant *where to look*, grounded in
which specific controls flagged.
"""

from __future__ import annotations

from biasguard.validation.report import IntegrityReport, Status

#: check.key -> a concrete investigation action for a FAIL/WARN on that check.
_ACTION_BY_KEY: dict[str, str] = {
    "lookahead": (
        "Run the truncation test manually: does on_bar read the current bar's "
        "close/high/low or any future bar? A decision must depend only on bars "
        "strictly before the one it acts on."
    ),
    "fill_realism": (
        "Re-run under TradeThroughFill vs TouchFill and compare net P&L. A large "
        "gap means profit is coming from optimistic fill mechanics, not alpha."
    ),
    "costs": (
        "Enable realistic commission and slippage (a non-zero ExecutionProfile). "
        "A zero-cost run overstates returns, especially for high-frequency signals."
    ),
    "slippage_sensitivity": (
        "Sweep slippage upward (e.g. 1 -> 2 -> 3 ticks) and watch how fast the edge "
        "decays. An edge that dies within a tick or two is fill-fragile."
    ),
    "regime": (
        "Break P&L down by year and by regime. If a few days/months carry most of "
        "the net, the result is a concentrated bet, not a stable edge."
    ),
    "out_of_sample": (
        "Compare in-sample vs out-of-sample metrics on the same split. A profit that "
        "only exists in-sample is a sign of overfitting or a look-back in tuning."
    ),
    "monte_carlo": (
        "Inspect the block-bootstrap distribution: if P(profit) under resampling is "
        "low, the historical gain is likely luck of trade sequence, not a repeatable edge."
    ),
    "ohlc": (
        "Fix the underlying data: check for OHLC violations, duplicate timestamps, "
        "timezone mislabeling, or gaps before trusting any downstream number."
    ),
}

_PROFILE_ACTION = (
    "Re-run under the Real Market execution profile. The active profile uses "
    "optimistic assumptions, so these results describe a simulated environment."
)


def investigation_targets(
    report: IntegrityReport, *, profile_realistic: bool | None = None
) -> tuple[str, ...]:
    """Ordered, de-duplicated diagnostic actions for a report's flagged checks.

    ``report.results`` is already ordered most-severe first, so the returned
    targets lead with the checks that most threaten trust. ``profile_realistic``,
    when ``False``, appends a target to re-run under a realistic profile.
    """
    targets: list[str] = []
    for result in report.results:
        if result.status in (Status.FAIL, Status.WARN):
            action = _ACTION_BY_KEY.get(result.key)
            targets.append(action if action else f"Review '{result.name}': {result.summary}")
    if profile_realistic is False:
        targets.append(_PROFILE_ACTION)
    # De-duplicate while preserving order (two WARN/FAILs could share an action).
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            ordered.append(target)
    return tuple(ordered)


__all__ = ["investigation_targets"]
