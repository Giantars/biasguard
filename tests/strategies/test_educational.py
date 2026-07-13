"""Tests for the educational strategy catalog.

Each case documents the integrity verdict it is meant to demonstrate; these tests
pin those verdicts so the examples never silently drift, and confirm every
strategy is labelled and (for the honest ones) genuinely causal.
"""

from __future__ import annotations

import pytest

from biasguard.strategies import CATALOG, EDUCATIONAL_LABEL, ExampleCase
from biasguard.strategies.educational import (
    FreeLunchChurn,
    LookaheadStrategy,
    MovingAverageCrossover,
    OverfitMeanReversion,
    RsiMeanReversion,
    ZeroSlippageScalper,
)
from biasguard.validation import BacktestSpec, assess_integrity

ALL_STRATEGIES = [
    MovingAverageCrossover,
    RsiMeanReversion,
    LookaheadStrategy,
    OverfitMeanReversion,
    ZeroSlippageScalper,
    FreeLunchChurn,
]


#: Assessing a case is expensive (fill-realism replays dozens of null runs), and
#: several tests reuse the same case, so memoize by key — each is assessed once.
_ASSESSED: dict[str, tuple[object, object]] = {}


def _assess(case: ExampleCase) -> tuple[object, object]:
    if case.key not in _ASSESSED:
        spec = BacktestSpec.from_profile(
            data=case.make_data(),
            strategy_factory=case.make_strategy,
            instrument=case.instrument,
            profile=case.profile,
        )
        _ASSESSED[case.key] = (spec.run(), assess_integrity(spec))
    return _ASSESSED[case.key]


@pytest.mark.parametrize("case", CATALOG, ids=lambda c: c.key)
def test_watched_check_matches_expected(case: ExampleCase) -> None:
    _, report = _assess(case)
    watched = report.get(case.watch)
    assert watched is not None, f"{case.key}: watched check {case.watch!r} did not run"
    assert watched.status.value == case.expected_status, (
        f"{case.key}: {case.watch} was {watched.status} ({watched.summary}), "
        f"expected {case.expected_status}"
    )


def test_honest_examples_pass_the_lookahead_gate() -> None:
    for key in ("ma_crossover", "rsi_reversion"):
        case = next(c for c in CATALOG if c.key == key)
        _, report = _assess(case)
        assert report.gates_passed, f"{key} should be causal (gate must pass)"


def test_lookahead_example_fails_the_gate_and_caps_score() -> None:
    case = next(c for c in CATALOG if c.key == "lookahead")
    _, report = _assess(case)
    assert not report.gates_passed
    assert report.score <= 25.0  # a gate failure caps the whole score


def test_fail_examples_are_not_trustworthy() -> None:
    # A FAIL (lookahead gate, unmodelled costs) disqualifies trust outright.
    for key in ("lookahead", "free_lunch"):
        case = next(c for c in CATALOG if c.key == key)
        _, report = _assess(case)
        assert report.fails, f"{key} should have a FAIL"
        assert not report.trustworthy, f"{key} should not be trustworthy"


def test_warn_examples_are_flagged_but_not_top_grade() -> None:
    # A WARN (overfit concentration, thin slippage edge) is a caution, not a
    # disqualification: the run can still be "trustworthy" (grade C+), but it is
    # visibly flagged and never earns an A.
    for key in ("overfit", "zero_slippage"):
        case = next(c for c in CATALOG if c.key == key)
        _, report = _assess(case)
        assert report.warns, f"{key} should raise a WARN"
        assert report.grade != "A", f"{key} should not earn a top grade"


def test_every_strategy_carries_the_educational_label() -> None:
    for cls in ALL_STRATEGIES:
        assert cls().label == EDUCATIONAL_LABEL


def test_label_is_surfaced_in_audit_and_report_but_not_the_fingerprint() -> None:
    from biasguard.analytics.fingerprint import build_manifest
    from biasguard.audit import build_audit
    from biasguard.report import build_html_report

    case = next(c for c in CATALOG if c.key == "ma_crossover")
    data = case.make_data()
    run, report = _assess(case)
    manifest = build_manifest(data, case.make_strategy())

    # Captured as metadata, and NOT part of the fingerprint (it's a class
    # attribute, so it never leaks into the fingerprinted strategy params).
    assert manifest.strategy_label == EDUCATIONAL_LABEL
    assert "label" not in manifest.strategy["params"]

    audit = build_audit(report, manifest=manifest, profile=case.profile, metrics=run.metrics())  # type: ignore[attr-defined]
    assert EDUCATIONAL_LABEL in audit.to_ai_prompt()
    assert EDUCATIONAL_LABEL in audit.to_markdown()

    html = build_html_report(
        metrics=run.metrics(),  # type: ignore[attr-defined]
        equity=run.equity,  # type: ignore[attr-defined]
        trades=run.trades,  # type: ignore[attr-defined]
        manifest=manifest,
        profile=case.profile,
    )
    assert EDUCATIONAL_LABEL in html


def test_catalog_keys_are_unique() -> None:
    keys = [c.key for c in CATALOG]
    assert len(keys) == len(set(keys))
