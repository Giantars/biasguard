"""Tests for deterministic investigation-target derivation."""

from __future__ import annotations

from biasguard.audit.targets import investigation_targets
from biasguard.validation.report import CheckResult, IntegrityReport, Status


def _r(key: str, status: Status, *, summary: str = "s") -> CheckResult:
    return CheckResult(key, key, "test", status, 0.5, summary)


def _report(*results: CheckResult) -> IntegrityReport:
    return IntegrityReport.build(results)


class TestTargets:
    def test_fail_and_warn_produce_actions_worst_first(self) -> None:
        report = _report(
            _r("lookahead", Status.FAIL),
            _r("costs", Status.WARN),
            _r("ohlc", Status.PASS),
        )
        targets = investigation_targets(report)
        assert len(targets) == 2
        # FAIL (lookahead) is ordered before WARN (costs).
        assert "truncation test" in targets[0].lower()
        assert "commission" in targets[1].lower()

    def test_passing_report_has_no_targets(self) -> None:
        report = _report(_r("lookahead", Status.PASS), _r("costs", Status.PASS))
        assert investigation_targets(report) == ()

    def test_unrealistic_profile_appends_action(self) -> None:
        report = _report(_r("ohlc", Status.PASS))
        targets = investigation_targets(report, profile_realistic=False)
        assert any("Real Market" in t for t in targets)

    def test_realistic_profile_adds_nothing(self) -> None:
        report = _report(_r("ohlc", Status.PASS))
        assert investigation_targets(report, profile_realistic=True) == ()

    def test_unknown_key_uses_fallback(self) -> None:
        report = _report(_r("some_new_check", Status.WARN, summary="weird thing happened"))
        targets = investigation_targets(report)
        assert len(targets) == 1
        assert "weird thing happened" in targets[0]

    def test_duplicate_actions_deduplicated(self) -> None:
        # Two results mapping to the same action collapse to one target.
        report = _report(_r("lookahead", Status.FAIL), _r("lookahead", Status.WARN))
        targets = investigation_targets(report)
        assert len(targets) == 1
