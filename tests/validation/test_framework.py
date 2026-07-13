"""Tests for the integrity-framework core: report, score, registry, context."""

from __future__ import annotations

import math

import pytest
from tests.conftest import make_ohlcv
from tests.known_bad.strategies import UptickCausal

from biasguard.execution.costs import PerContractCommission, TickSlippage
from biasguard.execution.instrument import NQ
from biasguard.validation import (
    BacktestSpec,
    CheckResult,
    IntegrityCheck,
    IntegrityContext,
    IntegrityRegistry,
    IntegrityReport,
    Status,
    aggregate_score,
    assess_integrity,
    grade_for,
    register_check,
)
from biasguard.validation.registry import DEFAULT_REGISTRY


def _result(
    key: str, status: Status, score: float, *, gate: bool = False, weight: float = 1.0
) -> CheckResult:
    return CheckResult(key, key, "test", status, score, "s", is_gate=gate, weight=weight)


class TestScoring:
    def test_weighted_mean(self) -> None:
        results = (
            _result("a", Status.PASS, 1.0, weight=1.0),
            _result("b", Status.PASS, 0.0, weight=1.0),
        )
        assert aggregate_score(results) == pytest.approx(50.0)

    def test_skip_excluded(self) -> None:
        results = (_result("a", Status.PASS, 1.0), _result("b", Status.SKIP, 0.0))
        assert aggregate_score(results) == pytest.approx(100.0)

    def test_gate_failure_caps_score(self) -> None:
        # Everything else is perfect, but a gate failed -> capped at 25.
        results = (
            _result("gate", Status.FAIL, 0.0, gate=True, weight=2.0),
            _result("a", Status.PASS, 1.0, weight=1.0),
            _result("b", Status.PASS, 1.0, weight=1.0),
        )
        assert aggregate_score(results) <= 25.0

    def test_all_skip_is_nan(self) -> None:
        assert math.isnan(aggregate_score((_result("a", Status.SKIP, 0.0),)))

    def test_gate_skip_caps_score(self) -> None:
        # A gate that did not run (SKIP) caps the score: the causality control
        # was never verified, so a leaky run whose gate crashed can't score A.
        results = (
            _result("gate", Status.SKIP, 0.0, gate=True, weight=2.0),
            _result("a", Status.PASS, 1.0, weight=1.0),
            _result("b", Status.PASS, 1.0, weight=1.0),
        )
        assert aggregate_score(results) <= 60.0

    def test_gate_pass_not_capped(self) -> None:
        results = (
            _result("gate", Status.PASS, 1.0, gate=True, weight=2.0),
            _result("a", Status.PASS, 1.0, weight=1.0),
        )
        assert aggregate_score(results) == pytest.approx(100.0)

    def test_grades(self) -> None:
        assert grade_for(95)[0] == "A"
        assert grade_for(80)[0] == "B"
        assert grade_for(30)[0] == "F"
        assert grade_for(float("nan"))[0] == "N/A"


class TestReport:
    def test_build_orders_worst_first(self) -> None:
        report = IntegrityReport.build(
            (_result("ok", Status.PASS, 1.0), _result("bad", Status.FAIL, 0.0))
        )
        assert report.results[0].status is Status.FAIL
        assert not report.trustworthy

    def test_summary_and_dict_and_html(self) -> None:
        report = IntegrityReport.build((_result("ok", Status.PASS, 1.0),))
        assert "Integrity score" in report.summary()
        assert report.to_dict()["grade"] == report.grade
        assert "Integrity" in report.to_html()

    def test_get(self) -> None:
        report = IntegrityReport.build((_result("ok", Status.PASS, 1.0),))
        assert report.get("ok") is not None
        assert report.get("missing") is None

    def test_gate_skip_is_not_trustworthy(self) -> None:
        report = IntegrityReport.build(
            (_result("gate", Status.SKIP, 0.0, gate=True), _result("a", Status.PASS, 1.0))
        )
        assert not report.gates_passed
        assert not report.trustworthy

    def test_summary_guards_nan_score(self) -> None:
        report = IntegrityReport.build((_result("a", Status.SKIP, 0.0),))
        text = report.summary()
        assert "nan/100" not in text
        assert "—/100" in text


class TestRegistry:
    def test_register_get_unregister(self) -> None:
        reg = IntegrityRegistry()

        class C(IntegrityCheck):
            key = "c"

            def run(self, ctx: IntegrityContext) -> CheckResult:
                return self.result(Status.PASS, 1.0, "ok")

        reg.register(C())
        assert reg.get("c") is not None
        assert [c.key for c in reg.checks()] == ["c"]
        reg.unregister("c")
        assert reg.get("c") is None

    def test_no_replace_raises(self) -> None:
        reg = IntegrityRegistry()

        class C(IntegrityCheck):
            key = "c"

            def run(self, ctx: IntegrityContext) -> CheckResult:
                return self.result(Status.PASS, 1.0, "ok")

        reg.register(C())
        with pytest.raises(ValueError):
            reg.register(C(), replace=False)

    def test_builtins_registered(self) -> None:
        keys = {c.key for c in DEFAULT_REGISTRY.checks()}
        assert {"ohlc", "lookahead", "costs", "fill_realism", "slippage_sensitivity"} <= keys


def _spec(strategy_cls: type = UptickCausal) -> BacktestSpec:
    return BacktestSpec(
        data=make_ohlcv(n=60),
        strategy_factory=strategy_cls,
        instrument=NQ,
        commission=PerContractCommission(1.90),
        slippage=TickSlippage(1.0),
    )


class TestContextAndSpec:
    def test_run_and_rerun(self) -> None:
        spec = _spec()
        out = spec.run()
        assert out.equity is not None and len(out.equity) == 60
        # A perturbed re-run (truncated) returns fewer bars.
        short = spec.run(data=spec.data.iloc[:20])
        assert len(short.equity) == 20

    def test_context_build(self) -> None:
        ctx = IntegrityContext.build(_spec())
        assert ctx.baseline is not None
        assert ctx.rerun().net_pnl == ctx.baseline.net_pnl  # deterministic


class TestAssessDriver:
    def test_runs_all_and_scores(self) -> None:
        report = assess_integrity(_spec())
        keys = {r.key for r in report.results}
        assert "lookahead" in keys and "fill_realism" in keys
        assert math.isnan(report.score) or 0.0 <= report.score <= 100.0

    def test_include_exclude(self) -> None:
        report = assess_integrity(_spec(), include={"ohlc"})
        assert {r.key for r in report.results} == {"ohlc"}

    def test_buggy_check_is_sandboxed(self) -> None:
        reg = DEFAULT_REGISTRY.copy()

        class Boom(IntegrityCheck):
            key = "boom"

            def run(self, ctx: IntegrityContext) -> CheckResult:
                raise RuntimeError("kaboom")

        reg.register(Boom())
        report = assess_integrity(_spec(), registry=reg, include={"boom"})
        boom = report.get("boom")
        assert boom is not None and boom.status is Status.SKIP
        assert "kaboom" in boom.detail

    def test_custom_check_plugs_in(self) -> None:
        reg = DEFAULT_REGISTRY.copy()

        class Custom(IntegrityCheck):
            key = "custom_probe"
            name = "Custom probe"
            category = "robustness"

            def run(self, ctx: IntegrityContext) -> CheckResult:
                return self.result(Status.PASS, 1.0, "custom check ran")

        register_check(Custom(), registry=reg)
        report = assess_integrity(_spec(), registry=reg, include={"custom_probe"})
        assert report.get("custom_probe") is not None
