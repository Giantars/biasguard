"""Example 8 - Educational strategy catalog: how BiasGuard evaluates behavior.

Run:  python examples/08_educational_strategies.py

Runs every strategy in ``biasguard.strategies.CATALOG`` end-to-end and, for each,
generates an HTML report, an Integrity Report, and an AI Audit Export, then
prints why it passes or fails. Two strategies are honest causal templates
(moving-average crossover, RSI mean reversion); four are intentionally flawed to
show what the Integrity Framework catches:

    lookahead bias        -> FAIL (lookahead gate)
    overfit parameters    -> WARN (regime concentration)
    zero-slippage scalper -> WARN (slippage sensitivity)
    unrealistic costs     -> FAIL (transaction costs)

Every strategy is labelled "Educational example - not intended as a profitable
trading strategy." The point is the *evaluation*, not the edge.
"""

from __future__ import annotations

from pathlib import Path

from biasguard.analytics.fingerprint import build_manifest
from biasguard.audit import build_audit
from biasguard.montecarlo import MonteCarloResult, MonteCarloSimulator
from biasguard.report import build_html_report
from biasguard.strategies import CATALOG
from biasguard.validation import BacktestSpec, assess_integrity

OUTPUT_DIR = Path(__file__).parent / "output" / "phase8"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing reports and audits under: {OUTPUT_DIR}\n")

    for case in CATALOG:
        data = case.make_data()
        spec = BacktestSpec.from_profile(
            data=data,
            strategy_factory=case.make_strategy,
            instrument=case.instrument,
            profile=case.profile,
        )
        run = spec.run()
        report = assess_integrity(spec)
        manifest = build_manifest(data, case.make_strategy())

        # Monte Carlo needs a meaningful number of trades to be worth showing.
        mc: MonteCarloResult | None = None
        if len(run.trades) >= 10:
            mc = MonteCarloSimulator(n_paths=2000, seed=1).run(run.trades)

        audit = build_audit(
            report,
            title=f"biasguard - {case.title}",
            manifest=manifest,
            profile=case.profile,
            metrics=run.metrics(),
            monte_carlo=mc,
        )

        case_dir = OUTPUT_DIR / case.key
        audit.write(case_dir)
        build_html_report(
            metrics=run.metrics(),
            equity=run.equity,
            trades=run.trades,
            manifest=manifest,
            validation_html=report.to_html(),
            profile=case.profile,
            ai_prompt=audit.to_ai_prompt(),  # adds the Download/Copy AI-prompt button
            monte_carlo=mc,  # adds the resampled-equity fan chart
            title=case.title,
            path=case_dir / "report.html",
        )

        watched = report.get(case.watch)
        watched_status = watched.status.value if watched is not None else "MISSING"
        match = "OK" if watched_status == case.expected_status else "MISMATCH"
        strategy = case.make_strategy()

        print("=" * 76)
        print(f"{case.title}")
        print(f"  {getattr(strategy, 'label', '')}")
        print(
            f"  net ${run.net_pnl:,.0f} over {len(run.trades)} trades  |  "
            f"integrity {report.score:.0f}/100 ({report.grade})"
        )
        print(
            f"  watched check '{case.watch}': {watched_status} "
            f"(expected {case.expected_status}) -> {match}"
        )
        if watched is not None:
            print(f"    {watched.summary}")
        print(f"  Why: {case.explanation}")
        print(f"  Artifacts: {case_dir}\\ (report.html, audit_report.md/json, ai_debug_prompt.txt)")
        print()

    print("=" * 76)
    print("Reminder: these are teaching artifacts. A PASS means 'causal and honestly")
    print("modelled', NOT 'profitable'. BiasGuard evaluates integrity, not edge.")


if __name__ == "__main__":
    main()
