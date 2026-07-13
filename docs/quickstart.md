# Quick start

The shortest useful thing `biasguard` does: run a strategy and print a **trust verdict**.

```python
from biasguard.strategies import MovingAverageCrossover, swinging_data
from biasguard.execution import NQ, REAL_MARKET
from biasguard.validation import BacktestSpec, assess_integrity

spec = BacktestSpec.from_profile(
    data=swinging_data(),                 # a bundled synthetic sample frame
    strategy_factory=MovingAverageCrossover,
    instrument=NQ,
    profile=REAL_MARKET,                  # realistic commission + slippage + conservative fills
)

report = assess_integrity(spec)
print(report.summary())
```

```
Integrity score: 97/100 (A — High integrity). 0 fail, 0 warn, 5 pass, 3 skip.
  [PASS] Fill realism (alpha vs. mechanics): $7,871 of alpha beyond mechanics; null beats it 8% of the time
  [PASS] Lookahead / repaint (truncation test): decisions byte-identical under truncation — causal
  [PASS] Transaction costs: median trade is 312.1x round-trip cost; costs modelled
  [PASS] OHLC & timezone integrity: data passes OHLC and timezone validation
  [PASS] Slippage sensitivity: edge survives 4.0 ticks of slippage
  [SKIP] Monte Carlo / regime / out-of-sample: too few trades in this tiny sample
```

*(Big numbers on a bundled sample are just synthetic data — a `PASS` means the run is causal and honestly
costed, not that the strategy is profitable.)*

## What just happened

- **`BacktestSpec.from_profile(...)`** describes a reproducible run: the data, a strategy *factory* (so
  each internal re-run starts fresh), the instrument, and an execution profile that supplies commission,
  slippage, and fill models.
- **`assess_integrity(spec)`** runs every applicable integrity check — each re-executing the backtest
  under a perturbation (truncated history, more slippage, a stricter fill model, a random-direction
  null) — and aggregates them into a 0–100 score with an A–F grade.
- **`report.summary()`** prints the verdict. `report.score`, `report.grade`, `report.trustworthy`, and
  `report.get("lookahead")` give you the structured pieces.

## Run it and get the artifacts

```python
run = spec.run()                          # the underlying backtest (equity, trades, fills)
print(f"net ${run.net_pnl:,.0f} over {len(run.trades)} trades")
print(run.metrics())                      # Sharpe, Sortino, max DD, profit factor, ...
```

Generate a shareable HTML report and an AI-ready audit bundle:

```python
from biasguard.analytics import build_manifest
from biasguard.audit import build_audit
from biasguard.report import build_html_report

manifest = build_manifest(spec.data, MovingAverageCrossover())
audit = build_audit(report, manifest=manifest, profile=REAL_MARKET, metrics=run.metrics())

build_html_report(
    metrics=run.metrics(), equity=run.equity, trades=run.trades,
    manifest=manifest, validation_html=report.to_html(), profile=REAL_MARKET,
    ai_prompt=audit.to_ai_prompt(),       # embeds a "Download / Copy AI debug prompt" button
    path="report.html",
)

audit.write("audit/")                     # audit_report.md, audit_report.json, ai_debug_prompt.txt
print(audit.to_ai_prompt())               # or paste this straight into Claude / ChatGPT
```

The report now has a **"Debug with an AI assistant"** section with **Download** and **Copy** buttons —
the prompt is embedded in the (offline, self-contained) HTML, so anyone you share the report with can
grab it without touching the files.

## Try a flawed strategy

Every failure mode has a bundled demonstration. Swap the strategy for one with a planted lookahead and
watch the gate fail:

```python
from biasguard.strategies import LookaheadStrategy, mean_reverting_data

spec = BacktestSpec.from_profile(
    data=mean_reverting_data(), strategy_factory=LookaheadStrategy,
    instrument=NQ, profile=REAL_MARKET,
)
report = assess_integrity(spec)
print(report.get("lookahead").status, "-", report.get("lookahead").summary)
# FAIL - decisions change when history is truncated at bar 200 — lookahead/repaint
print(report.score)   # capped at 25 by the gate
```

Next: the [first-backtest tutorial](tutorial.md) runs this on your own CSV.
