# biasguard

**A backtester that tries to prove your backtest wrong.**

[![CI](https://github.com/Giantars/biasguard/actions/workflows/ci.yml/badge.svg)](https://github.com/Giantars/biasguard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2.svg)](https://mypy-lang.org/)

`biasguard` is a bias-resistant, event-driven backtesting framework for Python. Most backtesters
flatter you with a green equity curve. This one ships a **trust verdict** next to the Sharpe ratio —
a structured, 0–100 Integrity Score that answers the only question that matters before you risk money:

> **How much should you actually trust these results?**

> ⚠️ **This project reveals no trading edge.** It is research *infrastructure*. Every bundled strategy
> is a trivial, clearly-labelled educational example. The value is in the engine's honesty and the
> validation layer — not in any signal.

---

## Why another backtester?

Because almost every "profitable" backtest is lying, and the lies repeat. `biasguard` is built from a
catalog of real, forensically-audited failure modes (see
[docs/how-backtests-lie.md](docs/how-backtests-lie.md)) and turns each one into an automated check that
runs on *your* strategy:

| The lie | What `biasguard` does about it |
| --- | --- |
| **Lookahead / repaint** | Signals on bar *i*'s close fill no earlier than bar *i+1*, enforced *by construction*. The lookahead check re-runs your strategy on truncated history (`data[:T]`) and asserts the past decisions are **byte-identical** — the mechanical way to catch a future leak. It's a **gate**: a failure caps the whole score. |
| **Phantom / optimistic fills** | Fill models are first-class and swappable (`TouchFill`, `TradeThroughFill`). The fill-realism check runs the zero-slip, trade-through, and **random-direction-null** tests and reports **alpha vs. bracket mechanics**. |
| **Missing costs** | Commission and slippage are mandatory and logged per fill. A zero-cost run is a **FAIL**, not a footnote; a sub-2×-cost edge is a WARN. |
| **Overfitting / regime luck** | Per-year P&L concentration, in-sample vs. out-of-sample consistency, and a **block-bootstrap Monte Carlo** that preserves win/loss streaks all feed the score. |
| **Same-bar stop + target** | Resolved **stop-first (pessimistic)** by default — the sim never gives you the favourable exit for free. |
| **Bad data** | Mandatory timezone on load; OHLC-invariant, duplicate-timestamp, and gap checks before a single bar is traded. |

A green backtest that *survives* these checks is worth something. One that doesn't gets caught here
instead of in production.

## Quick start

```bash
pip install -e ".[dev]"      # from a clone; see docs/installation.md
```

```python
from biasguard.strategies import MovingAverageCrossover, swinging_data
from biasguard.execution import NQ, REAL_MARKET
from biasguard.validation import BacktestSpec, assess_integrity

# Describe a run: data + strategy + instrument + a realistic execution profile.
spec = BacktestSpec.from_profile(
    data=swinging_data(),                 # a bundled synthetic sample
    strategy_factory=MovingAverageCrossover,
    instrument=NQ,
    profile=REAL_MARKET,
)

report = assess_integrity(spec)           # run every integrity check
print(report.summary())
# Integrity score: 97/100 (A — High integrity). 0 fail, 0 warn, 5 pass, 3 skip.
#   [PASS] Lookahead / repaint (truncation test): decisions byte-identical under truncation — causal
#   [PASS] Fill realism (alpha vs. mechanics): $7,871 of alpha beyond mechanics; null beats it 8% ...
#   ...
```

Turn it into a shareable HTML report and a paste-ready AI debugging prompt:

```python
from biasguard.analytics import build_manifest
from biasguard.audit import build_audit
from biasguard.report import build_html_report

run = spec.run()
manifest = build_manifest(spec.data, MovingAverageCrossover())
audit = build_audit(report, manifest=manifest, profile=REAL_MARKET, metrics=run.metrics())

build_html_report(
    metrics=run.metrics(), equity=run.equity, trades=run.trades,
    manifest=manifest, validation_html=report.to_html(), profile=REAL_MARKET,
    ai_prompt=audit.to_ai_prompt(),       # adds a "Download / Copy AI debug prompt" button
    path="report.html",
)
audit.write("audit/")                     # audit_report.md/json + ai_debug_prompt.txt
```

See the [first-backtest tutorial](docs/tutorial.md) to run this on your own CSV.

## What you get

- **Backtest Integrity Framework** — a pluggable set of checks (lookahead, fill realism, costs, regime
  concentration, in-/out-of-sample, slippage sensitivity, Monte Carlo) aggregated into a single 0–100
  Integrity Score and a plain-English grade. Add your own check without touching the engine.
- **Execution profiles** — swap the whole cost/slippage/fill environment in one line (`REAL_MARKET`,
  `PROP_FIRM_SIM`, or a custom profile). Optimistic assumptions are **probed** and the report states
  when results represent a *simulated* execution environment.
- **Deterministic replay fingerprint** — every run is identified by a SHA-256 of its inputs. Two
  identical backtests produce the same fingerprint; change any input and it changes.
- **AI audit export** — turns the integrity report into `audit_report.md`, `audit_report.json`, and a
  ready-to-paste `ai_debug_prompt.txt` so you can hand the full context to an LLM for debugging.
- **Monte Carlo & prop-firm risk** — block-bootstrap resampling that preserves streaks, with
  configurable account limits (trailing drawdown, daily loss, max loss) and breach probabilities.
- **Self-contained HTML report** — equity curve, drawdown, monthly heatmap, trade distribution, the
  trust verdict, and the full replay manifest, in one offline file.

## Design principles

- **Strict causality by construction.** `MarketEvent → SignalEvent → OrderEvent → FillEvent`, one bar
  at a time. Nothing may read past the current bar's close.
- **Truncation-testable.** The engine is a pure function of `(data, strategy, config)`, so running on
  `data[:T]` is cheap — which is what makes lookahead detection *mechanical* rather than aspirational.
- **Power-validated.** The suite ships deliberately-broken strategies in `tests/known_bad/`; a
  causality test that can't catch a planted leak is worthless.
- **Deterministic.** No wall-clock or RNG in the decision path. Same input → same output, to the cent.
  All Monte Carlo is explicitly seeded.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Installation](docs/installation.md) | Requirements, venv, editable install, optional extras |
| [Quick start](docs/quickstart.md) | The five-line integrity check |
| [First backtest tutorial](docs/tutorial.md) | Load your own CSV → run → report → audit |
| [How backtests lie](docs/how-backtests-lie.md) | The failure-mode catalog and the check that catches each |
| [Architecture](docs/architecture.md) | Layers, event flow, extension points |
| [API reference](docs/api-reference.md) | The public surface, module by module |
| [Configuration reference](docs/configuration.md) | Profiles, account limits, the check registry |
| [Example outputs](docs/examples.md) | What each bundled example demonstrates |
| [Deploying responsibly](docs/deploying-responsibly.md) | Integrity ≠ edge; from backtest to live |
| [Contributing](docs/contributing.md) · [Roadmap](docs/roadmap.md) · [Changelog](CHANGELOG.md) | |

## Project layout

```
src/biasguard/
  data/         load + validate OHLCV (tz, duplicates, gaps, OHLC invariants)
  events/       MarketEvent / SignalEvent / OrderEvent / FillEvent (frozen)
  engine/       the causal event loop + MarketView + truncation entrypoint
  strategy/     the Strategy ABC and StrategyContext
  execution/    instruments, cost/slippage/fill models, broker, portfolio, execution profiles
  indicators/   causal SMA / EMA / RSI / ATR / rolling statistics
  strategies/   educational example strategies + the demonstration catalog
  analytics/    performance metrics + the deterministic replay fingerprint
  validation/   the Backtest Integrity Framework (pluggable checks + score)
  montecarlo/   block-bootstrap resampling + prop-firm account limits
  audit/        AI audit export (markdown / json / prompt)
  report/       self-contained Plotly HTML report
tests/          mirrors src/ + known_bad/ power-validation fixtures
examples/       runnable end-to-end scripts on sample data
docs/           architecture, the failure-mode catalog, deploying responsibly, ...
```

## Status

**v1.0.0** — first public release. Stable public API under [semantic versioning](https://semver.org/).
See the [changelog](CHANGELOG.md) and [roadmap](docs/roadmap.md).

## Contributing

Contributions are welcome — see [docs/contributing.md](docs/contributing.md). Every change must pass the
same gate CI runs: `ruff`, `black --check`, `mypy`, and `pytest`.

## License

MIT — see [LICENSE](LICENSE).
