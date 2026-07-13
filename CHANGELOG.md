# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [1.0.0] — 2026-07-13

First public release: a complete, bias-resistant, event-driven backtesting framework with a Backtest
Integrity Framework as its differentiator. The public API (everything exported from a package's
`__init__`) is now stable under semantic versioning.

### Added

- **Data layer** (`biasguard.data`) — CSV/Parquet loaders with **mandatory timezone**, and a validation
  suite: OHLC-invariant, duplicate-timestamp, monotonic-index, and missing-bar/gap detection, surfaced as
  a structured `DataQualityReport`.
- **Event-driven engine** (`biasguard.engine`) — `MarketEvent → SignalEvent → OrderEvent → FillEvent`
  with a fixed fill-first/decide/size/rest/mark step order, a completed-bars-only gate, the causal
  `MarketView`, and `run_backtest(..., upto=T)` as the **truncation harness** for mechanical lookahead
  detection.
- **Strategy interface** (`biasguard.strategy`) — the `Strategy` ABC and read-only `StrategyContext`.
- **Execution core** (`biasguard.execution`) — instruments (NQ/MNQ/ES/MES/GC/SI), mandatory commission
  and slippage models, swappable fill models (`TouchFill`/`TradeThroughFill`), a simulated broker with
  **stop-first** same-bar resolution, positions, trades, and portfolio.
- **Execution profiles** — named bundles of cost/slippage/fill assumptions (`REAL_MARKET`,
  `PROP_FIRM_SIM`, `custom_profile`) whose realism is **probed behaviourally**; the report states when
  results represent a simulated environment.
- **Analytics & replay fingerprint** (`biasguard.analytics`) — the full metric set (Sharpe, Sortino,
  CAGR, max drawdown, profit factor, win rate, expectancy, per-year, IS/OOS) and a **deterministic
  SHA-256 replay fingerprint** over a canonical manifest of every run input.
- **Backtest Integrity Framework** (`biasguard.validation`) — a pluggable registry of eight checks
  (`ohlc`, `lookahead` gate, `costs`, `fill_realism`, `slippage_sensitivity`, `regime`, `out_of_sample`,
  `monte_carlo`) aggregated into a 0–100 Integrity Score and an A–F grade; power-validated against
  planted-bug fixtures in `tests/known_bad/`.
- **Monte Carlo & prop-firm risk** (`biasguard.montecarlo`) — streak-preserving block bootstrap,
  configurable `AccountConfig` limits (trailing drawdown, intraday daily loss, max loss, profit target),
  and breach-probability reporting.
- **AI audit export** (`biasguard.audit`) — turns an integrity report into `audit_report.md`,
  `audit_report.json`, and a paste-ready `ai_debug_prompt.txt`, deterministically.
- **Causal indicators** (`biasguard.indicators`) — SMA, EMA, RSI (Wilder), ATR (Wilder), and rolling
  mean/std/z-score, each strictly trailing and **truncation-stable**.
- **Educational strategies** (`biasguard.strategies`) — two honest templates (MA crossover, RSI
  reversion) and four intentionally-flawed demonstrations (lookahead, overfit, zero-slippage-dependent,
  unrealistic-cost), each labelled and paired in a `CATALOG` with the integrity verdict it demonstrates.
- **Self-contained HTML report** (`biasguard.report`) — equity curve, drawdown, monthly heatmap, trade
  distribution, the trust verdict, and the embedded replay manifest, in one offline file. Optional
  `ai_prompt=` embeds a **"Download / Copy AI debug prompt"** control (safely escaped, downloaded
  client-side), and optional `monte_carlo=` adds a **resampled-equity fan chart** with the distribution
  summary.
- **Docs** — architecture, the "how backtests lie" failure-mode catalog, deploying responsibly,
  installation, quick start, a first-backtest tutorial, example outputs, an API reference, and a
  configuration reference.
- **Tooling** — `ruff` + `black` + `mypy --strict` + `pytest` gate, GitHub Actions CI (Python 3.11/3.12),
  pre-commit hooks, and issue/PR templates.

### Notes

- This framework **reveals no trading edge**; every bundled strategy is a clearly-labelled educational
  example. A passing Integrity Score means "causal and honestly modelled", not "profitable".

[Unreleased]: https://github.com/Giantars/biasguard/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Giantars/biasguard/releases/tag/v1.0.0
