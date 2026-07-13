# Roadmap

`biasguard` v1.0.0 is a complete, honest backtesting-and-integrity framework. This document is candid
about what it **does not** yet do, and what might come next. Nothing here is a committed date — it is a
list of directions, and community input is welcome via [issues](https://github.com/Giantars/biasguard/issues).

## Stability commitment

v1.0.0 marks a **stable public API** under [semantic versioning](https://semver.org/):

- The **public API** is everything exported from a package's `__init__` (e.g. `biasguard.validation`,
  `biasguard.execution`). Names prefixed with `_` are private and may change any time.
- Breaking changes to the public API will bump the **major** version and be listed in the
  [changelog](../CHANGELOG.md).
- New checks, profiles, indicators, and options are **minor** versions.
- Bug fixes and doc changes are **patch** versions.

The replay fingerprint has its own schema version (`MANIFEST_VERSION`); a change that alters fingerprints
is called out explicitly, because reproducibility is a promise.

## Deferred from v1.0.0 (known gaps)

These were scoped out deliberately to ship a focused, correct release — not because they don't matter.

- **Advanced overfitting statistics.** The Deflated Sharpe Ratio (Bailey & López de Prado), PBO/CSCV, and
  automated walk-forward were in the original vision. v1.0.0 covers overfitting via regime concentration,
  in-/out-of-sample consistency, and block-bootstrap Monte Carlo. DSR/PBO would slot in as new
  `IntegrityCheck` plugins over a parameter sweep.
- **A live / paper runner.** The engine is deliberately built *backtest-as-brain* (a strategy is a pure
  function of its causal context), but v1.0.0 ships no live driver or parity harness. See
  [deploying-responsibly.md](deploying-responsibly.md) for the principles it would follow.
- **Tick-level fills & a queue model.** Fills are bar-resolution. A `fill_probability`/queue model and
  tick-data fills would sharpen the phantom-fill analysis.
- **`on_tick` strategy hook.** The `Strategy` interface is designed so `on_tick` can be added without
  breaking `on_bar` users.
- **Continuous-contract roll handling.** Guidance exists (back-adjusted for returns, raw for fills), but
  there is no built-in roll-seam handler or adjusted-price-reference warning yet.
- **Multi-asset portfolios & resampling.** The engine runs one instrument per backtest today.

## Candidate near-term improvements

- More execution profiles for common retail brokers and prop firms.
- A parameter-sweep helper that runs the multiplicity-aware checks automatically.
- Additional bootstraps (e.g. stationary bootstrap variants) and regime-conditioned MC presets.
- A CSV-schema doctor that suggests the right `load_csv` arguments for a messy file.

## How to influence it

The most valuable contributions are **new failure modes** — a way a backtest lies that
[how-backtests-lie.md](how-backtests-lie.md) doesn't cover, ideally with a planted-bug fixture and a
check that catches it. That is the heart of the project. See [contributing.md](contributing.md).
