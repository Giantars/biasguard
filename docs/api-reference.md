# API reference

The public API is what each package exports from its `__init__`. Import from the subpackage
(`from biasguard.validation import assess_integrity`) rather than reaching into private modules. Anything
prefixed with `_` is private and may change between minor versions.

Every public class and function has a docstring; this page is the map. `help(biasguard.validation)` or
your editor's go-to-definition gives the details.

---

## `biasguard`

`__version__` — the package version string (`"1.0.0"`).

## `biasguard.data` — load & validate OHLCV

| Name | What it is |
| --- | --- |
| `load_csv(path, *, tz, ...)` | Load a CSV into a canonical OHLCV frame. **`tz` is required.** Attaches a quality report to `df.attrs["quality_report"]`. |
| `load_parquet(path, *, tz, ...)` | Same for Parquet. |
| `validate_ohlcv(df, ...)` | Return a `DataQualityReport` without loading. |
| `Bar` | Immutable OHLCV bar (validates invariants at construction). |
| `DataQualityReport`, `DataIssue`, `Severity` | Structured validation output. |
| `DataValidationError`, `DataQualityWarning` | Raised / warned on invalid data. |
| `find_gaps`, `find_duplicate_timestamps`, `find_ohlc_violations`, `infer_frequency`, `canonicalize_columns` | Lower-level validation helpers. |

## `biasguard.types` — shared enums

`Direction` (`LONG`/`SHORT`/`FLAT`), `OrderSide` (`BUY`/`SELL`, with `.sign`), `OrderType`
(`MARKET`/`LIMIT`/`STOP`), `Severity` (`INFO`/`WARNING`/`ERROR`).

## `biasguard.events` — the event types

`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent` — frozen dataclasses. Value equality is what
makes the truncation test possible.

## `biasguard.engine` — the causal loop

| Name | What it is |
| --- | --- |
| `Backtester(handler, strategy, *, portfolio=, broker=)` | Runs one strategy over one data handler. |
| `run_backtest(data, strategy_factory, *, upto=, ...)` | Functional entrypoint **and the truncation harness** (`upto=T` runs on `data[:T]`). |
| `DataHandler(data, *, validate=True)` | The completed-bars-only feed; vends causal views. |
| `MarketView` | Read-only window bounded at the current bar (`.opens/.highs/.lows/.closes/.volumes`, `len()`, `.window(field, n)`, `.as_frame()`). |
| `Portfolio`, `Broker` | Protocols the engine depends on (concrete impls live in `execution`). |
| `NullPortfolio`, `BacktestResult` | The no-op portfolio (pure decision engine) and the run record. |

## `biasguard.strategy` — the interface

`Strategy` (ABC; implement `on_bar(ctx)`, optional `on_start`/`on_finish`, `params()` for the
fingerprint), `StrategyContext` (`.view`, `.position`, `.index`, `.bar`; factories `.long()`, `.short()`,
`.exit()`, `.signal(dir)`), `NoOpStrategy`.

<a id="indicators"></a>

## `biasguard.indicators` — causal indicators

All strictly trailing and truncation-stable; warm-up returns `NaN`.

`sma`, `rolling_mean`, `rolling_std`, `rolling_zscore`, `ema`, `rsi`, `atr`, `true_range`.

```python
from biasguard.indicators import rsi
value = rsi(ctx.view.closes, 14)[-1]   # only sees completed bars
```

## `biasguard.execution` — the execution model

| Group | Names |
| --- | --- |
| Instruments | `Instrument`, presets `NQ`, `MNQ`, `ES`, `MES`, `GC`, `SI`, and `PRESETS` |
| Commission | `CommissionModel` (ABC), `ZeroCommission`, `PerContractCommission`, `PercentCommission` |
| Slippage | `SlippageModel` (ABC), `NoSlippage`, `FixedSlippage`, `TickSlippage`, `PercentSlippage` |
| Fills | `FillModel` (ABC), `TouchFill`, `TradeThroughFill`, `FillRequest`, `FillDecision` |
| Orders / positions | `Order`, `OrderStatus`, `Position`, `ApplyResult`, `Trade` |
| Broker / portfolio | `SimulatedBroker`, `SameBarPolicy`, `Portfolio`, `Sizer`, `FixedSizer` |
| Profiles | `ExecutionProfile`, `ExecutionRealism`, `REAL_MARKET`, `PROP_FIRM_SIM`, `custom_profile`, `assess_realism`, `PROFILES`, `register_profile`, `get_profile` |

## `biasguard.analytics` — metrics & fingerprint

| Name | What it is |
| --- | --- |
| `compute_metrics(equity, trades, ...)` → `PerformanceMetrics` | Total return, CAGR, Sharpe, Sortino, max DD, profit factor, win rate, expectancy, … |
| `sharpe_ratio`, `sortino_ratio`, `cagr`, `max_drawdown`, `profit_factor`, `win_rate`, `expectancy`, `drawdown_series`, `per_year_breakdown`, `split_is_oos` | The underlying functions. |
| `build_manifest(data, strategy, *, broker=, portfolio=)` → `ReplayManifest` | The deterministic replay manifest + fingerprint. |
| `hash_market_data`, `describe`, `MANIFEST_VERSION` | Fingerprint internals you can reuse. |

## `biasguard.validation` — the Integrity Framework

| Name | What it is |
| --- | --- |
| `BacktestSpec` | A reproducible recipe. `BacktestSpec.from_profile(...)` builds one from an `ExecutionProfile`. `.run(**perturbations)` re-executes. |
| `assess_integrity(spec, *, include=, exclude=, seed=, oos_cut=, config=)` → `IntegrityReport` | Run the checks and aggregate the score. |
| `IntegrityReport` | `.score`, `.grade`, `.trustworthy`, `.results`, `.get(key)`, `.fails`, `.warns`, `.summary()`, `.to_dict()`, `.to_html()`. |
| `CheckResult`, `Status` | A single check's verdict (`PASS`/`WARN`/`FAIL`/`SKIP`) + score + metrics. |
| `IntegrityCheck` | The plugin base class — subclass to add a check. |
| `IntegrityRegistry`, `DEFAULT_REGISTRY`, `register_check` | The registry (plugin surface). |
| `IntegrityContext`, `RunOutput` | What a check receives / a run's outcome. |
| `aggregate_score`, `grade_for`, `GATE_FAIL_CAP` | Scoring internals. |

The eight built-in checks (`biasguard.validation.checks`): `ohlc`, `lookahead` (gate), `costs`,
`fill_realism`, `slippage_sensitivity`, `regime`, `out_of_sample`, `monte_carlo`.

## `biasguard.montecarlo` — resampling & risk

| Name | What it is |
| --- | --- |
| `MonteCarloSimulator(*, bootstrap=, n_paths=, seed=)` | `.run(trades, *, account=, regime_mask=)` → `MonteCarloResult`. |
| `MonteCarloResult` | `.prob_profit`, `.prob_breach`, percentiles, `.worst_case_drawdown`, `.equity_bands`, `.summary()`, `.to_html()`. |
| `AccountConfig` | Prop-firm limits: starting balance, trailing drawdown, daily loss, max loss, profit target. |
| `Bootstrap`, `StationaryBootstrap`, `CircularBlockBootstrap`, `IIDBootstrap` | Resampling schemes (block by default; IID for contrast only). |
| `recent_regime_mask`, `trade_pnls`, `evaluate_paths` | Helpers. |

## `biasguard.audit` — AI audit export

| Name | What it is |
| --- | --- |
| `build_audit(report, *, manifest=, profile=, metrics=, monte_carlo=)` → `AuditExport` | Assemble the audit context. |
| `AuditExport` | `.to_json()`, `.to_markdown()`, `.to_ai_prompt()`, `.write(dir)`, `.targets()`, `.warnings()`. |
| `investigation_targets(report)` | The deterministic "what to investigate next" list. |

## `biasguard.report` — HTML report

`build_html_report(*, metrics, equity, trades, manifest=, validation_html=, profile=, disclaimer=,
ai_prompt=, monte_carlo=, path=)` → the self-contained HTML string (and writes it + the manifest when
`path` is given). Pass `ai_prompt=audit.to_ai_prompt()` to embed a Download/Copy AI-prompt button, and
`monte_carlo=<MonteCarloResult>` to add a resampled-equity fan chart.

## `biasguard.strategies` — educational examples

`CATALOG` (tuple of `ExampleCase`), `get_case(key)`, `EDUCATIONAL_LABEL`, `EducationalStrategy`, the six
strategy classes (`MovingAverageCrossover`, `RsiMeanReversion`, `LookaheadStrategy`,
`OverfitMeanReversion`, `ZeroSlippageScalper`, `FreeLunchChurn`), the data generators (`trending_data`,
`swinging_data`, `mean_reverting_data`, `regime_break_data`, `choppy_data`), and `ZERO_COST_PROFILE`.
