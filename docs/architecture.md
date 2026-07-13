# Architecture

`biasguard` is a small stack of single-responsibility layers with a strict dependency direction. The
design goals, in priority order: **causality by construction**, **truncation-testability**,
**determinism**, and **extensibility without engine changes**.

## The layers

Dependencies point strictly downward — a higher layer may import a lower one, never the reverse. This is
what keeps the engine unaware of execution details and the execution core unaware of validation.

```
types                      enums shared across packages (Direction, OrderSide, OrderType, Severity)
  └─ data                  load + validate OHLCV; the Bar value object
      └─ events            MarketEvent / SignalEvent / OrderEvent / FillEvent (frozen dataclasses)
          └─ engine        the event loop, MarketView (causal window), truncation entrypoint
              └─ strategy  the Strategy ABC + StrategyContext
                  ├─ indicators    causal SMA/EMA/RSI/ATR/rolling stats
                  ├─ execution     instruments, cost/slippage/fill models, broker, portfolio, profiles
                  │   └─ strategies  educational examples + the demonstration catalog
                  ├─ analytics     performance metrics + the replay fingerprint
                  │   └─ montecarlo block-bootstrap resampling + prop-firm account limits
                  │       └─ validation   the Backtest Integrity Framework
                  │           └─ audit    AI audit export
                  └─ report        self-contained Plotly HTML report
```

The engine depends on the `Portfolio` and `Broker` **Protocols**, not on the concrete execution classes,
so it never imports `execution` — there is no cycle, and the execution model can evolve without touching
the loop.

## Event flow

The engine processes one bar at a time in a **fixed step order** that makes the causal contract
structural rather than conventional:

```
for each bar i:
  1. FILL-FIRST   broker matches orders resting from earlier bars against bar i
  2. DECIDE       strategy sees a MarketView of bars [0..i] and emits signals
  3. SIZE         portfolio turns signals into orders
  4. REST         orders are placed but not matched until step 1 of bar i+1
  5. MARK         portfolio records equity at bar i's close
```

The consequence: a signal decided on bar *i*'s close cannot fill before bar *i+1*. There is no code path
that fills an order against the bar the signal was decided on. `MarketEvent → SignalEvent → OrderEvent →
FillEvent` is the only route, and each event is a frozen dataclass stamped with the time it was decided.

## Causality by construction — and how it's proven

Two mechanisms enforce "never read the future":

1. **The `MarketView`** hands a strategy read-only numpy slices bounded at the current bar. Indexing
   past it raises `IndexError` — it fails loud rather than silently returning a future value.
2. **The truncation test** is the ultimate arbiter. Because the engine is a pure function of
   `(data, strategy, config)`, running on `data[:T]` is cheap and shares no state with the full run
   (`run_backtest` takes *factories*, not instances, so nothing is warm). Any real dependence on future
   data — including deliberate circumvention via a numpy view's `.base` — changes the truncated result
   and is caught. The `lookahead` integrity check automates this and is a **gate**.

This is *power-validated*: `tests/known_bad/` ships strategies with planted leaks, and the causality
tests must catch them. A green causality suite that can't go red on purpose proves nothing.

## The Backtest Integrity Framework

The headline differentiator. A `BacktestSpec` is a reproducible recipe (`data + strategy_factory +
instrument + cost/slippage/fill models`) with one method, `run(...)`, that re-executes with optional
perturbations (truncated data, a different fill model, more slippage, a different strategy). That single
perturbation seam is what lets every check be a plugin:

- Each **`IntegrityCheck`** receives an `IntegrityContext` (the spec + a baseline run + a seed) and
  returns a `CheckResult` (status + score in [0,1] + metrics).
- The **registry** holds the checks; `assess_integrity(spec)` runs every applicable one, sandboxing each
  in `try/except` so a buggy plugin degrades to `SKIP` rather than sinking the report.
- The **score** is a weighted mean of non-skipped checks, capped by gates: a failed gate caps at 25, a
  gate that couldn't run caps at 60. The result is a 0–100 Integrity Score and an A–F grade.

The eight built-in checks: `ohlc`, `lookahead` (gate), `costs`, `fill_realism`, `slippage_sensitivity`,
`regime`, `out_of_sample`, `monte_carlo`. See [how-backtests-lie.md](how-backtests-lie.md) for what each
one is for.

## The execution model

Execution realism is where edges live and die, so every piece is a swappable object:

- **Instrument** — multiplier + tick size; drives P&L-to-dollars, tick slippage, per-contract commission.
- **Commission models** — `ZeroCommission` / `PerContractCommission` / `PercentCommission`.
- **Slippage models** — `NoSlippage` / `FixedSlippage` / `TickSlippage` / `PercentSlippage`, adverse by
  construction (a BUY executes higher, a SELL lower), applied only to liquidity-*taking* fills.
- **Fill models** — `TouchFill` (optimistic) vs `TradeThroughFill` (conservative default). A limit fills
  at its limit price, never an intrabar favourable extreme.
- **Execution profiles** — a named bundle of the above (`REAL_MARKET`, `PROP_FIRM_SIM`, or a custom one).
  Realism is *probed* (does slippage move the price? does commission cost anything? does the fill model
  fill on a mere touch?), so an optimistic assumption — even one hidden in a user's own model — is
  surfaced and the report states results are a *simulated* environment.

## Determinism & the replay fingerprint

No wall-clock or RNG in the decision path; all Monte Carlo is explicitly seeded. Every run is identified
by a **replay fingerprint** — a SHA-256 over a canonical manifest of every input (market-data content
hash, strategy identity + params, execution settings, cost/slippage config, framework version). Two
identical backtests produce the same fingerprint; change any input and it changes. The manifest is
embedded in the HTML report and the audit export.

## Extension points

Everything below can be added **without modifying the engine**:

| To add… | Do this |
| --- | --- |
| An **integrity check** | Subclass `IntegrityCheck`, implement `run`, `register_check(MyCheck())`. |
| An **execution profile** | Build an `ExecutionProfile` (or `custom_profile(...)`) and `register_profile(...)`. |
| A **cost / slippage / fill model** | Subclass the relevant ABC; realism is probed behaviourally, not by type. |
| An **indicator** | Add a pure, trailing function to `indicators/`; add a truncation test. |
| A **strategy** | Subclass `Strategy`, implement `on_bar`; read only through `StrategyContext`. |
| A **bootstrap** | Subclass `Bootstrap` for a new resampling scheme. |

See [contributing.md](contributing.md) for the mechanics and the quality gate.
