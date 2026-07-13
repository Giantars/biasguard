# Configuration reference

Everything you can tune, in one place. `biasguard` favours explicit configuration objects over global
settings, so a run is fully described by its inputs (and captured in the replay fingerprint).

## Instruments

An `Instrument` sets contract economics. Use a preset or define your own:

```python
from biasguard.execution import NQ, Instrument

NQ                                   # multiplier 20, tick 0.25 (a preset)
MYFUT = Instrument("MYFUT", multiplier=20.0, tick_size=0.25)
```

Presets: `NQ`, `MNQ`, `ES`, `MES`, `GC`, `SI` (verify against your broker; `PRESETS` is the dict).

## Execution profiles

A profile bundles commission + slippage + fill assumptions. Pick one, or build a custom profile.

```python
from biasguard.execution import REAL_MARKET, PROP_FIRM_SIM, custom_profile, register_profile
from biasguard.execution import PerContractCommission, TickSlippage, TouchFill

REAL_MARKET      # $2/contract, 1 tick slippage, conservative TradeThroughFill — realistic (default)
PROP_FIRM_SIM    # commissions on, but touch fills + no slippage — flagged as a SIMULATED environment

mine = custom_profile(
    "My Broker",
    commission=PerContractCommission(1.25),
    slippage=TickSlippage(1.0),
    fill_model_factory=TouchFill,        # optional; defaults to TradeThroughFill
    description="My broker's real fees.",
)
register_profile(mine)                   # optional: make it discoverable by name
```

**Realism is probed, not declared.** `profile.realism` runs each model through a behavioural probe (does
slippage move the price adversely? does commission cost anything? does the fill model fill on a mere
touch?). If any assumption is optimistic — even inside a model you wrote — `profile.is_realistic` is
`False` and the report states results are a simulated environment.

### Cost, slippage, and fill models

| Model | Options |
| --- | --- |
| Commission | `ZeroCommission()`, `PerContractCommission(per_contract)`, `PercentCommission(pct)` |
| Slippage | `NoSlippage()`, `FixedSlippage(amount)`, `TickSlippage(ticks)`, `PercentSlippage(pct)` |
| Fill | `TouchFill()` (optimistic), `TradeThroughFill(min_ticks_through=1.0)` (conservative default) |

Slippage is adverse by construction (BUY higher, SELL lower) and applies only to liquidity-taking fills
(market/stop). A limit fills at its limit price.

### Same-bar policy

The broker resolves a bar that spans both a stop and a target **stop-first** by default
(`SameBarPolicy.STOP_FIRST`). This is the pessimistic, honest choice.

## `assess_integrity` options

```python
report = assess_integrity(
    spec,
    include={"lookahead", "costs"},   # run only these checks (default: all applicable)
    exclude={"monte_carlo"},          # or drop specific ones
    seed=12345,                       # seed for stochastic checks (reproducible)
    oos_cut=pd.Timestamp("2024-06-01"),  # explicit in-/out-of-sample cut (default: 70% time split)
    config={"account": account},      # passed through to checks (e.g. Monte Carlo account limits)
)
```

### Tuning a check

Checks read tunables from class attributes, so you can subclass-and-register a variant into a copy of the
registry without touching the engine:

| Check | Tunable (default) |
| --- | --- |
| `MonteCarloCheck` | `n_paths` (2000), `min_trades` (10) |
| `FillRealismCheck` | `n_null_runs` (50) |
| `SlippageSensitivityCheck` | `levels_ticks` ((0, 0.5, 1, 2, 4)) |
| `RegimeConcentrationCheck` | WARN threshold: >70% of net from one year |

```python
from biasguard.validation.registry import DEFAULT_REGISTRY
from biasguard.validation.checks.montecarlo import MonteCarloCheck

class FastMC(MonteCarloCheck):
    n_paths = 500

reg = DEFAULT_REGISTRY.copy()
reg.register(FastMC())
report = assess_integrity(spec, registry=reg)
```

## Prop-firm account limits

Encode a funded account and let Monte Carlo estimate breach probabilities. `None` disables a limit; all
limits are **positive magnitudes**.

```python
from biasguard.montecarlo import AccountConfig

AccountConfig(
    starting_balance=50_000.0,
    trailing_drawdown_limit=2_000.0,   # breach if drawdown from the running peak exceeds this
    daily_loss_limit=1_000.0,          # intraday running loss within one (synthetic) day
    max_loss_limit=2_500.0,            # absolute floor: starting_balance - this
    profit_target=3_000.0,             # optional pass target
    trades_per_day=None,               # for daily-loss chunking; inferred from the ledger if None
)
```

Pass it via `config={"account": account}` to `assess_integrity` (the `monte_carlo` check uses it) or
directly to `MonteCarloSimulator.run(trades, account=account)`.

## Monte Carlo

```python
from biasguard.montecarlo import MonteCarloSimulator, CircularBlockBootstrap

MonteCarloSimulator(
    bootstrap=None,        # default StationaryBootstrap (streak-preserving); or CircularBlockBootstrap(...)
    n_paths=10_000,
    seed=1,                # deterministic given the seed
    sample_curves=100,     # how many raw equity paths to keep for plotting
)
```

Use `IIDBootstrap` only to *contrast* — it destroys streaks and understates drawdown, so it is never the
default.

## HTML report & audit

```python
build_html_report(
    metrics=..., equity=..., trades=...,
    manifest=...,              # adds the fingerprint + replay manifest
    validation_html=report.to_html(),   # the trust verdict block
    profile=REAL_MARKET,       # shows the profile + a banner if it's a simulated environment
    disclaimer="...",          # optional; auto-filled from a strategy's label if present
    ai_prompt=audit.to_ai_prompt(),     # optional; embeds a Download/Copy "AI debug prompt" button
    monte_carlo=mc_result,     # optional; adds a resampled-equity fan chart + distribution summary
    path="report.html",        # writes the report + a companion .manifest.json
)

build_audit(report, manifest=..., profile=..., metrics=..., monte_carlo=...).write("audit/")
```

Everything above is deterministic: the same inputs produce byte-identical reports, audits, and
fingerprints.
