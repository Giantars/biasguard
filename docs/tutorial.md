# Your first backtest

This walks through a complete run on **your own data**: load a CSV, write a small strategy, execute it
under a realistic profile, get a trust verdict, and export a report. It assumes you've
[installed](installation.md) the package.

## 1. Prepare your data

`biasguard` works on OHLCV bars. A CSV needs a timestamp column and open/high/low/close (volume
optional). Column names are flexible — `load_csv` maps common aliases (`Open`, `o`, `Last`, `Vol`, …)
onto the canonical schema.

```
timestamp,open,high,low,close,volume
2024-01-02 09:30:00,15000.25,15004.50,14999.00,15002.75,1234
2024-01-02 09:31:00,15002.75,15008.00,15001.50,15006.25,987
...
```

## 2. Load it — timezone is mandatory

The single most common silent bug is a timezone mismatch (an "ET" file that is really exchange-local).
`biasguard` refuses to guess:

```python
from biasguard.data import load_csv

df = load_csv("mydata.csv", tz="America/New_York")   # tz is required
print(df.attrs["quality_report"].summary())          # duplicates, gaps, OHLC violations
```

If the file has invalid OHLC bars, duplicate timestamps, or an unsorted index, the quality report says
so. Errors block the run; warnings (like session gaps) don't.

## 3. Write a strategy

A strategy subclasses `Strategy` and implements `on_bar`. It reads the world **only** through the
`StrategyContext` — a causal view of the past plus its current position — and returns zero or more
signals. Use the bundled [causal indicators](api-reference.md#indicators) and read the latest value with
`[-1]`.

```python
from collections.abc import Sequence

from biasguard.strategy import Strategy, StrategyContext
from biasguard.events import SignalEvent
from biasguard.indicators import ema


class EmaCross(Strategy):
    """Long when the fast EMA is above the slow EMA; flat otherwise."""

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        self.fast = fast          # public attrs are captured in the replay fingerprint
        self.slow = slow

    def on_bar(self, ctx: StrategyContext) -> Sequence[SignalEvent]:
        closes = ctx.view.closes          # numpy slice of closes up to *this* bar
        if len(closes) < self.slow:
            return ()
        fast = ema(closes, self.fast)[-1]
        slow = ema(closes, self.slow)[-1]
        if fast > slow and ctx.position <= 0:
            return (ctx.long(),)
        if fast < slow and ctx.position > 0:
            return (ctx.exit(),)
        return ()
```

Because `on_bar` can only see `ctx.view` (bars `[0..i]`), it is causal by construction — and the
truncation test will prove it.

## 4. Pick an instrument and an execution profile

The instrument sets contract economics (multiplier, tick size). The profile bundles commission,
slippage, and fill assumptions. Use a preset or define your own:

```python
from biasguard.execution import Instrument, REAL_MARKET

MYFUT = Instrument("MYFUT", multiplier=20.0, tick_size=0.25)   # or a preset: NQ, ES, GC, ...
```

## 5. Run and assess

```python
from biasguard.validation import BacktestSpec, assess_integrity

spec = BacktestSpec.from_profile(
    data=df,
    strategy_factory=EmaCross,       # a factory, so each internal re-run starts fresh
    instrument=MYFUT,
    profile=REAL_MARKET,
)

run = spec.run()
print(f"net ${run.net_pnl:,.0f} over {len(run.trades)} trades")
print(run.metrics())                 # Sharpe, Sortino, max drawdown, profit factor, ...

report = assess_integrity(spec)
print(report.summary())              # the trust verdict
```

## 6. Read the verdict

Don't start with the Sharpe. Start here:

```python
print("score:", report.score, report.grade)
print("trustworthy:", report.trustworthy)          # no FAILs, gates passed, grade C+
for r in report.results:
    print(f"  [{r.status}] {r.name}: {r.summary}")
```

- A **FAIL on `lookahead`** means your strategy read the future — fix that before anything else; the
  score is capped and the rest is moot.
- A **FAIL on `costs`** means you ran with zero costs. Use a realistic profile.
- **WARNs** on `regime`, `slippage_sensitivity`, `out_of_sample`, or `monte_carlo` are cautions: the
  result may be real but is fragile. Investigate before trusting it.

## 7. Export a report and an AI audit

```python
from biasguard.analytics import build_manifest
from biasguard.audit import build_audit
from biasguard.report import build_html_report

manifest = build_manifest(df, EmaCross())
audit = build_audit(report, manifest=manifest, profile=REAL_MARKET, metrics=run.metrics())

build_html_report(
    metrics=run.metrics(), equity=run.equity, trades=run.trades,
    manifest=manifest, validation_html=report.to_html(), profile=REAL_MARKET,
    ai_prompt=audit.to_ai_prompt(),       # embeds a "Download / Copy AI debug prompt" button
    path="report.html",
)

audit.write("audit/")
```

Open `report.html` (self-contained, works offline). If something looks off, use the **"Debug with an AI
assistant"** buttons in the report — or paste `audit/ai_debug_prompt.txt` — into an LLM. It contains the
full, deterministic context (fingerprint, profile, findings, metrics, Monte Carlo summary, and concrete
investigation targets) to debug your strategy.

## Next steps

- [How backtests lie](how-backtests-lie.md) — what each check is actually protecting you from.
- [Configuration reference](configuration.md) — profiles, account limits, tuning checks.
- [Deploying responsibly](deploying-responsibly.md) — integrity is not edge.
