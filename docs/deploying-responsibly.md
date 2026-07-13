# Deploying responsibly

> `biasguard` measures **integrity**, not **edge**. A high Integrity Score means "this backtest is causal
> and honestly modelled" — *not* "this strategy is profitable". Read this before you risk money.

## Integrity is necessary, not sufficient

The Integrity Score answers "how much should I trust that these results are real?" It does **not** answer
"will this make money?" A strategy can earn a perfect 100/100 and still have no edge — it just means the
number you're looking at isn't a lie. The score's job is to stop you from trusting a *fictional* result;
it cannot manufacture a real one.

Concretely, a `PASS` on the lookahead gate means your strategy is causal. A `PASS` on `costs` means costs
are modelled. Neither means the equity curve will repeat out of sample. That is what the `regime`,
`out_of_sample`, and `monte_carlo` checks are for — and even those are cautions, not guarantees.

## A responsible workflow

1. **Assess integrity first.** Run `assess_integrity(spec)` before you look at the Sharpe. If the
   lookahead gate fails, the number is meaningless — fix the leak before anything else.
2. **Cost it realistically.** Use `REAL_MARKET` (or your broker's real fees), never a zero-cost profile,
   for any number you intend to believe. Watch the `costs` and `slippage_sensitivity` checks: an edge
   thinner than routine slippage is a fill artifact.
3. **Stress the sequence.** The `monte_carlo` check block-bootstraps your trade ledger. If the strategy
   is profitable in only a minority of realistic re-orderings, the historical gain is luck of sequence.
4. **Respect the drawdown, not the return.** Size against the *worst-case* drawdown distribution and, if
   you trade a funded/prop account, the breach probabilities from `AccountConfig` — not the median.
5. **Paper-trade before live.** Treat the shadow/paper period as the real out-of-sample test. Measure
   whether live fills match the backtest's assumptions.

## The `PROP_FIRM_SIM` profile is deliberately optimistic

Prop-firm evaluation/sim accounts often fill more generously than a live book (touch fills, little
slippage). `PROP_FIRM_SIM` models that, and `biasguard` **flags it as a simulated environment** on
purpose — so you never mistake a rosy sim result for what a live book will give you. If a strategy only
works under `PROP_FIRM_SIM` and dies under `REAL_MARKET`, that gap is the warning.

## Backtest → live drift (the expensive part)

The single most expensive failure in practice is not in the backtest — it's the live engine silently
diverging from it. Two rules, learned the hard way:

- **Backtest-as-brain.** Do not re-implement the strategy for live. A `biasguard` strategy is a pure
  function of its causal context (`on_bar(ctx)`), so the *same* function can drive both backtest and
  live. Before any capital, run a parity harness that replays history through the live decision path and
  asserts **N/N = 100%** decision match. Anything less than 100% is a bug, not a rounding difference.
- **Never skip a bar silently.** A forward-only "fetch from last_bar+1" live loop will eventually drop a
  late or out-of-order bar, and from then on live and backtest make different decisions. Use a rolling
  re-fetch (re-pull a trailing window each cycle, dedup keep-last) and a **gap detector** that logs
  loudly when a hole appears. Make a skip impossible to be silent.

`biasguard` v1.0.0 is a backtesting framework and ships **no live runner** — but the engine is built so a
future one can share the exact decision path. See the [roadmap](roadmap.md).

## Prop-firm risk analysis

If you trade a funded account with a trailing drawdown or daily loss limit, encode it and let Monte Carlo
estimate your breach probability:

```python
from biasguard.montecarlo import AccountConfig, MonteCarloSimulator

account = AccountConfig(
    starting_balance=50_000.0,
    trailing_drawdown_limit=2_000.0,   # blow-up if drawdown from the peak exceeds this
    daily_loss_limit=1_000.0,          # intraday, not end-of-day net
)
result = MonteCarloSimulator(n_paths=10_000, seed=1).run(run.trades, account=account)
print(result.summary())   # P(profit), p5/p50/p95, worst-case drawdown, P(breach each limit)
```

The daily-loss check is **intraday** (the running minimum within a day), so it catches a day that dipped
below the limit and recovered — the way a real prop firm would.

## The disclaimer, restated

Every bundled strategy is labelled *"Educational example — not intended as a profitable trading
strategy."* The framework reveals no edge and makes no recommendation. Use it to test **your own**
strategies, and treat a clean integrity report as permission to keep investigating — not as a signal to
trade.
