# biasguard v1.0.0

**A backtester that tries to prove your backtest wrong.**

This is the first public release of `biasguard` — a bias-resistant, event-driven backtesting framework
whose differentiator isn't speed or a strategy library, but **honesty**. Alongside the usual metrics it
ships a 0–100 **Integrity Score** that answers the only question worth asking before you risk money:
*how much should you actually trust these results?*

> ⚠️ This project reveals **no trading edge**. It is research infrastructure. Every bundled strategy is a
> trivial, clearly-labelled educational example.

## Highlights

- **Causality by construction, proven by truncation.** Signals fill on the next bar, and the `lookahead`
  gate re-runs your strategy on truncated history to assert past decisions are byte-identical. It's
  power-validated against planted-bug fixtures — a causality test that can't go red proves nothing.
- **Fill realism as a first-class concern.** Swappable fill models, and checks that separate real alpha
  from bracket mechanics (zero-slip survival, trade-through, random-direction null).
- **Costs are mandatory.** A zero-cost run is a FAIL, not a footnote.
- **Overfitting & regime luck** surfaced via per-year concentration, in-/out-of-sample consistency, and a
  streak-preserving block-bootstrap Monte Carlo — with prop-firm account-limit breach probabilities.
- **Execution profiles** that swap the whole cost/slippage/fill environment in one line, with realism
  *probed* so optimistic assumptions can't hide.
- **Deterministic replay fingerprint**, a **self-contained HTML report**, and an **AI audit export** that
  hands an LLM the full context to debug your strategy.

## Getting started

```bash
git clone https://github.com/Giantars/biasguard.git && cd biasguard
pip install -e ".[dev]"
python examples/08_educational_strategies.py     # see every check in action
```

Then read the [quick start](quickstart.md) and the [first-backtest tutorial](tutorial.md). The
[how-backtests-lie catalog](how-backtests-lie.md) is the doc that explains *why* every check exists.

## Stability

v1.0.0 marks a stable public API under [semantic versioning](https://semver.org/). See the
[changelog](../CHANGELOG.md) for the full contents and the [roadmap](roadmap.md) for what's next
(advanced overfitting statistics, a live/paper parity runner, tick-level fills).

## Thanks

`biasguard`'s validation layer is distilled from forensically auditing many "profitable" backtests to
death. If you find a way a backtest lies that it doesn't catch yet, that's the most valuable
[contribution](contributing.md) you can make.
