# How backtests lie

> A backtest is a *claim*. `biasguard`'s job is to make the claim checkable and to refuse to flatter
> you. This document is the catalog of failure modes the framework was built against — each one drawn
> from real, forensically-audited "profitable" backtests that turned out to be lying — and the check
> that catches it.

Almost every backtest that looks great is wrong, and the ways it is wrong cluster into a small set of
repeatable traps. Below, each entry gives **the trap**, **how it bites**, and **how biasguard catches
it** (with the integrity-check key you'll see in a report).

The checks run via [`assess_integrity`](api-reference.md); each returns `PASS` / `WARN` / `FAIL` / `SKIP`
and a number, aggregated into the 0–100 Integrity Score.

---

## A. Lookahead & future leakage

### 1. Entry-bar lookahead — `lookahead` (gate)

**Trap.** Deciding a signal on bar *i* and filling it on that same still-forming bar, using the bar's
eventual close/high/low. In one audited case an engine showed **+$1.59M**; the no-leak variant **lost
$620K** — the entire "edge" was a 15-minute entry-bar leak.

**How biasguard catches it.** Causality is structural: a strategy sees a `MarketView` bounded at the
current bar, and a signal on bar *i*'s close fills no earlier than bar *i+1*'s open. On top of that, the
`lookahead` check runs the **truncation test** — it re-runs your strategy on `data[:T]` for several `T`
and asserts every past decision is **byte-identical** to the full-history run. If truncating the future
changes a past decision, the strategy used information it could not have had live. This is a **gate**: a
failure caps the entire Integrity Score, because a causality leak means "do not trust", however good
everything else looks.

### 2. Repainting indicators — `lookahead`

**Trap.** An indicator or "level" that *retroactively changes* as future bars arrive — a centered moving
average, a zig-zag, a level recomputed from later bars. Live can never see the repainted value.

**How biasguard catches it.** Repaint is just lookahead by another name, and the same truncation test
catches it: if the signal set computed on `data[:T]` differs from the full run's first `T` signals
(0 added, 0 removed, 0 changed), the indicator repainted. The bundled [indicators](api-reference.md#indicators)
are all strictly trailing and **truncation-stable** by construction, and there is a test that proves it.

### 3. Exit-bar lookahead — `lookahead`, power-validated

**Trap.** Exit logic peeking one bar or tick ahead to pick target-vs-stop or to exit early.

**How biasguard catches it.** The truncation test covers exits too. Crucially, the test suite is
**power-validated**: `tests/known_bad/` ships strategies containing deliberate leaks, and the causality
tests must catch them. *A causality test that has never gone red is worthless.*

### 4. Same-bar stop + target ambiguity — engine default

**Trap.** A bar spans both the stop and the target; assuming the target filled is optimistic.

**How biasguard catches it.** The broker resolves same-bar stop+target **stop-first (pessimistic)** by
default (`SameBarPolicy.STOP_FIRST`) — the sim never hands you the favourable exit for free.

### 5. Partial / forming-bar leakage — engine default

**Trap.** Feeding an incomplete bar to an indicator that then "knows" the bar's final OHLC.

**How biasguard catches it.** A **completed-bars-only gate**: the engine emits a bar to the strategy only
after it has closed. The `MarketView` cannot index past the current bar (it raises `IndexError`).

### 6. Timezone mislabeling — `ohlc`

**Trap.** Intraday data assumed to be one timezone but actually exchange-local (e.g. an NQ CSV that is
Central time, not Eastern) — every session filter is then silently wrong.

**How biasguard catches it.** Timezone is **mandatory** on load (`load_csv(..., tz=...)`), and the `ohlc`
data-integrity check validates the index is tz-aware, sorted, and unique before a bar is traded.

---

## B. Fill fiction & execution realism

This is the most under-modelled area in every framework, and where real strategies die.

### 7. Phantom fills — `fill_realism`

**Trap.** A resting limit assumed to fill because price *touched* the level, when the market only tapped
it (you were behind the queue) or gapped through it. Tick audits found ~70% of passive fills at levels
*never actually printed at the level*. The tell: **resting-order-fills-at-touch + a directional filter
that picks which side = a phantom "edge" that survives zero slippage, because it is order mechanics, not
alpha.**

**How biasguard catches it.** The fill model is a first-class, swappable object, and the `fill_realism`
check runs the three killer tests:

- **Zero-slip survival** — if removing slippage *is* the whole P&L, the edge is a fill artifact.
- **Trade-through requirement** — re-run under `TradeThroughFill` (price must trade *through* the limit,
  not merely touch it); a real edge keeps most of its P&L, a phantom collapses.
- **Random-direction null** — book the same entry/exit *timing* with a coin-flip side. If the null earns
  as much, the P&L is direction-free bracket mechanics; the report subtracts it and reports
  `alpha = net − null_mean`.

### 8. Fill-price realism ≠ fill existence — fill-model default

**Trap.** The subtlest, and it kills real strategies. The sim confirms a fill *happened* but books it at
an **unachievable price** — a passive limit filled at the *favourable open of a wide bar* instead of at
the limit. On intraday bars spanning 40–80+ points, crediting a resting limit with the bar's opening
extreme is tens of points of price improvement no limit can receive. One fade backtest showed
**+$9,075/unit**; the fill-at-open improvement was **164% of net**; repriced at the actual limit it was
**−$10k to −$16k** — a losing strategy whose entire edge was the fill price. It passed truncation,
repaint, and an independent re-pricer *to the cent*, because all of those verified the engine was
**self-consistent**, not that the price was **achievable**.

**How biasguard catches it.** The fill models fill a limit **at its limit price** — never at an intrabar
favourable extreme. A better-than-limit fill is only booked when the bar's *open* is already through the
level (a genuine between-bar gap, which is real price improvement). **Key lesson: "reproduces to the
cent" and "no time-lookahead" are necessary but not sufficient — fill-price achievability is a separate
audit axis.**

### 9. Fill-optimism, generally — `slippage_sensitivity`

**Trap.** The sim fills easier than reality. The zero-slippage sim is the *maximally optimistic* case.
It is worst for mean-reversion/fades: the winning trades (clean touch-and-reverse) are exactly the ones
least likely to fill, so the backtest keeps winners it wouldn't get.

**How biasguard catches it.** The `slippage_sensitivity` check re-runs with escalating slippage and finds
the break-even level. An edge that dies within a tick or two of slippage is a fill artifact, not alpha,
and is flagged WARN/FAIL. Order types are distinguished by fill reliability — market/stop fill through
(reliable), limits are queue-dependent (optimistic).

### 10. Missing / unrealistic transaction costs — `costs`

**Trap.** Zero commission and slippage. Many high-frequency edges are sub-cost and flip negative once
realistic costs apply.

**How biasguard catches it.** Cost models are **mandatory**. The `costs` check **FAILs** (not warns) if
commission and slippage are both zero, and WARNs if the median trade P&L is under 2× round-trip cost.

### 11. Instrument / basis mismatch — replay fingerprint

**Trap.** Backtest on NQ, deploy on MNQ (or a different feed) — different fill prices and bar density.

**How biasguard catches it.** The instrument and every execution input are captured in the replay
manifest and folded into the deterministic fingerprint, so a mismatch is visible in the run's identity.

---

## C. Statistical & measurement

### 12. Overfitting / multiplicity — `monte_carlo`, `out_of_sample`

**Trap.** Trying N configurations and reporting the best — the winner is drawn from a best-of-N maximum,
not a random trial.

**How biasguard catches it.** The `monte_carlo` check resamples the trade ledger with a **block
bootstrap** (preserving win/loss streaks) and asks how often the strategy stays profitable under
realistic re-orderings; a historical profit that survives only a minority of the time is luck of
sequence. The `out_of_sample` check splits the run in time and flags a profit that exists in-sample but
not out. *(Deflated Sharpe / PBO / CSCV are on the [roadmap](roadmap.md), not in v1.0.0.)*

### 13. Regime dependence — `regime`

**Trap.** An all-green sample window shows a tiny drawdown that is a *floor, not a ceiling* — the bad
regime simply isn't in the data.

**How biasguard catches it.** The `regime` check breaks P&L down per calendar year and WARNs when too
much of the net comes from a single year. The HTML report always shows the per-year breakdown.

### 14. In-sample / dev-window reproduction — `out_of_sample`

**Trap.** Reproducing a result on the *same* data the strategy was tuned on is not out-of-sample.

**How biasguard catches it.** The `out_of_sample` check makes the in-sample/out-of-sample split explicit
and labels the comparison; a strategy whose edge evaporates in the later period is flagged.

### 15. Trust-me statistics — replay fingerprint

**Trap.** Headline numbers that can't be reproduced because the data/config weren't shipped.

**How biasguard catches it.** Every run carries a **replay fingerprint** and a manifest of all its
inputs; the report and audit export are deterministic, so the same run reproduces byte-for-byte.

---

## D. Backtest → live drift

These are live-trading traps. `biasguard` v1.0.0 is a backtesting framework (no live runner), so these
live in [deploying-responsibly.md](deploying-responsibly.md) — but they are the reason the engine is
built the way it is.

### 16. Deploy == backtest drift

**Trap.** The live engine *re-implements* the strategy and silently diverges — the single most expensive
class in practice. **Principle: backtest-as-brain** — the live driver should call the *same* strategy
functions on data-up-to-now, and prove parity (N/N = 100% decision match) before any capital. Because a
`biasguard` strategy is a pure function of its causal context, the same `on_bar` can drive both.

### 17. Silent bar-skipping (live ingestion)

**Trap.** A forward-only "fetch from last_bar+1" live loop drops a late/out-of-order bar permanently and
silently, so live makes different decisions than the backtest. The fix is a rolling re-fetch plus a loud
gap detector — see [deploying-responsibly.md](deploying-responsibly.md).

---

*This catalog is the distilled experience of auditing many "profitable" backtests to death. If you find
a failure mode it doesn't cover, that's a [contribution](contributing.md) worth making.*
