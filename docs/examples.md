# Example outputs

The `examples/` directory has eight runnable scripts, one per layer of the framework. Each is
self-contained, uses bundled synthetic data, and prints its results (some also write an HTML report or
audit files). Run any of them with:

```bash
python examples/08_educational_strategies.py
```

## The scripts

| Script | Demonstrates |
| --- | --- |
| `01_data_layer.py` | Loading + validating OHLCV: timezone handling, duplicate/gap/OHLC-violation detection, the data quality report. |
| `02_causal_engine.py` | The event loop and the causality guarantee: signals fill on the next bar, and the truncation test reproduces `data[:T]` decisions byte-for-byte. |
| `03_execution_core.py` | Broker, portfolio, positions, and order lifecycle under commission/slippage/fill models — the first realistic equity curve. |
| `04_report_and_fingerprint.py` | A **full showcase report** on multi-year data (every section populated: heatmap, trade distribution, Monte Carlo fan chart, per-year) written to `examples/output/showcase/report.html`, plus the deterministic replay fingerprint. |
| `05_integrity_framework.py` | The Backtest Integrity Framework: running the checks, the 0–100 score, and how a planted bug flips a verdict. |
| `06_monte_carlo.py` | Block-bootstrap Monte Carlo vs. IID, prop-firm account limits, and Monte Carlo feeding the Integrity Score. |
| `07_execution_profiles_and_audit.py` | Swapping execution profiles (and the "simulated environment" flag) plus the AI audit export. |
| `08_educational_strategies.py` | The full catalog below — every strategy run end-to-end with its report, integrity verdict, and audit. |

## The educational strategy catalog

`examples/08` runs `biasguard.strategies.CATALOG`: two honest templates and four intentionally-flawed
strategies, each paired with the verdict it is designed to demonstrate. Every strategy is labelled
*"Educational example — not intended as a profitable trading strategy."*

| Strategy | Watched check | Verdict | Why |
| --- | --- | --- | --- |
| **Moving Average Crossover** | `lookahead` | **PASS** | Causal trend-follower; SMAs read from the completed-bars view. Passing means *honest*, not profitable. |
| **RSI Mean Reversion** | `lookahead` | **PASS** | Causal reverter; RSI computed only from past closes. |
| **Lookahead Bias** | `lookahead` (gate) | **FAIL** | Reaches into the backing array to read tomorrow's close; truncation catches it and caps the score. |
| **Overfit Parameters** | `regime` | **WARN** | Causal, but ~84% of the P&L comes from one regime — a concentration, not a durable edge. |
| **Depends on Zero Slippage** | `slippage_sensitivity` | **WARN** | A thin 1–2 tick scalp edge that evaporates within a tick or two of slippage. |
| **Unrealistic Costs** | `costs` | **FAIL** | A high-turnover strategy run with zero fees; the cost check refuses it. |

Each case's `ExampleCase` records the strategy, its data generator, the execution profile, the watched
check, the expected status, and a plain-English explanation — so the examples double as executable
documentation, and the test-suite pins every verdict so they can never silently drift.

## Generated artifacts

`examples/08` writes, per strategy, into `examples/output/phase8/<key>/` (git-ignored):

- `report.html` — the self-contained visual report (open it in any browser),
- `report.html.manifest.json` — the replay manifest + fingerprint,
- `audit_report.md` / `audit_report.json` — the structured audit,
- `ai_debug_prompt.txt` — the paste-ready LLM debugging prompt.

Inspecting these side by side (a PASS report vs. a FAIL report) is the fastest way to build intuition for
what the Integrity Framework is telling you.
