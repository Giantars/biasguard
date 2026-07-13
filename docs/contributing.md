# Contributing

Thanks for considering a contribution. `biasguard` is research infrastructure: correctness and honesty
matter more than features. This guide covers the setup, the quality gate every change must pass, and the
conventions that keep the framework trustworthy.

## Setup

```bash
git clone https://github.com/Giantars/biasguard.git
cd biasguard
python -m venv .venv && . .venv/Scripts/Activate.ps1   # or: source .venv/bin/activate
pip install -e ".[dev]"
```

## The quality gate

CI runs exactly these four commands, and so should you before opening a PR. All must pass:

```bash
ruff check src tests examples      # lint
black --check src tests examples   # formatting (line length 100)
mypy src                           # strict type checking
pytest -q                          # the full suite
```

A [pre-commit](https://pre-commit.com/) config is provided — `pre-commit install` runs ruff + black on
every commit so you never push a formatting failure.

## Non-negotiable conventions

These are the properties that make `biasguard` worth using; a PR that breaks one will be asked to change.

- **Layering points downward.** A package may import lower layers only (see
  [architecture.md](architecture.md)). The engine must never import `execution`, `validation`, etc. —
  it depends on the `Portfolio`/`Broker` Protocols.
- **Causality by construction.** A strategy reads the world only through `StrategyContext` /
  `MarketView`. Nothing may read past the current bar. If you add anything that touches the decision
  path, it must survive the truncation test.
- **Power-validate causality claims.** If you add or change a causality check, add a planted-bug fixture
  in `tests/known_bad/` and prove the check catches it. A causality test that can't go red is worthless.
- **Determinism.** No wall-clock or RNG in the decision path. Seed all Monte Carlo. Test data is
  generated deterministically (no RNG in fixtures) so results are reproducible to the cent.
- **Costs are first-class.** Don't add a path that silently trades for free.

## Adding things (no engine change required)

| To add… | How |
| --- | --- |
| **Integrity check** | Subclass `IntegrityCheck`, set `key`/`name`/`category`/`weight`/`is_gate`, implement `run(ctx)` returning `self.result(...)`. Register with `register_check(MyCheck())`. Add a test asserting PASS on a clean run and the flagged status on a deliberately-bad one. |
| **Execution profile** | Build an `ExecutionProfile` (or `custom_profile(...)`) and `register_profile(...)`. Realism is *probed*, so a new optimistic model is flagged automatically. |
| **Cost / slippage / fill model** | Subclass the ABC in `execution/costs.py` or `execution/fill_models.py`. Keep it stateless and deterministic. |
| **Indicator** | Add a pure, strictly-trailing function to `indicators/core.py`. **Add a truncation-causality test** (`indicator(values[:k]) == indicator(values)[:k]`, exact). |
| **Bootstrap** | Subclass `Bootstrap` in `montecarlo/bootstrap.py`. Preserve determinism given a seeded generator. |

## Testing conventions

- Tests mirror `src/` (e.g. `tests/validation/` for `src/biasguard/validation/`).
- Prefer deterministic synthetic data (`tests/conftest.py::make_ohlcv`, or the generators in
  `biasguard.strategies`) over RNG.
- Expensive assessments (fill-realism replays dozens of null runs) should be **memoized** when reused
  across tests — see `tests/strategies/test_educational.py`.
- New public behaviour needs a test that would fail without it (no tautologies).

## Code style

- Type hints on every function; the package ships `py.typed` and must stay `mypy --strict`-clean.
- Docstrings explain *why*, not just *what* — the codebase is meant to be read.
- Match the surrounding code's naming and idiom. Public config is a plain attribute; mutable state is
  underscore-prefixed (this convention drives the fingerprint's parameter introspection).

## Pull requests

1. Branch from `main`.
2. Make the change + tests; run the gate locally.
3. Fill in the [pull request template](../.github/pull_request_template.md) — describe the failure mode
   or capability, and how you verified it.
4. CI must be green. Reviews focus on correctness, causality, and whether the change preserves the
   framework's honesty guarantees.

By contributing you agree your work is licensed under the project's [MIT license](../LICENSE).
