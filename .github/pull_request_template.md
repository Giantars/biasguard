## What & why

What does this change, and why? If it addresses a failure mode, link the issue and name the trap.

## How it was verified

- [ ] `ruff check src tests examples` passes
- [ ] `black --check src tests examples` passes
- [ ] `mypy src` passes
- [ ] `pytest` passes
- [ ] Added/updated tests that would fail without this change

For changes touching causality or execution:

- [ ] The decision path stays causal (survives the truncation test)
- [ ] If this adds/changes a causality check, a planted-bug fixture in `tests/known_bad/` proves it catches the bug
- [ ] Determinism preserved (no wall-clock/RNG in the decision path; Monte Carlo seeded)

## Notes for reviewers

Anything non-obvious: a design trade-off, a naming choice, a follow-up you're deferring.
