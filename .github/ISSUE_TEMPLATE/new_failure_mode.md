---
name: New failure mode
about: A way a backtest lies that biasguard doesn't catch yet (the most valuable contribution)
title: "[failure-mode] "
labels: "failure-mode, enhancement"
---

biasguard's validation layer is a catalog of ways backtests lie (see
[how-backtests-lie.md](../../docs/how-backtests-lie.md)). Found one it misses? That's the heart of the
project.

## The trap

How does the backtest lie? What produces the fictional edge?

## How it bites

A concrete (anonymized) example: the fictional result vs. the honest one.

## How to detect it

The test or perturbation that separates the lie from a real edge. Ideally: what re-run, comparison, or
null makes the fiction visible?

## Suggested hook

How it might become a check or an engine guarantee. Bonus: a **planted-bug fixture** (a strategy/engine
that exhibits the flaw) so the new check can be power-validated — a causality/realism test that can't go
red on purpose is worthless.
