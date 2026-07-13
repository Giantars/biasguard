---
name: Bug report
about: Something behaves incorrectly (a wrong number, a crash, a non-deterministic result)
title: "[bug] "
labels: bug
---

## What happened

A clear description of the incorrect behaviour.

## Expected behaviour

What you expected instead.

## Minimal reproduction

A small, self-contained snippet (use bundled data / `biasguard.strategies` generators if possible):

```python
# ...
```

## Determinism

- [ ] The result is reproducible (same inputs → same output). If it is **not** deterministic, that is
      itself the bug — please say so.

## Environment

- biasguard version: (`python -c "import biasguard; print(biasguard.__version__)"`)
- Python version:
- OS:

## Anything else

Stack traces, the replay fingerprint, or an `ai_debug_prompt.txt` snippet are all helpful.
