"""Causal technical indicators (SMA, EMA, RSI, ATR, rolling statistics).

Every function is causal by construction — the value at index ``i`` depends only
on inputs at ``<= i`` — and truncation-stable, matching the framework's
no-lookahead guarantee. See :mod:`biasguard.indicators.core`.

    from biasguard.indicators import sma, rsi
    signal = rsi(view.closes, 14)[-1]   # only sees completed bars
"""

from __future__ import annotations

from biasguard.indicators.core import (
    ArrayLike,
    atr,
    ema,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    rsi,
    sma,
    true_range,
)

__all__ = [
    "ArrayLike",
    "atr",
    "ema",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "rsi",
    "sma",
    "true_range",
]
