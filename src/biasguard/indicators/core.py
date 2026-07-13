"""Causal technical indicators — they only ever look backward.

Every function here is **causal by construction**: the value at index ``i``
depends only on inputs at indices ``<= i``. Combined with the engine's
:class:`~biasguard.engine.data_handler.MarketView` (which hands a strategy a
slice ending at the current bar), a strategy that indexes ``indicator(...)[-1]``
sees only completed history.

This is verifiable the same way the whole framework is: **truncation**. For any
prefix length ``k``, ``indicator(values[:k])`` equals ``indicator(values)[:k]``
element-for-element — no future value can leak into a past reading. Warm-up
periods return ``NaN`` rather than a peeked-ahead guess.

Inputs are 1-D array-likes of floats; outputs are ``float64`` arrays of the same
length.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ArrayLike = Sequence[float] | np.ndarray


def _asarray(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got {arr.ndim}-D")
    return arr


def _check_period(period: int) -> None:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")


def rolling_mean(values: ArrayLike, period: int) -> np.ndarray:
    """Trailing simple moving average; ``NaN`` for the first ``period - 1`` bars."""
    v = _asarray(values)
    _check_period(period)
    out = np.full(v.shape, np.nan)
    if v.size >= period:
        out[period - 1 :] = sliding_window_view(v, period).mean(axis=1)
    return out


def rolling_std(values: ArrayLike, period: int, *, ddof: int = 1) -> np.ndarray:
    """Trailing rolling standard deviation (sample ``ddof=1`` by default)."""
    v = _asarray(values)
    _check_period(period)
    out = np.full(v.shape, np.nan)
    if v.size >= period and period > ddof:
        out[period - 1 :] = sliding_window_view(v, period).std(axis=1, ddof=ddof)
    return out


def sma(values: ArrayLike, period: int) -> np.ndarray:
    """Simple moving average — an alias for :func:`rolling_mean`."""
    return rolling_mean(values, period)


def rolling_zscore(values: ArrayLike, period: int, *, ddof: int = 1) -> np.ndarray:
    """Trailing z-score: ``(value - rolling_mean) / rolling_std`` at each bar.

    Where the rolling standard deviation is zero (a flat window) the z-score is
    undefined and returned as ``NaN`` rather than ``+/-inf``.
    """
    v = _asarray(values)
    mean = rolling_mean(v, period)
    std = rolling_std(v, period, ddof=ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.asarray((v - mean) / std, dtype="float64")
    z[~np.isfinite(z)] = np.nan
    return z


def ema(values: ArrayLike, period: int) -> np.ndarray:
    """Exponential moving average (``adjust=False`` recursion, span = ``period``).

    Seeded with the first value and updated recursively, so it is causal and
    truncation-stable (each value depends only on the previous EMA and the
    current input).
    """
    v = _asarray(values)
    _check_period(period)
    out = np.empty(v.shape)
    if v.size == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = v[0]
    for i in range(1, v.size):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(values: ArrayLike, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index in ``[0, 100]``.

    The first defined value is at index ``period`` (it needs ``period + 1``
    prices); earlier bars are ``NaN``. Uses Wilder's recursive smoothing of
    average gains/losses, which is causal and truncation-stable.
    """
    v = _asarray(values)
    _check_period(period)
    n = v.size
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    delta = np.diff(v)  # delta[j] = v[j+1] - v[j]
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    avg_gain = float(gain[:period].mean())
    avg_loss = float(loss[:period].mean())
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for j in range(period, n - 1):  # delta index; price index is j + 1
        avg_gain = (avg_gain * (period - 1) + gain[j]) / period
        avg_loss = (avg_loss * (period - 1) + loss[j]) / period
        out[j + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """True range per bar. ``tr[0]`` is the first bar's range (no prior close)."""
    hi = _asarray(high)
    lo = _asarray(low)
    cl = _asarray(close)
    if not (hi.size == lo.size == cl.size):
        raise ValueError("high, low and close must be the same length")
    tr = np.empty(hi.shape)
    if hi.size == 0:
        return tr
    tr[0] = hi[0] - lo[0]
    prev_close = cl[:-1]
    tr[1:] = np.maximum.reduce(
        [hi[1:] - lo[1:], np.abs(hi[1:] - prev_close), np.abs(lo[1:] - prev_close)]
    )
    return tr


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> np.ndarray:
    """Wilder's Average True Range.

    The first defined value is at index ``period``; earlier bars are ``NaN``.
    Causal and truncation-stable (Wilder recursion over the true range).
    """
    _check_period(period)
    tr = true_range(high, low, close)
    n = tr.size
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    atr_val = float(tr[1 : period + 1].mean())
    out[period] = atr_val
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        out[i] = atr_val
    return out


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
