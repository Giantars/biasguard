"""Tests for the causal indicators — correctness, warm-up, and truncation-causality.

The headline property is *causality by truncation*: for any prefix length ``k``,
``indicator(values[:k])`` must equal ``indicator(values)[:k]`` element-for-element.
If it does not, a future value leaked into a past reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from biasguard.indicators import (
    atr,
    ema,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    rsi,
    sma,
    true_range,
)

# A deterministic, non-trivial series (no RNG — reproducible to the bit).
_T = np.arange(60, dtype="float64")
SERIES = 100.0 + np.sin(_T / 3.0) * 6.0 + _T * 0.15
HIGH = SERIES + 1.2
LOW = SERIES - 1.1
CLOSE = SERIES


def _assert_prefix_stable(full: np.ndarray, prefix: np.ndarray, k: int) -> None:
    # The claim is *byte-identical*, not merely close: the trailing/recursive
    # implementations perform the identical FP op sequence on a prefix, so exact
    # equality must hold. assert_allclose (rtol 1e-7) would be too weak.
    assert np.array_equal(prefix, full[:k], equal_nan=True)


class TestSma:
    def test_known_values(self) -> None:
        out = sma([1.0, 2.0, 3.0, 4.0], 2)
        np.testing.assert_allclose(out, [np.nan, 1.5, 2.5, 3.5], equal_nan=True)

    def test_warmup_is_nan(self) -> None:
        out = sma(SERIES, 5)
        assert np.all(np.isnan(out[:4]))
        assert not np.isnan(out[4])

    def test_is_alias_of_rolling_mean(self) -> None:
        np.testing.assert_array_equal(sma(SERIES, 7), rolling_mean(SERIES, 7))

    def test_causal_under_truncation(self) -> None:
        full = sma(SERIES, 5)
        for k in (3, 5, 6, 30, 59):
            _assert_prefix_stable(full, sma(SERIES[:k], 5), k)

    def test_period_longer_than_series(self) -> None:
        assert np.all(np.isnan(sma([1.0, 2.0], 5)))

    def test_rejects_bad_period(self) -> None:
        with pytest.raises(ValueError):
            sma(SERIES, 0)


class TestEma:
    def test_recursive_formula(self) -> None:
        out = ema([1.0, 2.0, 3.0], 2)  # alpha = 2/3
        assert out[0] == 1.0
        assert out[1] == pytest.approx(2.0 / 3.0 * 2.0 + 1.0 / 3.0 * 1.0)
        assert out[2] == pytest.approx(2.0 / 3.0 * 3.0 + 1.0 / 3.0 * out[1])

    def test_no_nan(self) -> None:
        assert not np.any(np.isnan(ema(SERIES, 10)))

    def test_causal_under_truncation(self) -> None:
        full = ema(SERIES, 10)
        for k in (1, 2, 20, 59):
            _assert_prefix_stable(full, ema(SERIES[:k], 10), k)

    def test_empty(self) -> None:
        assert ema([], 5).size == 0


class TestRollingStdZscore:
    def test_std_matches_numpy(self) -> None:
        out = rolling_std([1.0, 2.0, 3.0, 4.0, 5.0], 3, ddof=1)
        assert out[2] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
        assert out[4] == pytest.approx(np.std([3.0, 4.0, 5.0], ddof=1))

    def test_zscore_flat_window_is_nan(self) -> None:
        # A perfectly flat window has zero std -> z-score undefined (NaN, not inf).
        out = rolling_zscore([5.0, 5.0, 5.0, 5.0], 3)
        assert np.all(np.isnan(out))

    def test_zscore_causal_under_truncation(self) -> None:
        full = rolling_zscore(SERIES, 10)
        for k in (9, 10, 25, 59):
            _assert_prefix_stable(full, rolling_zscore(SERIES[:k], 10), k)


class TestRsi:
    def test_bounded_0_100(self) -> None:
        out = rsi(SERIES, 14)
        defined = out[~np.isnan(out)]
        assert np.all((defined >= 0.0) & (defined <= 100.0))

    def test_all_gains_is_100(self) -> None:
        out = rsi(np.arange(1.0, 30.0), 14)  # strictly increasing
        assert out[-1] == pytest.approx(100.0)

    def test_all_losses_is_0(self) -> None:
        out = rsi(np.arange(30.0, 1.0, -1.0), 14)  # strictly decreasing
        assert out[-1] == pytest.approx(0.0)

    def test_first_value_at_period(self) -> None:
        out = rsi(SERIES, 14)
        assert np.all(np.isnan(out[:14]))
        assert not np.isnan(out[14])

    def test_causal_under_truncation(self) -> None:
        full = rsi(SERIES, 14)
        for k in (10, 15, 16, 40, 59):
            _assert_prefix_stable(full, rsi(SERIES[:k], 14), k)


class TestAtr:
    def test_true_range_first_bar(self) -> None:
        tr = true_range([10.0, 12.0], [9.0, 11.0], [9.5, 11.5])
        assert tr[0] == pytest.approx(1.0)  # high - low
        # bar 1: max(12-11, |12-9.5|, |11-9.5|) = 2.5
        assert tr[1] == pytest.approx(2.5)

    def test_positive_and_warmup(self) -> None:
        out = atr(HIGH, LOW, CLOSE, 14)
        assert np.all(np.isnan(out[:14]))
        defined = out[~np.isnan(out)]
        assert np.all(defined > 0.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            true_range([1.0, 2.0], [1.0], [1.0, 2.0])

    def test_causal_under_truncation(self) -> None:
        full = atr(HIGH, LOW, CLOSE, 14)
        for k in (10, 15, 16, 40, 59):
            _assert_prefix_stable(full, atr(HIGH[:k], LOW[:k], CLOSE[:k], 14), k)
