"""Tests for the bootstrap resamplers — determinism, streaks, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from biasguard.montecarlo.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    StationaryBootstrap,
)

VALUES = np.arange(10, dtype="float64")  # value == index, so we can inspect structure


def rng(seed: int = 1) -> np.random.Generator:
    return np.random.default_rng(seed)


def _is_dup(path: np.ndarray) -> bool:
    return bool(path[0] == path[1])


class TestStationary:
    def test_deterministic(self) -> None:
        a = StationaryBootstrap().resample(VALUES, rng(7))
        b = StationaryBootstrap().resample(VALUES, rng(7))
        np.testing.assert_array_equal(a, b)

    def test_length_and_membership(self) -> None:
        out = StationaryBootstrap().resample(VALUES, rng(), size=25)
        assert len(out) == 25
        assert set(out).issubset(set(VALUES))

    def test_large_block_preserves_streaks(self) -> None:
        # With a huge mean block, consecutive draws should mostly be adjacent
        # (index+1), i.e. streaks are preserved; with block 1 they should not.
        big = StationaryBootstrap(mean_block=1000).resample(VALUES, rng(3), size=500)
        small = StationaryBootstrap(mean_block=1.0).resample(VALUES, rng(3), size=500)

        def adjacent_fraction(path: np.ndarray) -> float:
            diffs = (path[1:] - path[:-1]) % len(VALUES)
            return float((diffs == 1).mean())

        assert adjacent_fraction(big) > 0.9
        assert adjacent_fraction(small) < 0.3

    def test_no_head_duplication_bias(self) -> None:
        # Regression: on a restart, position 1 must be an independent draw, not a
        # forced duplicate of position 0. With mean_block=2 (p=0.5) over 100
        # values the duplication rate should be ~0.5% (=p/m), not ~50%.
        r = rng(11)
        values = np.arange(100.0)
        boot = StationaryBootstrap(mean_block=2.0)
        trials = 4000
        dups = sum(1 for _ in range(trials) if _is_dup(boot.resample(values, r, size=2)))
        assert dups / trials < 0.05

    def test_rejects_bad_block(self) -> None:
        with pytest.raises(ValueError):
            StationaryBootstrap(mean_block=0.5)

    def test_empty(self) -> None:
        assert StationaryBootstrap().resample(np.empty(0), rng()).size == 0


class TestCircularBlock:
    def test_deterministic_and_length(self) -> None:
        a = CircularBlockBootstrap(block_length=3).resample(VALUES, rng(2), size=20)
        b = CircularBlockBootstrap(block_length=3).resample(VALUES, rng(2), size=20)
        np.testing.assert_array_equal(a, b)
        assert len(a) == 20

    def test_membership(self) -> None:
        out = CircularBlockBootstrap(block_length=4).resample(VALUES, rng())
        assert set(out).issubset(set(VALUES))

    def test_block_structure(self) -> None:
        # Within a block, values are consecutive (mod n).
        out = CircularBlockBootstrap(block_length=5).resample(VALUES, rng(9), size=10)
        first_block = out[:5]
        diffs = (first_block[1:] - first_block[:-1]) % len(VALUES)
        assert np.all(diffs == 1)

    def test_rejects_bad_block(self) -> None:
        with pytest.raises(ValueError):
            CircularBlockBootstrap(block_length=0)


class TestIID:
    def test_membership_and_length(self) -> None:
        out = IIDBootstrap().resample(VALUES, rng(), size=50)
        assert len(out) == 50
        assert set(out).issubset(set(VALUES))
