"""Bootstrap resamplers for Monte Carlo over a trade ledger.

The default is a **stationary block bootstrap**: it resamples *blocks* of
consecutive trades with random (geometric) lengths, which preserves win/loss
streaks and clustering. Plain IID resampling is provided only for contrast —
it destroys exactly the autocorrelation that drives real drawdowns, so it is
never the default.

Every resampler is a pure function of ``(values, rng)`` — deterministic given a
seeded :class:`numpy.random.Generator`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class Bootstrap(ABC):
    """Produces one resampled path of length ``size`` from ``values``."""

    name: str = "bootstrap"

    @abstractmethod
    def resample(
        self, values: np.ndarray, rng: np.random.Generator, size: int | None = None
    ) -> np.ndarray:
        """Return a resampled sequence of length ``size`` (default ``len(values)``)."""
        raise NotImplementedError


class StationaryBootstrap(Bootstrap):
    """Politis-Romano stationary bootstrap (geometric block lengths).

    Blocks of consecutive trades are wrapped circularly; a new random block is
    started with probability ``1 / mean_block`` at each step, giving an expected
    block length of ``mean_block`` (default ``max(2, sqrt(n))``). Preserves
    streaks without the artifacts of a single fixed block length.
    """

    name = "stationary_block"

    def __init__(self, mean_block: float | None = None) -> None:
        if mean_block is not None and mean_block < 1.0:
            raise ValueError("mean_block must be >= 1")
        self.mean_block = mean_block

    def resample(
        self, values: np.ndarray, rng: np.random.Generator, size: int | None = None
    ) -> np.ndarray:
        m = len(values)
        n = m if size is None else size
        if m == 0 or n == 0:
            return np.empty(0, dtype=values.dtype)
        mean_block = self.mean_block if self.mean_block is not None else max(2.0, math.sqrt(m))
        p = 1.0 / mean_block
        # Pre-draw randomness (2 rng calls) then a tight index loop. Each
        # restart uses its own fresh draw ``starts[k]``; ``starts[0]`` seeds the
        # first position only, so no draw is reused (Politis-Romano requires an
        # independent uniform start for every new block).
        starts = rng.integers(0, m, size=n)
        new_block = rng.random(n) < p
        idx = np.empty(n, dtype=np.intp)
        idx[0] = int(starts[0])
        for k in range(1, n):
            prev = int(idx[k - 1])
            idx[k] = int(starts[k]) if new_block[k] else (prev + 1 if prev + 1 < m else 0)
        return values[idx]


class CircularBlockBootstrap(Bootstrap):
    """Fixed-length circular block bootstrap.

    Concatenates ``ceil(size / block_length)`` blocks of ``block_length``
    consecutive (circularly-wrapped) trades, truncated to ``size``. Simpler than
    the stationary bootstrap but imposes one block length.
    """

    name = "circular_block"

    def __init__(self, block_length: int | None = None) -> None:
        if block_length is not None and block_length < 1:
            raise ValueError("block_length must be >= 1")
        self.block_length = block_length

    def resample(
        self, values: np.ndarray, rng: np.random.Generator, size: int | None = None
    ) -> np.ndarray:
        m = len(values)
        n = m if size is None else size
        if m == 0 or n == 0:
            return np.empty(0, dtype=values.dtype)
        length = self.block_length if self.block_length is not None else max(2, round(math.sqrt(m)))
        length = min(length, m)
        n_blocks = math.ceil(n / length)
        starts = rng.integers(0, m, size=n_blocks)
        offsets = np.arange(length)
        idx = ((starts[:, None] + offsets[None, :]) % m).reshape(-1)[:n]
        return np.asarray(values[idx])


class IIDBootstrap(Bootstrap):
    """Plain resample-with-replacement. **Destroys streaks** — provided only to
    contrast with the block methods; do not use as the primary Monte Carlo."""

    name = "iid"

    def resample(
        self, values: np.ndarray, rng: np.random.Generator, size: int | None = None
    ) -> np.ndarray:
        m = len(values)
        n = m if size is None else size
        if m == 0 or n == 0:
            return np.empty(0, dtype=values.dtype)
        return np.asarray(values[rng.integers(0, m, size=n)])


__all__ = [
    "Bootstrap",
    "CircularBlockBootstrap",
    "IIDBootstrap",
    "StationaryBootstrap",
]
