"""Shared test fixtures.

All synthetic data is generated deterministically (no RNG) so that tests are
reproducible to the cent — the same discipline the framework itself follows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 120,
    *,
    freq: str = "1min",
    tz: str | None = "America/Chicago",
    start: str = "2024-01-02 08:30",
) -> pd.DataFrame:
    """Build a clean, valid OHLCV frame with a gentle deterministic uptrend.

    The construction guarantees the OHLC invariants (high is the max, low is the
    min) so the frame passes validation by design.
    """
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    close = 15000.0 + np.arange(n) * 0.25
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 100.0 + (np.arange(n) % 10) * 5.0
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A pristine 120-bar 1-minute NQ-like frame in America/Chicago."""
    return make_ohlcv()


@pytest.fixture
def ohlcv_factory() -> Callable[..., pd.DataFrame]:
    """Return the :func:`make_ohlcv` builder for tests needing custom frames."""
    return make_ohlcv


def write_csv(df: pd.DataFrame, path: str) -> str:
    """Write a frame to CSV with the index as a ``timestamp`` column."""
    df.to_csv(path, index=True)
    return path
