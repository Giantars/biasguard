"""Example 1 — the data layer and its trust verdict.

Run:  python examples/01_data_layer.py

Demonstrates the two things biasguard's data layer is for:

1. Loading OHLCV data into a canonical, timezone-aware contract.
2. Refusing to flatter you — the quality report catches duplicate timestamps,
   invalid OHLC geometry and missing candles instead of silently loading them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from biasguard.data import load_csv


def build_sample(n: int = 60, start: str = "2024-01-02 08:30") -> pd.DataFrame:
    """A small, clean, deterministic 1-minute NQ-like session (no RNG)."""
    idx = pd.date_range(start=start, periods=n, freq="1min")  # naive on purpose
    close = 15000.0 + np.arange(n) * 0.25
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 100.0 + (np.arange(n) % 10) * 5.0
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
    df.index.name = "timestamp"
    return df


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # 1) A clean file loads and passes ---------------------------------- #
        clean_path = tmpdir / "nq_clean.csv"
        build_sample().to_csv(clean_path)
        clean = load_csv(clean_path, tz="America/Chicago", session_gap="30min")

        print("=" * 70)
        print("CLEAN FILE")
        print("=" * 70)
        print(f"rows={len(clean)}  index_tz={clean.index.tz}  columns={list(clean.columns)}")
        print(clean.attrs["quality_report"].summary())
        print()

        # 2) A dirty file gets caught --------------------------------------- #
        dirty = build_sample()
        dirty.iloc[10, dirty.columns.get_loc("high")] = dirty.iloc[10]["low"] - 1.0  # high < low
        dirty = pd.concat([dirty, dirty.iloc[[0]]])  # duplicate timestamp
        dirty = dirty.drop(dirty.index[30])  # a missing candle
        dirty_path = tmpdir / "nq_dirty.csv"
        dirty.to_csv(dirty_path)

        # on_invalid="ignore" so we can print the verdict rather than raise.
        loaded = load_csv(
            dirty_path, tz="America/Chicago", session_gap="30min", on_invalid="ignore"
        )
        print("=" * 70)
        print("DIRTY FILE")
        print("=" * 70)
        report = loaded.attrs["quality_report"]
        print(report.summary())
        print()
        print(f"Trust verdict: {'PASS' if report.ok else 'FAIL — do not backtest on this data'}")


if __name__ == "__main__":
    main()
