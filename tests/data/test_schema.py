"""Tests for the OHLCV schema contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from biasguard.data.schema import (
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    VOLUME,
    Bar,
    canonicalize_columns,
    resolve_timestamp_column,
)


class TestCanonicalizeColumns:
    def test_renames_mixed_case_and_aliases(self) -> None:
        df = pd.DataFrame({"Open": [1.0], "HIGH": [2.0], "low": [0.5], "Last": [1.5], "Vol": [10]})
        out = canonicalize_columns(df)
        assert set(out.columns) == {OPEN, HIGH, LOW, CLOSE, VOLUME}

    def test_explicit_map_wins(self) -> None:
        df = pd.DataFrame({"px_open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
        out = canonicalize_columns(df, column_map={"px_open": OPEN})
        assert OPEN in out.columns

    def test_unrecognized_columns_pass_through(self) -> None:
        df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "oi": [42]})
        out = canonicalize_columns(df)
        assert "oi" in out.columns


class TestResolveTimestampColumn:
    def test_finds_alias(self) -> None:
        cols = pd.Index(["datetime", "open", "close"])
        assert resolve_timestamp_column(cols) == "datetime"

    def test_explicit_takes_precedence(self) -> None:
        cols = pd.Index(["t", "open"])
        assert resolve_timestamp_column(cols, explicit="t") == "t"

    def test_explicit_missing_raises(self) -> None:
        cols = pd.Index(["open", "close"])
        with pytest.raises(KeyError):
            resolve_timestamp_column(cols, explicit="nope")

    def test_returns_none_when_absent(self) -> None:
        cols = pd.Index(["open", "high", "low", "close"])
        assert resolve_timestamp_column(cols) is None


class TestBar:
    def test_valid_bar(self) -> None:
        ts = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")
        bar = Bar(timestamp=ts, open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0)
        assert bar.close == 100.5
        assert bar.volume == 10.0

    def test_from_row_without_volume(self) -> None:
        ts = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")
        row = pd.Series({OPEN: 100.0, HIGH: 101.0, LOW: 99.0, CLOSE: 100.5})
        bar = Bar.from_row(ts, row)
        assert bar.volume is None

    def test_from_row_with_nan_volume(self) -> None:
        ts = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")
        row = pd.Series({OPEN: 100.0, HIGH: 101.0, LOW: 99.0, CLOSE: 100.5, VOLUME: float("nan")})
        bar = Bar.from_row(ts, row)
        assert bar.volume is None

    @pytest.mark.parametrize(
        "o,h,lo,c",
        [
            (100.0, 99.0, 98.0, 99.5),  # high < low
            (100.0, 100.5, 99.0, 101.0),  # high not the max (close above high)
            (100.0, 101.0, 100.5, 100.2),  # low not the min (open below low)
        ],
    )
    def test_invalid_bar_raises(self, o: float, h: float, lo: float, c: float) -> None:
        ts = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")
        with pytest.raises(ValueError):
            Bar(timestamp=ts, open=o, high=h, low=lo, close=c)

    def test_bar_is_frozen(self) -> None:
        ts = pd.Timestamp("2024-01-02 08:30", tz="America/Chicago")
        bar = Bar(timestamp=ts, open=100.0, high=101.0, low=99.0, close=100.5)
        with pytest.raises(FrozenInstanceError):
            bar.close = 200.0  # type: ignore[misc]
