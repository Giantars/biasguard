"""Tests for the CSV / Parquet loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import make_ohlcv, write_csv

from biasguard.data.loaders import DataQualityWarning, load_csv, load_parquet
from biasguard.data.validation import DataQualityReport, DataValidationError


class TestLoadCsvHappyPath:
    def test_returns_canonical_contract(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        path = write_csv(clean_df, str(tmp_path / "clean.csv"))
        df = load_csv(path, tz="America/Chicago", session_gap="30min")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None
        assert df.index.is_monotonic_increasing

    def test_report_attached_to_attrs(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        path = write_csv(clean_df, str(tmp_path / "clean.csv"))
        df = load_csv(path, tz="America/Chicago", session_gap="30min")
        report = df.attrs["quality_report"]
        assert isinstance(report, DataQualityReport)
        assert report.ok
        assert df.attrs["tz"] == "America/Chicago"
        assert df.attrs["source"].endswith("clean.csv")


class TestTimezoneHandling:
    def test_tz_is_required(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        path = write_csv(clean_df, str(tmp_path / "c.csv"))
        with pytest.raises(ValueError, match="tz is required"):
            load_csv(path, tz="")

    def test_naive_timestamps_are_localized(self, tmp_path: Path) -> None:
        naive = make_ohlcv(tz=None)
        path = write_csv(naive, str(tmp_path / "naive.csv"))
        df = load_csv(path, tz="America/Chicago", session_gap="30min")
        assert df.index.tz is not None
        assert str(df.index.tz) == "America/Chicago"

    def test_aware_timestamps_are_converted(self, tmp_path: Path) -> None:
        utc = make_ohlcv(tz="UTC", start="2024-01-02 08:30")
        path = write_csv(utc, str(tmp_path / "utc.csv"))
        df = load_csv(path, tz="America/New_York", session_gap="30min")
        # 08:30 UTC in January is 03:30 America/New_York (EST, UTC-5).
        first = df.index[0]
        assert first.hour == 3 and first.minute == 30
        assert first.tzinfo is not None


class TestColumnResolution:
    def test_aliases_and_explicit_map(self, tmp_path: Path) -> None:
        raw = pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-01-02 08:30", periods=3, freq="1min"),
                "Open": [100.0, 100.5, 101.0],
                "High": [101.0, 101.5, 102.0],
                "Low": [99.5, 100.0, 100.5],
                "Last": [100.5, 101.0, 101.5],
                "Vol": [10, 12, 11],
            }
        )
        path = str(tmp_path / "weird.csv")
        raw.to_csv(path, index=False)
        df = load_csv(path, tz="America/Chicago", column_map={"Last": "close"})
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_explicit_timestamp_col(self, tmp_path: Path) -> None:
        raw = pd.DataFrame(
            {
                "bar_open_ts": pd.date_range("2024-01-02 08:30", periods=3, freq="1min"),
                "open": [100.0, 100.5, 101.0],
                "high": [101.0, 101.5, 102.0],
                "low": [99.5, 100.0, 100.5],
                "close": [100.5, 101.0, 101.5],
            }
        )
        path = str(tmp_path / "explicit.csv")
        raw.to_csv(path, index=False)
        df = load_csv(path, tz="UTC", timestamp_col="bar_open_ts")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_timestamp_raises(self, tmp_path: Path) -> None:
        raw = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
        path = str(tmp_path / "notime.csv")
        raw.to_csv(path, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_csv(path, tz="UTC")


class TestSortingAndValidationModes:
    def test_unsorted_input_is_sorted(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        shuffled = pd.concat([clean_df.iloc[60:], clean_df.iloc[:60]])
        path = write_csv(shuffled, str(tmp_path / "shuffled.csv"))
        df = load_csv(path, tz="America/Chicago", session_gap="30min")
        assert df.index.is_monotonic_increasing

    def test_duplicate_timestamps_raise_by_default(
        self, clean_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        dup = pd.concat([clean_df, clean_df.iloc[[0]]])
        path = write_csv(dup, str(tmp_path / "dup.csv"))
        with pytest.raises(DataValidationError):
            load_csv(path, tz="America/Chicago", session_gap="30min")

    def test_on_invalid_warn(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        dup = pd.concat([clean_df, clean_df.iloc[[0]]])
        path = write_csv(dup, str(tmp_path / "dup.csv"))
        with pytest.warns(DataQualityWarning):
            df = load_csv(path, tz="America/Chicago", session_gap="30min", on_invalid="warn")
        assert not df.attrs["quality_report"].ok

    def test_on_invalid_ignore(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        dup = pd.concat([clean_df, clean_df.iloc[[0]]])
        path = write_csv(dup, str(tmp_path / "dup.csv"))
        df = load_csv(path, tz="America/Chicago", session_gap="30min", on_invalid="ignore")
        assert not df.attrs["quality_report"].ok  # attached, but not raised

    def test_validate_false_skips_report(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        path = write_csv(clean_df, str(tmp_path / "c.csv"))
        df = load_csv(path, tz="America/Chicago", validate=False)
        assert "quality_report" not in df.attrs


class TestParquet:
    def test_parquet_roundtrip(self, clean_df: pd.DataFrame, tmp_path: Path) -> None:
        path = str(tmp_path / "clean.parquet")
        clean_df.to_parquet(path)
        df = load_parquet(path, tz="America/Chicago", session_gap="30min")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.tz is not None
        assert df.attrs["quality_report"].ok

    def test_parquet_converts_timezone(self, tmp_path: Path) -> None:
        utc = make_ohlcv(tz="UTC", start="2024-01-02 08:30")
        path = str(tmp_path / "utc.parquet")
        utc.to_parquet(path)
        df = load_parquet(path, tz="America/New_York", session_gap="30min")
        assert df.index[0].hour == 3
