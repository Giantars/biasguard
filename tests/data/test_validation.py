"""Tests for the data-quality detection primitives."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from biasguard.data.validation import (
    DataValidationError,
    find_duplicate_timestamps,
    find_gaps,
    find_ohlc_violations,
    infer_frequency,
    validate_ohlcv,
)
from biasguard.types import Severity


class TestCleanFrame:
    def test_clean_frame_passes(self, clean_df: pd.DataFrame) -> None:
        report = validate_ohlcv(clean_df, session_gap="30min")
        assert report.ok
        assert report.errors == ()
        assert report.inferred_freq == pd.Timedelta("1min")

    def test_summary_is_readable(self, clean_df: pd.DataFrame) -> None:
        report = validate_ohlcv(clean_df)
        assert "DataQualityReport" in report.summary()


class TestIndexChecks:
    def test_naive_timezone_is_error(self) -> None:
        df = make_ohlcv(tz=None)
        report = validate_ohlcv(df)
        assert not report.ok
        assert any(i.code == "naive_timezone" for i in report.errors)

    def test_missing_columns_is_error(self, clean_df: pd.DataFrame) -> None:
        report = validate_ohlcv(clean_df.drop(columns=["close"]))
        assert any(i.code == "missing_columns" for i in report.errors)

    def test_non_monotonic_is_error(self, clean_df: pd.DataFrame) -> None:
        shuffled = pd.concat([clean_df.iloc[50:], clean_df.iloc[:50]])
        report = validate_ohlcv(shuffled)
        assert any(i.code == "not_monotonic" for i in report.errors)

    def test_non_datetime_index_is_error(self) -> None:
        df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]}, index=[0])
        report = validate_ohlcv(df)
        assert any(i.code == "index_not_datetime" for i in report.errors)


class TestDuplicates:
    def test_duplicate_timestamps_detected(self, clean_df: pd.DataFrame) -> None:
        dup = pd.concat([clean_df, clean_df.iloc[[0]]]).sort_index()
        found = find_duplicate_timestamps(dup)
        assert len(found) == 1
        report = validate_ohlcv(dup)
        assert any(i.code == "duplicate_timestamps" for i in report.errors)


class TestOHLCViolations:
    def test_high_below_low(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 1.0
        viol = find_ohlc_violations(df)
        assert "high_lt_low" in viol
        report = validate_ohlcv(df)
        assert any(i.code == "high_lt_low" for i in report.errors)

    def test_high_not_max(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["close"] - 0.01
        assert "high_not_max" in find_ohlc_violations(df)

    def test_low_not_min(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.iloc[7, df.columns.get_loc("low")] = df.iloc[7]["open"] + 0.01
        assert "low_not_min" in find_ohlc_violations(df)

    def test_nan_price_is_error(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.iloc[3, df.columns.get_loc("close")] = float("nan")
        report = validate_ohlcv(df)
        assert any(i.code == "nan_price" for i in report.errors)

    def test_nonpositive_price_is_warning_not_error(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        # Make an entire bar non-positive but internally consistent.
        loc = 4
        for col in ("open", "high", "low", "close"):
            df.iloc[loc, df.columns.get_loc(col)] = -1.0
        report = validate_ohlcv(df)
        assert any(
            i.code == "nonpositive_price" and i.severity is Severity.WARNING for i in report.issues
        )
        assert not any(i.code == "nonpositive_price" for i in report.errors)


class TestVolume:
    def test_missing_volume_warns(self, clean_df: pd.DataFrame) -> None:
        report = validate_ohlcv(clean_df.drop(columns=["volume"]))
        assert any(i.code == "no_volume" and i.severity is Severity.WARNING for i in report.issues)

    def test_negative_volume_is_error(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.iloc[2, df.columns.get_loc("volume")] = -5.0
        report = validate_ohlcv(df)
        assert any(i.code == "negative_volume" for i in report.errors)


class TestFrequencyAndGaps:
    def test_infer_frequency(self, clean_df: pd.DataFrame) -> None:
        assert infer_frequency(clean_df.index) == pd.Timedelta("1min")

    def test_find_gaps_detects_hole(self, clean_df: pd.DataFrame) -> None:
        holed = clean_df.drop(clean_df.index[10])  # remove one 1-min bar
        gaps = find_gaps(holed)
        assert len(gaps) == 1
        assert int(gaps["missing_bars"].iloc[0]) == 1

    def test_missing_bars_warns_without_session_gap(self, clean_df: pd.DataFrame) -> None:
        holed = clean_df.drop(clean_df.index[10])
        report = validate_ohlcv(holed)
        assert any(
            i.code == "missing_bars" and i.severity is Severity.WARNING for i in report.issues
        )

    def test_session_gap_reclassifies_overnight_break(self) -> None:
        # Two 3-bar sessions separated by a multi-hour overnight gap.
        day1 = make_ohlcv(n=3, start="2024-01-02 14:57")
        day2 = make_ohlcv(n=3, start="2024-01-03 08:30")
        df = pd.concat([day1, day2])
        # Without session_gap the overnight break looks like missing bars.
        naive = validate_ohlcv(df)
        assert any(i.code == "missing_bars" for i in naive.issues)
        # With a session_gap the break is downgraded to INFO.
        aware = validate_ohlcv(df, session_gap="30min")
        assert any(i.code == "session_breaks" and i.severity is Severity.INFO for i in aware.issues)
        assert not any(i.code == "missing_bars" for i in aware.issues)


class TestReport:
    def test_raise_if_failed(self, clean_df: pd.DataFrame) -> None:
        report = validate_ohlcv(clean_df.drop(columns=["close"]))
        with pytest.raises(DataValidationError) as exc:
            report.raise_if_failed()
        assert exc.value.report is report

    def test_clean_report_does_not_raise(self, clean_df: pd.DataFrame) -> None:
        validate_ohlcv(clean_df, session_gap="30min").raise_if_failed()

    def test_issues_sorted_most_severe_first(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.drop(columns=["volume"]).copy()  # a warning
        df.iloc[3, df.columns.get_loc("close")] = float("nan")  # an error
        report = validate_ohlcv(df)
        severities = [i.severity for i in report.issues]
        assert severities == sorted(severities, reverse=True)
