"""Tests for the DataHandler and the causal MarketView.

These are the "never expose the future" tests at the data-access level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from biasguard.data.validation import DataValidationError
from biasguard.engine import DataHandler
from biasguard.events import MarketEvent


class TestIteration:
    def test_len(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        assert len(dh) == len(clean_df)

    def test_iter_yields_market_events_in_order(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        events = list(dh)
        assert len(events) == len(clean_df)
        assert all(isinstance(e, MarketEvent) for e in events)
        assert [e.index for e in events] == list(range(len(clean_df)))
        assert events[0].timestamp == clean_df.index[0]
        assert events[-1].bar.close == clean_df["close"].iloc[-1]

    def test_iteration_is_repeatable(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        first = [e.index for e in dh]
        second = [e.index for e in dh]
        assert first == second


class TestMarketViewIsCausal:
    def test_view_length_is_i_plus_one(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        view = dh.view(5)
        assert len(view) == 6
        assert view.index == 5

    def test_view_current_bar_matches(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        view = dh.view(5)
        assert view.current.close == clean_df["close"].iloc[5]
        assert view.timestamp == clean_df.index[5]

    def test_closes_stop_at_current_bar(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        i = 5
        view = dh.view(i)
        closes = view.closes
        assert len(closes) == i + 1
        np.testing.assert_array_equal(closes, clean_df["close"].to_numpy()[: i + 1])

    def test_future_access_raises_indexerror(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        i = 5
        closes = dh.view(i).closes
        with pytest.raises(IndexError):
            _ = closes[i + 1]  # bar i+1 must be unreachable

    def test_view_arrays_are_read_only(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        closes = dh.view(5).closes
        with pytest.raises(ValueError):
            closes[0] = 0.0  # must not be able to mutate market data

    def test_all_ohlcv_accessors_are_causal(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        i = 8
        view = dh.view(i)
        for field, prop in (
            ("open", "opens"),
            ("high", "highs"),
            ("low", "lows"),
            ("close", "closes"),
            ("volume", "volumes"),
        ):
            arr = getattr(view, prop)
            assert len(arr) == i + 1
            np.testing.assert_array_equal(arr, clean_df[field].to_numpy()[: i + 1])

    def test_volumes_are_nan_when_absent(self) -> None:
        df = make_ohlcv().drop(columns=["volume"])
        dh = DataHandler(df)
        vols = dh.view(3).volumes
        assert len(vols) == 4
        assert bool(np.isnan(vols).all())
        assert dh.view(3).current.volume is None

    def test_window_rejects_nonpositive(self, clean_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="window size"):
            dh = DataHandler(clean_df)
            dh.view(5).window("close", 0)

    def test_window_returns_last_n(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        i = 10
        win = dh.view(i).window("close", 3)
        np.testing.assert_array_equal(win, clean_df["close"].to_numpy()[i - 2 : i + 1])

    def test_window_clips_at_start(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        win = dh.view(1).window("close", 10)  # only 2 bars exist
        assert len(win) == 2

    def test_as_frame_is_a_causal_copy(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        frame = dh.view(4).as_frame()
        assert len(frame) == 5
        frame.iloc[0, 0] = -999.0  # mutating the copy must not affect the source
        assert dh.view(4).as_frame().iloc[0, 0] != -999.0

    def test_view_out_of_range_raises(self, clean_df: pd.DataFrame) -> None:
        dh = DataHandler(clean_df)
        with pytest.raises(IndexError):
            dh.view(len(clean_df))


class TestValidationIntegration:
    def test_datahandler_rejects_invalid_data_by_default(self, clean_df: pd.DataFrame) -> None:
        bad = clean_df.copy()
        bad.iloc[3, bad.columns.get_loc("close")] = float("nan")
        with pytest.raises(DataValidationError):
            DataHandler(bad)

    def test_datahandler_can_skip_validation(self) -> None:
        naive = make_ohlcv(tz=None)  # would FAIL validation (naive tz)
        dh = DataHandler(naive, validate=False)
        assert len(dh) == len(naive)

    def test_missing_bars_do_not_block_construction(self, clean_df: pd.DataFrame) -> None:
        holed = clean_df.drop(clean_df.index[10])  # a gap -> WARNING, not ERROR
        dh = DataHandler(holed)  # should not raise
        assert len(dh) == len(holed)
