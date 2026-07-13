"""The completed-bars-only data feed and the causal :class:`MarketView`.

This is where "never expose the future" is enforced at the data-access level.
A strategy is handed a :class:`MarketView` for bar ``i`` and can read columns
only up to and including index ``i``. Every array returned is a **read-only**
slice, so a strategy can neither mutate market data nor index past the current
bar (indexing ``i+1`` raises ``IndexError`` — it fails loud rather than
silently returning a future value).

The ultimate arbiter of causality is not this view but the truncation test
(:func:`biasguard.engine.backtester.run_backtest` with ``upto=T``): running on
``data[:T]`` must reproduce the first ``T`` bars' decisions exactly. Because the
truncated data's underlying arrays are physically shorter, *any* real
dependence on future data — including deliberate circumvention via a numpy
view's ``.base`` — changes the result and is caught.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd

from biasguard.data.schema import CLOSE, HIGH, LOW, OPEN, VOLUME, Bar
from biasguard.data.validation import validate_ohlcv
from biasguard.events import MarketEvent

_PRICE_FIELDS = (OPEN, HIGH, LOW, CLOSE)


class MarketView:
    """A read-only, causal window over market data up to bar ``index``.

    Constructed by a :class:`DataHandler`. Cheap to create (holds no copies);
    every accessor returns a read-only numpy slice bounded at the current bar.
    """

    __slots__ = ("_dh", "_i")

    def __init__(self, data_handler: DataHandler, i: int) -> None:
        self._dh = data_handler
        self._i = i

    @property
    def index(self) -> int:
        """0-based index of the current (most recent completed) bar."""
        return self._i

    def __len__(self) -> int:
        return self._i + 1

    @property
    def timestamp(self) -> pd.Timestamp:
        # Wrap to give mypy a concrete Timestamp (pandas' Index is loosely typed).
        return pd.Timestamp(self._dh.timestamps[self._i])

    @property
    def current(self) -> Bar:
        """The current bar as an immutable :class:`~biasguard.data.schema.Bar`."""
        return self._dh.bar_at(self._i)

    def _bounded(self, field: str, lo: int) -> np.ndarray:
        arr = self._dh.column(field)[lo : self._i + 1]
        arr.flags.writeable = False
        return arr

    @property
    def opens(self) -> np.ndarray:
        return self._bounded(OPEN, 0)

    @property
    def highs(self) -> np.ndarray:
        return self._bounded(HIGH, 0)

    @property
    def lows(self) -> np.ndarray:
        return self._bounded(LOW, 0)

    @property
    def closes(self) -> np.ndarray:
        return self._bounded(CLOSE, 0)

    @property
    def volumes(self) -> np.ndarray:
        return self._bounded(VOLUME, 0)

    def window(self, field: str, n: int) -> np.ndarray:
        """The last ``n`` values of ``field`` up to and including the current bar.

        Clipped at the start of history, so early bars return fewer than ``n``
        values rather than reaching before index 0.
        """
        if n <= 0:
            raise ValueError(f"window size must be positive, got {n}")
        lo = max(0, self._i + 1 - n)
        return self._bounded(field, lo)

    def as_frame(self) -> pd.DataFrame:
        """A **copy** of the OHLCV frame up to and including the current bar."""
        return self._dh.frame.iloc[: self._i + 1].copy()


class DataHandler:
    """Feeds completed bars in chronological order and vends causal views.

    Iteration is stateless and repeatable: ``for event in handler`` yields a
    :class:`~biasguard.events.MarketEvent` per bar, and ``handler.view(i)``
    returns the causal window for any bar independently of iteration.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        symbol: str = "",
        validate: bool = True,
        expected_freq: pd.Timedelta | str | None = None,
        session_gap: pd.Timedelta | str | None = None,
    ) -> None:
        """Wrap a canonical OHLCV frame.

        Parameters
        ----------
        data:
            A frame following the canonical contract (tz-aware, sorted,
            ``open/high/low/close`` columns). Typically the output of
            :func:`biasguard.data.load_csv`.
        symbol:
            Instrument label recorded on emitted signals/fills.
        validate:
            If ``True`` (default) the frame is validated and construction
            **raises** on any ERROR-level finding — the engine refuses to run on
            data that failed the quality gate. Missing candles (a WARNING) do
            not block construction.
        """
        if validate:
            report = validate_ohlcv(data, expected_freq=expected_freq, session_gap=session_gap)
            report.raise_if_failed()

        self.symbol = symbol
        self.frame = data
        self.timestamps = data.index
        self._n = len(data)
        self._has_volume = VOLUME in data.columns

        # Own the price memory and freeze it: market data is immutable input.
        # ``.copy()`` guarantees each column owns a clean 1-D buffer (base None),
        # fully decoupled from the source frame's block layout.
        self._cols: dict[str, np.ndarray] = {}
        for field in _PRICE_FIELDS:
            arr = data[field].to_numpy(dtype="float64").copy()
            arr.flags.writeable = False
            self._cols[field] = arr
        if self._has_volume:
            vol = data[VOLUME].to_numpy(dtype="float64").copy()
        else:
            vol = np.full(self._n, np.nan, dtype="float64")
        vol.flags.writeable = False
        self._cols[VOLUME] = vol

    def __len__(self) -> int:
        return self._n

    def column(self, field: str) -> np.ndarray:
        """The full (read-only) array for ``field``. Prefer :class:`MarketView`."""
        return self._cols[field]

    def bar_at(self, i: int) -> Bar:
        """The bar at index ``i`` as an immutable :class:`Bar`."""
        raw_vol = self._cols[VOLUME][i]
        volume = None if (not self._has_volume or math.isnan(raw_vol)) else float(raw_vol)
        return Bar(
            timestamp=self.timestamps[i],
            open=float(self._cols[OPEN][i]),
            high=float(self._cols[HIGH][i]),
            low=float(self._cols[LOW][i]),
            close=float(self._cols[CLOSE][i]),
            volume=volume,
        )

    def view(self, i: int) -> MarketView:
        """A causal :class:`MarketView` ending at bar ``i``."""
        if not 0 <= i < self._n:
            raise IndexError(f"bar index {i} out of range [0, {self._n})")
        return MarketView(self, i)

    def __iter__(self) -> Iterator[MarketEvent]:
        for i in range(self._n):
            yield MarketEvent(timestamp=self.timestamps[i], index=i, bar=self.bar_at(i))


__all__ = ["DataHandler", "MarketView"]
