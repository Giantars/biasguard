"""The OHLCV data contract.

A single, canonical shape that the rest of biasguard can rely on:

* the index is a timezone-aware :class:`pandas.DatetimeIndex` named ``timestamp``,
  sorted strictly ascending;
* columns ``open``, ``high``, ``low``, ``close`` are always present;
* column ``volume`` is present when the source provides it.

Real-world files disagree wildly on casing and naming (``Open``, ``OPEN``, ``o``,
``Last``, ``Vol`` ...). :func:`canonicalize_columns` maps those onto the contract
so nothing downstream has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical names
# --------------------------------------------------------------------------- #
TIMESTAMP = "timestamp"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

#: Columns every OHLCV frame must contain.
REQUIRED_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE)
#: Full OHLCV column set (``volume`` is optional but expected).
OHLCV_COLUMNS: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE, VOLUME)

#: Case-insensitive aliases mapped onto canonical column names.
_COLUMN_ALIASES: dict[str, str] = {
    "o": OPEN,
    "open": OPEN,
    "openprice": OPEN,
    "open_price": OPEN,
    "h": HIGH,
    "high": HIGH,
    "highprice": HIGH,
    "high_price": HIGH,
    "max": HIGH,
    "l": LOW,
    "low": LOW,
    "lowprice": LOW,
    "low_price": LOW,
    "min": LOW,
    "c": CLOSE,
    "close": CLOSE,
    "closeprice": CLOSE,
    "close_price": CLOSE,
    "last": CLOSE,
    "settle": CLOSE,
    "v": VOLUME,
    "vol": VOLUME,
    "volume": VOLUME,
    "qty": VOLUME,
}

#: Case-insensitive aliases that identify the timestamp column.
_TIMESTAMP_ALIASES: frozenset[str] = frozenset(
    {
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "date",
        "ts",
        "bar_time",
        "opentime",
        "open_time",
    }
)


@dataclass(frozen=True, slots=True)
class Bar:
    """A single completed OHLCV bar.

    This is the immutable unit the engine hands to strategies. ``volume`` is
    optional because some feeds do not provide it.
    """

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __post_init__(self) -> None:
        # A Bar is a value object; enforce the OHLC invariants at construction
        # so an invalid bar can never silently flow through the engine.
        if not (self.high >= self.low):
            raise ValueError(f"Bar high < low at {self.timestamp}: {self.high} < {self.low}")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"Bar high is not the maximum at {self.timestamp}")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Bar low is not the minimum at {self.timestamp}")

    @classmethod
    def from_row(cls, timestamp: pd.Timestamp, row: pd.Series) -> Bar:
        """Build a :class:`Bar` from a canonical OHLCV row."""
        vol = row[VOLUME] if VOLUME in row.index and pd.notna(row[VOLUME]) else None
        return cls(
            timestamp=timestamp,
            open=float(row[OPEN]),
            high=float(row[HIGH]),
            low=float(row[LOW]),
            close=float(row[CLOSE]),
            volume=None if vol is None else float(vol),
        )


def resolve_timestamp_column(columns: pd.Index, explicit: str | None = None) -> str | None:
    """Return the name of the timestamp column, or ``None`` if not found.

    Parameters
    ----------
    columns:
        The columns of the raw frame.
    explicit:
        A caller-specified column name. If given it is returned verbatim (and
        must exist in ``columns``); otherwise the aliases are searched.
    """
    if explicit is not None:
        if explicit not in columns:
            raise KeyError(
                f"timestamp_col={explicit!r} not found; available columns: {list(columns)}"
            )
        return explicit
    for col in columns:
        if str(col).strip().lower() in _TIMESTAMP_ALIASES:
            return str(col)
    return None


def canonicalize_columns(
    df: pd.DataFrame, column_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Rename recognized OHLCV columns to their canonical lowercase names.

    An explicit ``column_map`` (``{source_name: canonical_name}``) always wins
    over the built-in aliases. Unrecognized columns are passed through
    untouched so extra series (e.g. ``open_interest``) survive.
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        if column_map and col in column_map:
            mapping[col] = column_map[col]
            continue
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            mapping[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=mapping)


__all__ = [
    "CLOSE",
    "HIGH",
    "LOW",
    "OHLCV_COLUMNS",
    "OPEN",
    "REQUIRED_COLUMNS",
    "TIMESTAMP",
    "VOLUME",
    "Bar",
    "canonicalize_columns",
    "resolve_timestamp_column",
]
