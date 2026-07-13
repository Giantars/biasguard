"""Loaders for OHLCV market data from CSV and Parquet.

Every loader funnels through :func:`_finalize`, which enforces the one rule that
prevents a whole class of silent bugs: **the timezone must be declared.** There
is no default ``tz`` — an intraday file whose timezone is guessed is an intraday
file whose session filters are a coin flip (brief trap #7).

The result is always the canonical contract from :mod:`biasguard.data.schema`:
a tz-aware, strictly-ascending ``DatetimeIndex`` named ``timestamp`` with
``open/high/low/close`` (and ``volume`` when available). The
:class:`~biasguard.data.validation.DataQualityReport` is attached to
``df.attrs["quality_report"]`` so the verdict travels with the frame.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from biasguard.data.schema import (
    TIMESTAMP,
    canonicalize_columns,
    resolve_timestamp_column,
)
from biasguard.data.validation import (
    DataQualityReport,
    validate_ohlcv,
)

OnInvalid = Literal["raise", "warn", "ignore"]
#: Units accepted for numeric epoch timestamps (forwarded to ``pandas.to_datetime``).
TimestampUnit = Literal["D", "s", "ms", "us", "ns"]
#: How to resolve DST-ambiguous wall-clock times when localizing naive timestamps.
AmbiguousPolicy = Literal["infer", "NaT", "raise"] | bool
#: How to resolve DST-nonexistent wall-clock times when localizing naive timestamps.
NonexistentPolicy = Literal["shift_forward", "shift_backward", "NaT", "raise"]


class DataQualityWarning(UserWarning):
    """Emitted by loaders when ``on_invalid="warn"`` and findings exist."""


def _resolve_datetime_index(
    df: pd.DataFrame,
    *,
    timestamp_col: str | None,
    timestamp_unit: TimestampUnit | None,
) -> pd.DataFrame:
    """Attach a (possibly naive) DatetimeIndex, dropping the source column."""
    ts_col = resolve_timestamp_column(df.columns, timestamp_col)
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], unit=timestamp_unit, errors="raise")
        df = df.drop(columns=[ts_col])
        df.index = pd.DatetimeIndex(ts.to_numpy())
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df  # e.g. a Parquet file that already carries a datetime index
    raise ValueError(
        "could not find a timestamp column and the index is not datetime-like; "
        f"pass timestamp_col=... explicitly. Columns seen: {list(df.columns)}"
    )


def _apply_timezone(
    df: pd.DataFrame,
    *,
    tz: str,
    ambiguous: AmbiguousPolicy,
    nonexistent: NonexistentPolicy,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz, ambiguous=ambiguous, nonexistent=nonexistent)
    else:
        idx = idx.tz_convert(tz)
    df.index = idx
    df.index.name = TIMESTAMP
    return df


def _finalize(
    raw: pd.DataFrame,
    *,
    tz: str,
    source: str,
    timestamp_col: str | None,
    timestamp_unit: TimestampUnit | None,
    column_map: dict[str, str] | None,
    ambiguous: AmbiguousPolicy,
    nonexistent: NonexistentPolicy,
    validate: bool,
    on_invalid: OnInvalid,
    expected_freq: pd.Timedelta | str | None,
    session_gap: pd.Timedelta | str | None,
) -> pd.DataFrame:
    if not tz:
        raise ValueError(
            "tz is required: declare the source timezone explicitly "
            "(e.g. tz='America/Chicago', tz='America/New_York', or tz='UTC')."
        )

    df = canonicalize_columns(raw.copy(), column_map)
    df = _resolve_datetime_index(df, timestamp_col=timestamp_col, timestamp_unit=timestamp_unit)
    df = _apply_timezone(df, tz=tz, ambiguous=ambiguous, nonexistent=nonexistent)
    # The loader guarantees chronological order; a stable sort preserves the
    # original ordering of any exact-duplicate timestamps for later detection.
    df = df.sort_index(kind="stable")

    if validate:
        report = validate_ohlcv(df, expected_freq=expected_freq, session_gap=session_gap)
        df.attrs["quality_report"] = report
        df.attrs["tz"] = tz
        df.attrs["source"] = source
        _handle_report(report, on_invalid)
    return df


def _handle_report(report: DataQualityReport, on_invalid: OnInvalid) -> None:
    if on_invalid == "raise":
        report.raise_if_failed()  # raises only on ERROR-level findings
    elif on_invalid == "warn":
        if report.errors or report.warnings:
            warnings.warn(report.summary(), DataQualityWarning, stacklevel=3)
    elif on_invalid == "ignore":
        return
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"on_invalid must be raise|warn|ignore, got {on_invalid!r}")


def load_csv(
    path: str | Path,
    *,
    tz: str,
    timestamp_col: str | None = None,
    timestamp_unit: TimestampUnit | None = None,
    column_map: dict[str, str] | None = None,
    ambiguous: AmbiguousPolicy = "infer",
    nonexistent: NonexistentPolicy = "shift_forward",
    validate: bool = True,
    on_invalid: OnInvalid = "raise",
    expected_freq: pd.Timedelta | str | None = None,
    session_gap: pd.Timedelta | str | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into the canonical contract.

    Parameters
    ----------
    path:
        Path to the CSV file.
    tz:
        **Required.** The timezone the timestamps represent (naive timestamps are
        localized to it; tz-aware timestamps are converted to it). There is no
        default on purpose — see the module docstring.
    timestamp_col:
        Name of the timestamp column. If omitted, common aliases (``timestamp``,
        ``datetime``, ``date`` ...) are searched.
    timestamp_unit:
        Passed to :func:`pandas.to_datetime` for numeric epoch timestamps
        (e.g. ``"s"`` or ``"ms"``).
    column_map:
        Explicit ``{source_name: canonical_name}`` overrides for column renaming.
    ambiguous, nonexistent:
        Forwarded to :meth:`pandas.DatetimeIndex.tz_localize` to resolve DST
        transitions for naive timestamps.
    validate:
        Run data-quality validation after loading (default ``True``).
    on_invalid:
        ``"raise"`` (default) raises on ERROR-level findings; ``"warn"`` emits a
        :class:`DataQualityWarning`; ``"ignore"`` attaches the report silently.
    expected_freq, session_gap:
        Forwarded to :func:`~biasguard.data.validation.validate_ohlcv`.

    Returns
    -------
    pandas.DataFrame
        Canonical OHLCV frame. The quality report is on
        ``df.attrs["quality_report"]``.
    """
    raw = pd.read_csv(path, **read_csv_kwargs)
    return _finalize(
        raw,
        tz=tz,
        source=str(path),
        timestamp_col=timestamp_col,
        timestamp_unit=timestamp_unit,
        column_map=column_map,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
        validate=validate,
        on_invalid=on_invalid,
        expected_freq=expected_freq,
        session_gap=session_gap,
    )


def load_parquet(
    path: str | Path,
    *,
    tz: str,
    timestamp_col: str | None = None,
    timestamp_unit: TimestampUnit | None = None,
    column_map: dict[str, str] | None = None,
    ambiguous: AmbiguousPolicy = "infer",
    nonexistent: NonexistentPolicy = "shift_forward",
    validate: bool = True,
    on_invalid: OnInvalid = "raise",
    expected_freq: pd.Timedelta | str | None = None,
    session_gap: pd.Timedelta | str | None = None,
    **read_parquet_kwargs: Any,
) -> pd.DataFrame:
    """Load OHLCV data from a Parquet file into the canonical contract.

    Behaves like :func:`load_csv`. Parquet frequently preserves a DatetimeIndex
    and its timezone; when present that timezone is *converted* to ``tz`` rather
    than assumed.
    """
    raw = pd.read_parquet(path, **read_parquet_kwargs)
    return _finalize(
        raw,
        tz=tz,
        source=str(path),
        timestamp_col=timestamp_col,
        timestamp_unit=timestamp_unit,
        column_map=column_map,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
        validate=validate,
        on_invalid=on_invalid,
        expected_freq=expected_freq,
        session_gap=session_gap,
    )


__all__ = [
    "DataQualityWarning",
    "load_csv",
    "load_parquet",
]
