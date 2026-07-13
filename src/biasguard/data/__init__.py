"""Data layer: loading and validating OHLCV market data.

Typical use::

    from biasguard.data import load_csv

    df = load_csv("nq_1min.csv", tz="America/Chicago", session_gap="30min")
    print(df.attrs["quality_report"].summary())
"""

from __future__ import annotations

from biasguard.data.loaders import (
    DataQualityWarning,
    load_csv,
    load_parquet,
)
from biasguard.data.schema import (
    CLOSE,
    HIGH,
    LOW,
    OHLCV_COLUMNS,
    OPEN,
    REQUIRED_COLUMNS,
    TIMESTAMP,
    VOLUME,
    Bar,
    canonicalize_columns,
    resolve_timestamp_column,
)
from biasguard.data.validation import (
    DataIssue,
    DataQualityReport,
    DataValidationError,
    check_missing_bars,
    check_ohlc_relationships,
    find_duplicate_timestamps,
    find_gaps,
    find_ohlc_violations,
    infer_frequency,
    validate_ohlcv,
)

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
    "DataIssue",
    "DataQualityReport",
    "DataQualityWarning",
    "DataValidationError",
    "canonicalize_columns",
    "check_missing_bars",
    "check_ohlc_relationships",
    "find_duplicate_timestamps",
    "find_gaps",
    "find_ohlc_violations",
    "infer_frequency",
    "load_csv",
    "load_parquet",
    "resolve_timestamp_column",
    "validate_ohlcv",
]
