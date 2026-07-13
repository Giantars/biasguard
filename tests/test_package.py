"""Smoke tests for the package itself."""

from __future__ import annotations

import biasguard


def test_version_exposed() -> None:
    assert isinstance(biasguard.__version__, str)
    assert biasguard.__version__.count(".") >= 1


def test_data_public_api_imports() -> None:
    from biasguard.data import load_csv, load_parquet, validate_ohlcv

    assert callable(load_csv)
    assert callable(load_parquet)
    assert callable(validate_ohlcv)
