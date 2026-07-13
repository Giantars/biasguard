"""Replay fingerprint — a deterministic hash that identifies a backtest run.

Two identical backtests must produce the same fingerprint; changing *any* input
must change it. The fingerprint is the SHA-256 of a canonical JSON manifest that
captures every input to a run:

* **market data** — a content hash of the OHLCV bytes (endianness-normalized,
  so it is stable across platforms), plus symbol / columns / row count / span;
* **strategy identity** — the fully-qualified class name;
* **strategy parameters** — from ``strategy.params()``;
* **execution settings** — instrument, fill model, same-bar policy;
* **commission / slippage** — model type + parameters;
* **framework version** — plus a manifest-schema version.

Best practice: build the manifest **before** ``run()`` so no post-run state can
leak into the strategy parameters.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from biasguard import __version__
from biasguard.strategy.base import Strategy

MANIFEST_VERSION = "bg-manifest-2"
_FINGERPRINT_PREFIX = "bg1"


# --------------------------------------------------------------------------- #
# Canonicalization helpers
# --------------------------------------------------------------------------- #
def _qualname(obj: type) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _sort_key(value: Any) -> str:
    """A stable, type-aware sort key for already-canonicalized values."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _all_slots(cls: type) -> list[str]:
    """Every ``__slots__`` name declared across the class MRO."""
    slots: list[str] = []
    for klass in cls.__mro__:
        raw = getattr(klass, "__slots__", ())
        names = (raw,) if isinstance(raw, str) else tuple(raw)
        slots.extend(str(s) for s in names)
    return slots


def _canon(value: Any) -> Any:
    """Convert a value into a canonical, deterministic, JSON-serializable form.

    Determinism is the whole point: numerically or semantically identical
    configs must canonicalize identically, and distinct configs must differ. A
    value that cannot be represented deterministically raises ``TypeError``
    rather than silently producing a non-reproducible fingerprint.
    """
    if isinstance(value, Enum):
        return {"__enum__": f"{_qualname(type(value))}.{value.name}"}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, float):
        # Normalize non-finite values and negative zero, and collapse integral
        # floats to ints so numerically-identical configs hash the same.
        if value != value:  # NaN
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        if value == 0.0:
            return 0  # collapses -0.0 and +0.0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _canon(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo is not None else value
        return {"__timestamp__": ts.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"__timedelta__": str(value)}
    if isinstance(value, type):
        return {"__type__": _qualname(value)}
    if callable(value):
        module = getattr(value, "__module__", "?")
        qual = getattr(value, "__qualname__", repr(value))
        return {"__callable__": f"{module}.{qual}"}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _canon(v) for k, v in sorted(dataclasses.asdict(value).items())}
    if isinstance(value, (set, frozenset)):
        # Sort canonicalized elements so element order is irrelevant (a set's
        # iteration order is hash-seed-dependent and thus non-deterministic).
        return {"__set__": sorted((_canon(v) for v in value), key=_sort_key)}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        # Type-tag keys so int 1 and str '1' cannot collapse to one entry.
        return {
            f"{type(k).__name__}:{k}": _canon(v)
            for k, v in sorted(value.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))
        }
    if hasattr(value, "__dict__") or _all_slots(type(value)):
        return describe(value)
    raise TypeError(
        f"cannot deterministically fingerprint a value of type {type(value).__name__!r}; "
        "give it a params() override, make it a dataclass, or use a primitive/enum"
    )


def describe(obj: object) -> dict[str, Any]:
    """Describe a configurable object as ``{"type": qualname, "params": {...}}``.

    Introspects public instance attributes from ``__dict__`` and/or
    ``__slots__`` (convention: configuration is public, mutable state is
    underscore-prefixed). Callable attributes are represented by their qualified
    name rather than dropped, so changing a policy function changes the
    fingerprint.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        params = {k: _canon(v) for k, v in sorted(dataclasses.asdict(obj).items())}
    else:
        attrs: dict[str, Any] = {}
        if hasattr(obj, "__dict__"):
            attrs.update(vars(obj))
        for slot in _all_slots(type(obj)):
            if slot != "__dict__" and hasattr(obj, slot):
                attrs[slot] = getattr(obj, slot)
        params = {k: _canon(v) for k, v in sorted(attrs.items()) if not k.startswith("_")}
    return {"type": _qualname(type(obj)), "params": params}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# --------------------------------------------------------------------------- #
# Market-data content hash
# --------------------------------------------------------------------------- #
def hash_market_data(data: pd.DataFrame, *, symbol: str = "") -> str:
    """A deterministic, cross-platform content hash of a market-data frame.

    Hashes **every** column (not just OHLCV) so a strategy that trades off an
    auxiliary column (e.g. ``open_interest`` reached via ``view.as_frame()``)
    cannot collide with different data. Bytes are little-endian and negative
    zero is normalized, so numerically-identical frames hash identically.
    """
    hasher = hashlib.sha256()
    hasher.update(_FINGERPRINT_PREFIX.encode("utf-8"))
    hasher.update(symbol.encode("utf-8"))
    hasher.update(f"nrows={len(data)}".encode())

    idx = pd.DatetimeIndex(data.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)  # naive UTC wall time
    # int64 nanoseconds since epoch (UTC), little-endian for stable bytes.
    ns = idx.to_numpy().astype("datetime64[ns]").view("int64")
    hasher.update(b"ts")
    hasher.update(np.ascontiguousarray(ns, dtype="<i8").tobytes())

    for col in sorted(data.columns, key=str):
        hasher.update(str(col).encode("utf-8"))
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype="float64") + 0.0  # normalize -0.0 -> +0.0
            hasher.update(b"f8")
            hasher.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        else:
            hashed = pd.util.hash_pandas_object(series, index=False).to_numpy()
            hasher.update(b"obj")
            hasher.update(np.ascontiguousarray(hashed, dtype="<u8").tobytes())
    return hasher.hexdigest()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReplayManifest:
    """A structured, serializable record of every input to a run."""

    framework_version: str
    manifest_version: str
    data: dict[str, Any]
    strategy: dict[str, Any]
    execution: dict[str, Any]
    costs: dict[str, Any]
    portfolio: dict[str, Any]
    extra: dict[str, Any]
    fingerprint: str
    #: An optional human disclaimer captured from the strategy (e.g. an
    #: "educational example" label). Metadata only — deliberately NOT part of the
    #: fingerprint payload, so labelling a strategy never changes its identity.
    strategy_label: str | None = None

    @property
    def short(self) -> str:
        """A short display form, e.g. ``bg1:1a2b3c4d5e6f7a8b``."""
        return f"{_FINGERPRINT_PREFIX}:{self.fingerprint[:16]}"

    def payload(self) -> dict[str, Any]:
        """The exact dict the fingerprint is computed over (excludes itself)."""
        return {
            "framework_version": self.framework_version,
            "manifest_version": self.manifest_version,
            "data": self.data,
            "strategy": self.strategy,
            "execution": self.execution,
            "costs": self.costs,
            "portfolio": self.portfolio,
            "extra": self.extra,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload(),
            "fingerprint": self.fingerprint,
            "strategy_label": self.strategy_label,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, ensure_ascii=True)


def _fingerprint_of(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_manifest(
    data: pd.DataFrame,
    strategy: Strategy,
    *,
    broker: object | None = None,
    portfolio: object | None = None,
    strategy_params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> ReplayManifest:
    """Assemble a :class:`ReplayManifest` (and its fingerprint) from run inputs.

    ``broker`` and ``portfolio`` are the concrete execution objects (duck-typed
    on their public config attributes). Pass ``strategy_params`` to override the
    default introspection.
    """
    # Normalize the span to naive UTC so a tz relabeling of the same instants
    # (which hash_market_data already normalizes) does not change the fingerprint.
    idx = pd.DatetimeIndex(data.index)
    norm = idx.tz_convert("UTC").tz_localize(None) if (len(idx) and idx.tz is not None) else idx
    span_start = norm[0].isoformat() if len(norm) else None
    span_end = norm[-1].isoformat() if len(norm) else None
    instrument = getattr(portfolio, "instrument", None) if portfolio is not None else None
    if instrument is None and broker is not None:
        instrument = getattr(broker, "instrument", None)
    symbol = str(getattr(instrument, "symbol", "")) if instrument is not None else ""

    data_section = {
        "symbol": symbol,
        "n_bars": len(data),
        "columns": [str(c) for c in data.columns],
        "start": span_start,
        "end": span_end,
        "sha256": hash_market_data(data, symbol=symbol),
    }

    params = strategy_params if strategy_params is not None else strategy.params()
    strategy_section = {
        "identity": _qualname(type(strategy)),
        "params": {k: _canon(v) for k, v in sorted(params.items())},
    }
    raw_label = getattr(strategy, "label", None)
    strategy_label = raw_label if isinstance(raw_label, str) and raw_label else None

    execution_section: dict[str, Any] = {}
    costs_section: dict[str, Any] = {}
    if broker is not None:
        execution_section = {
            "instrument": _canon(instrument) if instrument is not None else None,
            "fill_model": describe(broker.fill_model) if hasattr(broker, "fill_model") else None,
            "same_bar_policy": _canon(getattr(broker, "same_bar_policy", None)),
        }
        costs_section = {
            "commission": describe(broker.commission) if hasattr(broker, "commission") else None,
            "slippage": describe(broker.slippage) if hasattr(broker, "slippage") else None,
        }

    portfolio_section: dict[str, Any] = {}
    if portfolio is not None:
        portfolio_section = {
            "initial_capital": _canon(getattr(portfolio, "initial_capital", None)),
            "sizer": describe(portfolio.sizer) if hasattr(portfolio, "sizer") else None,
        }

    payload = {
        "framework_version": __version__,
        "manifest_version": MANIFEST_VERSION,
        "data": data_section,
        "strategy": strategy_section,
        "execution": execution_section,
        "costs": costs_section,
        "portfolio": portfolio_section,
        "extra": _canon(extra or {}),
    }
    fingerprint = _fingerprint_of(payload)
    return ReplayManifest(
        framework_version=__version__,
        manifest_version=MANIFEST_VERSION,
        data=data_section,
        strategy=strategy_section,
        execution=execution_section,
        costs=costs_section,
        portfolio=portfolio_section,
        extra=payload["extra"],
        fingerprint=fingerprint,
        strategy_label=strategy_label,
    )


__all__ = [
    "MANIFEST_VERSION",
    "ReplayManifest",
    "build_manifest",
    "describe",
    "hash_market_data",
]
