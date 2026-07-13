"""Data-quality validation for OHLCV frames.

These are the low-level *detection primitives*: pure functions that inspect a
frame and return facts (which timestamps are duplicated, which bars violate the
OHLC invariants, where the gaps are). They are deliberately independent of the
richer strategy-level :mod:`biasguard.validation` module built in a later phase,
which reuses them.

Design stance: this module would rather *over*-warn than hide a hole. A
backtesting tool whose entire premise is "refuse to flatter the user" must not
silently swallow missing candles or dubious prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biasguard.data.schema import (
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    REQUIRED_COLUMNS,
    VOLUME,
)
from biasguard.types import Severity

_MAX_SAMPLE = 5  # how many offending timestamps to attach to an issue


class DataValidationError(ValueError):
    """Raised when a frame fails validation and ``on_invalid="raise"``.

    The originating :class:`DataQualityReport` is attached as ``.report`` so
    callers can inspect every finding, not just the message.
    """

    def __init__(self, report: DataQualityReport) -> None:
        super().__init__(report.summary())
        self.report = report


@dataclass(frozen=True, slots=True)
class DataIssue:
    """A single data-quality finding."""

    code: str
    severity: Severity
    message: str
    count: int = 0
    sample: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"[{self.severity}] {self.code}: {self.message}"
        if self.sample:
            head += f" (e.g. {', '.join(self.sample)})"
        return head


@dataclass(frozen=True)
class DataQualityReport:
    """The result of validating an OHLCV frame."""

    n_rows: int
    issues: tuple[DataIssue, ...] = field(default_factory=tuple)
    inferred_freq: pd.Timedelta | None = None

    @property
    def errors(self) -> tuple[DataIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[DataIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def infos(self) -> tuple[DataIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.INFO)

    @property
    def ok(self) -> bool:
        """True when there are no ERROR-level findings."""
        return not self.errors

    def summary(self) -> str:
        freq = "unknown" if self.inferred_freq is None else str(self.inferred_freq)
        lines = [
            f"DataQualityReport: {self.n_rows} rows, inferred bar interval {freq}, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{len(self.infos)} info(s)."
        ]
        lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise :class:`DataValidationError` if any ERROR-level issue exists."""
        if not self.ok:
            raise DataValidationError(self)


# --------------------------------------------------------------------------- #
# Detection primitives
# --------------------------------------------------------------------------- #
def _sample(index: pd.Index) -> tuple[str, ...]:
    return tuple(str(ts) for ts in index[:_MAX_SAMPLE])


def check_required_columns(df: pd.DataFrame) -> list[DataIssue]:
    """ERROR for each missing required OHLC column."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if not missing:
        return []
    return [
        DataIssue(
            code="missing_columns",
            severity=Severity.ERROR,
            message=f"required column(s) absent: {missing}",
            count=len(missing),
        )
    ]


def check_index(df: pd.DataFrame) -> list[DataIssue]:
    """Checks on the index itself: type, timezone, monotonicity."""
    issues: list[DataIssue] = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        issues.append(
            DataIssue(
                code="index_not_datetime",
                severity=Severity.ERROR,
                message=f"index must be a DatetimeIndex, got {type(idx).__name__}",
            )
        )
        return issues  # remaining index checks are meaningless without datetimes
    if idx.tz is None:
        # Trap #7: timezone mislabeling starts with a missing timezone.
        issues.append(
            DataIssue(
                code="naive_timezone",
                severity=Severity.ERROR,
                message=(
                    "index is timezone-naive; declare the source timezone on load "
                    "(e.g. tz='America/Chicago') so session filters cannot be silently wrong"
                ),
            )
        )
    if not idx.is_monotonic_increasing:
        issues.append(
            DataIssue(
                code="not_monotonic",
                severity=Severity.ERROR,
                message="timestamps are not sorted strictly ascending",
            )
        )
    return issues


def find_duplicate_timestamps(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the timestamps that appear more than once."""
    idx = df.index
    dup_mask = idx.duplicated(keep=False)
    return idx[dup_mask].unique()  # type: ignore[return-value]


def check_duplicate_timestamps(df: pd.DataFrame) -> list[DataIssue]:
    if not isinstance(df.index, pd.DatetimeIndex):
        return []
    dups = find_duplicate_timestamps(df)
    if len(dups) == 0:
        return []
    return [
        DataIssue(
            code="duplicate_timestamps",
            severity=Severity.ERROR,
            message=f"{len(dups)} timestamp(s) occur more than once",
            count=len(dups),
            sample=_sample(dups),
        )
    ]


def find_ohlc_violations(df: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    """Return a mapping ``violation_code -> offending timestamps``.

    Comparisons against NaN evaluate to ``False`` in pandas, so NaN bars do not
    masquerade as relationship violations; they are reported separately under
    ``nan_price``.
    """
    if not all(c in df.columns for c in REQUIRED_COLUMNS):
        return {}
    o, h, lo, c = df[OPEN], df[HIGH], df[LOW], df[CLOSE]
    prices = df[list(REQUIRED_COLUMNS)]
    out: dict[str, pd.DatetimeIndex] = {}

    high_low = h < lo
    high_not_max = (h < o) | (h < c)
    low_not_min = (lo > o) | (lo > c)
    nan_price = prices.isna().any(axis=1)
    nonpositive = (prices <= 0).any(axis=1)

    for code, mask in {
        "high_lt_low": high_low,
        "high_not_max": high_not_max,
        "low_not_min": low_not_min,
        "nan_price": nan_price,
        "nonpositive_price": nonpositive,
    }.items():
        hits = df.index[mask.to_numpy()]
        if len(hits) > 0:
            out[code] = hits  # type: ignore[assignment]
    return out


def check_ohlc_relationships(df: pd.DataFrame) -> list[DataIssue]:
    """ERROR on impossible OHLC geometry; WARN on non-positive prices."""
    violations = find_ohlc_violations(df)
    issues: list[DataIssue] = []
    messages = {
        "high_lt_low": ("high is below low", Severity.ERROR),
        "high_not_max": ("high is not the bar maximum", Severity.ERROR),
        "low_not_min": ("low is not the bar minimum", Severity.ERROR),
        "nan_price": ("NaN in an OHLC price", Severity.ERROR),
        # Non-positive prices are unusual but legitimate for some instruments
        # (spreads, and crude oil in April 2020), so warn rather than error.
        "nonpositive_price": (
            "non-positive price (unusual; verify the instrument)",
            Severity.WARNING,
        ),
    }
    for code, hits in violations.items():
        msg, sev = messages[code]
        issues.append(
            DataIssue(
                code=code,
                severity=sev,
                message=f"{len(hits)} bar(s) where {msg}",
                count=len(hits),
                sample=_sample(hits),
            )
        )
    return issues


def check_volume(df: pd.DataFrame) -> list[DataIssue]:
    """WARN if volume is missing; ERROR on negative volume."""
    if VOLUME not in df.columns:
        return [
            DataIssue(
                code="no_volume",
                severity=Severity.WARNING,
                message="no volume column; timezone/session profile checks are unavailable",
            )
        ]
    vol = df[VOLUME]
    neg = df.index[(vol < 0).to_numpy()]
    if len(neg) == 0:
        return []
    return [
        DataIssue(
            code="negative_volume",
            severity=Severity.ERROR,
            message=f"{len(neg)} bar(s) with negative volume",
            count=len(neg),
            sample=_sample(neg),
        )
    ]


def infer_frequency(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Infer the bar interval as the *modal* gap between consecutive bars.

    The mode (not the min or mean) is robust to overnight/weekend gaps and the
    occasional missing bar, which is exactly the regularity we want to measure
    intraday.
    """
    if len(index) < 2:
        return None
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty:
        return None
    return pd.Timedelta(mode.iloc[0])


def find_gaps(df: pd.DataFrame, *, freq: pd.Timedelta | None = None) -> pd.DataFrame:
    """Return every interval strictly larger than the bar frequency.

    Returns a frame indexed by ``prev_ts`` with columns ``next_ts``, ``gap`` and
    ``missing_bars`` (an estimate: ``round(gap / freq) - 1``). Overnight and
    weekend breaks appear here too — that is intentional; callers separate them
    via ``session_gap`` in :func:`validate_ohlcv`.
    """
    idx = df.index
    empty = pd.DataFrame(
        {
            "next_ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "gap": pd.Series(dtype="timedelta64[ns]"),
            "missing_bars": pd.Series(dtype="int64"),
        }
    )
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return empty
    if freq is None:
        freq = infer_frequency(idx)
    if freq is None or freq <= pd.Timedelta(0):
        return empty

    prev = idx[:-1]
    nxt = idx[1:]
    gaps = nxt - prev
    over = np.asarray(gaps > freq)
    if not over.any():
        return empty
    # Estimate missing bars per gap; use numpy for the float arithmetic so the
    # result is a plain int array (pandas' Index has no ``.round``).
    ratio = np.asarray(gaps / freq, dtype="float64")
    missing = np.rint(ratio).astype("int64") - 1
    result = pd.DataFrame(
        {"next_ts": nxt[over], "gap": gaps[over], "missing_bars": missing[over]},
        index=pd.Index(prev[over], name="prev_ts"),
    )
    return result


def check_missing_bars(
    df: pd.DataFrame,
    *,
    freq: pd.Timedelta | None = None,
    session_gap: pd.Timedelta | None = None,
) -> list[DataIssue]:
    """WARN on intra-session gaps; INFO on likely session/weekend breaks.

    Without an exchange calendar we cannot know *for certain* which absent bars
    are holidays vs. dropped data. ``session_gap`` lets the caller draw the line:
    any gap greater than or equal to it is treated as a legitimate session break
    (INFO); smaller gaps are treated as missing bars (WARNING). When
    ``session_gap`` is ``None`` every gap is surfaced as a potential hole.
    """
    gaps = find_gaps(df, freq=freq)
    if gaps.empty:
        return []

    if session_gap is None:
        intrasession = gaps
        breaks = gaps.iloc[0:0]
    else:
        is_break = gaps["gap"] >= session_gap
        breaks = gaps[is_break]
        intrasession = gaps[~is_break]

    issues: list[DataIssue] = []
    if not intrasession.empty:
        total_missing = int(intrasession["missing_bars"].sum())
        issues.append(
            DataIssue(
                code="missing_bars",
                severity=Severity.WARNING,
                message=(
                    f"{len(intrasession)} gap(s) totalling ~{total_missing} missing bar(s)"
                    + (
                        ""
                        if session_gap
                        else "; set session_gap to exclude overnight/weekend breaks"
                    )
                ),
                count=total_missing,
                sample=_sample(intrasession.index),
            )
        )
    if not breaks.empty:
        issues.append(
            DataIssue(
                code="session_breaks",
                severity=Severity.INFO,
                message=f"{len(breaks)} gap(s) >= session_gap treated as session/weekend breaks",
                count=len(breaks),
                sample=_sample(breaks.index),
            )
        )
    return issues


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    expected_freq: pd.Timedelta | str | None = None,
    session_gap: pd.Timedelta | str | None = None,
) -> DataQualityReport:
    """Run every data-quality check and return a structured report.

    Parameters
    ----------
    df:
        A frame that is expected to follow the canonical OHLCV contract.
    expected_freq:
        The bar interval you *expect* (e.g. ``"1min"``). If omitted it is
        inferred from the data's modal gap. Passing it lets the gap detector
        catch a wholesale frequency mismatch.
    session_gap:
        Gaps at least this large are classified as legitimate session/weekend
        breaks rather than missing bars.
    """
    freq = pd.Timedelta(expected_freq) if expected_freq is not None else None
    sess = pd.Timedelta(session_gap) if session_gap is not None else None

    issues: list[DataIssue] = []
    issues += check_required_columns(df)
    issues += check_index(df)

    inferred: pd.Timedelta | None = None
    if isinstance(df.index, pd.DatetimeIndex):
        inferred = freq if freq is not None else infer_frequency(df.index)
        issues += check_duplicate_timestamps(df)
        # Gap detection needs a sorted index to be meaningful.
        if df.index.is_monotonic_increasing:
            issues += check_missing_bars(df, freq=freq, session_gap=sess)

    issues += check_ohlc_relationships(df)
    issues += check_volume(df)

    # Sort issues most-severe first for readable summaries.
    issues.sort(key=lambda i: i.severity, reverse=True)
    return DataQualityReport(n_rows=len(df), issues=tuple(issues), inferred_freq=inferred)


__all__ = [
    "DataIssue",
    "DataQualityReport",
    "DataValidationError",
    "check_duplicate_timestamps",
    "check_index",
    "check_missing_bars",
    "check_ohlc_relationships",
    "check_required_columns",
    "check_volume",
    "find_duplicate_timestamps",
    "find_gaps",
    "find_ohlc_violations",
    "infer_frequency",
    "validate_ohlcv",
]
