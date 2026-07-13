"""Account constraints (prop-firm style) and vectorized path evaluation.

An :class:`AccountConfig` describes the rules a simulated account must respect.
:func:`evaluate_paths` scores an entire ``(paths x steps)`` equity matrix against
those rules at once, so a Monte Carlo run can report the probability of breaching
each limit. The design accommodates future constraints by adding fields to
``AccountConfig`` and a branch to :func:`evaluate_paths`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AccountConfig:
    """Account rules for the risk layer. ``None`` disables a given limit.

    Limit values are positive magnitudes (e.g. ``trailing_drawdown_limit=2000``
    means "breach if drawdown from the running peak exceeds \\$2000").
    """

    starting_balance: float = 100_000.0
    max_loss_limit: float | None = None  # absolute floor: start - this
    trailing_drawdown_limit: float | None = None  # max drawdown from running peak
    daily_loss_limit: float | None = None  # max loss within one (synthetic) day
    profit_target: float | None = None  # optional pass target
    trades_per_day: int | None = None  # for daily-loss chunking under resampling

    def __post_init__(self) -> None:
        for name in (
            "max_loss_limit",
            "trailing_drawdown_limit",
            "daily_loss_limit",
            "profit_target",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be a positive magnitude or None, got {value}")
        if self.trades_per_day is not None and self.trades_per_day < 1:
            raise ValueError("trades_per_day must be >= 1 or None")

    @property
    def has_limits(self) -> bool:
        return any(
            x is not None
            for x in (self.max_loss_limit, self.trailing_drawdown_limit, self.daily_loss_limit)
        )


@dataclass(frozen=True, slots=True)
class PathOutcome:
    """The evaluation of a single equity path against an :class:`AccountConfig`."""

    final_pnl: float
    max_drawdown: float  # dollars, >= 0
    min_equity: float
    reached_target: bool
    breaches: frozenset[str]

    @property
    def breached(self) -> bool:
        return bool(self.breaches)


def path_equity(pnl: np.ndarray, starting_balance: float) -> np.ndarray:
    """Cumulative equity including the starting point (length ``len(pnl) + 1``)."""
    equity = np.empty(len(pnl) + 1, dtype="float64")
    equity[0] = starting_balance
    np.cumsum(pnl, out=equity[1:])
    equity[1:] += starting_balance
    return equity


def _daily_breach(paths: np.ndarray, trades_per_day: int, limit: float) -> np.ndarray:
    """Per-path flag: did any synthetic day's *intraday* loss exceed ``limit``?

    A prop-firm daily loss limit is an intraday threshold — it is breached the
    moment cumulative P&L since the day's start crosses ``-limit``, even if the
    day later recovers. So we test the deepest intraday drawdown from the day's
    start (the running minimum of the within-day cumulative sum), not the day's
    net. Trailing zero-padding of the final partial day does not move that
    minimum, so it neither masks nor invents a breach.
    """
    n_paths, n = paths.shape
    n_days = math.ceil(n / trades_per_day)
    pad = n_days * trades_per_day - n
    padded = np.pad(paths, ((0, 0), (0, pad))) if pad else paths
    reshaped = padded.reshape(n_paths, n_days, trades_per_day)
    day_intraday_low = np.cumsum(reshaped, axis=2).min(axis=2)
    return np.asarray((day_intraday_low < -limit).any(axis=1))


def evaluate_paths(
    equity: np.ndarray,
    paths: np.ndarray,
    account: AccountConfig,
    *,
    trades_per_day: int | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized evaluation of a ``(n_paths, steps)`` equity matrix.

    ``equity`` is ``(n_paths, n+1)`` (includes the starting balance); ``paths``
    is the ``(n_paths, n)`` per-trade P&L. Returns per-path arrays: ``final_pnl``,
    ``max_drawdown``, ``min_equity``, ``reached_target``, ``any_breach``, and a
    ``breach_<limit>`` boolean array for each configured limit.
    """
    start = account.starting_balance
    peak = np.maximum.accumulate(equity, axis=1)
    dd = peak - equity
    n_paths = equity.shape[0]

    out: dict[str, np.ndarray] = {
        "final_pnl": equity[:, -1] - start,
        "max_drawdown": dd.max(axis=1),
        "min_equity": equity.min(axis=1),
    }
    any_breach = np.zeros(n_paths, dtype=bool)

    if account.max_loss_limit is not None:
        b = np.asarray((equity < start - account.max_loss_limit).any(axis=1))
        out["breach_max_loss"] = b
        any_breach |= b
    if account.trailing_drawdown_limit is not None:
        b = np.asarray((dd > account.trailing_drawdown_limit).any(axis=1))
        out["breach_trailing_dd"] = b
        any_breach |= b
    if account.daily_loss_limit is not None:
        tpd = trades_per_day if trades_per_day is not None else account.trades_per_day
        if tpd is None or tpd < 1:
            raise ValueError(
                "daily_loss_limit requires trades_per_day: pass it to evaluate_paths or set "
                "AccountConfig.trades_per_day (the simulator infers it from the ledger)"
            )
        b = _daily_breach(paths, tpd, account.daily_loss_limit)
        out["breach_daily_loss"] = b
        any_breach |= b

    out["any_breach"] = any_breach
    out["reached_target"] = (
        np.asarray(equity.max(axis=1) - start >= account.profit_target)
        if account.profit_target is not None
        else np.zeros(n_paths, dtype=bool)
    )
    return out


def evaluate_path(
    pnl: np.ndarray, account: AccountConfig, *, trades_per_day: int | None = None
) -> PathOutcome:
    """Evaluate a single per-trade P&L path (convenience over :func:`evaluate_paths`)."""
    equity = path_equity(pnl, account.starting_balance).reshape(1, -1)
    result = evaluate_paths(equity, pnl.reshape(1, -1), account, trades_per_day=trades_per_day)
    breaches = frozenset(
        limit
        for limit in ("max_loss", "trailing_dd", "daily_loss")
        if bool(result.get(f"breach_{limit}", np.zeros(1))[0])
    )
    return PathOutcome(
        final_pnl=float(result["final_pnl"][0]),
        max_drawdown=float(result["max_drawdown"][0]),
        min_equity=float(result["min_equity"][0]),
        reached_target=bool(result["reached_target"][0]),
        breaches=breaches,
    )


__all__ = [
    "AccountConfig",
    "PathOutcome",
    "evaluate_path",
    "evaluate_paths",
    "path_equity",
]
