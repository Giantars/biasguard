"""Performance metrics computed from an equity curve and a trade ledger.

Every metric is a pure function with explicit edge-case behaviour (documented
per function), so a backtest never crashes on a degenerate input — it returns
``nan``/``inf`` with a defined meaning instead.

Annualization uses the *observed* sampling density by default:
``periods_per_year = (n - 1) / years_elapsed``. This is self-consistent for
intraday data (it does not assume a session calendar) and is overridable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from biasguard.execution.orders import Trade

_DAYS_PER_YEAR = 365.25


# --------------------------------------------------------------------------- #
# Pure metric functions
# --------------------------------------------------------------------------- #
def periodic_returns(equity: pd.Series) -> np.ndarray:
    """Simple per-period returns of the equity curve (NaNs dropped)."""
    if len(equity) < 2:
        return np.empty(0, dtype=float)
    return equity.astype(float).pct_change().dropna().to_numpy()


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observed periods per year from the index span; ``nan`` if indeterminable."""
    if len(index) < 3:
        return float("nan")
    years = (index[-1] - index[0]) / pd.Timedelta(days=_DAYS_PER_YEAR)
    if years <= 0:
        return float("nan")
    return (len(index) - 1) / years


def sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe.

    ``nan`` if <2 returns. If the excess returns have zero dispersion, the ratio
    is undefined in the usual sense; we return the sign-consistent limit
    (``+inf``/``-inf`` for a constant positive/negative mean, ``0.0`` for a
    constant zero mean) to match :func:`sortino_ratio`'s convention.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2 or not math.isfinite(periods_per_year):
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    sd = float(excess.std(ddof=1))
    mean_excess = float(excess.mean())
    if sd == 0.0:
        return 0.0 if mean_excess == 0.0 else math.copysign(float("inf"), mean_excess)
    return float(mean_excess / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float] | np.ndarray,
    periods_per_year: float,
    mar: float = 0.0,
) -> float:
    """Annualized Sortino using downside deviation.

    ``nan`` if <2 returns; ``inf`` if there is no downside and the mean excess
    is positive; ``0.0`` if no downside and non-positive mean.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2 or not math.isfinite(periods_per_year):
        return float("nan")
    excess = r - mar / periods_per_year
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    mean_excess = float(excess.mean())
    if dd == 0.0:
        return float("inf") if mean_excess > 0 else 0.0
    return float(mean_excess / dd * math.sqrt(periods_per_year))


def annual_volatility(returns: Sequence[float] | np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2 or not math.isfinite(periods_per_year):
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """The underwater curve: ``equity / running_peak - 1`` (values <= 0)."""
    eq = equity.astype(float)
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a fraction (<= 0)."""
    if len(equity) < 1:
        return 0.0
    return float(drawdown_series(equity).min())


def max_drawdown_dollars(equity: pd.Series) -> float:
    if len(equity) < 1:
        return 0.0
    eq = equity.astype(float)
    return float((eq - eq.cummax()).min())


def max_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """Longest time spent below a prior peak."""
    if len(equity) < 2:
        return pd.Timedelta(0)
    eq = equity.astype(float)
    underwater = (eq < eq.cummax()).to_numpy()
    index = pd.DatetimeIndex(equity.index)
    longest = pd.Timedelta(0)
    start_i: int | None = None
    for i, is_under in enumerate(underwater):
        if is_under and start_i is None:
            start_i = i
        elif not is_under and start_i is not None:
            longest = max(longest, index[i] - index[start_i])
            start_i = None
    if start_i is not None:
        longest = max(longest, index[-1] - index[start_i])
    return longest


def cagr(equity: pd.Series, initial_capital: float) -> float:
    """Compound annual growth rate. ``nan`` if base/time is degenerate."""
    if len(equity) < 2 or initial_capital <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]) / pd.Timedelta(days=_DAYS_PER_YEAR)
    if years <= 0:
        return float("nan")
    growth = float(equity.iloc[-1]) / initial_capital
    if growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def profit_factor(trades: Sequence[Trade]) -> float:
    """Gross profit / gross loss (net of costs). ``inf`` if no losses."""
    wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    losses = sum(-t.net_pnl for t in trades if t.net_pnl < 0)
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def win_rate(trades: Sequence[Trade]) -> float:
    if not trades:
        return float("nan")
    return sum(1 for t in trades if t.is_win) / len(trades)


def expectancy(trades: Sequence[Trade]) -> float:
    """Mean net P&L per trade."""
    if not trades:
        return float("nan")
    return float(np.mean([t.net_pnl for t in trades]))


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerformanceMetrics:
    """The full metric suite for a backtest run."""

    start: pd.Timestamp | None
    end: pd.Timestamp | None
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_dollars: float
    max_drawdown_duration: pd.Timedelta
    periods_per_year: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    gross_profit: float
    gross_loss: float
    total_pnl: float
    total_commission: float

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in asdict(self).items():
            if isinstance(value, pd.Timestamp):
                out[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                out[key] = str(value)
            else:
                out[key] = value
        return out


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    *,
    initial_capital: float,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full metric suite from an equity curve and trade ledger."""
    equity = equity.astype(float)
    has_curve = len(equity) >= 1
    ppy = (
        periods_per_year
        if periods_per_year is not None
        else infer_periods_per_year(pd.DatetimeIndex(equity.index))
    )
    returns = periodic_returns(equity)

    final_equity = float(equity.iloc[-1]) if has_curve else initial_capital
    total_return = (final_equity / initial_capital - 1.0) if initial_capital > 0 else float("nan")
    _cagr = cagr(equity, initial_capital) if has_curve else float("nan")
    mdd = max_drawdown(equity) if has_curve else 0.0
    calmar = (_cagr / abs(mdd)) if (mdd < 0 and math.isfinite(_cagr)) else float("nan")

    net = [t.net_pnl for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x < 0]

    return PerformanceMetrics(
        start=pd.Timestamp(equity.index[0]) if has_curve else None,
        end=pd.Timestamp(equity.index[-1]) if has_curve else None,
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        cagr=_cagr,
        annual_volatility=annual_volatility(returns, ppy),
        sharpe=sharpe_ratio(returns, ppy, risk_free_rate),
        sortino=sortino_ratio(returns, ppy, mar=risk_free_rate),
        calmar=calmar,
        max_drawdown=mdd,
        max_drawdown_dollars=max_drawdown_dollars(equity) if has_curve else 0.0,
        max_drawdown_duration=max_drawdown_duration(equity) if has_curve else pd.Timedelta(0),
        periods_per_year=ppy,
        n_trades=len(trades),
        win_rate=win_rate(trades),
        profit_factor=profit_factor(trades),
        expectancy=expectancy(trades),
        avg_trade=float(np.mean(net)) if net else float("nan"),
        avg_win=float(np.mean(wins)) if wins else float("nan"),
        avg_loss=float(np.mean(losses)) if losses else float("nan"),
        largest_win=float(max(wins)) if wins else float("nan"),
        largest_loss=float(min(losses)) if losses else float("nan"),
        gross_profit=float(sum(wins)),
        gross_loss=float(sum(losses)),
        total_pnl=float(sum(net)),
        total_commission=float(sum(t.commission for t in trades)),
    )


def per_year_breakdown(equity: pd.Series, trades: Sequence[Trade]) -> pd.DataFrame:
    """Per-calendar-year net P&L, trade count, and year return.

    Surfaces regime concentration (brief trap #13): the report flags when one
    year dominates the total.
    """
    years: dict[int, dict[str, float]] = {}
    for t in trades:
        yr = int(t.exit_time.year)
        row = years.setdefault(yr, {"net_pnl": 0.0, "n_trades": 0.0})
        row["net_pnl"] += t.net_pnl
        row["n_trades"] += 1

    if len(equity) >= 2:
        eq = equity.astype(float)
        year_end = eq.resample("YE").last()
        year_start = eq.resample("YE").first()
        for ts in year_end.index:
            yr = int(ts.year)
            row = years.setdefault(yr, {"net_pnl": 0.0, "n_trades": 0.0})
            start_val = float(year_start.loc[ts])
            row["return_pct"] = (
                float(year_end.loc[ts]) / start_val - 1.0 if start_val > 0 else float("nan")
            )

    if not years:
        return pd.DataFrame(columns=["net_pnl", "n_trades", "return_pct"])
    frame = pd.DataFrame.from_dict(years, orient="index").sort_index()
    frame.index.name = "year"
    if "n_trades" in frame:
        frame["n_trades"] = frame["n_trades"].astype("int64")
    return frame


def split_is_oos(
    equity: pd.Series,
    trades: Sequence[Trade],
    cut: pd.Timestamp,
    *,
    initial_capital: float,
    periods_per_year: float | None = None,
) -> tuple[PerformanceMetrics, PerformanceMetrics]:
    """Metrics for the in-sample (<= cut) and out-of-sample (> cut) splits."""
    eq_is = equity[equity.index <= cut]
    eq_oos = equity[equity.index > cut]
    tr_is = [t for t in trades if t.exit_time <= cut]
    tr_oos = [t for t in trades if t.exit_time > cut]
    cap_oos = float(eq_is.iloc[-1]) if len(eq_is) else initial_capital
    metrics_is = compute_metrics(
        eq_is, tr_is, initial_capital=initial_capital, periods_per_year=periods_per_year
    )
    metrics_oos = compute_metrics(
        eq_oos, tr_oos, initial_capital=cap_oos, periods_per_year=periods_per_year
    )
    return metrics_is, metrics_oos


__all__ = [
    "PerformanceMetrics",
    "annual_volatility",
    "cagr",
    "compute_metrics",
    "drawdown_series",
    "expectancy",
    "infer_periods_per_year",
    "max_drawdown",
    "max_drawdown_dollars",
    "max_drawdown_duration",
    "per_year_breakdown",
    "periodic_returns",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "split_is_oos",
    "win_rate",
]
