"""Execution profiles — named bundles of cost / slippage / fill assumptions.

A backtest's realism lives almost entirely in three swappable objects: the
commission model, the slippage model, and the fill model. An
:class:`ExecutionProfile` groups those (plus a human-readable assumption set)
under a name like ``"Real Market"`` or ``"Prop Firm Simulation"`` so a whole
execution environment can be selected in one line and displayed in the report.

The engine never learns about profiles — a profile only *supplies* the same
model objects the broker already consumes, so new profiles are added with zero
engine change (construct an :class:`ExecutionProfile`, optionally
:func:`register_profile` it).

**Realism is probed, not declared.** :meth:`ExecutionProfile.realism` runs each
model through a tiny probe (does slippage move the price? does commission cost
anything? does the fill model fill on a mere touch?) so an unrealistic
assumption — *including one hidden inside a user's custom model* — is always
surfaced. If any is found, the report states that results represent a
**simulated execution environment**.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from biasguard.data.schema import Bar
from biasguard.events import OrderEvent
from biasguard.execution.costs import (
    CommissionModel,
    NoSlippage,
    PerContractCommission,
    SlippageModel,
    TickSlippage,
)
from biasguard.execution.fill_models import (
    FillModel,
    FillRequest,
    TouchFill,
    TradeThroughFill,
)
from biasguard.execution.instrument import Instrument
from biasguard.types import OrderSide, OrderType

# A neutral instrument used only to probe cost/slippage/fill models for realism.
_PROBE_INSTRUMENT = Instrument("PROBE", multiplier=1.0, tick_size=0.25)
_PROBE_PRICE = 100.0
_PROBE_TS = pd.Timestamp("2020-01-01")


def _slippage_warning(slippage: SlippageModel) -> str | None:
    """Behavioral probe: is slippage adverse (worse than reference) on both sides?

    Adverse means a BUY fills at ``>=`` the reference price and a SELL at ``<=``
    it. A model that leaves either side unchanged (zero) or *improves* it
    (favorable) is optimistic and cannot be certified realistic. Because this
    probes behavior rather than type, it flags any custom model — a favorable or
    one-sided one, not just an exact no-op — matching the "cannot be bypassed"
    guarantee.
    """
    buy = slippage.apply(_PROBE_PRICE, OrderSide.BUY, _PROBE_INSTRUMENT)
    sell = slippage.apply(_PROBE_PRICE, OrderSide.SELL, _PROBE_INSTRUMENT)
    if buy < _PROBE_PRICE or sell > _PROBE_PRICE:
        return "Favorable slippage: at least one side fills better than the reference (optimistic)."
    if buy == _PROBE_PRICE and sell == _PROBE_PRICE:
        return "Zero slippage: fills assume no spread cost or market impact."
    if buy == _PROBE_PRICE or sell == _PROBE_PRICE:
        return "Zero slippage on one side: half of fills assume no cost."
    return None


def _commission_warning(commission: CommissionModel) -> str | None:
    """Behavioral probe: does a one-contract fill cost nothing?"""
    if commission.calculate(1.0, _PROBE_PRICE, _PROBE_INSTRUMENT) == 0.0:
        return "Zero commission: trading costs are not modelled."
    return None


def _fill_warning(fill_model: FillModel) -> str | None:
    """Behavioral probe: does the model fill a limit on less than a tick of trade-through?

    A synthetic BUY limit sits at ``_PROBE_PRICE``; the probe bar dips only 0.9
    of a tick below it. The conservative default (``TradeThroughFill`` at one
    tick) does not fill here; a ``TouchFill`` — or *any* custom model that fills
    on a near-touch — does. Probing behavior (not ``isinstance``) is what lets
    this catch an optimistic fill model hidden inside a user's own subclass.
    """
    tick = _PROBE_INSTRUMENT.tick_size
    order = OrderEvent(
        timestamp=_PROBE_TS,
        symbol="PROBE",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.LIMIT,
        limit_price=_PROBE_PRICE,
    )
    bar = Bar(
        timestamp=_PROBE_TS,
        open=_PROBE_PRICE,
        high=_PROBE_PRICE,
        low=_PROBE_PRICE - 0.9 * tick,
        close=_PROBE_PRICE,
        volume=1.0,
    )
    decision = fill_model.fill(FillRequest(order=order, bar=bar, remaining_qty=1.0, tick_size=tick))
    if decision.filled:
        return (
            "Optimistic fills: a limit is assumed to fill on a near-touch "
            "(under one tick of trade-through; queue position ignored)."
        )
    return None


def _model_description(model: object) -> dict[str, Any]:
    """A small, dependency-free structured view of a model for export.

    Deliberately does not use :func:`biasguard.analytics.fingerprint.describe`
    so this module stays within the ``execution`` layer (no upward import).
    Tolerates ``__slots__``-only custom models (``vars`` would raise): it reads
    ``__dict__`` when present and falls back to declared slots otherwise.
    """
    attrs: dict[str, Any] = dict(getattr(model, "__dict__", {}))
    for klass in type(model).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if slot != "__dict__" and hasattr(model, slot):
                attrs.setdefault(slot, getattr(model, slot))
    params = {
        k: v
        for k, v in sorted(attrs.items())
        if not k.startswith("_") and isinstance(v, (int, float, str, bool))
    }
    return {"type": type(model).__name__, "params": params}


@dataclass(frozen=True, slots=True)
class ExecutionRealism:
    """The realism verdict for a profile's assumptions."""

    is_realistic: bool
    warnings: tuple[str, ...]

    @property
    def banner(self) -> str:
        """One-line status suitable for a report banner."""
        if self.is_realistic:
            return "Realistic execution assumptions."
        return "Results represent a SIMULATED execution environment (optimistic assumptions)."


def assess_realism(
    commission: CommissionModel, slippage: SlippageModel, fill_model: FillModel
) -> ExecutionRealism:
    """Behaviorally probe the three models and collect optimistic-assumption warnings.

    Every axis is probed by *running* the model (not by checking its class), so
    an unrealistic assumption hidden inside a user's own model is surfaced too.
    """
    warnings = [
        warning
        for warning in (
            _slippage_warning(slippage),
            _commission_warning(commission),
            _fill_warning(fill_model),
        )
        if warning is not None
    ]
    return ExecutionRealism(is_realistic=not warnings, warnings=tuple(warnings))


@dataclass(frozen=True)
class ExecutionProfile:
    """A named, reusable bundle of execution assumptions.

    Supplies the commission, slippage, and fill models the broker consumes. The
    fill model is a *factory* (matching :class:`~biasguard.validation.context.BacktestSpec`)
    because a fresh instance is created per run; commission and slippage models
    are stateless and shared. ``assumptions`` is a human-readable list shown in
    the report alongside the probed realism verdict.
    """

    name: str
    description: str
    commission: CommissionModel
    slippage: SlippageModel
    fill_model_factory: Callable[[], FillModel] = TradeThroughFill
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def realism(self) -> ExecutionRealism:
        """Probe the configured models for optimistic assumptions."""
        return assess_realism(self.commission, self.slippage, self.fill_model_factory())

    @property
    def is_realistic(self) -> bool:
        return self.realism.is_realistic

    def describe(self) -> dict[str, Any]:
        """A deterministic, JSON-serializable description for the audit export."""
        realism = self.realism
        return {
            "name": self.name,
            "description": self.description,
            "commission": _model_description(self.commission),
            "slippage": _model_description(self.slippage),
            "fill_model": _model_description(self.fill_model_factory()),
            "assumptions": list(self.assumptions),
            "is_realistic": realism.is_realistic,
            "warnings": list(realism.warnings),
        }


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
#: The default. Costs on, conservative trade-through fills, one tick of adverse
#: slippage — a defensible estimate of live retail futures execution.
REAL_MARKET = ExecutionProfile(
    name="Real Market",
    description=(
        "Best-estimate live retail execution: per-contract commission, one tick of "
        "adverse slippage on liquidity-taking fills, and conservative trade-through "
        "limit fills."
    ),
    commission=PerContractCommission(2.0),
    slippage=TickSlippage(1.0),
    fill_model_factory=TradeThroughFill,
    assumptions=(
        "Commission: $2.00 per contract per side.",
        "Slippage: 1 tick adverse on market/stop fills.",
        "Limit fills require price to trade through the level (queue risk modelled).",
    ),
)

#: Mirrors a prop-firm evaluation / sim account, which tends to fill optimistically
#: (sim fills at the touch, little to no slippage). Deliberately optimistic and
#: therefore flagged as a simulated environment — do not mistake it for live.
PROP_FIRM_SIM = ExecutionProfile(
    name="Prop Firm Simulation",
    description=(
        "Approximates a prop-firm evaluation / sim account: commissions are charged "
        "but fills are optimistic (touch fills, no modelled slippage), as sim engines "
        "often fill better than a live book would. Flagged as simulated."
    ),
    commission=PerContractCommission(2.0),
    slippage=NoSlippage(),
    fill_model_factory=TouchFill,
    assumptions=(
        "Commission: $2.00 per contract per side.",
        "Slippage: none (sim fills at the reference price).",
        "Limit fills on a touch (optimistic; ignores queue position).",
    ),
)


def custom_profile(
    name: str,
    *,
    commission: CommissionModel,
    slippage: SlippageModel,
    fill_model_factory: Callable[[], FillModel] = TradeThroughFill,
    description: str = "User-defined execution profile.",
    assumptions: tuple[str, ...] = (),
) -> ExecutionProfile:
    """Build a fully custom :class:`ExecutionProfile` from your own models."""
    return ExecutionProfile(
        name=name,
        description=description,
        commission=commission,
        slippage=slippage,
        fill_model_factory=fill_model_factory,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------------- #
# Registry (extensible without touching the engine)
# --------------------------------------------------------------------------- #
#: Named profiles available by name. Add to it with :func:`register_profile`.
PROFILES: dict[str, ExecutionProfile] = {}


def register_profile(profile: ExecutionProfile, *, replace: bool = True) -> ExecutionProfile:
    """Register a profile under its name (into the module-level :data:`PROFILES`)."""
    if profile.name in PROFILES and not replace:
        raise ValueError(f"a profile named {profile.name!r} is already registered")
    PROFILES[profile.name] = profile
    return profile


def get_profile(name: str) -> ExecutionProfile | None:
    """Look up a registered profile by name (``None`` if absent)."""
    return PROFILES.get(name)


register_profile(REAL_MARKET)
register_profile(PROP_FIRM_SIM)


__all__ = [
    "PROFILES",
    "PROP_FIRM_SIM",
    "REAL_MARKET",
    "ExecutionProfile",
    "ExecutionRealism",
    "assess_realism",
    "custom_profile",
    "get_profile",
    "register_profile",
]
