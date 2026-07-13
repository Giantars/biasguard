"""Instrument (contract) specifications.

A minimal spec: how many dollars one price point is worth per contract
(``multiplier``) and the minimum price increment (``tick_size``). These drive
commission-per-contract, tick-based slippage, and P&L-to-dollars conversion.

Presets are provided for the common CME index and metals futures the framework
was designed against. They are conveniences, not endorsements — verify contract
specs against your broker before trusting a dollar figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Instrument:
    """Contract specification used by the broker and portfolio."""

    symbol: str
    multiplier: float = 1.0
    tick_size: float = 0.01

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick per contract."""
        return self.multiplier * self.tick_size

    def round_to_tick(self, price: float) -> float:
        """Round a price to the nearest valid tick."""
        return round(price / self.tick_size) * self.tick_size


# --- Common CME futures presets (verify against your broker) --------------- #
NQ = Instrument("NQ", multiplier=20.0, tick_size=0.25)
MNQ = Instrument("MNQ", multiplier=2.0, tick_size=0.25)
ES = Instrument("ES", multiplier=50.0, tick_size=0.25)
MES = Instrument("MES", multiplier=5.0, tick_size=0.25)
GC = Instrument("GC", multiplier=100.0, tick_size=0.10)
SI = Instrument("SI", multiplier=5000.0, tick_size=0.005)

PRESETS: dict[str, Instrument] = {i.symbol: i for i in (NQ, MNQ, ES, MES, GC, SI)}


__all__ = ["ES", "GC", "MES", "MNQ", "NQ", "PRESETS", "SI", "Instrument"]
