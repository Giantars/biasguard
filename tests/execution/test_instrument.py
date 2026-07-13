"""Tests for the Instrument spec and presets."""

from __future__ import annotations

import pytest

from biasguard.execution.instrument import MNQ, NQ, PRESETS, Instrument


def test_tick_value() -> None:
    assert NQ.tick_value == pytest.approx(20.0 * 0.25)
    assert MNQ.tick_value == pytest.approx(2.0 * 0.25)


def test_round_to_tick() -> None:
    assert NQ.round_to_tick(15000.1) == pytest.approx(15000.0)
    assert NQ.round_to_tick(15000.2) == pytest.approx(15000.25)


def test_presets_indexed_by_symbol() -> None:
    assert PRESETS["NQ"] is NQ
    assert set(PRESETS) >= {"NQ", "MNQ", "ES", "MES", "GC", "SI"}


@pytest.mark.parametrize("bad", [{"multiplier": 0.0}, {"tick_size": -1.0}])
def test_invalid_spec_raises(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Instrument("X", **bad)
