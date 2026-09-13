"""Причины закрытия записей статистики."""

from __future__ import annotations

from app.analysis.signals import Direction
from app.bot.outcomes import EXIT_REASONS, Outcome
from app.core.timeframes import Timeframe
from tests.synthetic import START


def _record(reason: str | None) -> Outcome:
    return Outcome(
        id=1,
        symbol="TESTUSDT",
        timeframe=Timeframe.H1,
        direction=Direction.BULL,
        degree=2,
        signal_time=START,
        entry_time=START,
        entry_price=100.0,
        exit_reason=reason,
        status="closed",
    )


def test_replaced_is_a_known_reason():
    assert "replaced" in EXIT_REASONS
    assert EXIT_REASONS["replaced"] == "замена контрольной точки"


def test_reason_label_is_russian_for_every_reason():
    for reason in EXIT_REASONS:
        label = _record(reason).reason_label
        assert label != reason, f"сырое английское слово в выводе: {reason}"
        assert label.strip()


def test_unknown_reason_does_not_crash():
    assert _record("wat").reason_label == "wat"
    assert _record(None).reason_label == "—"
