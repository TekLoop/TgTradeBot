"""Таймфреймы и выравнивание границ свечей по UTC.

Модуль намеренно не зависит ни от чего, кроме стандартной библиотеки:
его импортируют и analysis/, и data/, и bot/.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class Timeframe(str, Enum):
    """Поддерживаемые таймфреймы. Значение = каноничное строковое имя."""

    H1 = "1H"
    H2 = "2H"
    H3 = "3H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"

    # --- парсинг ------------------------------------------------------------

    @classmethod
    def parse(cls, raw: str | "Timeframe") -> "Timeframe":
        if isinstance(raw, Timeframe):
            return raw
        key = str(raw).strip().upper().replace(" ", "")
        if key in _ALIASES:
            return _ALIASES[key]
        raise ValueError(
            f"Неизвестный таймфрейм: {raw!r}. "
            f"Доступны: {', '.join(tf.value for tf in Timeframe)}"
        )

    @classmethod
    def try_parse(cls, raw: str) -> "Timeframe | None":
        try:
            return cls.parse(raw)
        except ValueError:
            return None

    # --- метрики ------------------------------------------------------------

    @property
    def minutes(self) -> int:
        return _MINUTES[self]

    @property
    def duration(self) -> timedelta:
        return timedelta(minutes=self.minutes)

    @property
    def pandas_rule(self) -> str:
        """Правило для DataFrame.resample()."""
        return _PANDAS_RULE[self]

    def divides(self, other: "Timeframe") -> bool:
        """True, если из self можно собрать other простым ресемплингом."""
        return self.minutes < other.minutes and other.minutes % self.minutes == 0

    def ratio_to(self, other: "Timeframe") -> int:
        if not self.divides(other):
            raise ValueError(f"{self.value} не делит {other.value} нацело")
        return other.minutes // self.minutes

    # --- границы свечей -----------------------------------------------------

    def floor(self, ts: datetime) -> datetime:
        """Время открытия свечи, в которую попадает ts.

        Часовые/дневные ТФ выравниваются от epoch (UTC-полночь),
        недельный — от понедельника 00:00 UTC (как на большинстве бирж).
        """
        ts = _as_utc(ts)
        if self is Timeframe.W1:
            monday = ts - timedelta(days=ts.weekday())
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = ts - EPOCH
        buckets = elapsed // self.duration
        return EPOCH + buckets * self.duration

    def open_time(self, ts: datetime) -> datetime:
        return self.floor(ts)

    def close_time(self, open_time: datetime) -> datetime:
        """Момент закрытия свечи, открытой в open_time (правая граница, исключающая)."""
        return _as_utc(open_time) + self.duration

    def next_close(self, ts: datetime) -> datetime:
        """Ближайшее строго будущее закрытие свечи относительно ts."""
        return self.floor(ts) + self.duration

    def is_closed(self, open_time: datetime, now: datetime) -> bool:
        return self.close_time(open_time) <= _as_utc(now)


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


_MINUTES: dict[Timeframe, int] = {
    Timeframe.H1: 60,
    Timeframe.H2: 120,
    Timeframe.H3: 180,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
}

_PANDAS_RULE: dict[Timeframe, str] = {
    Timeframe.H1: "1h",
    Timeframe.H2: "2h",
    Timeframe.H3: "3h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1D",
    Timeframe.W1: "W-MON",
}


def _build_aliases() -> dict[str, Timeframe]:
    aliases: dict[str, Timeframe] = {}
    extra = {
        Timeframe.H1: ("1H", "H1", "1HR", "1HRS", "60M", "60MIN"),
        Timeframe.H2: ("2H", "H2", "2HRS"),
        Timeframe.H3: ("3H", "H3", "3HRS"),
        Timeframe.H4: ("4H", "H4", "4HRS"),
        Timeframe.D1: ("1D", "D1", "D", "1DAY", "DAILY"),
        Timeframe.W1: ("1W", "W1", "W", "7DAY", "WEEKLY"),
    }
    for tf, keys in extra.items():
        for key in keys:
            aliases[key] = tf
    return aliases


_ALIASES = _build_aliases()

ALL_TIMEFRAMES: tuple[Timeframe, ...] = tuple(Timeframe)
