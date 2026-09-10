"""Статистика по дивергенциям: что происходило с ценой ПОСЛЕ сигнала.

На каждую подтверждённую дивергенцию заводится запись («исход»):

    вход      = close свечи, подтвердившей фрактал (момент, когда пришёл алерт);
    ход в пользу сигнала  = лучшая цена в сторону сигнала после входа;
    ход против сигнала    = худшая цена против сигнала после входа;
    выход     = обратная дивергенция, дивергенция того же направления
                или таймаут по числу баров.

Все проценты считаются В СТОРОНУ СИГНАЛА: для медвежьей дивергенции падение
цены даёт плюс, рост — минус. Так бычьи и медвежьи записи можно усреднять
вместе, не путаясь в знаках.

Запись обновляется инкрементально, теми же свечами, которые и так пришли
на очередном опросе. Пересчитывать из окна нельзя: lookback_bars ограничен,
а запись на 1D легко живёт дольше, чем окно анализа.

Таблица создаётся лениво, при первом обращении, поэтому storage.py трогать
не нужно — существующая БД доживёт до новой схемы сама.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.analysis.signals import Direction, Signal
from app.bot.storage import Storage
from app.core.timeframes import Timeframe

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    direction     TEXT NOT NULL,
    degree        INTEGER NOT NULL DEFAULT 2,
    signal_time   TEXT NOT NULL,
    entry_time    TEXT NOT NULL,
    entry_price   REAL NOT NULL,
    entry_rsi     REAL,
    last_time     TEXT,
    bars_held     INTEGER NOT NULL DEFAULT 0,
    max_high      REAL,
    max_high_time TEXT,
    min_low       REAL,
    min_low_time  TEXT,
    exit_time     TEXT,
    exit_price    REAL,
    exit_reason   TEXT,
    status        TEXT NOT NULL DEFAULT 'open',
    created_at    TEXT NOT NULL,
    UNIQUE (symbol, timeframe, direction, signal_time)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_status
    ON outcomes(status, symbol, timeframe);
"""

#: Человеческие названия причин закрытия записи.
EXIT_REASONS = {
    "reverse": "обратная дивергенция",
    "same": "новая дивергенция того же направления",
    "timeout": "таймаут",
}


@dataclass(slots=True)
class Outcome:
    id: int
    symbol: str
    timeframe: Timeframe
    direction: Direction
    degree: int
    signal_time: datetime
    entry_time: datetime
    entry_price: float
    entry_rsi: float | None = None
    last_time: datetime | None = None
    bars_held: int = 0
    max_high: float | None = None
    max_high_time: datetime | None = None
    min_low: float | None = None
    min_low_time: datetime | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    status: str = "open"

    # --- проценты «в сторону сигнала» --------------------------------------

    @property
    def sign(self) -> float:
        return 1.0 if self.direction is Direction.BULL else -1.0

    def pnl_pct(self, price: float | None) -> float | None:
        if price is None or not self.entry_price:
            return None
        return self.sign * (price / self.entry_price - 1.0) * 100.0

    @property
    def favorable_price(self) -> float | None:
        return self.max_high if self.direction is Direction.BULL else self.min_low

    @property
    def favorable_time(self) -> datetime | None:
        return self.max_high_time if self.direction is Direction.BULL else self.min_low_time

    @property
    def adverse_price(self) -> float | None:
        return self.min_low if self.direction is Direction.BULL else self.max_high

    @property
    def adverse_time(self) -> datetime | None:
        return self.min_low_time if self.direction is Direction.BULL else self.max_high_time

    @property
    def favorable_pct(self) -> float | None:
        """Максимальный ход в сторону сигнала, % (обычно ≥ 0)."""
        return self.pnl_pct(self.favorable_price)

    @property
    def adverse_pct(self) -> float | None:
        """Максимальный ход против сигнала, % (обычно ≤ 0)."""
        return self.pnl_pct(self.adverse_price)

    @property
    def result_pct(self) -> float | None:
        """Результат на момент закрытия записи, %."""
        return self.pnl_pct(self.exit_price)

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def reason_label(self) -> str:
        return EXIT_REASONS.get(self.exit_reason or "", self.exit_reason or "—")


class OutcomeStore:
    """Доступ к таблице outcomes. Работает на соединении из Storage."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._lock = asyncio.Lock()
        self._ready = False

    async def ensure_schema(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await self.storage.conn.executescript(SCHEMA)
            await self.storage.conn.commit()
            self._ready = True

    # --- запись -------------------------------------------------------------

    async def open_for_signal(self, signal: Signal) -> Outcome | None:
        """Заводит запись по дивергенции. Повторный вызов вернёт существующую."""
        if signal.confirmation_time is None or signal.confirmation_price is None:
            return None  # фрактал ещё не подтверждён — входа нет
        await self.ensure_schema()
        async with self._lock:
            await self.storage.conn.execute(
                """INSERT OR IGNORE INTO outcomes
                   (symbol, timeframe, direction, degree, signal_time, entry_time,
                    entry_price, entry_rsi, last_time, bars_held,
                    max_high, max_high_time, min_low, min_low_time,
                    status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 'open', ?)""",
                (
                    signal.symbol.upper(),
                    signal.timeframe.value,
                    signal.direction.value,
                    signal.degree,
                    _iso(signal.candle_time),
                    _iso(signal.confirmation_time),
                    signal.confirmation_price,
                    signal.rsi,
                    _iso(signal.confirmation_time),
                    signal.confirmation_price,
                    _iso(signal.confirmation_time),
                    signal.confirmation_price,
                    _iso(signal.confirmation_time),
                    _now_iso(),
                ),
            )
            await self.storage.conn.commit()
        return await self.get(
            signal.symbol, signal.timeframe, signal.direction, signal.candle_time
        )

    async def advance(
        self,
        outcome: Outcome,
        *,
        high: float,
        high_time: datetime,
        low: float,
        low_time: datetime,
        last_time: datetime,
        bars: int,
    ) -> None:
        """Досчитывает бегущие экстремумы по новым барам."""
        if outcome.max_high is None or high > outcome.max_high:
            outcome.max_high, outcome.max_high_time = high, high_time
        if outcome.min_low is None or low < outcome.min_low:
            outcome.min_low, outcome.min_low_time = low, low_time
        outcome.last_time = last_time
        outcome.bars_held += bars

        await self.ensure_schema()
        async with self._lock:
            await self.storage.conn.execute(
                """UPDATE outcomes SET
                     max_high=?, max_high_time=?, min_low=?, min_low_time=?,
                     last_time=?, bars_held=?
                   WHERE id=?""",
                (
                    outcome.max_high,
                    _iso_or_none(outcome.max_high_time),
                    outcome.min_low,
                    _iso_or_none(outcome.min_low_time),
                    _iso_or_none(outcome.last_time),
                    outcome.bars_held,
                    outcome.id,
                ),
            )
            await self.storage.conn.commit()

    async def close(
        self,
        outcome: Outcome,
        exit_time: datetime,
        exit_price: float,
        reason: str,
    ) -> None:
        outcome.exit_time = exit_time
        outcome.exit_price = float(exit_price)
        outcome.exit_reason = reason
        outcome.status = "closed"

        await self.ensure_schema()
        async with self._lock:
            await self.storage.conn.execute(
                """UPDATE outcomes SET
                     exit_time=?, exit_price=?, exit_reason=?, status='closed'
                   WHERE id=?""",
                (_iso(exit_time), float(exit_price), reason, outcome.id),
            )
            await self.storage.conn.commit()
        log.info(
            "%s:%s статистика закрыта (%s): итог %.2f%%",
            outcome.symbol, outcome.timeframe.value, reason,
            outcome.result_pct if outcome.result_pct is not None else float("nan"),
        )

    # --- чтение -------------------------------------------------------------

    async def get(
        self,
        symbol: str,
        timeframe: Timeframe,
        direction: Direction,
        signal_time: datetime,
    ) -> Outcome | None:
        await self.ensure_schema()
        cur = await self.storage.conn.execute(
            """SELECT * FROM outcomes
               WHERE symbol=? AND timeframe=? AND direction=? AND signal_time=?""",
            (symbol.upper(), timeframe.value, direction.value, _iso(signal_time)),
        )
        row = await cur.fetchone()
        return _row_to_outcome(row) if row else None

    async def list_open(
        self, symbol: str | None = None, timeframe: Timeframe | None = None
    ) -> list[Outcome]:
        return await self._query("open", symbol, timeframe, limit=500, oldest_first=True)

    async def list_closed(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 500,
    ) -> list[Outcome]:
        return await self._query("closed", symbol, timeframe, limit=limit)

    async def counts(self, symbol: str | None = None) -> tuple[int, int]:
        """(открытых, закрытых)."""
        await self.ensure_schema()
        sql = "SELECT status, COUNT(*) AS n FROM outcomes"
        args: list = []
        if symbol:
            sql += " WHERE symbol=?"
            args.append(symbol.upper())
        sql += " GROUP BY status"
        cur = await self.storage.conn.execute(sql, args)
        stats = {row["status"]: int(row["n"]) for row in await cur.fetchall()}
        return stats.get("open", 0), stats.get("closed", 0)

    async def _query(
        self,
        status: str,
        symbol: str | None,
        timeframe: Timeframe | None,
        limit: int,
        oldest_first: bool = False,
    ) -> list[Outcome]:
        await self.ensure_schema()
        sql = "SELECT * FROM outcomes WHERE status=?"
        args: list = [status]
        if symbol:
            sql += " AND symbol=?"
            args.append(symbol.upper())
        if timeframe is not None:
            sql += " AND timeframe=?"
            args.append(timeframe.value)
        sql += f" ORDER BY entry_time {'ASC' if oldest_first else 'DESC'} LIMIT ?"
        args.append(int(limit))
        cur = await self.storage.conn.execute(sql, args)
        return [_row_to_outcome(row) for row in await cur.fetchall()]


def _row_to_outcome(row) -> Outcome:
    return Outcome(
        id=int(row["id"]),
        symbol=row["symbol"],
        timeframe=Timeframe.parse(row["timeframe"]),
        direction=Direction(row["direction"]),
        degree=int(row["degree"]),
        signal_time=_parse(row["signal_time"]),
        entry_time=_parse(row["entry_time"]),
        entry_price=float(row["entry_price"]),
        entry_rsi=_float_or_none(row["entry_rsi"]),
        last_time=_parse_or_none(row["last_time"]),
        bars_held=int(row["bars_held"] or 0),
        max_high=_float_or_none(row["max_high"]),
        max_high_time=_parse_or_none(row["max_high_time"]),
        min_low=_float_or_none(row["min_low"]),
        min_low_time=_parse_or_none(row["min_low_time"]),
        exit_time=_parse_or_none(row["exit_time"]),
        exit_price=_float_or_none(row["exit_price"]),
        exit_reason=row["exit_reason"],
        status=row["status"],
    )


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def _parse(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _parse_or_none(value: str | None) -> datetime | None:
    return None if value is None else _parse(value)


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _iso_or_none(ts: datetime | None) -> str | None:
    return None if ts is None else _iso(ts)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
