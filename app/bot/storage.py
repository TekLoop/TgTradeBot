"""Состояние бота в SQLite: активы, таймфреймы, пороги, опорные точки,
отправленные алерты (дедупликация), подписчики.

Всё переживает рестарт. Ключ дедупликации:
(symbol, timeframe, alert_type, candle_time).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.analysis.signals import AlertType, Direction, PivotPoint
from app.core.timeframes import Timeframe

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    symbol   TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    enabled  INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS asset_timeframes (
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    enabled   INTEGER NOT NULL DEFAULT 1,
    seeded    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, timeframe),
    FOREIGN KEY (symbol) REFERENCES assets(symbol) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS params (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sent_alerts (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    alert_type  TEXT NOT NULL,
    candle_time TEXT NOT NULL,
    payload     TEXT,
    notified    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, alert_type, candle_time)
);

CREATE TABLE IF NOT EXISTS anchors (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    direction  TEXT NOT NULL,
    candle_time TEXT,
    price      REAL,
    rsi        REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, direction)
);

CREATE TABLE IF NOT EXISTS subscribers (
    chat_id  INTEGER PRIMARY KEY,
    title    TEXT,
    added_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_created ON sent_alerts(created_at);
"""


@dataclass(slots=True)
class AssetRecord:
    symbol: str
    provider: str
    enabled: bool = True
    timeframes: dict[Timeframe, bool] = field(default_factory=dict)

    def active_timeframes(self) -> list[Timeframe]:
        return [tf for tf, on in sorted(
            self.timeframes.items(), key=lambda kv: kv[0].minutes
        ) if on]


@dataclass(frozen=True, slots=True)
class TrackingUnit:
    """Независимая единица отслеживания: актив × таймфрейм."""

    symbol: str
    provider: str
    timeframe: Timeframe

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.timeframe.value}"


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        log.info("SQLite готова: %s", self.db_path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Storage.connect() не вызван")
        return self._conn

    # --- активы -------------------------------------------------------------

    async def upsert_asset(
        self,
        symbol: str,
        provider: str,
        timeframes: list[Timeframe] | None = None,
        *,
        force_enable: bool = True,
    ) -> None:
        """force_enable=False используется при синхронизации из YAML:
        новые ТФ добавляются, но выключенные вручную через /tf остаются выключенными."""
        symbol = symbol.upper()
        async with self._lock:
            await self.conn.execute(
                """INSERT INTO assets(symbol, provider, enabled, created_at)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(symbol) DO UPDATE SET provider=excluded.provider""",
                (symbol, provider.lower(), _now_iso()),
            )
            conflict = (
                "ON CONFLICT(symbol, timeframe) DO UPDATE SET enabled=1"
                if force_enable
                else "ON CONFLICT(symbol, timeframe) DO NOTHING"
            )
            for tf in timeframes or []:
                await self.conn.execute(
                    "INSERT INTO asset_timeframes(symbol, timeframe, enabled, seeded) "
                    f"VALUES (?, ?, 1, 0) {conflict}",
                    (symbol, tf.value),
                )
            await self.conn.commit()

    async def remove_asset(self, symbol: str) -> bool:
        symbol = symbol.upper()
        async with self._lock:
            cur = await self.conn.execute("DELETE FROM assets WHERE symbol = ?", (symbol,))
            await self.conn.execute("DELETE FROM asset_timeframes WHERE symbol = ?", (symbol,))
            await self.conn.execute("DELETE FROM anchors WHERE symbol = ?", (symbol,))
            await self.conn.commit()
            return cur.rowcount > 0

    async def set_timeframe(self, symbol: str, tf: Timeframe, enabled: bool) -> bool:
        symbol = symbol.upper()
        async with self._lock:
            cur = await self.conn.execute(
                "SELECT 1 FROM assets WHERE symbol = ?", (symbol,)
            )
            if await cur.fetchone() is None:
                return False
            await self.conn.execute(
                """INSERT INTO asset_timeframes(symbol, timeframe, enabled, seeded)
                   VALUES (?, ?, ?, 0)
                   ON CONFLICT(symbol, timeframe) DO UPDATE SET enabled=excluded.enabled""",
                (symbol, tf.value, int(enabled)),
            )
            await self.conn.commit()
            return True

    async def list_assets(self) -> list[AssetRecord]:
        cur = await self.conn.execute(
            "SELECT symbol, provider, enabled FROM assets ORDER BY symbol"
        )
        rows = await cur.fetchall()
        records = {
            row["symbol"]: AssetRecord(
                symbol=row["symbol"],
                provider=row["provider"],
                enabled=bool(row["enabled"]),
            )
            for row in rows
        }
        cur = await self.conn.execute(
            "SELECT symbol, timeframe, enabled FROM asset_timeframes"
        )
        for row in await cur.fetchall():
            record = records.get(row["symbol"])
            tf = Timeframe.try_parse(row["timeframe"])
            if record is None or tf is None:
                continue
            record.timeframes[tf] = bool(row["enabled"])
        return list(records.values())

    async def get_asset(self, symbol: str) -> AssetRecord | None:
        symbol = symbol.upper()
        for record in await self.list_assets():
            if record.symbol == symbol:
                return record
        return None

    async def active_units(self) -> list[TrackingUnit]:
        units: list[TrackingUnit] = []
        for asset in await self.list_assets():
            if not asset.enabled:
                continue
            for tf in asset.active_timeframes():
                units.append(TrackingUnit(asset.symbol, asset.provider, tf))
        return units

    # --- параметры ----------------------------------------------------------

    async def get_param_overrides(self) -> dict:
        cur = await self.conn.execute("SELECT key, value FROM params")
        out: dict = {}
        for row in await cur.fetchall():
            try:
                out[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                continue
        return out

    async def set_param(self, key: str, value) -> None:
        async with self._lock:
            await self.conn.execute(
                """INSERT INTO params(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (key, json.dumps(value)),
            )
            await self.conn.commit()

    async def clear_params(self) -> None:
        async with self._lock:
            await self.conn.execute("DELETE FROM params")
            await self.conn.commit()

    # --- дедупликация алертов ----------------------------------------------

    async def try_register_alert(
        self,
        symbol: str,
        timeframe: Timeframe,
        alert_type: AlertType,
        candle_time: datetime,
        payload: dict | None = None,
        notified: bool = False,
    ) -> bool:
        """True — сигнал новый (его нужно обработать), False — уже был."""
        async with self._lock:
            cur = await self.conn.execute(
                """INSERT OR IGNORE INTO sent_alerts
                   (symbol, timeframe, alert_type, candle_time, payload, notified, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol.upper(),
                    timeframe.value,
                    alert_type.value,
                    _iso(candle_time),
                    json.dumps(payload or {}, default=str),
                    int(notified),
                    _now_iso(),
                ),
            )
            await self.conn.commit()
            return cur.rowcount > 0

    async def mark_notified(
        self, symbol: str, timeframe: Timeframe, alert_type: AlertType, candle_time: datetime
    ) -> None:
        async with self._lock:
            await self.conn.execute(
                """UPDATE sent_alerts SET notified=1
                   WHERE symbol=? AND timeframe=? AND alert_type=? AND candle_time=?""",
                (symbol.upper(), timeframe.value, alert_type.value, _iso(candle_time)),
            )
            await self.conn.commit()

    async def recent_alerts(self, symbol: str, limit: int = 5) -> list[dict]:
        cur = await self.conn.execute(
            """SELECT timeframe, alert_type, candle_time, notified
               FROM sent_alerts WHERE symbol = ? AND notified = 1
               ORDER BY candle_time DESC LIMIT ?""",
            (symbol.upper(), limit),
        )
        return [dict(row) for row in await cur.fetchall()]

    # --- опорные точки ------------------------------------------------------

    async def save_anchor(
        self,
        symbol: str,
        timeframe: Timeframe,
        direction: Direction,
        point: PivotPoint | None,
    ) -> None:
        async with self._lock:
            await self.conn.execute(
                """INSERT INTO anchors(symbol, timeframe, direction, candle_time, price, rsi, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, timeframe, direction) DO UPDATE SET
                     candle_time=excluded.candle_time,
                     price=excluded.price,
                     rsi=excluded.rsi,
                     updated_at=excluded.updated_at""",
                (
                    symbol.upper(),
                    timeframe.value,
                    direction.value,
                    _iso(point.time) if point else None,
                    point.price if point else None,
                    point.rsi if point else None,
                    _now_iso(),
                ),
            )
            await self.conn.commit()

    async def get_anchor(
        self, symbol: str, timeframe: Timeframe, direction: Direction = Direction.BULL
    ) -> dict | None:
        cur = await self.conn.execute(
            """SELECT candle_time, price, rsi FROM anchors
               WHERE symbol=? AND timeframe=? AND direction=?""",
            (symbol.upper(), timeframe.value, direction.value),
        )
        row = await cur.fetchone()
        if row is None or row["candle_time"] is None:
            return None
        return dict(row)

    # --- seeding (первый прогон без спама) ---------------------------------

    async def is_seeded(self, symbol: str, timeframe: Timeframe) -> bool:
        cur = await self.conn.execute(
            "SELECT seeded FROM asset_timeframes WHERE symbol=? AND timeframe=?",
            (symbol.upper(), timeframe.value),
        )
        row = await cur.fetchone()
        return bool(row and row["seeded"])

    async def mark_seeded(self, symbol: str, timeframe: Timeframe) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE asset_timeframes SET seeded=1 WHERE symbol=? AND timeframe=?",
                (symbol.upper(), timeframe.value),
            )
            await self.conn.commit()

    # --- подписчики ---------------------------------------------------------

    async def add_subscriber(self, chat_id: int, title: str | None = None) -> None:
        async with self._lock:
            await self.conn.execute(
                """INSERT INTO subscribers(chat_id, title, added_at) VALUES (?, ?, ?)
                   ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title""",
                (chat_id, title, _now_iso()),
            )
            await self.conn.commit()

    async def remove_subscriber(self, chat_id: int) -> None:
        async with self._lock:
            await self.conn.execute("DELETE FROM subscribers WHERE chat_id = ?", (chat_id,))
            await self.conn.commit()

    async def list_subscribers(self) -> list[int]:
        cur = await self.conn.execute("SELECT chat_id FROM subscribers")
        return [int(row["chat_id"]) for row in await cur.fetchall()]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()
