"""Состояние бота в SQLite: активы, таймфреймы, пороги, опорные точки,
отправленные алерты (дедупликация), подписчики.

Всё переживает рестарт. Ключ дедупликации:
(symbol, timeframe, alert_type, candle_time).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
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

-- Оверрайды порогов. Ключ составной: (key, timeframe).
-- ВАЖНО: маркер глобального оверрайда — пустая строка '', НЕ NULL.
-- В SQLite NULL в составном PRIMARY KEY не конфликтует сам с собой, поэтому
-- с NULL пролезли бы дубли одного и того же глобального ключа.
CREATE TABLE IF NOT EXISTS params (
    key       TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '',
    value     TEXT NOT NULL,
    PRIMARY KEY (key, timeframe)
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

-- candle_time/price/rsi хранят ОПОРНУЮ точку chain[0]; chain_json — цепочку
-- целиком. Старые колонки оставлены, чтобы не ломать прежние обработчики.
CREATE TABLE IF NOT EXISTS anchors (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    direction  TEXT NOT NULL,
    candle_time TEXT,
    price      REAL,
    rsi        REAL,
    chain_json TEXT,
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
        await self._migrate()
        log.info("SQLite готова: %s", self.db_path)

    # --- миграции -----------------------------------------------------------

    async def _has_column(self, table: str, column: str) -> bool:
        cur = await self.conn.execute(f"pragma table_info({table})")
        return any(row["name"] == column for row in await cur.fetchall())

    async def _migrate(self) -> None:
        """Обе миграции идемпотентны: наличие колонки проверяется до запуска."""
        await self._migrate_params()
        await self._migrate_anchors()

    async def _migrate_params(self) -> None:
        """params: key TEXT PRIMARY KEY → составной ключ (key, timeframe).

        В SQLite нельзя добавить колонку в PRIMARY KEY через ALTER TABLE,
        поэтому таблица пересобирается целиком, в одной транзакции.
        Существующие записи становятся глобальными (timeframe = '').

        Заодно переезжает переименованный параметр: max_bars_between раньше
        делал две работы сразу — ограничивал возраст цепочки И расстояние
        в паре. Поэтому его значение раскладывается в ОБА новых параметра,
        иначе прежнее поведение поехало бы, а max > ttl не прошёл бы валидацию.
        """
        if await self._has_column("params", "timeframe"):
            return
        log.info("Миграция params: составной ключ (key, timeframe)")
        await self.conn.executescript(
            """
            BEGIN;
            CREATE TABLE params_new (
                key       TEXT NOT NULL,
                timeframe TEXT NOT NULL DEFAULT '',
                value     TEXT NOT NULL,
                PRIMARY KEY (key, timeframe)
            );
            INSERT INTO params_new(key, timeframe, value)
                SELECT key, '', value FROM params
                WHERE key <> 'max_bars_between';
            INSERT OR IGNORE INTO params_new(key, timeframe, value)
                SELECT 'max_bars_between_points', '', value FROM params
                WHERE key = 'max_bars_between';
            INSERT OR IGNORE INTO params_new(key, timeframe, value)
                SELECT 'anchor_ttl_bars', '', value FROM params
                WHERE key = 'max_bars_between';
            DROP TABLE params;
            ALTER TABLE params_new RENAME TO params;
            COMMIT;
            """
        )
        await self.conn.commit()

    async def _migrate_anchors(self) -> None:
        """anchors: добавление chain_json.

        Ограничение SQLite на ALTER TABLE касается колонок первичного ключа;
        здесь PK не меняется, и ADD COLUMN отрабатывает одной строкой.

        Старые строки содержат ПОСЛЕДНИЕ точки, помеченные как опорные.
        Данные отображательные, в расчёт не читаются и перезапишутся
        при первом же опросе.
        """
        if await self._has_column("anchors", "chain_json"):
            return
        log.info("Миграция anchors: ADD COLUMN chain_json")
        await self.conn.execute("ALTER TABLE anchors ADD COLUMN chain_json TEXT")
        await self.conn.commit()

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

    async def get_param_overrides(self) -> tuple[dict, dict[Timeframe, dict]]:
        """(глобальные оверрайды, оверрайды по таймфреймам).

        Глобальный оверрайд хранится с timeframe = '' — см. комментарий
        к схеме: NULL здесь использовать нельзя.
        """
        cur = await self.conn.execute("SELECT key, timeframe, value FROM params")
        global_over: dict = {}
        by_tf: dict[Timeframe, dict] = {}
        for row in await cur.fetchall():
            try:
                value = json.loads(row["value"])
            except json.JSONDecodeError:
                continue
            raw_tf = row["timeframe"] or ""
            if not raw_tf:
                global_over[row["key"]] = value
                continue
            tf = Timeframe.try_parse(raw_tf)
            if tf is None:
                continue
            by_tf.setdefault(tf, {})[row["key"]] = value
        return global_over, by_tf

    async def set_param(self, key: str, value, timeframe: Timeframe | None = None) -> None:
        async with self._lock:
            await self.conn.execute(
                """INSERT INTO params(key, timeframe, value) VALUES (?, ?, ?)
                   ON CONFLICT(key, timeframe) DO UPDATE SET value=excluded.value""",
                (key, timeframe.value if timeframe else "", json.dumps(value)),
            )
            await self.conn.commit()

    async def reset_param(self, key: str, timeframe: Timeframe | None = None) -> bool:
        """Снимает один оверрайд. True — запись была и удалена."""
        async with self._lock:
            cur = await self.conn.execute(
                "DELETE FROM params WHERE key = ? AND timeframe = ?",
                (key, timeframe.value if timeframe else ""),
            )
            await self.conn.commit()
            return cur.rowcount > 0

    async def clear_params(self, timeframe: Timeframe | None = None) -> None:
        """Без аргумента — сброс всех оверрайдов, включая оверлеи таймфреймов."""
        async with self._lock:
            if timeframe is None:
                await self.conn.execute("DELETE FROM params")
            else:
                await self.conn.execute(
                    "DELETE FROM params WHERE timeframe = ?", (timeframe.value,)
                )
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
        chain: Sequence[PivotPoint] | None,
    ) -> None:
        """Пишет ОПОРНУЮ точку chain[0] в старые колонки и цепочку в chain_json."""
        points = list(chain or ())
        anchor = points[0] if points else None
        async with self._lock:
            await self.conn.execute(
                """INSERT INTO anchors(symbol, timeframe, direction, candle_time,
                                       price, rsi, chain_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, timeframe, direction) DO UPDATE SET
                     candle_time=excluded.candle_time,
                     price=excluded.price,
                     rsi=excluded.rsi,
                     chain_json=excluded.chain_json,
                     updated_at=excluded.updated_at""",
                (
                    symbol.upper(),
                    timeframe.value,
                    direction.value,
                    _iso(anchor.time) if anchor else None,
                    anchor.price if anchor else None,
                    anchor.rsi if anchor else None,
                    json.dumps([p.as_dict() for p in points]) if points else None,
                    _now_iso(),
                ),
            )
            await self.conn.commit()

    async def get_anchor(
        self, symbol: str, timeframe: Timeframe, direction: Direction = Direction.BULL
    ) -> dict | None:
        """Опорная точка + цепочка. Ключ 'chain' — список точек; при пустом
        chain_json (строка из старой БД) остаётся только опорная точка."""
        cur = await self.conn.execute(
            """SELECT candle_time, price, rsi, chain_json FROM anchors
               WHERE symbol=? AND timeframe=? AND direction=?""",
            (symbol.upper(), timeframe.value, direction.value),
        )
        row = await cur.fetchone()
        if row is None or row["candle_time"] is None:
            return None
        data = dict(row)
        raw_chain = data.pop("chain_json", None)
        chain: list[PivotPoint] = []
        if raw_chain:
            try:
                chain = [PivotPoint.from_dict(item) for item in json.loads(raw_chain)]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                log.warning(
                    "%s:%s: битый chain_json, откатываюсь на одну точку",
                    symbol.upper(), timeframe.value,
                )
                chain = []
        data["chain"] = chain
        return data

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
