"""Миграции БД и точечный откат параметров в рантайме."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from app.analysis.signals import Direction, PivotPoint
from app.bot.storage import Storage
from app.config import ParamsConfig
from app.core.timeframes import Timeframe
from tests.synthetic import START


def run(coro):
    return asyncio.run(coro)


def legacy_db(path) -> None:
    """Схема до миграций: params с одиночным ключом, anchors без chain_json."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE assets (symbol TEXT PRIMARY KEY, provider TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
        CREATE TABLE asset_timeframes (symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1, seeded INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (symbol, timeframe));
        CREATE TABLE params (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sent_alerts (symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            alert_type TEXT NOT NULL, candle_time TEXT NOT NULL, payload TEXT,
            notified INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            PRIMARY KEY (symbol, timeframe, alert_type, candle_time));
        CREATE TABLE anchors (symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            direction TEXT NOT NULL, candle_time TEXT, price REAL, rsi REAL,
            updated_at TEXT NOT NULL, PRIMARY KEY (symbol, timeframe, direction));
        CREATE TABLE subscribers (chat_id INTEGER PRIMARY KEY, title TEXT,
            added_at TEXT NOT NULL);
        """
    )
    connection.execute("INSERT INTO params VALUES ('oversold', '28')")
    connection.execute("INSERT INTO params VALUES ('max_bars_between', '60')")
    connection.execute(
        "INSERT INTO anchors VALUES ('BTCUSDT','1D','bull',"
        "'2026-01-01T00:00:00+00:00',100.0,25.0,'now')"
    )
    connection.commit()
    connection.close()


async def opened(path) -> Storage:
    storage = Storage(path)
    await storage.connect()
    return storage


# --- миграция params --------------------------------------------------------


def test_params_migration_preserves_records_as_global(tmp_path):
    path = tmp_path / "state.db"
    legacy_db(path)

    async def scenario():
        storage = await opened(path)
        global_over, by_tf = await storage.get_param_overrides()
        await storage.close()
        return global_over, by_tf

    global_over, by_tf = run(scenario())
    assert global_over["oversold"] == 28
    assert by_tf == {}


def test_old_max_bars_between_lands_in_both_new_params(tmp_path):
    """Старый параметр делал две работы сразу, поэтому раскладывается в оба:
    иначе max > ttl не прошёл бы валидацию и настройка молча потерялась бы."""
    path = tmp_path / "state.db"
    legacy_db(path)

    async def scenario():
        storage = await opened(path)
        global_over, _ = await storage.get_param_overrides()
        await storage.close()
        return global_over

    global_over = run(scenario())
    assert global_over["max_bars_between_points"] == 60
    assert global_over["anchor_ttl_bars"] == 60
    assert "max_bars_between" not in global_over


def test_params_migration_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    legacy_db(path)

    async def scenario():
        for _ in range(3):
            storage = await opened(path)
            await storage.close()
        storage = await opened(path)
        global_over, _ = await storage.get_param_overrides()
        await storage.close()
        return global_over

    assert run(scenario())["oversold"] == 28


def test_global_key_cannot_duplicate(tmp_path):
    """Маркер глобального оверрайда — пустая строка, а не NULL: с NULL
    в составном ключе пролезли бы дубли одного и того же ключа."""
    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        await storage.set_param("oversold", 28)
        await storage.set_param("oversold", 26)
        await storage.set_param("oversold", 24, Timeframe.D1)
        global_over, by_tf = await storage.get_param_overrides()
        await storage.close()
        return global_over, by_tf

    global_over, by_tf = run(scenario())
    assert global_over == {"oversold": 26}          # перезапись, не дубль
    assert by_tf[Timeframe.D1] == {"oversold": 24}


def test_reset_param_removes_only_requested_scope(tmp_path):
    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        await storage.set_param("oversold", 26)
        await storage.set_param("oversold", 24, Timeframe.D1)
        removed = await storage.reset_param("oversold", Timeframe.D1)
        global_over, by_tf = await storage.get_param_overrides()
        await storage.close()
        return removed, global_over, by_tf

    removed, global_over, by_tf = run(scenario())
    assert removed is True
    assert global_over == {"oversold": 26}
    assert by_tf.get(Timeframe.D1, {}) == {}


# --- миграция anchors -------------------------------------------------------


def test_anchors_migration_adds_column_and_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    legacy_db(path)

    async def scenario():
        for _ in range(2):
            storage = await opened(path)
            await storage.close()
        storage = await opened(path)
        anchor = await storage.get_anchor("BTCUSDT", Timeframe.D1, Direction.BULL)
        await storage.close()
        return anchor

    anchor = run(scenario())
    assert anchor is not None
    assert anchor["price"] == 100.0
    assert anchor["chain"] == []   # старая строка: откат на одну точку


def test_chain_round_trip(tmp_path):
    path = tmp_path / "state.db"
    chain = (
        PivotPoint(index=10, time=START, price=100.0, rsi=25.0),
        PivotPoint(index=20, time=START, price=96.0, rsi=30.0),
    )

    async def scenario():
        storage = await opened(path)
        await storage.save_anchor("BTCUSDT", Timeframe.D1, Direction.BULL, chain)
        anchor = await storage.get_anchor("BTCUSDT", Timeframe.D1, Direction.BULL)
        await storage.close()
        return anchor

    anchor = run(scenario())
    assert anchor["price"] == 100.0            # старые колонки хранят chain[0]
    assert len(anchor["chain"]) == 2
    assert anchor["chain"][-1].price == 96.0


def test_empty_chain_clears_anchor(tmp_path):
    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        await storage.save_anchor("BTCUSDT", Timeframe.D1, Direction.BULL, ())
        anchor = await storage.get_anchor("BTCUSDT", Timeframe.D1, Direction.BULL)
        await storage.close()
        return anchor

    assert run(scenario()) is None


def test_broken_chain_json_falls_back_without_error(tmp_path):
    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        await storage.save_anchor(
            "BTCUSDT", Timeframe.D1, Direction.BULL,
            (PivotPoint(index=1, time=START, price=100.0, rsi=25.0),),
        )
        await storage.conn.execute("UPDATE anchors SET chain_json = '{{{'")
        await storage.conn.commit()
        anchor = await storage.get_anchor("BTCUSDT", Timeframe.D1, Direction.BULL)
        await storage.close()
        return anchor

    anchor = run(scenario())
    assert anchor["chain"] == []
    assert anchor["price"] == 100.0


# --- точечный откат в рантайме ----------------------------------------------


def test_broken_override_of_one_timeframe_does_not_affect_others(tmp_path):
    from app.service import TrackingService

    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        service = TrackingService(
            storage, {}, ParamsConfig(anchor_ttl_bars=40),
            params_by_timeframe={Timeframe.D1: {"anchor_ttl_bars": 60,
                                                "max_bars_between_points": 60}},
        )
        # заведомо невалидно: max > ttl для 4H
        await storage.set_param("max_bars_between_points", 999, Timeframe.H4)
        broken = await service.effective_params(Timeframe.H4)
        healthy = await service.effective_params(Timeframe.D1)
        await storage.close()
        return broken, healthy

    broken, healthy = run(scenario())
    # битый ТФ откатился на базу + YAML, но не утащил за собой остальные
    assert broken.max_bars_between_points == 40
    assert healthy.anchor_ttl_bars == 60
    assert healthy.max_bars_between_points == 60


def test_invalid_value_is_rejected_and_not_written(tmp_path):
    from app.service import TrackingService

    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        service = TrackingService(storage, {}, ParamsConfig(anchor_ttl_bars=40))
        problem = await service.validate_override(
            "max_bars_between_points", 999, Timeframe.D1
        )
        _, by_tf = await storage.get_param_overrides()
        await storage.close()
        return problem, by_tf

    problem, by_tf = run(scenario())
    assert problem is not None
    assert "max_bars_between_points" in problem
    assert by_tf == {}          # в БД не записалось ничего


def test_global_change_is_checked_against_every_timeframe(tmp_path):
    from app.service import TrackingService

    path = tmp_path / "state.db"

    async def scenario():
        storage = await opened(path)
        service = TrackingService(
            storage, {}, ParamsConfig(anchor_ttl_bars=40, max_bars_between_points=40),
            params_by_timeframe={Timeframe.D1: {"anchor_ttl_bars": 60,
                                                "max_bars_between_points": 60}},
        )
        # глобальный ttl=30 несовместим с max=60, который стоит у 1D
        problem = await service.validate_override("anchor_ttl_bars", 30, None)
        await storage.close()
        return problem

    problem = run(scenario())
    assert problem is not None
    assert "1D" in problem
