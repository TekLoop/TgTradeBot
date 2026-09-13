"""Тесты кэша свечей (app/backtest_cache.py).

Сеть не трогается: выкачка подменяется фейковым fetcher'ом, который отдаёт
детерминированный ряд и записывает, о каких диапазонах его спросили.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.backtest_cache import CacheError, CacheSettings, load_klines, read_cache

STEP_MS = 60 * 60_000                     # 1H
NOW = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=5)


def fake_fetcher(calls: list[tuple[int, int]]):
    """Ряд, однозначно восстановимый по времени: цена = номер часа."""

    def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        calls.append((start_ms, end_ms))
        rows = []
        cursor = start_ms - start_ms % STEP_MS
        while cursor < end_ms:
            index = cursor // STEP_MS
            rows.append(
                {
                    "open_time": pd.to_datetime(cursor, unit="ms", utc=True),
                    "open": float(index),
                    "high": float(index) + 1,
                    "low": float(index) - 1,
                    "close": float(index),
                    "volume": 1.0,
                }
            )
            cursor += STEP_MS
        return pd.DataFrame(rows)

    return fetch


def settings_for(tmp_path, **kwargs) -> CacheSettings:
    return CacheSettings(cache_dir=tmp_path / "cache", quiet=True, **kwargs)


def test_first_run_downloads_and_writes_cache(tmp_path):
    calls: list[tuple[int, int]] = []
    settings = settings_for(tmp_path)

    df = load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher(calls), now=NOW,
    )

    path = settings.path_for("BTCUSDT", "1h")
    assert path.exists()
    assert path.name == "binance_BTCUSDT_1h.csv.gz"
    assert len(calls) == 1
    assert len(df) == 5 * 24
    assert df["open_time"].is_monotonic_increasing
    # на диске то же самое, что вернули
    assert len(read_cache(path)) == len(df)


def test_second_run_uses_cache_without_network(tmp_path):
    settings = settings_for(tmp_path)
    first = load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher([]), now=NOW,
    )

    offline = settings_for(tmp_path, offline=True)
    second = load_klines(
        "BTCUSDT", "1h", START, STEP_MS, settings=offline, now=NOW,
    )

    assert len(second) == len(first)
    pd.testing.assert_frame_equal(first, second)


def test_offline_run_is_repeatable(tmp_path):
    settings = settings_for(tmp_path)
    load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher([]), now=NOW,
    )
    offline = settings_for(tmp_path, offline=True)

    runs = [
        load_klines("BTCUSDT", "1h", START, STEP_MS, settings=offline, now=NOW)
        for _ in range(2)
    ]

    pd.testing.assert_frame_equal(runs[0], runs[1])


def test_offline_without_cache_fails_clearly(tmp_path):
    offline = settings_for(tmp_path, offline=True)

    with pytest.raises(CacheError) as exc:
        load_klines("BTCUSDT", "1h", START, STEP_MS, settings=offline, now=NOW)

    assert "BTCUSDT" in str(exc.value)


def test_offline_with_short_history_fails(tmp_path):
    settings = settings_for(tmp_path)
    load_klines(
        "BTCUSDT", "1h", NOW - timedelta(days=1), STEP_MS,
        settings=settings, fetcher=fake_fetcher([]), now=NOW,
    )

    offline = settings_for(tmp_path, offline=True)
    with pytest.raises(CacheError):
        load_klines(
            "BTCUSDT", "1h", NOW - timedelta(days=30), STEP_MS,
            settings=offline, now=NOW,
        )


def test_only_the_missing_tail_is_downloaded(tmp_path):
    settings = settings_for(tmp_path)
    earlier = NOW - timedelta(days=2)
    load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher([]), now=earlier,
    )

    calls: list[tuple[int, int]] = []
    df = load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher(calls), now=NOW,
    )

    assert len(calls) == 1
    requested_from = pd.to_datetime(calls[0][0], unit="ms", utc=True)
    assert requested_from >= pd.Timestamp(earlier) - pd.Timedelta(hours=2)
    assert len(df) == 5 * 24


def test_missing_head_is_downloaded(tmp_path):
    settings = settings_for(tmp_path)
    load_klines(
        "BTCUSDT", "1h", NOW - timedelta(days=2), STEP_MS,
        settings=settings, fetcher=fake_fetcher([]), now=NOW,
    )

    calls: list[tuple[int, int]] = []
    df = load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher(calls), now=NOW,
    )

    assert calls, "начало периода должно было докачаться"
    assert len(df) == 5 * 24
    assert df["open_time"].iloc[0] <= pd.Timestamp(START) + pd.Timedelta(hours=1)
    assert not df["open_time"].duplicated().any()


def test_refresh_redownloads(tmp_path):
    settings = settings_for(tmp_path)
    load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=settings, fetcher=fake_fetcher([]), now=NOW,
    )

    calls: list[tuple[int, int]] = []
    refreshed = settings_for(tmp_path, refresh=True)
    df = load_klines(
        "BTCUSDT", "1h", START, STEP_MS,
        settings=refreshed, fetcher=fake_fetcher(calls), now=NOW,
    )

    assert len(calls) == 1
    assert calls[0][0] == int(START.timestamp() * 1000)
    assert len(df) == 5 * 24
