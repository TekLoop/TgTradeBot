"""Тесты ресемплинга OHLCV, в первую очередь 1H → 3H."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.core.timeframes import Timeframe
from app.data.resampling import resample_ohlcv

START = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # понедельник 00:00 UTC


def hourly(count: int, start: datetime = START) -> pd.DataFrame:
    rows = []
    for i in range(count):
        base = 100.0 + i
        rows.append(
            {
                "open_time": start + timedelta(hours=i),
                "open": base,
                "high": base + 2.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": 10.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_resample_1h_to_3h_aggregation():
    df = hourly(9)
    out = resample_ohlcv(df, Timeframe.H3, source=Timeframe.H1)

    assert len(out) == 3
    assert list(out["open_time"]) == [
        pd.Timestamp(START),
        pd.Timestamp(START + timedelta(hours=3)),
        pd.Timestamp(START + timedelta(hours=6)),
    ]

    first = out.iloc[0]
    assert first["open"] == df.iloc[0]["open"]                    # open = first
    assert first["high"] == df.iloc[0:3]["high"].max()            # high = max
    assert first["low"] == df.iloc[0:3]["low"].min()              # low = min
    assert first["close"] == df.iloc[2]["close"]                  # close = last
    assert first["volume"] == pytest.approx(df.iloc[0:3]["volume"].sum())

    last = out.iloc[-1]
    assert last["open"] == df.iloc[6]["open"]
    assert last["close"] == df.iloc[8]["close"]
    assert last["volume"] == pytest.approx(df.iloc[6:9]["volume"].sum())


def test_resample_drops_incomplete_tail_bucket():
    """7 часовых свечей = 2 полных трёхчасовки; хвост из 1 бара отбрасывается,
    чтобы не появилась «недорисованная» свеча."""
    out = resample_ohlcv(hourly(7), Timeframe.H3, source=Timeframe.H1)
    assert len(out) == 2
    assert out["open_time"].iloc[-1] == pd.Timestamp(START + timedelta(hours=3))


def test_resample_buckets_are_aligned_to_utc_midnight():
    """Границы 3H идут от 00:00 UTC, а не от первой свечи в выборке."""
    start = START + timedelta(hours=1)  # начинаем с 01:00
    out = resample_ohlcv(hourly(12, start=start), Timeframe.H3, source=Timeframe.H1)
    for ts in out["open_time"]:
        assert ts.hour % 3 == 0


def test_resample_1h_to_4h_and_1d():
    df = hourly(48)
    h4 = resample_ohlcv(df, Timeframe.H4, source=Timeframe.H1)
    assert len(h4) == 12
    assert all(ts.hour % 4 == 0 for ts in h4["open_time"])

    d1 = resample_ohlcv(df, Timeframe.D1, source=Timeframe.H1)
    assert len(d1) == 2
    assert d1.iloc[0]["high"] == df.iloc[0:24]["high"].max()
    assert d1.iloc[0]["low"] == df.iloc[0:24]["low"].min()


def test_resample_1d_to_1w_starts_on_monday():
    days = []
    for i in range(21):
        base = 100.0 + i
        days.append(
            {
                "open_time": START + timedelta(days=i),
                "open": base,
                "high": base + 2,
                "low": base - 1,
                "close": base + 0.5,
                "volume": 1.0,
            }
        )
    out = resample_ohlcv(pd.DataFrame(days), Timeframe.W1, source=Timeframe.D1)

    assert len(out) == 3
    for ts in out["open_time"]:
        assert ts.weekday() == 0  # понедельник
    assert out.iloc[0]["volume"] == pytest.approx(7.0)


def test_resample_is_idempotent_on_empty_frame():
    empty = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    out = resample_ohlcv(empty, Timeframe.H3, source=Timeframe.H1)
    assert out.empty
