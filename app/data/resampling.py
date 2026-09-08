"""Ресемплинг OHLCV в старший таймфрейм средствами pandas.

Агрегация: open=first, high=max, low=min, close=last, volume=sum.
Границы выравниваются так же, как считает биржа: часовые/дневные — от epoch
(UTC-полночь), недельные — от понедельника.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.timeframes import Timeframe

log = logging.getLogger(__name__)

AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит DataFrame к каноничному виду: колонки, типы, UTC, сортировка."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В OHLCV нет колонок: {missing}")
    out = df.loc[:, OHLCV_COLUMNS].copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset="open_time", keep="last")
    out = out.sort_values("open_time").reset_index(drop=True)
    return out


def resample_ohlcv(
    df: pd.DataFrame,
    target: Timeframe,
    source: Timeframe | None = None,
    *,
    drop_incomplete_tail: bool = True,
) -> pd.DataFrame:
    """Собирает свечи target из более мелких свечей source.

    Последний бакет отбрасывается, если исходных данных не хватает,
    чтобы полностью его закрыть (иначе получилась бы «недорисованная» свеча).
    """
    df = normalize_ohlcv(df)
    if df.empty:
        return df

    indexed = df.set_index("open_time")
    if target.minutes < Timeframe.D1.minutes:
        # Только tick-подобные частоты (h/min/s) принимают origin.
        resampler = indexed.resample(
            target.pandas_rule, origin="epoch", closed="left", label="left"
        )
    else:
        # 1D и W-MON и так выравниваются по UTC-полуночи / понедельнику.
        resampler = indexed.resample(target.pandas_rule, closed="left", label="left")

    agg = resampler.agg(AGGREGATION).dropna(subset=["open", "high", "low", "close"])
    agg = agg.reset_index().rename(columns={"index": "open_time"})

    if drop_incomplete_tail and not agg.empty:
        source_step = source.duration if source else _infer_step(df)
        if source_step is not None:
            last_source_close = df["open_time"].iloc[-1] + source_step
            last_bucket_close = agg["open_time"].iloc[-1] + target.duration
            if last_source_close < last_bucket_close:
                agg = agg.iloc[:-1]

    return agg.reset_index(drop=True)


def _infer_step(df: pd.DataFrame) -> pd.Timedelta | None:
    if len(df) < 2:
        return None
    diffs = df["open_time"].diff().dropna()
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]
