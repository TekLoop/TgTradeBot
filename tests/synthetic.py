"""Генераторы синтетических рядов для тестов."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

START = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def make_prices(segments: list[tuple[float, int]], start: float = 100.0) -> list[float]:
    """segments = [(шаг за бар, количество баров), ...]."""
    prices: list[float] = []
    current = start
    for step, count in segments:
        for _ in range(count):
            current += step
            prices.append(round(current, 4))
    return prices


def build_df(prices: list[float], step: timedelta = timedelta(hours=1)) -> pd.DataFrame:
    """«Линейный» ряд: low = high = close = цена бара, open = цена предыдущего.

    Такой ряд удобен тем, что фрактал по low совпадает с разворотом close,
    а значит RSI и экстремумы считаются по одним и тем же точкам.
    """
    rows = []
    for i, price in enumerate(prices):
        prev = prices[i - 1] if i else price
        rows.append(
            {
                "open_time": START + i * step,
                "open": prev,
                "high": max(prev, price),
                "low": price,
                "close": price,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_bear_df(prices: list[float]) -> pd.DataFrame:
    """То же, но экстремумы — максимумы: high = цена бара."""
    df = build_df(prices)
    df["high"] = df["close"]
    df["low"] = df[["open", "close"]].min(axis=1)
    return df


def mirror(prices: list[float], axis: float = 200.0) -> list[float]:
    return [round(axis - p, 4) for p in prices]


# --- готовые сценарии -------------------------------------------------------

#: Есть и ALERT #1, и ALERT #2: второй минимум ниже по цене, но выше по RSI.
DIVERGENCE_SEGMENTS = [(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (1.0, 4), (-1.05, 6), (1.0, 5)]

#: Тот же ряд, продлённый третьим минимумом: ещё ниже по цене, ещё выше по RSI.
TRIPLE_SEGMENTS = DIVERGENCE_SEGMENTS + [(-0.5, 12), (1.0, 5)]

#: Второй минимум ВЫШЕ первого — цена не обновила минимум, дивергенции нет.
NO_DIVERGENCE_SEGMENTS = [(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (1.5, 5), (-1.0, 4), (1.0, 6)]

#: Между минимумами RSI ушёл выше 50 — опорная точка инвалидируется.
INVALIDATION_SEGMENTS = [(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (3.0, 6), (-1.6, 13), (1.0, 5)]

#: Ровный аптренд — сигналов быть не должно.
UPTREND_SEGMENTS = [(0.5, 60)]


def divergence_df() -> pd.DataFrame:
    return build_df(make_prices(DIVERGENCE_SEGMENTS))


def triple_df() -> pd.DataFrame:
    return build_df(make_prices(TRIPLE_SEGMENTS))


def no_divergence_df() -> pd.DataFrame:
    return build_df(make_prices(NO_DIVERGENCE_SEGMENTS))


def invalidation_df() -> pd.DataFrame:
    return build_df(make_prices(INVALIDATION_SEGMENTS))


def uptrend_df() -> pd.DataFrame:
    return build_df(make_prices(UPTREND_SEGMENTS))


# --- точные сценарии по позициям минимумов ----------------------------------


def lows_scenario(
    lows: list[tuple[int, float]],
    *,
    total: int = 95,
    warmup: int = 25,
    warmup_step: float = 0.5,
    drift: float = -0.05,
    base: float = 100.0,
) -> pd.DataFrame:
    """Ряд с минимумами заданной глубины в заданных позициях.

    Сначала warmup баров роста — он нужен, чтобы RSI не залипал на нуле
    (в чистом падении avg_gain = 0 и RSI ровно 0 во всех точках). Дальше
    пологий дрейф, на который накладываются провалы из lows.

    lows = [(индекс бара, насколько ниже линии дрейфа), ...]. Позиции должны
    отстоять друг от друга минимум на fractal_n + 1 баров, иначе соседние
    провалы съедят фракталы друг друга.
    """
    prices: list[float] = []
    current = base
    for _ in range(warmup):
        current += warmup_step
        prices.append(round(current, 6))
    for _ in range(total - warmup):
        current += drift
        prices.append(round(current, 6))
    for index, depth in lows:
        if not 0 <= index < total:
            raise ValueError(f"минимум вне ряда: {index}")
        prices[index] = round(prices[index] - depth, 6)
    return build_df(prices)


# --- свечи с фитилями -------------------------------------------------------


def wick_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Ряд из явных OHLC: (open, high, low, close) на каждый бар."""
    data = []
    for i, (open_, high, low, close) in enumerate(rows):
        data.append(
            {
                "open_time": START + i * timedelta(hours=1),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(data)


def flat_with_candle(
    total: int,
    index: int,
    candle: tuple[float, float, float, float],
    base: float = 100.0,
) -> pd.DataFrame:
    """Ровный пилообразный ряд, в позиции index — свеча с заданным OHLC."""
    rows: list[tuple[float, float, float, float]] = []
    for i in range(total):
        if i == index:
            rows.append(candle)
            continue
        open_ = base
        close = base + (0.05 if i % 2 else -0.05)
        rows.append((open_, max(open_, close), min(open_, close), close))
    return wick_df(rows)


def rsi_dive_df(total: int = 60, drop: float = 2.0, tail: int = 0) -> pd.DataFrame:
    """Затяжное падение: RSI надолго уезжает в крайнюю зону.

    tail — сколько баров отскока добавить в конец (для проверки перезарядки
    по RSI, а не по барам).
    """
    body = total - 10 - tail
    segments: list[tuple[float, int]] = [(0.2, 10), (-drop, body)]
    if tail:
        segments.append((drop * 1.5, tail))
    return build_df(make_prices(segments))
