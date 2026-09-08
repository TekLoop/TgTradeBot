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


# --- готовые сценарии -------------------------------------------------------

#: Есть и ALERT #1, и ALERT #2: второй минимум ниже по цене, но выше по RSI.
DIVERGENCE_SEGMENTS = [(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (1.0, 4), (-1.05, 6), (1.0, 5)]

#: Второй минимум ВЫШЕ первого — цена не обновила минимум, дивергенции нет.
NO_DIVERGENCE_SEGMENTS = [(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (1.5, 5), (-1.0, 4), (1.0, 6)]

#: Между минимумами RSI ушёл выше 50 — опорная точка инвалидируется.
INVALIDATION_SEGMENTS = [(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (3.0, 6), (-1.6, 13), (1.0, 5)]

#: Ровный аптренд — сигналов быть не должно.
UPTREND_SEGMENTS = [(0.5, 60)]


def divergence_df() -> pd.DataFrame:
    return build_df(make_prices(DIVERGENCE_SEGMENTS))


def no_divergence_df() -> pd.DataFrame:
    return build_df(make_prices(NO_DIVERGENCE_SEGMENTS))


def invalidation_df() -> pd.DataFrame:
    return build_df(make_prices(INVALIDATION_SEGMENTS))


def uptrend_df() -> pd.DataFrame:
    return build_df(make_prices(UPTREND_SEGMENTS))
