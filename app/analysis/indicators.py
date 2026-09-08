"""Индикаторы. Чистые функции над pandas/numpy, никаких внешних зависимостей."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Классический RSI по Уайлдеру (RMA-сглаживание, seed = SMA первых `period` дельт).

    Первые `period` значений — NaN (прогрев). Значения считаются по close.
    """
    if period < 2:
        raise ValueError("rsi_period должен быть >= 2")

    values = np.asarray(close, dtype=float)
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if n <= period:
        return pd.Series(out, index=close.index, name="rsi")

    deltas = np.diff(values)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)

    return pd.Series(out, index=close.index, name="rsi")
