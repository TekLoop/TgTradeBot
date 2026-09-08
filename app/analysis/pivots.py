"""Фракталы: локальные экстремумы, подтверждённые N свечами слева и справа."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _extremum_indices(values: Sequence[float], n: int, *, is_low: bool) -> list[int]:
    if n < 1:
        raise ValueError("fractal_n должен быть >= 1")
    arr = np.asarray(values, dtype=float)
    size = arr.size
    result: list[int] = []
    # i пробегает только те бары, у которых есть N соседей слева и справа,
    # т.е. фрактал автоматически «подтверждён».
    for i in range(n, size - n):
        pivot = arr[i]
        if np.isnan(pivot):
            continue
        window_left = arr[i - n : i]
        window_right = arr[i + 1 : i + 1 + n]
        if is_low:
            ok = np.all(pivot < window_left) and np.all(pivot < window_right)
        else:
            ok = np.all(pivot > window_left) and np.all(pivot > window_right)
        if ok:
            result.append(i)
    return result


def pivot_low_indices(lows: Sequence[float], n: int = 2) -> list[int]:
    """Индексы подтверждённых локальных минимумов (строгое сравнение с соседями)."""
    return _extremum_indices(lows, n, is_low=True)


def pivot_high_indices(highs: Sequence[float], n: int = 2) -> list[int]:
    """Индексы подтверждённых локальных максимумов."""
    return _extremum_indices(highs, n, is_low=False)
