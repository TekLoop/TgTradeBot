"""Тесты RSI, фракталов и арифметики таймфреймов."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.analysis.indicators import rsi
from app.analysis.pivots import pivot_high_indices, pivot_low_indices
from app.core.timeframes import Timeframe


def test_rsi_warmup_is_nan():
    series = pd.Series([100 + i * 0.5 for i in range(40)])
    values = rsi(series, 14)
    assert values.iloc[:14].isna().all()
    assert not np.isnan(values.iloc[14])


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(100, 160, dtype=float))
    down = pd.Series(np.arange(160, 100, -1, dtype=float))

    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)

    noisy = pd.Series(100 + np.sin(np.arange(80)) * 5)
    values = rsi(noisy, 14).dropna()
    assert values.between(0, 100).all()


def test_rsi_matches_manual_wilder_calculation():
    closes = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
    ]
    series = pd.Series(closes)
    values = rsi(series, 14)

    deltas = np.diff(np.array(closes))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain, avg_loss = gains[:14].mean(), losses[:14].mean()
    expected_14 = 100 - 100 / (1 + avg_gain / avg_loss)
    assert values.iloc[14] == pytest.approx(expected_14, abs=1e-9)

    avg_gain = (avg_gain * 13 + gains[14]) / 14
    avg_loss = (avg_loss * 13 + losses[14]) / 14
    expected_15 = 100 - 100 / (1 + avg_gain / avg_loss)
    assert values.iloc[15] == pytest.approx(expected_15, abs=1e-9)


def test_pivot_low_requires_n_bars_on_both_sides():
    lows = [5, 4, 3, 2, 1, 2, 3, 4, 5]
    assert pivot_low_indices(lows, 2) == [4]
    # без двух баров справа экстремум ещё не подтверждён
    assert pivot_low_indices(lows[:6], 2) == []


def test_pivot_high_is_symmetric():
    highs = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    assert pivot_high_indices(highs, 2) == [4]


def test_pivot_ignores_plateau():
    """Равные минимумы не считаются фракталом (строгое сравнение)."""
    assert pivot_low_indices([5, 4, 1, 1, 4, 5], 2) == []


def test_timeframe_parsing_and_ratios():
    assert Timeframe.parse("3h") is Timeframe.H3
    assert Timeframe.parse("1D") is Timeframe.D1
    assert Timeframe.try_parse("13M") is None

    assert Timeframe.H1.divides(Timeframe.H3)
    assert not Timeframe.H2.divides(Timeframe.H3)
    assert Timeframe.H1.ratio_to(Timeframe.H4) == 4
    assert Timeframe.D1.ratio_to(Timeframe.W1) == 7


def test_timeframe_boundaries_are_utc_aligned():
    now = datetime(2026, 9, 5, 13, 47, tzinfo=timezone.utc)  # суббота

    assert Timeframe.H3.floor(now) == datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
    assert Timeframe.H4.next_close(now) == datetime(2026, 9, 5, 16, tzinfo=timezone.utc)
    assert Timeframe.D1.next_close(now) == datetime(2026, 9, 6, tzinfo=timezone.utc)
    # недельная свеча открывается в понедельник
    assert Timeframe.W1.floor(now) == datetime(2026, 8, 31, tzinfo=timezone.utc)
    assert Timeframe.W1.next_close(now) == datetime(2026, 9, 7, tzinfo=timezone.utc)


def test_timeframe_next_close_on_exact_boundary():
    boundary = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
    assert Timeframe.H4.next_close(boundary) == datetime(
        2026, 9, 5, 16, 0, tzinfo=timezone.utc
    )
