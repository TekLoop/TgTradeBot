"""Тесты детектора сигналов на синтетических рядах."""

from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.signals import AlertType, DivergenceEngine, SignalParams
from app.core.timeframes import Timeframe
from tests.synthetic import (
    build_df,
    divergence_df,
    invalidation_df,
    make_prices,
    no_divergence_df,
    uptrend_df,
)

PARAMS = SignalParams()


def analyze(df: pd.DataFrame, params: SignalParams = PARAMS):
    return DivergenceEngine(params).analyze(df, "TESTUSDT", Timeframe.H1)


def types_of(result) -> list[AlertType]:
    return [s.type for s in result.signals]


def test_divergence_is_detected():
    """Ряд, где дивергенция заведомо ЕСТЬ: цена обновила минимум, RSI — нет."""
    result = analyze(divergence_df())

    assert AlertType.OVERSOLD_PIVOT in types_of(result)
    divergences = [s for s in result.signals if s.type is AlertType.BULLISH_DIVERGENCE]
    assert len(divergences) == 1

    signal = divergences[0]
    assert signal.reference is not None
    assert signal.price < signal.reference.price      # цена ниже
    assert signal.rsi > signal.reference.rsi          # RSI выше
    assert signal.rsi <= PARAMS.divergence_rsi_max    # всё ещё в нижней зоне
    assert signal.reference.rsi <= PARAMS.oversold    # опорная точка перепродана
    assert signal.price_delta < 0
    assert signal.rsi_delta > 0
    # после ALERT #2 опорная точка сброшена
    assert result.bull_anchor is None


def test_alert1_precedes_alert2_and_points_match():
    result = analyze(divergence_df())
    alert1 = next(s for s in result.signals if s.type is AlertType.OVERSOLD_PIVOT)
    alert2 = next(s for s in result.signals if s.type is AlertType.BULLISH_DIVERGENCE)

    assert alert1.candle_time < alert2.candle_time
    assert alert2.reference is not None
    assert alert2.reference.time == alert1.candle_time
    assert alert2.reference.price == pytest.approx(alert1.price)


def test_no_divergence_when_second_low_is_higher():
    """Ряд, где дивергенции заведомо НЕТ: второй минимум выше первого."""
    result = analyze(no_divergence_df())

    assert AlertType.OVERSOLD_PIVOT in types_of(result)
    assert AlertType.BULLISH_DIVERGENCE not in types_of(result)


def test_no_signals_on_clean_uptrend():
    result = analyze(uptrend_df())
    assert result.signals == []
    assert result.bull_anchor is None


def test_anchor_invalidated_when_rsi_crosses_reset_level():
    """RSI ушёл выше RESET_RSI между минимумами → дивергенция отменяется,
    хотя цена и обновила минимум."""
    df = invalidation_df()
    result = analyze(df)

    assert AlertType.BULLISH_DIVERGENCE not in types_of(result)
    # второй минимум сам по себе перепродан → он становится новой опорной точкой
    alerts1 = [s for s in result.signals if s.type is AlertType.OVERSOLD_PIVOT]
    assert len(alerts1) == 2
    assert alerts1[1].price < alerts1[0].price

    # если поднять порог сброса выше достигнутого RSI — дивергенция появляется
    relaxed = SignalParams(reset_rsi=95.0)
    assert AlertType.BULLISH_DIVERGENCE in types_of(analyze(df, relaxed))


def test_max_bars_between_limits_divergence():
    """Та же дивергенция не засчитывается, если точки слишком далеко друг от друга."""
    df = divergence_df()
    assert AlertType.BULLISH_DIVERGENCE in types_of(analyze(df, SignalParams()))

    tight = SignalParams(max_bars_between=3)
    assert AlertType.BULLISH_DIVERGENCE not in types_of(analyze(df, tight))


def test_divergence_rsi_max_gate():
    """Если второй минимум вышел из нижней зоны — дивергенция не засчитывается."""
    df = divergence_df()
    strict = SignalParams(divergence_rsi_max=15.0)
    assert AlertType.BULLISH_DIVERGENCE not in types_of(analyze(df, strict))


def test_unclosed_candle_is_not_analyzed_by_caller_contract():
    """Движок считает по тем барам, что ему дали: добавление новой свечи
    не должно менять уже выданные сигналы (нет перерисовки)."""
    df = divergence_df()
    base = analyze(df)
    extended = analyze(pd.concat([df, df.tail(1)], ignore_index=True).iloc[:-1])

    assert [(s.type, s.candle_time) for s in base.signals] == [
        (s.type, s.candle_time) for s in extended.signals
    ]


def test_bearish_mirror_logic():
    """Медвежий сценарий — зеркальное отражение бычьего."""
    bullish = make_prices([(0.4, 8), (-0.4, 8), (0.4, 8), (-2.0, 10), (1.0, 4), (-1.05, 6), (1.0, 5)])
    mirrored = [200.0 - p for p in bullish]
    df = build_df(mirrored)
    # для зеркального ряда экстремумы — это максимумы, поэтому high = цена бара
    df["high"] = df["close"]
    df["low"] = df[["open", "close"]].min(axis=1)

    off = analyze(df, SignalParams(bearish_enabled=False))
    assert AlertType.BEARISH_DIVERGENCE not in types_of(off)

    on = analyze(df, SignalParams(bearish_enabled=True))
    divergences = [s for s in on.signals if s.type is AlertType.BEARISH_DIVERGENCE]
    assert len(divergences) == 1
    signal = divergences[0]
    assert signal.reference is not None
    assert signal.price > signal.reference.price   # цена обновила максимум
    assert signal.rsi < signal.reference.rsi       # RSI максимум не обновил


def test_short_series_returns_nothing():
    df = build_df(make_prices([(1.0, 5)]))
    result = analyze(df)
    assert result.signals == []
    assert result.bars == 5


def test_fractal_confirmation_delay():
    """Экстремум подтверждается только через N баров — раньше сигнала нет."""
    df = divergence_df()
    full = analyze(df)
    last_divergence = next(
        s for s in full.signals if s.type is AlertType.BULLISH_DIVERGENCE
    )
    pivot_pos = df.index[df["open_time"] == last_divergence.candle_time][0]

    # обрезаем ряд так, что справа от экстремума остаётся N-1 баров
    truncated = df.iloc[: pivot_pos + PARAMS.fractal_n]
    assert AlertType.BULLISH_DIVERGENCE not in types_of(analyze(truncated))

    # добавляем недостающий бар — сигнал появляется
    confirmed = df.iloc[: pivot_pos + PARAMS.fractal_n + 1]
    assert AlertType.BULLISH_DIVERGENCE in types_of(analyze(confirmed))
