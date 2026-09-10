"""Тесты независимых детекторов: extreme-алерты и фитили."""

from __future__ import annotations

from app.analysis.signals import AlertType, DivergenceEngine, SignalParams
from app.core.timeframes import Timeframe
from tests.synthetic import flat_with_candle, rsi_dive_df

BASE = dict(extreme_alerts_enabled=True, bearish_enabled=False)


def analyze(df, **kw):
    return DivergenceEngine(SignalParams(**kw)).analyze(df, "TESTUSDT", Timeframe.H1)


def of_type(result, alert_type):
    return [s for s in result.signals if s.type is alert_type]


# --- Задача 4: extreme ------------------------------------------------------


def test_extreme_is_off_by_default():
    result = analyze(rsi_dive_df())
    assert not of_type(result, AlertType.EXTREME_OVERSOLD)


def test_extreme_fires_in_zone():
    result = analyze(rsi_dive_df(), **BASE)
    fired = of_type(result, AlertType.EXTREME_OVERSOLD)
    assert fired
    assert all(s.rsi <= SignalParams().extreme_oversold for s in fired)


def test_extreme_candle_is_its_own_confirmation():
    """confirm_idx = i, иначе фильтр notify_max_age_bars сработает случайно."""
    df = rsi_dive_df()
    result = analyze(df, **BASE)
    last_idx = len(df) - 1
    for signal in of_type(result, AlertType.EXTREME_OVERSOLD):
        assert signal.confirmation_time == signal.candle_time
        assert signal.confirmation_price == signal.price
        assert signal.bars_since_confirmation == last_idx - signal.pivot.index


def test_extreme_rearms_by_bars():
    """Затяжной экстремум даёт напоминания раз в extreme_rearm_bars баров."""
    df = rsi_dive_df(total=70, drop=3.0)
    tight = of_type(analyze(df, extreme_rearm_bars=3, **BASE),
                    AlertType.EXTREME_OVERSOLD)
    loose = of_type(analyze(df, extreme_rearm_bars=20, **BASE),
                    AlertType.EXTREME_OVERSOLD)
    assert len(tight) > len(loose)

    gaps = [
        b.pivot.index - a.pivot.index for a, b in zip(tight, tight[1:])
    ]
    assert gaps and min(gaps) >= 3


def test_extreme_rearms_by_rsi_only():
    """extreme_rearm_bars=0 → перезарядка только по возврату RSI."""
    df = rsi_dive_df(total=80, drop=3.0, tail=25)
    fired = of_type(
        analyze(df, extreme_rearm_bars=0, **BASE), AlertType.EXTREME_OVERSOLD
    )
    assert len(fired) == 1  # один заход в зону — один алерт


def test_extreme_bear_ignores_bearish_enabled():
    """У extreme свой выключатель: bearish_enabled его не касается."""
    prices = rsi_dive_df(total=70, drop=3.0)
    mirrored = prices.copy()
    for column in ("open", "high", "low", "close"):
        mirrored[column] = 300.0 - prices[column]
    mirrored["high"], mirrored["low"] = (
        mirrored[["open", "close"]].max(axis=1),
        mirrored[["open", "close"]].min(axis=1),
    )
    result = analyze(mirrored, extreme_alerts_enabled=True, bearish_enabled=False)
    assert of_type(result, AlertType.EXTREME_OVERBOUGHT)


def test_extreme_not_in_outcomes():
    from app.analysis.signals import DIVERGENCE_TYPES

    assert AlertType.EXTREME_OVERSOLD not in DIVERGENCE_TYPES
    assert AlertType.EXTREME_OVERBOUGHT not in DIVERGENCE_TYPES


# --- Задача 5: фитили -------------------------------------------------------

WICK = dict(wick_alerts_enabled=True)


def test_wick_is_off_by_default():
    df = flat_with_candle(60, 40, (100.0, 120.0, 99.0, 101.0))
    assert not analyze(df).signals


def test_upper_wick_detected():
    # тело 100→101, верхний фитиль до 110: (110-101)/101*100 ≈ 8.9% > 5%
    df = flat_with_candle(60, 40, (100.0, 110.0, 99.9, 101.0))
    fired = of_type(analyze(df, **WICK), AlertType.WICK_UPPER)
    assert len(fired) == 1
    assert fired[0].wick_pct > 5.0
    assert fired[0].ohlc == (100.0, 110.0, 99.9, 101.0)


def test_lower_wick_detected():
    df = flat_with_candle(60, 40, (100.0, 101.1, 90.0, 101.0))
    fired = of_type(analyze(df, **WICK), AlertType.WICK_LOWER)
    assert len(fired) == 1
    assert fired[0].wick_pct > 5.0


def test_both_wicks_give_two_separate_signals():
    df = flat_with_candle(60, 40, (100.0, 112.0, 90.0, 101.0))
    result = analyze(df, **WICK)
    assert len(of_type(result, AlertType.WICK_UPPER)) == 1
    assert len(of_type(result, AlertType.WICK_LOWER)) == 1


def test_doji_filter_skips_candle():
    """Свеча с почти нулевым телом — это размах, а не отбой."""
    doji = (100.0, 112.0, 90.0, 100.05)
    df = flat_with_candle(60, 40, doji)
    assert analyze(df, **WICK).signals            # без фильтра алерты есть
    assert not analyze(df, wick_min_body_pct=20.0, **WICK).signals


def test_degenerate_candle_is_skipped():
    """high == low: деления на ноль быть не должно."""
    df = flat_with_candle(60, 40, (100.0, 100.0, 100.0, 100.0))
    assert not analyze(df, **WICK).signals


def test_wick_candle_is_its_own_confirmation():
    df = flat_with_candle(60, 40, (100.0, 110.0, 99.9, 101.0))
    signal = of_type(analyze(df, **WICK), AlertType.WICK_UPPER)[0]
    assert signal.confirmation_time == signal.candle_time
    assert signal.confirmation_price == signal.price
    assert signal.bars_since_confirmation == len(df) - 1 - signal.pivot.index


def test_wick_denominator_is_body_edge_not_close():
    """Знаменатель — граница тела. Проверяем формулу числом."""
    df = flat_with_candle(60, 40, (100.0, 110.0, 99.9, 101.0))
    signal = of_type(analyze(df, **WICK), AlertType.WICK_UPPER)[0]
    expected = (110.0 - 101.0) / 101.0 * 100.0
    assert abs(signal.wick_pct - expected) < 1e-9


def test_detectors_do_not_touch_chains():
    """Включение детекторов не меняет ни одного сигнала дивергенции."""
    from tests.synthetic import divergence_df

    df = divergence_df()
    plain = analyze(df)
    loud = analyze(df, extreme_alerts_enabled=True, wick_alerts_enabled=True)

    def divergence_part(result):
        return [
            (s.type, s.candle_time, s.degree)
            for s in result.signals
            if s.type not in {
                AlertType.EXTREME_OVERSOLD, AlertType.EXTREME_OVERBOUGHT,
                AlertType.WICK_UPPER, AlertType.WICK_LOWER,
            }
        ]

    assert divergence_part(plain) == divergence_part(loud)
    assert plain.bull_chain == loud.bull_chain
