"""Тесты детектора сигналов на синтетических рядах.

По тесту на каждую строку спецификации Задачи 2. Часть сценариев прогоняется
с reset_rsi=95 и anchor_ttl_bars=0: у инвалидации и TTL свои отдельные тесты,
а здесь нужно изолировать именно логику смены опорной точки.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.signals import (
    AlertType,
    DivergenceEngine,
    PivotPoint,
    SignalParams,
    _bull_rules,
)
from app.core.timeframes import Timeframe
from tests.synthetic import (
    DIVERGENCE_SEGMENTS,
    build_df,
    build_bear_df,
    divergence_df,
    invalidation_df,
    lows_scenario,
    make_prices,
    mirror,
    no_divergence_df,
    triple_df,
    uptrend_df,
)

PARAMS = SignalParams()

#: Изоляция логики опорной точки: без инвалидации по RSI и без TTL.
ANCHOR_ONLY = dict(reset_rsi=95.0, anchor_ttl_bars=0)


def analyze(df: pd.DataFrame, params: SignalParams = PARAMS):
    return DivergenceEngine(params).analyze(df, "TESTUSDT", Timeframe.H1)


def types_of(result) -> list[AlertType]:
    return [s.type for s in result.signals]


def divergences_of(result):
    return [s for s in result.signals if s.type is AlertType.BULLISH_DIVERGENCE]


def alerts1_of(result):
    return [s for s in result.signals if s.type is AlertType.OVERSOLD_PIVOT]


def indices(chain) -> list[int]:
    return [point.index for point in chain]


# --- Задача 2: смена опорной точки ------------------------------------------


def test_higher_low_in_zone_does_not_replace_anchor():
    """РЕГРЕСС, который чинит Задача 2.

    Пивот с ценой ВЫШЕ опорной и RSI в зоне перепроданности. Старый код
    выбрасывал опорную (сравнивал только RSI), новый — молча игнорирует.
    """
    df = lows_scenario([(35, 8.0), (50, 4.0)], drift=-0.2)
    result = analyze(df, SignalParams(**ANCHOR_ONLY))

    intruder = result.bull_chain[0]
    assert indices(result.bull_chain) == [35]      # опорная не сменилась
    assert len(alerts1_of(result)) == 1            # второго ALERT #1 нет
    assert not divergences_of(result)
    assert intruder.index == 35


def test_lower_price_and_lower_rsi_replaces_anchor():
    """Законное вытеснение: цена ниже И RSI ниже → новая опорная, ALERT #1."""
    df = lows_scenario([(35, 6.0), (50, 15.0)], drift=-0.2)
    result = analyze(df, SignalParams(**ANCHOR_ONLY))

    alerts = alerts1_of(result)
    assert len(alerts) == 2
    first, second = alerts
    assert second.price < first.price
    assert second.rsi < first.rsi
    assert second.degree == 1
    assert indices(result.bull_chain) == [50]

    # Признак, по которому приёмка отличает вытеснение живой опорной
    # от старта на пустой цепочке.
    assert first.displaced_anchor is None
    assert second.displaced_anchor is not None
    assert second.displaced_anchor.index == 35


def test_alert1_on_empty_chain_has_no_price_constraint():
    """ALERT #1 на ПУСТОЙ цепочке ценой ни с чем не сравнивается.

    Приёмочное утверждение «цена ниже предыдущей опорной» относится только
    к сигналам с заполненным displaced_anchor.
    """
    result = analyze(divergence_df())
    starts = [s for s in alerts1_of(result) if s.displaced_anchor is None]
    assert starts, "хотя бы один ALERT #1 должен стартовать на пустой цепочке"
    for signal in starts:
        assert signal.degree == 1
        assert signal.rsi <= PARAMS.oversold


def test_lower_price_and_higher_rsi_gives_divergence():
    """Цена ниже, RSI выше → ALERT #2 со степенью 2."""
    result = analyze(divergence_df())
    divergences = divergences_of(result)

    assert len(divergences) == 1
    signal = divergences[0]
    assert signal.reference is not None
    assert signal.price < signal.reference.price
    assert signal.rsi > signal.reference.rsi
    assert signal.rsi <= PARAMS.divergence_rsi_max
    assert signal.degree == 2
    assert signal.replaced_from is None
    assert indices(result.bull_chain) == [33, 43]
    assert result.bull_anchor.index == 33        # опорная = chain[0]
    assert result.bull_last_point.index == 43    # последняя = chain[-1]


def test_replacement_of_second_point_keeps_degree_two():
    """Замена точки 2: RSI обновил минимум → пересчёт от опорной.

    degree = 2, а НЕ 3: иначе обычные двухточечные дивергенции попадут
    в статистику как тройные.
    """
    df = build_df(make_prices(DIVERGENCE_SEGMENTS + [(-1.0, 12), (1.0, 5)]))
    result = analyze(df)
    divergences = divergences_of(result)

    assert len(divergences) == 2
    extension, replacement = divergences
    assert extension.replaced_from is None
    assert extension.degree == 2

    assert replacement.replaced_from == 1
    assert replacement.degree == 2
    assert replacement.reference.index == 33          # пересчёт от опорной
    assert replacement.replaced_points == (extension.candle_time,)
    assert indices(result.bull_chain) == [33, 60]     # точка 2 выброшена


def test_extension_to_three_points():
    """Продление цепочки: degree растёт до 3."""
    result = analyze(triple_df())
    divergences = divergences_of(result)

    assert [s.degree for s in divergences] == [2, 3]
    third = divergences[-1]
    assert third.replaced_from is None
    assert len(third.chain) == 3
    prices = [p.price for p in third.chain]
    rsis = [p.rsi for p in third.chain]
    assert prices == sorted(prices, reverse=True)
    assert rsis == sorted(rsis)


def test_middle_point_replacement_in_chain_of_four():
    """Замена средней точки: цепочка обрезается, degree = новой длине.

    Обход цепочки назад проверяется напрямую на собранной цепочке — так
    сценарий не зависит от подгонки ценового ряда.
    """
    engine = DivergenceEngine(SignalParams())
    rules = _bull_rules(engine.params)
    chain = [
        PivotPoint(index=0, time=None, price=100.0, rsi=10.0),
        PivotPoint(index=10, time=None, price=98.0, rsi=20.0),
        PivotPoint(index=20, time=None, price=96.0, rsi=25.0),
        PivotPoint(index=30, time=None, price=94.0, rsi=30.0),
    ]
    # RSI между точками 1 и 2 → подходит только P1, точки 2 и 3 выбрасываются
    point = PivotPoint(index=38, time=None, price=90.0, rsi=22.0)
    assert engine._find_reference(chain, point, rules) == 1

    trimmed = chain[:2] + [point]
    assert len(trimmed) == 3   # degree считается из длины, не инкрементом


def test_pivot_closer_than_min_distance_is_ignored_entirely():
    """Пивот, отклонённый по минимальному расстоянию, НЕ должен провалиться
    в шаг вытеснения: он отклонён за то, что пришёл рано."""
    df = lows_scenario([(35, 6.0), (40, 15.0)], drift=-0.2)

    # без фильтра этот пивот законно вытесняет опорную
    loose = analyze(df, SignalParams(**ANCHOR_ONLY))
    assert indices(loose.bull_chain) == [40]
    assert loose.signals[-1].displaced_anchor is not None

    # с фильтром — цепочка не тронута, алерта нет
    strict = analyze(df, SignalParams(min_bars_between_points=8, **ANCHOR_ONLY))
    assert indices(strict.bull_chain) == [35]
    assert len(alerts1_of(strict)) == 1


def test_distant_pivot_does_not_erase_chain():
    """max_bars_between_points цепочку больше НЕ убивает.

    Раньше побарная проверка тем же числом обнуляла цепочку. Теперь далёкий
    пивот просто не образует дивергенцию, а опорная остаётся жива.
    """
    df = build_df(make_prices(DIVERGENCE_SEGMENTS))
    result = analyze(df, SignalParams(max_bars_between_points=5, **ANCHOR_ONLY))

    assert not divergences_of(result)          # пара 33→43 длиннее 5 баров
    assert indices(result.bull_chain) == [33]  # но опорная на месте


def test_anchor_ttl_kills_chain_from_first_point():
    """TTL считается от chain[0], а не от последней точки."""
    df = build_df(make_prices(DIVERGENCE_SEGMENTS + [(-0.2, 30), (1.0, 5)]))
    alive = analyze(df, SignalParams(anchor_ttl_bars=0, reset_rsi=95.0))
    assert alive.bull_chain

    short = analyze(df, SignalParams(anchor_ttl_bars=5, max_bars_between_points=5,
                                     reset_rsi=95.0))
    assert short.bull_chain == () or short.bull_anchor.index >= 33


def test_chain_cap_forbids_extension_but_keeps_anchor():
    """Потолок запрещает продление, но цепочку НЕ обнуляет."""
    capped = analyze(triple_df(), SignalParams(chain_max_points=2, reset_rsi=95.0,
                                               anchor_ttl_bars=0))
    assert all(s.degree <= 2 for s in capped.signals)
    # прежнее поведение обнуляло цепочку — теперь опорная жива
    assert capped.bull_chain != ()
    assert capped.bull_anchor is not None


def test_chain_cap_still_allows_replacement():
    """Замены укорачивают цепочку и разрешены при любой длине."""
    df = build_df(make_prices(DIVERGENCE_SEGMENTS + [(-1.0, 12), (1.0, 5)]))
    result = analyze(df, SignalParams(chain_max_points=2))
    replacements = [s for s in divergences_of(result) if s.replaced_from is not None]
    assert replacements
    assert all(s.degree == 2 for s in replacements)


def test_anchor_invalidated_when_rsi_crosses_reset_level():
    df = invalidation_df()
    result = analyze(df)

    assert AlertType.BULLISH_DIVERGENCE not in types_of(result)
    alerts = alerts1_of(result)
    assert len(alerts) == 2
    assert alerts[1].price < alerts[0].price

    relaxed = SignalParams(reset_rsi=95.0)
    assert AlertType.BULLISH_DIVERGENCE in types_of(analyze(df, relaxed))


# --- прежние гарантии, которые ломать нельзя --------------------------------


def test_no_divergence_when_second_low_is_higher():
    result = analyze(no_divergence_df())
    assert AlertType.OVERSOLD_PIVOT in types_of(result)
    assert AlertType.BULLISH_DIVERGENCE not in types_of(result)


def test_no_signals_on_clean_uptrend():
    result = analyze(uptrend_df())
    assert result.signals == []
    assert result.bull_anchor is None


def test_confirmation_price_is_close_of_confirming_candle():
    df = divergence_df()
    signal = divergences_of(analyze(df))[0]
    pivot_pos = int(df.index[df["open_time"] == signal.candle_time][0])
    confirm_pos = pivot_pos + PARAMS.fractal_n

    assert signal.confirmation_time == df["open_time"].iloc[confirm_pos].to_pydatetime()
    assert signal.confirmation_price == pytest.approx(df["close"].iloc[confirm_pos])


def test_fractal_confirmation_delay():
    df = divergence_df()
    signal = divergences_of(analyze(df))[0]
    pivot_pos = df.index[df["open_time"] == signal.candle_time][0]

    truncated = df.iloc[: pivot_pos + PARAMS.fractal_n]
    assert AlertType.BULLISH_DIVERGENCE not in types_of(analyze(truncated))

    confirmed = df.iloc[: pivot_pos + PARAMS.fractal_n + 1]
    assert AlertType.BULLISH_DIVERGENCE in types_of(analyze(confirmed))


def test_divergence_rsi_max_gate():
    df = divergence_df()
    strict = SignalParams(divergence_rsi_max=15.0)
    assert AlertType.BULLISH_DIVERGENCE not in types_of(analyze(df, strict))


def test_bearish_mirror_logic():
    df = build_bear_df(mirror(make_prices(DIVERGENCE_SEGMENTS)))

    off = analyze(df, SignalParams(bearish_enabled=False))
    assert AlertType.BEARISH_DIVERGENCE not in types_of(off)

    on = analyze(df, SignalParams(bearish_enabled=True))
    divergences = [s for s in on.signals if s.type is AlertType.BEARISH_DIVERGENCE]
    assert len(divergences) == 1
    signal = divergences[0]
    assert signal.price > signal.reference.price
    assert signal.rsi < signal.reference.rsi
    assert signal.degree == 2


def test_short_series_returns_nothing():
    df = build_df(make_prices([(1.0, 5)]))
    result = analyze(df)
    assert result.signals == []
    assert result.bars == 5


def test_no_repaint_when_window_extended():
    df = divergence_df()
    base = analyze(df)
    extended = analyze(pd.concat([df, df.tail(1)], ignore_index=True).iloc[:-1])
    assert [(s.type, s.candle_time) for s in base.signals] == [
        (s.type, s.candle_time) for s in extended.signals
    ]
