"""Тесты правил выхода: лесенка тейков со стопом (app/analysis/trade_rules.py).

Сделки здесь собираются вручную из баров, без детектора: проверяются именно
правила выхода, а не то, где нашлась дивергенция.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.analysis.signals import Direction
from app.analysis.trade_rules import (
    REASON_BE,
    REASON_OPEN,
    REASON_SL,
    REASON_TP3,
    Bar,
    TradeEntry,
    TradeRules,
    default_tp_levels,
    simulate_trade,
)
from app.core.timeframes import Timeframe
from tests.synthetic import START

ENTRY_PRICE = 100.0
#: Экстремум дивергенции: с буфером 1% стоп встаёт на 94.05, то есть −5.95%.
PIVOT_PRICE = 95.0

H4 = TradeRules.for_timeframe(Timeframe.H4)      # 5 / 10 / 20
D1 = TradeRules.for_timeframe(Timeframe.D1)      # 10 / 20 / 50


def bar(hours: float, high: float, low: float) -> Bar:
    return Bar(time=START + timedelta(hours=hours), high=high, low=low)


def long_entry() -> TradeEntry:
    return TradeEntry(time=START, price=ENTRY_PRICE, pivot_price=PIVOT_PRICE)


def short_entry() -> TradeEntry:
    """Зеркало: экстремум выше входа ровно настолько же."""
    return TradeEntry(time=START, price=ENTRY_PRICE, pivot_price=105.0)


def run(bars, rules=H4, direction=Direction.BULL, entry=None, subbars=None):
    return simulate_trade(
        entry or (long_entry() if direction is Direction.BULL else short_entry()),
        direction,
        bars,
        rules,
        subbars=subbars,
    )


# --- базовые исходы ---------------------------------------------------------


def test_stop_on_first_bar_loses_exactly_the_stop_distance():
    result = run([bar(1, 101.0, 90.0)])

    assert result.exit_reason == REASON_SL
    assert result.result_pct == pytest.approx(-result.sl_dist_pct)
    assert result.bars_held == 1
    assert result.exit_time == START + timedelta(hours=1)


def test_tp1_then_return_to_entry_is_breakeven():
    """30% зафиксированы на +5%, остаток закрыт по входу: +1.5%."""
    result = run([bar(1, 106.0, 101.0), bar(2, 102.0, 99.0)])

    assert result.exit_reason == REASON_BE
    assert result.result_pct == pytest.approx(1.5)
    assert [fill.tag for fill in result.fills] == ["tp1", "be"]


def test_full_ladder_on_4h():
    result = run([bar(1, 106.0, 101.0), bar(2, 111.0, 105.0), bar(3, 121.0, 110.0)])

    assert result.exit_reason == REASON_TP3
    assert result.result_pct == pytest.approx(0.3 * 5 + 0.3 * 10 + 0.4 * 20)
    assert result.result_pct == pytest.approx(12.5)
    assert [fill.tag for fill in result.fills] == ["tp1", "tp2", "tp3"]


def test_full_ladder_on_1d():
    result = run(
        [bar(1, 111.0, 101.0), bar(2, 121.0, 112.0), bar(3, 151.0, 125.0)],
        rules=D1,
    )

    assert result.exit_reason == REASON_TP3
    assert result.result_pct == pytest.approx(0.3 * 10 + 0.3 * 20 + 0.4 * 50)
    assert result.result_pct == pytest.approx(29.0)


def test_short_mirrors_the_same_numbers():
    """Шорт: минимум → максимум, ниже → выше, рост цены → убыток."""
    bars = [bar(1, 99.0, 94.0), bar(2, 95.0, 89.0), bar(3, 90.0, 79.0)]
    result = run(bars, direction=Direction.BEAR)

    assert result.exit_reason == REASON_TP3
    assert result.result_pct == pytest.approx(12.5)

    stopped = run([bar(1, 110.0, 99.0)], direction=Direction.BEAR)
    assert stopped.exit_reason == REASON_SL
    assert stopped.result_pct == pytest.approx(-stopped.sl_dist_pct)


def test_r_multiple_is_result_over_stop_distance():
    result = run([bar(1, 106.0, 101.0), bar(2, 102.0, 99.0)])

    assert result.sl_dist_pct == pytest.approx(5.95)
    assert result.r_multiple == pytest.approx(1.5 / 5.95)


# --- стоп -------------------------------------------------------------------


def test_stop_from_pivot_not_from_entry_candle():
    """sl_set=low считает стоп от свечи экстремума и всегда ниже входа."""
    result = run([bar(1, 101.0, 100.5)])

    assert result.stop_price == pytest.approx(PIVOT_PRICE * 0.99)
    assert result.stop_price < ENTRY_PRICE
    assert result.sl_dist_pct == pytest.approx((100.0 - 94.05) / 100.0 * 100.0)


def test_stop_from_entry_when_sl_set_is_enter():
    rules = TradeRules.for_timeframe(Timeframe.H4, {"sl_set": "enter", "sl_percent": 3.0})
    result = run([bar(1, 101.0, 100.5)], rules=rules)

    assert result.stop_price == pytest.approx(97.0)
    assert result.sl_dist_pct == pytest.approx(3.0)
    # цена экстремума при sl_set=enter не участвует вовсе
    assert rules.stop_price(100.0, None, Direction.BULL) == pytest.approx(97.0)


def test_stop_above_entry_is_rejected():
    """Страховка от подсунутых руками данных: для лонга стоп обязан быть ниже."""
    broken = TradeEntry(time=START, price=100.0, pivot_price=120.0)
    with pytest.raises(ValueError):
        run([bar(1, 121.0, 99.0)], entry=broken)


# --- уровни по таймфреймам --------------------------------------------------


def test_defaults_come_from_the_timeframe_table():
    assert default_tp_levels(Timeframe.H4) == (5.0, 10.0, 20.0)
    assert default_tp_levels(Timeframe.D1) == (10.0, 20.0, 50.0)
    # для остальных таймфреймов дефолт — как у 4H
    assert default_tp_levels(Timeframe.H1) == (5.0, 10.0, 20.0)
    assert D1.tp_levels == (10.0, 20.0, 50.0)


def test_param_overrides_the_table():
    rules = TradeRules.for_timeframe(Timeframe.D1, {"tp1_pct": 7.5})

    assert rules.tp1_pct == 7.5
    assert (rules.tp2_pct, rules.tp3_pct) == (20.0, 50.0)


def test_sizes_and_validation():
    rules = TradeRules.for_timeframe(Timeframe.H4, {"tp1_size": 40.0, "tp2_size": 25.0})
    assert rules.tp3_size == pytest.approx(35.0)

    with pytest.raises(ValueError):
        TradeRules.for_timeframe(Timeframe.H4, {"tp1_size": 60.0, "tp2_size": 45.0})
    with pytest.raises(ValueError):
        TradeRules.for_timeframe(Timeframe.H4, {"tp1_pct": 25.0})  # tp1 выше tp2
    with pytest.raises(ValueError):
        TradeRules.for_timeframe(Timeframe.H4, {"sl_set": "middle"})
    with pytest.raises(ValueError):
        TradeRules.for_timeframe(Timeframe.H4, {"unknown_key": 1})


def test_custom_sizes_change_the_weighted_result():
    rules = TradeRules.for_timeframe(Timeframe.H4, {"tp1_size": 50.0})
    result = run([bar(1, 106.0, 101.0), bar(2, 102.0, 99.0)], rules=rules)

    assert result.result_pct == pytest.approx(0.5 * 5)


# --- порядок внутри бара ----------------------------------------------------


CONFLICT_BAR = bar(4, 106.0, 94.0)   # задет и tp1 (105), и стоп (94.05)

#: Часовые свечи того же бара: сначала тейк, потом возврат к входу.
HOURLY_TP_FIRST = [
    bar(4, 106.0, 100.5),
    bar(5, 101.0, 100.2),
    bar(6, 100.5, 94.0),
]

#: Те же часы, но провал случился раньше тейка.
HOURLY_STOP_FIRST = [
    bar(4, 101.0, 94.0),
    bar(5, 106.0, 99.0),
    bar(6, 106.0, 100.0),
]


def test_intrabar_hourly_shows_take_first():
    subbars = {CONFLICT_BAR.time: HOURLY_TP_FIRST}
    result = run([CONFLICT_BAR], subbars=subbars)

    # стопом сделка не закрыта: тейк сработал раньше, дальше только безубыток
    assert result.exit_reason == REASON_BE
    assert result.result_pct == pytest.approx(1.5)
    assert result.tp1_time == HOURLY_TP_FIRST[0].time


def test_intrabar_chain_inside_one_bar():
    """tp1 на первом часе, безубыток на третьем — цепочка внутри одного бара."""
    subbars = {CONFLICT_BAR.time: HOURLY_TP_FIRST}
    result = run([CONFLICT_BAR], subbars=subbars)

    assert [fill.tag for fill in result.fills] == ["tp1", "be"]
    assert result.fills[0].time == HOURLY_TP_FIRST[0].time
    assert result.fills[1].time == HOURLY_TP_FIRST[2].time
    assert result.bars_held == 1


def test_intrabar_hourly_shows_stop_first():
    subbars = {CONFLICT_BAR.time: HOURLY_STOP_FIRST}
    result = run([CONFLICT_BAR], subbars=subbars)

    assert result.exit_reason == REASON_SL
    assert result.result_pct == pytest.approx(-result.sl_dist_pct)


def test_intrabar_off_is_conservative():
    rules = TradeRules.for_timeframe(Timeframe.H4, {"intrabar_resolution": "off"})
    subbars = {CONFLICT_BAR.time: HOURLY_TP_FIRST}
    result = run([CONFLICT_BAR], rules=rules, subbars=subbars)

    assert result.exit_reason == REASON_SL
    assert result.result_pct == pytest.approx(-result.sl_dist_pct)


def test_missing_hourly_data_falls_back_to_conservative():
    """Часов на этот бар нет — работает правило «стоп первым»."""
    result = run([CONFLICT_BAR], subbars={})

    assert result.exit_reason == REASON_SL


def test_both_touched_inside_one_hour_is_conservative():
    subbars = {CONFLICT_BAR.time: [bar(4, 106.0, 94.0)]}
    result = run([CONFLICT_BAR], subbars=subbars)

    assert result.exit_reason == REASON_SL


# --- незакрытые сделки ------------------------------------------------------


def test_new_divergence_does_not_close_the_trade():
    """Закрыть могут только стоп или tp3: данные кончились — сделка открыта."""
    result = run([bar(1, 101.0, 99.0), bar(2, 102.0, 98.0)])

    assert result.exit_reason == REASON_OPEN
    assert result.result_pct is None
    assert result.r_multiple is None
    assert result.bars_held == 2


def test_open_trade_keeps_realized_part_visible():
    result = run([bar(1, 106.0, 101.0), bar(2, 107.0, 102.0)])

    assert result.exit_reason == REASON_OPEN
    assert result.result_pct is None
    assert result.realized_pct == pytest.approx(1.5)
    assert result.tp1_time is not None


def test_excursions_are_signed():
    result = run([bar(1, 104.0, 97.0), bar(2, 103.0, 96.0)])

    assert result.mfe_pct == pytest.approx(4.0)
    assert result.mae_pct == pytest.approx(-4.0)
