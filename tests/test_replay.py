"""Тесты офлайн-оценщика сделок (app/analysis/replay.py).

Правила выхода проверяются отдельно в tests/test_trade_rules.py. Здесь —
стык с детектором: откуда берётся вход, откуда стоп, как считается
first_in_series и что попадает в средние.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from app.analysis.replay import (
    REASON_OPEN,
    aggregate,
    bucket_of,
    build_subbars,
    evaluate,
    group_by_sl_distance,
    supports_intrabar,
)
from app.analysis.signals import (
    AlertType,
    Direction,
    DivergenceEngine,
    PivotPoint,
    Signal,
    SignalParams,
)
from app.analysis.trade_rules import TradeRules
from app.core.timeframes import Timeframe
from tests.synthetic import START, build_df, make_prices, triple_df, uptrend_df

PARAMS = SignalParams()
RULES = TradeRules.for_timeframe(Timeframe.H1)


def signals_for(df: pd.DataFrame, params: SignalParams = PARAMS):
    return DivergenceEngine(params).analyze(df, "TESTUSDT", Timeframe.H1).signals


def divergences(df: pd.DataFrame):
    return [s for s in signals_for(df) if s.type in {
        AlertType.BULLISH_DIVERGENCE, AlertType.BEARISH_DIVERGENCE
    }]


def position(df: pd.DataFrame, ts) -> int:
    return int(df.index[df["open_time"] == ts][0])


# --- вход и стоп ------------------------------------------------------------


def test_entry_is_close_of_confirming_candle():
    df = triple_df()
    signal = divergences(df)[0]
    trade = evaluate(df, signals_for(df), rules=RULES)[0]

    assert trade.entry_time == signal.confirmation_time
    assert trade.entry_price == pytest.approx(signal.confirmation_price)


def test_stop_comes_from_the_pivot_candle_not_the_entry_candle():
    """Свеча входа стоит на fractal_n баров позже экстремума, и её минимум
    по построению фрактала выше. Стоп обязан считаться от точки дивергенции."""
    df = triple_df()
    trades = evaluate(df, signals_for(df), rules=RULES)
    signals = divergences(df)

    assert trades
    for trade, signal in zip(trades, signals):
        pivot_low = signal.pivot.price
        entry_bar_low = float(df["low"].iloc[position(df, trade.entry_time)])

        assert trade.stop_price == pytest.approx(pivot_low * (1 - RULES.sl_buffer_pct / 100))
        assert trade.stop_price < trade.entry_price      # для лонга — всегда
        assert trade.stop_price < entry_bar_low          # не от свечи входа


def test_trades_appear_for_every_divergence():
    df = triple_df()
    trades = evaluate(df, signals_for(df), rules=RULES)

    assert [t.degree for t in trades] == [2, 3]
    assert all(t.direction is Direction.BULL for t in trades)


def test_no_signals_gives_no_trades():
    df = uptrend_df()
    assert evaluate(df, signals_for(df), rules=RULES) == []
    assert aggregate([])["count"] == 0


# --- новая дивергенция сделку не закрывает ----------------------------------


def make_signal(df: pd.DataFrame, index: int, direction: Direction, *, drop: float = 5.0):
    """Ручной сигнал: вход на закрытии бара index, экстремум на drop% в стороне."""
    times = pd.to_datetime(df["open_time"], utc=True)
    close = float(df["close"].iloc[index])
    shift = 1 - drop / 100 if direction is Direction.BULL else 1 + drop / 100
    moment = times.iloc[index].to_pydatetime()
    alert = (
        AlertType.BULLISH_DIVERGENCE
        if direction is Direction.BULL
        else AlertType.BEARISH_DIVERGENCE
    )
    return Signal(
        symbol="TESTUSDT",
        timeframe=Timeframe.H1,
        type=alert,
        price=close * shift,
        rsi=25.0,
        candle_time=moment,
        direction=direction,
        pivot=PivotPoint(index=index, time=moment, price=close * shift, rsi=25.0),
        confirmation_time=moment,
        confirmation_price=close,
        degree=2,
    )


def test_next_divergence_does_not_close_the_trade():
    """Обе сделки живут дальше: закрыть может только стоп или tp3."""
    df = build_df(make_prices([(0.02, 40), (-0.02, 40)]))   # ход меньше 1%
    trades = evaluate(
        df,
        [make_signal(df, 10, Direction.BULL), make_signal(df, 40, Direction.BULL)],
        rules=RULES,
    )

    assert [t.exit_reason for t in trades] == [REASON_OPEN, REASON_OPEN]
    assert all(t.result_pct is None for t in trades)


def test_first_in_series_separates_chains_from_independent_entries():
    df = build_df(make_prices([(0.02, 40), (-0.02, 40)]))
    trades = evaluate(
        df,
        [make_signal(df, 10, Direction.BULL), make_signal(df, 40, Direction.BULL)],
        rules=RULES,
    )

    # вторая сделка открыта, пока первая ещё висит → это продолжение цепочки
    assert [t.first_in_series for t in trades] == [True, False]


def test_first_in_series_is_per_direction_and_symbol():
    df = build_df(make_prices([(0.02, 40), (-0.02, 40)]))
    trades = evaluate(
        df,
        [make_signal(df, 10, Direction.BULL), make_signal(df, 40, Direction.BEAR)],
        rules=RULES,
    )

    assert [t.first_in_series for t in trades] == [True, True]


def test_closed_trade_does_not_block_the_next_one():
    """Стоп по первой сделке выбит раньше входа во вторую — вторая независима."""
    df = build_df(make_prices([(-0.5, 30), (0.02, 30)]))
    trades = evaluate(
        df,
        [make_signal(df, 5, Direction.BULL, drop=1.0), make_signal(df, 40, Direction.BULL)],
        rules=RULES,
    )

    assert trades[0].is_closed
    assert trades[0].exit_time < trades[1].entry_time
    assert [t.first_in_series for t in trades] == [True, True]


# --- сводка -----------------------------------------------------------------


def test_aggregate_counts_only_closed():
    df = build_df(make_prices([(-0.5, 30), (0.02, 30)]))
    trades = evaluate(
        df,
        [make_signal(df, 5, Direction.BULL, drop=1.0), make_signal(df, 40, Direction.BULL)],
        rules=RULES,
    )
    stats = aggregate(trades)

    assert stats["count"] == 1
    assert stats["open"] == 1
    assert stats["result"] is not None
    assert stats["win_rate"] in (0.0, 100.0)
    assert stats["reason_counts"]["sl"] + stats["reason_counts"]["be"] + \
        stats["reason_counts"]["tp3"] == stats["count"]


def test_sl_distance_buckets():
    assert bucket_of(1.5) == "до 2%"
    assert bucket_of(2.0) == "2–4%"
    assert bucket_of(6.9) == "4–7%"
    assert bucket_of(12.0) == "больше 7%"

    df = build_df(make_prices([(0.02, 40), (-0.02, 40)]))
    trades = evaluate(df, [make_signal(df, 10, Direction.BULL, drop=5.0)], rules=RULES)
    groups = group_by_sl_distance(trades)

    assert len(groups["4–7%"]) == 1
    assert sum(len(part) for part in groups.values()) == 1


def test_csv_row_has_every_required_field():
    df = triple_df()
    row = evaluate(df, signals_for(df), rules=RULES)[0].as_row()

    required = {
        "symbol", "timeframe", "direction", "degree", "signal_time", "entry_time",
        "entry_price", "stop_price", "sl_dist_pct", "tp1_price", "tp2_price",
        "tp3_price", "tp1_time", "tp2_time", "tp3_time", "exit_time", "exit_reason",
        "result_pct", "r_multiple", "mae_pct", "mfe_pct", "bars_held",
        "first_in_series",
    }
    assert required <= set(row)


# --- часовые подсвечки ------------------------------------------------------


def hourly_frame(hours: int = 96) -> pd.DataFrame:
    rows = []
    for i in range(hours):
        price = 100.0 + i * 0.1
        rows.append(
            {
                "open_time": START + timedelta(hours=i),
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_subbars_group_by_parent_bar():
    index = build_subbars(hourly_frame(), Timeframe.H4)

    group = index.get(START)
    assert len(group) == 4
    assert [b.time for b in group] == [START + timedelta(hours=i) for i in range(4)]
    assert index.get(START + timedelta(hours=4))[0].time == START + timedelta(hours=4)
    assert index.get(START - timedelta(days=5)) is None


def test_no_subbars_outside_supported_timeframes():
    """Разрешение работает для 4H и 1D. 1H его же часами не разложить,
    остальные ТФ в работе не используются — там консервативное правило."""
    assert build_subbars(hourly_frame(), Timeframe.H1) is None
    assert build_subbars(hourly_frame(), Timeframe.H3) is None
    assert build_subbars(hourly_frame(), Timeframe.W1) is None
    assert build_subbars(hourly_frame(), Timeframe.D1) is not None
    assert build_subbars(None, Timeframe.H4) is None
    assert supports_intrabar(Timeframe.H4) and not supports_intrabar(Timeframe.H1)


def test_subbars_change_the_outcome_on_a_conflicting_bar():
    """Один бар задел и tp1, и стоп: часовые данные решают, что было первым."""
    entry_index = 2
    rows = [
        {"open_time": START + timedelta(hours=4 * i), "open": 100.0,
         "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}
        for i in range(6)
    ]
    rows[entry_index + 1] = {
        "open_time": START + timedelta(hours=4 * (entry_index + 1)),
        "open": 100.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 1.0,
    }
    df = pd.DataFrame(rows)
    signal = make_signal(df, entry_index, Direction.BULL)

    conflict_time = df["open_time"].iloc[entry_index + 1].to_pydatetime()
    hours = [
        {"open_time": conflict_time + timedelta(hours=h), "open": 100.0,
         "high": high, "low": low, "close": 100.0, "volume": 1.0}
        for h, (high, low) in enumerate(
            [(106.0, 100.5), (101.0, 100.2), (100.5, 94.0), (100.5, 99.5)]
        )
    ]
    subbars = build_subbars(pd.DataFrame(hours), Timeframe.H4)

    rules = TradeRules.for_timeframe(Timeframe.H4)
    with_hours = evaluate(df, [signal], rules=rules, subbars=subbars)[0]
    without = evaluate(df, [signal], rules=rules)[0]

    assert with_hours.exit_reason == "be"
    assert with_hours.result_pct == pytest.approx(1.5)
    assert without.exit_reason == "sl"
