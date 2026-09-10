"""Тесты офлайн-оценщика сделок (app/analysis/replay.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.replay import (
    REASON_OPEN,
    REASON_REVERSE,
    REASON_SAME,
    REASON_TIMEOUT,
    aggregate,
    evaluate,
)
from app.analysis.signals import (
    AlertType,
    Direction,
    DivergenceEngine,
    Signal,
    SignalParams,
)
from app.core.timeframes import Timeframe
from tests.synthetic import build_df, make_prices, triple_df, uptrend_df

PARAMS = SignalParams()


def signals_for(df: pd.DataFrame, params: SignalParams = PARAMS):
    return DivergenceEngine(params).analyze(df, "TESTUSDT", Timeframe.H1).signals


def position(df: pd.DataFrame, ts) -> int:
    return int(df.index[df["open_time"] == ts][0])


def test_chain_produces_two_trades():
    df = triple_df()
    trades = evaluate(df, signals_for(df))

    assert [t.degree for t in trades] == [2, 3]
    first, second = trades
    # ×2 закрыт продлением цепочки, ×3 остался открытым — данные кончились
    assert first.exit_reason == REASON_SAME
    assert first.exit_time == second.entry_time
    assert second.exit_reason == REASON_OPEN
    assert second.result_pct is None


def test_entry_is_close_of_confirming_candle():
    df = triple_df()
    signal = next(s for s in signals_for(df) if s.type is AlertType.BULLISH_DIVERGENCE)
    trade = evaluate(df, signals_for(df))[0]

    assert trade.entry_time == signal.confirmation_time
    assert trade.entry_price == pytest.approx(signal.confirmation_price)


def test_excursions_match_manual_scan():
    """Экстремумы считаются от цены входа по барам после неё."""
    df = triple_df()
    trade = evaluate(df, signals_for(df))[0]

    start = position(df, trade.entry_time) + 1
    stop = position(df, trade.exit_time)
    segment = df.iloc[start : stop + 1]

    assert trade.max_high == pytest.approx(max(segment["high"].max(), trade.entry_price))
    assert trade.min_low == pytest.approx(min(segment["low"].min(), trade.entry_price))
    assert trade.bars_held == stop - start + 1
    # бычья сделка: ход в пользу не отрицателен, ход против не положителен
    assert trade.favorable_pct >= 0 >= trade.adverse_pct


def test_timeout_closes_trade():
    df = triple_df()
    trades = evaluate(df, signals_for(df), timeout_bars=5)

    first = trades[0]
    assert first.exit_reason == REASON_TIMEOUT
    assert first.bars_held == 5
    assert position(df, first.exit_time) - position(df, first.entry_time) == 5


def test_next_signal_wins_over_later_timeout():
    df = triple_df()
    without = evaluate(df, signals_for(df))[0]
    with_timeout = evaluate(df, signals_for(df), timeout_bars=500)[0]

    assert with_timeout.exit_reason == without.exit_reason == REASON_SAME
    assert with_timeout.exit_time == without.exit_time


def test_opposite_direction_closes_as_reverse():
    """Медвежий сигнал закрывает бычью сделку с пометкой «обратная»."""
    df = build_df(make_prices([(1.0, 30), (-1.0, 30)]))
    times = pd.to_datetime(df["open_time"], utc=True)

    def make(index: int, direction: Direction) -> Signal:
        alert = (
            AlertType.BULLISH_DIVERGENCE
            if direction is Direction.BULL
            else AlertType.BEARISH_DIVERGENCE
        )
        return Signal(
            symbol="TESTUSDT",
            timeframe=Timeframe.H1,
            type=alert,
            price=float(df["low"].iloc[index]),
            rsi=25.0,
            candle_time=times.iloc[index].to_pydatetime(),
            direction=direction,
            confirmation_time=times.iloc[index].to_pydatetime(),
            confirmation_price=float(df["close"].iloc[index]),
            degree=2,
        )

    trades = evaluate(df, [make(10, Direction.BULL), make(40, Direction.BEAR)])

    assert trades[0].exit_reason == REASON_REVERSE
    assert trades[0].direction is Direction.BULL
    # цена росла до 30-го бара, значит бычья сделка закрылась в плюс
    assert trades[0].result_pct > 0
    # медвежья сделка открыта на падении: ход в сторону сигнала положителен
    assert trades[1].direction is Direction.BEAR
    assert trades[1].favorable_pct >= 0


def test_no_signals_gives_no_trades():
    df = uptrend_df()
    assert evaluate(df, signals_for(df)) == []
    assert aggregate([])["count"] == 0


def test_aggregate_counts_only_closed():
    df = triple_df()
    stats = aggregate(evaluate(df, signals_for(df)))

    assert stats["count"] == 1
    assert stats["open"] == 1
    assert stats["result"] is not None
    assert stats["win_rate"] in (0.0, 100.0)
