"""Независимые детекторы: extreme-алерты (Задача 4) и фитили (Задача 5).

Оба работают по одной свече, состояния цепочек не имеют и повлиять на логику
опорных точек не могут — именно поэтому они вынесены отдельными классами,
а не ветками внутри DivergenceEngine._scan().

Контракт детектора: detect(window, symbol, timeframe) -> list[Signal].
Никакого хранимого состояния между вызовами: перезарядка вычисляется в рамках
той же переигровки окна и в БД не попадает.

Зависимости идут только внутрь: numpy/pandas и app.analysis.* / app.core.*.
Ничего из app/bot/, app/data/, app/config.py импортировать нельзя.
"""

from __future__ import annotations

import numpy as np

from app.analysis.signals import (
    AlertType,
    Direction,
    PivotPoint,
    Signal,
    SignalParams,
    Window,
)
from app.core.timeframes import Timeframe


def _instant_signal(
    window: Window,
    symbol: str,
    timeframe: Timeframe,
    alert_type: AlertType,
    direction: Direction,
    index: int,
    rsi_value: float,
    *,
    wick_pct: float | None = None,
    ohlc: tuple[float, float, float, float] | None = None,
) -> Signal:
    """Сигнал, для которого свеча сама себе подтверждение.

    confirm_idx = i, поэтому фильтр notify_max_age_bars в service.py работает
    для этих типов так же, как для дивергенций: свежий сигнал уходит,
    протухший — нет.
    """
    price = float(window.closes[index])
    moment = window.times[index]
    point = PivotPoint(index=index, time=moment, price=price, rsi=rsi_value)
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        type=alert_type,
        price=price,
        rsi=rsi_value,
        candle_time=moment,
        direction=direction,
        reference=None,
        pivot=point,
        confirmation_time=moment,
        confirmation_price=price,
        bars_since_confirmation=max(0, window.last_idx - index),
        degree=1,
        chain=(),
        wick_pct=wick_pct,
        ohlc=ohlc,
    )


class ExtremeDetector:
    """RSI в крайней зоне: алерт на той же свече, без ожидания фрактала.

    Перезарядка — логическое ИЛИ: RSI вернулся за порог перезарядки ИЛИ
    прошло extreme_rearm_bars баров. При затяжном сидении RSI на экстремуме
    напоминания приходят каждые extreme_rearm_bars баров — это осознанное
    поведение, а не баг.

    Медвежий алерт НЕ подчиняется bearish_enabled: у функции свой выключатель.
    """

    def __init__(self, params: SignalParams) -> None:
        self.params = params

    def detect(self, window: Window, symbol: str, timeframe: Timeframe) -> list[Signal]:
        p = self.params
        if not p.extreme_alerts_enabled:
            return []
        signals: list[Signal] = []
        signals += self._scan(
            window, symbol, timeframe,
            direction=Direction.BULL,
            alert_type=AlertType.EXTREME_OVERSOLD,
            triggered=lambda r: r <= p.extreme_oversold,
            rearmed=lambda r: r > p.extreme_rearm_rsi_bull,
        )
        signals += self._scan(
            window, symbol, timeframe,
            direction=Direction.BEAR,
            alert_type=AlertType.EXTREME_OVERBOUGHT,
            triggered=lambda r: r >= p.extreme_overbought,
            rearmed=lambda r: r < p.extreme_rearm_rsi_bear,
        )
        return signals

    def _scan(
        self,
        window: Window,
        symbol: str,
        timeframe: Timeframe,
        *,
        direction: Direction,
        alert_type: AlertType,
        triggered,
        rearmed,
    ) -> list[Signal]:
        p = self.params
        signals: list[Signal] = []
        armed = True
        fired_at: int | None = None

        for i in range(len(window)):
            r = window.rsi[i]
            if np.isnan(r):
                continue  # прогрев RSI
            value = float(r)

            if not armed:
                by_rsi = rearmed(value)
                by_bars = (
                    p.extreme_rearm_bars > 0
                    and fired_at is not None
                    and i - fired_at >= p.extreme_rearm_bars
                )
                if by_rsi or by_bars:
                    armed = True

            if armed and triggered(value):
                signals.append(
                    _instant_signal(
                        window, symbol, timeframe, alert_type, direction, i, value
                    )
                )
                armed = False
                fired_at = i

        return signals


class WickDetector:
    """Длинный фитиль свечи. Знаменатель — граница тела, а не close.

        верхний % = (high - max(open, close)) / max(open, close) * 100
        нижний  % = (min(open, close) - low)  / min(open, close) * 100

    Формулы зафиксированы: менять их нельзя без пересчёта всего, что на них
    построено. Оба направления под одним выключателем wick_alerts_enabled,
    верхний фитиль НЕ подчиняется bearish_enabled.

    Свечи, у которых RSI ещё не посчитан (прогрев), пропускаются: Signal.rsi
    не должен быть NaN, а первые rsi_period баров окна всё равно не интересны.
    """

    def __init__(self, params: SignalParams) -> None:
        self.params = params

    def detect(self, window: Window, symbol: str, timeframe: Timeframe) -> list[Signal]:
        p = self.params
        if not p.wick_alerts_enabled:
            return []

        df = window.df
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)

        signals: list[Signal] = []
        for i in range(len(window)):
            r = window.rsi[i]
            if np.isnan(r):
                continue
            value = float(r)

            open_, high, low, close = (
                float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
            )
            span = high - low
            if span <= 0.0:
                continue  # вырожденная свеча: деления на ноль быть не должно

            # Фильтр доджи: свеча с почти нулевым телом даёт большие проценты
            # с обеих сторон, но это не отбой, а просто размах.
            if p.wick_min_body_pct > 0.0:
                body_pct = abs(close - open_) / span * 100.0
                if body_pct < p.wick_min_body_pct:
                    continue

            body_high = max(open_, close)
            body_low = min(open_, close)
            ohlc = (open_, high, low, close)

            if body_high > 0.0:
                upper_pct = (high - body_high) / body_high * 100.0
                if upper_pct > p.wick_threshold_pct:
                    signals.append(
                        _instant_signal(
                            window, symbol, timeframe,
                            AlertType.WICK_UPPER, Direction.BEAR, i, value,
                            wick_pct=upper_pct, ohlc=ohlc,
                        )
                    )

            if body_low > 0.0:
                lower_pct = (body_low - low) / body_low * 100.0
                if lower_pct > p.wick_threshold_pct:
                    # Если превышены оба порога — два отдельных сообщения,
                    # по одному на направление. Объединённого алерта нет.
                    signals.append(
                        _instant_signal(
                            window, symbol, timeframe,
                            AlertType.WICK_LOWER, Direction.BULL, i, value,
                            wick_pct=lower_pct, ohlc=ohlc,
                        )
                    )

        return signals
