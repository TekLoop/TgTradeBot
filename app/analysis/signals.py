"""Детекция сигналов.

Слой ничего не знает ни про Telegram, ни про API провайдеров.
Вход — DataFrame с закрытыми свечами, выход — список Signal.

Движок работает в режиме «переигровки» (replay): на каждом запуске он заново
прогоняет всё окно баров через конечный автомат. Это даёт три полезных свойства:
  * состояние восстанавливается после рестарта и после пропущенных опросов;
  * результат детерминирован и легко тестируется (никакого скрытого состояния);
  * повторная отправка исключается на уровне БД по ключу
    (symbol, timeframe, alert_type, candle_time).

Опорная точка (anchor) всё равно сохраняется в БД — но только для /status
и для наглядности, логика от неё не зависит.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

from app.analysis.indicators import rsi as rsi_indicator
from app.analysis.pivots import pivot_high_indices, pivot_low_indices
from app.core.timeframes import Timeframe


class AlertType(str, Enum):
    OVERSOLD_PIVOT = "oversold_pivot"          # ALERT #1 (бычий сценарий)
    BULLISH_DIVERGENCE = "bullish_divergence"  # ALERT #2 (бычий сценарий)
    OVERBOUGHT_PIVOT = "overbought_pivot"      # ALERT #1 (медвежий сценарий)
    BEARISH_DIVERGENCE = "bearish_divergence"  # ALERT #2 (медвежий сценарий)


class Direction(str, Enum):
    BULL = "bull"
    BEAR = "bear"


@dataclass(frozen=True, slots=True)
class SignalParams:
    """Пороги детектора. Отдельный от pydantic-конфига объект,
    чтобы analysis/ оставался зависимым только от pandas/numpy."""

    rsi_period: int = 14
    fractal_n: int = 2
    oversold: float = 30.0
    overbought: float = 70.0
    divergence_rsi_max: float = 35.0   # верхняя граница RSI для бычьей дивергенции
    divergence_rsi_min: float = 65.0   # нижняя граница RSI для медвежьей дивергенции
    reset_rsi: float = 50.0            # инвалидация бычьей опорной точки (RSI выше)
    reset_rsi_bear: float = 50.0       # инвалидация медвежьей опорной точки (RSI ниже)
    max_bars_between: int = 40
    bearish_enabled: bool = False


@dataclass(frozen=True, slots=True)
class PivotPoint:
    index: int
    time: datetime          # open_time свечи-экстремума
    price: float
    rsi: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "time": self.time.isoformat(),
            "price": self.price,
            "rsi": self.rsi,
        }


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    timeframe: Timeframe
    type: AlertType
    price: float
    rsi: float
    candle_time: datetime               # open_time свечи-экстремума = ключ дедупликации
    direction: Direction = Direction.BULL
    reference: PivotPoint | None = None  # первая точка (для ALERT #2)
    pivot: PivotPoint | None = None      # вторая (текущая) точка
    confirmation_time: datetime | None = None  # open_time свечи, подтвердившей фрактал
    bars_since_confirmation: int = 0     # 0 = подтверждено на последней закрытой свече

    @property
    def price_delta(self) -> float | None:
        if self.reference is None or self.pivot is None:
            return None
        return self.pivot.price - self.reference.price

    @property
    def price_delta_pct(self) -> float | None:
        if self.reference is None or self.pivot is None or self.reference.price == 0:
            return None
        return (self.pivot.price / self.reference.price - 1.0) * 100.0

    @property
    def rsi_delta(self) -> float | None:
        if self.reference is None or self.pivot is None:
            return None
        return self.pivot.rsi - self.reference.rsi


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    symbol: str
    timeframe: Timeframe
    signals: list[Signal] = field(default_factory=list)
    bull_anchor: PivotPoint | None = None
    bear_anchor: PivotPoint | None = None
    last_price: float | None = None
    last_rsi: float | None = None
    last_candle_time: datetime | None = None
    bars: int = 0


@dataclass(frozen=True, slots=True)
class _Rules:
    """Набор компараторов, задающий бычий или медвежий сценарий."""

    direction: Direction
    alert1: AlertType
    alert2: AlertType
    price_source: str                       # 'low' | 'high'
    pivots: Callable[[np.ndarray, int], list[int]]
    is_extreme_rsi: Callable[[float], bool]  # условие ALERT #1
    price_broke: Callable[[float, float], bool]
    rsi_diverged: Callable[[float, float], bool]
    rsi_in_zone: Callable[[float], bool]     # зона для ALERT #2
    is_reset: Callable[[float], bool]        # инвалидация опорной точки


def _bull_rules(p: SignalParams) -> _Rules:
    return _Rules(
        direction=Direction.BULL,
        alert1=AlertType.OVERSOLD_PIVOT,
        alert2=AlertType.BULLISH_DIVERGENCE,
        price_source="low",
        pivots=pivot_low_indices,
        is_extreme_rsi=lambda r: r <= p.oversold,
        price_broke=lambda new, old: new < old,      # цена обновила минимум
        rsi_diverged=lambda new, old: new > old,     # RSI минимум НЕ обновил
        rsi_in_zone=lambda r: r <= p.divergence_rsi_max,
        is_reset=lambda r: r > p.reset_rsi,
    )


def _bear_rules(p: SignalParams) -> _Rules:
    return _Rules(
        direction=Direction.BEAR,
        alert1=AlertType.OVERBOUGHT_PIVOT,
        alert2=AlertType.BEARISH_DIVERGENCE,
        price_source="high",
        pivots=pivot_high_indices,
        is_extreme_rsi=lambda r: r >= p.overbought,
        price_broke=lambda new, old: new > old,      # цена обновила максимум
        rsi_diverged=lambda new, old: new < old,     # RSI максимум НЕ обновил
        rsi_in_zone=lambda r: r >= p.divergence_rsi_min,
        is_reset=lambda r: r < p.reset_rsi_bear,
    )


REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


class DivergenceEngine:
    """Конечный автомат ALERT #1 → ALERT #2 с инвалидацией опорной точки."""

    def __init__(self, params: SignalParams) -> None:
        self.params = params

    # --- публичный API ------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> AnalysisResult:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"В DataFrame нет колонок: {missing}")

        p = self.params
        if len(df) < p.rsi_period + 2 * p.fractal_n + 2:
            return AnalysisResult(symbol=symbol, timeframe=timeframe, bars=len(df))

        df = df.reset_index(drop=True)
        rsi_vals = rsi_indicator(df["close"], p.rsi_period).to_numpy(dtype=float)
        times = pd.DatetimeIndex(
            pd.to_datetime(df["open_time"], utc=True)
        ).to_pydatetime()
        last_idx = len(df) - 1

        signals: list[Signal] = []
        bull_anchor = self._scan(
            df, rsi_vals, times, symbol, timeframe, _bull_rules(p), signals
        )
        bear_anchor = None
        if p.bearish_enabled:
            bear_anchor = self._scan(
                df, rsi_vals, times, symbol, timeframe, _bear_rules(p), signals
            )

        signals.sort(key=lambda s: (s.candle_time, s.type.value))

        last_rsi = float(rsi_vals[last_idx]) if not np.isnan(rsi_vals[last_idx]) else None
        return AnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            signals=signals,
            bull_anchor=bull_anchor,
            bear_anchor=bear_anchor,
            last_price=float(df["close"].iloc[last_idx]),
            last_rsi=last_rsi,
            last_candle_time=times[last_idx],
            bars=len(df),
        )

    # --- внутреннее ---------------------------------------------------------

    def _scan(
        self,
        df: pd.DataFrame,
        rsi_vals: np.ndarray,
        times,
        symbol: str,
        timeframe: Timeframe,
        rules: _Rules,
        signals: list[Signal],
    ) -> PivotPoint | None:
        p = self.params
        n = p.fractal_n
        last_idx = len(df) - 1
        price_arr = df[rules.price_source].to_numpy(dtype=float)
        pivot_indices = set(rules.pivots(price_arr, n))

        anchor: PivotPoint | None = None

        for i in range(len(df)):
            r = rsi_vals[i]
            if np.isnan(r):
                continue  # прогрев RSI

            # 1. Протухание опорной точки по времени.
            if anchor is not None and i - anchor.index > p.max_bars_between:
                anchor = None

            # 2. Инвалидация по RSI (бары строго после опорной точки).
            if anchor is not None and i > anchor.index and rules.is_reset(float(r)):
                anchor = None

            if i not in pivot_indices:
                continue

            point = PivotPoint(
                index=i,
                time=times[i],
                price=float(price_arr[i]),
                rsi=float(r),
            )
            confirm_idx = i + n  # фрактал подтверждается через N баров

            if anchor is None:
                if rules.is_extreme_rsi(point.rsi):
                    anchor = point
                    signals.append(
                        self._make_signal(
                            symbol, timeframe, rules.alert1, rules.direction,
                            point, None, times, confirm_idx, last_idx,
                        )
                    )
                continue

            # Есть опорная точка → проверяем дивергенцию.
            is_divergence = (
                rules.price_broke(point.price, anchor.price)
                and rules.rsi_diverged(point.rsi, anchor.rsi)
                and rules.rsi_in_zone(point.rsi)
                and (point.index - anchor.index) <= p.max_bars_between
            )
            if is_divergence:
                signals.append(
                    self._make_signal(
                        symbol, timeframe, rules.alert2, rules.direction,
                        point, anchor, times, confirm_idx, last_idx,
                    )
                )
                anchor = None  # после ALERT #2 опорная точка сбрасывается
            elif rules.is_extreme_rsi(point.rsi):
                # Дивергенции нет, но новый экстремум сам по себе в зоне
                # перепроданности/перекупленности → он становится новой опорной точкой.
                anchor = point
                signals.append(
                    self._make_signal(
                        symbol, timeframe, rules.alert1, rules.direction,
                        point, None, times, confirm_idx, last_idx,
                    )
                )
            # иначе — опорная точка сохраняется, ждём следующий фрактал

        return anchor

    @staticmethod
    def _make_signal(
        symbol: str,
        timeframe: Timeframe,
        alert_type: AlertType,
        direction: Direction,
        point: PivotPoint,
        reference: PivotPoint | None,
        times,
        confirm_idx: int,
        last_idx: int,
    ) -> Signal:
        confirmation_time = times[confirm_idx] if confirm_idx <= last_idx else None
        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            type=alert_type,
            price=point.price,
            rsi=point.rsi,
            candle_time=point.time,
            direction=direction,
            reference=reference,
            pivot=point,
            confirmation_time=confirmation_time,
            bars_since_confirmation=max(0, last_idx - confirm_idx),
        )
