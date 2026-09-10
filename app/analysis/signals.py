"""Детекция сигналов.

Слой ничего не знает ни про Telegram, ни про API провайдеров.
Вход — DataFrame с закрытыми свечами, выход — список Signal.

Движок работает в режиме «переигровки» (replay): на каждом запуске он заново
прогоняет всё окно баров через конечный автомат. Это даёт три полезных свойства:
  * состояние восстанавливается после рестарта и после пропущенных опросов;
  * результат детерминирован и легко тестируется (никакого скрытого состояния);
  * повторная отправка исключается на уровне БД по ключу
    (symbol, timeframe, alert_type, candle_time).

ЦЕПОЧКИ ТОЧЕК. Вместо одной опорной точки движок держит цепочку экстремумов.
Каждый следующий подтверждённый фрактал продлевает её, если цена обновила
экстремум предыдущей точки, а RSI — нет. Степень сигнала (degree) равна числу
точек в цепочке: 2 — обычная дивергенция, 3 и больше — тройная и глубже.
Цепочка обнуляется при достижении chain_max_points, при инвалидации по RSI
и по max_bars_between. chain_max_points=2 воспроизводит прежнее поведение.
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


#: Типы, по которым заводится запись статистики (см. app/bot/storage.py).
DIVERGENCE_TYPES = frozenset(
    {AlertType.BULLISH_DIVERGENCE, AlertType.BEARISH_DIVERGENCE}
)


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
    reset_rsi: float = 50.0            # инвалидация бычьей цепочки (RSI выше)
    reset_rsi_bear: float = 50.0       # инвалидация медвежьей цепочки (RSI ниже)
    max_bars_between: int = 40
    bearish_enabled: bool = False
    chain_max_points: int = 4          # максимум точек в цепочке (2 = как раньше)


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
    reference: PivotPoint | None = None  # предыдущая точка цепочки (для ALERT #2+)
    pivot: PivotPoint | None = None      # текущая точка
    confirmation_time: datetime | None = None  # open_time свечи, подтвердившей фрактал
    confirmation_price: float | None = None    # close той же свечи = «цена входа»
    bars_since_confirmation: int = 0     # 0 = подтверждено на последней закрытой свече
    degree: int = 1                      # число точек в цепочке: 2 — обычная дивергенция
    chain: tuple[PivotPoint, ...] = ()   # вся цепочка целиком (для сообщения)

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

    @property
    def is_divergence(self) -> bool:
        return self.type in DIVERGENCE_TYPES


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    symbol: str
    timeframe: Timeframe
    signals: list[Signal] = field(default_factory=list)
    bull_anchor: PivotPoint | None = None       # последняя точка бычьей цепочки
    bear_anchor: PivotPoint | None = None
    bull_chain: tuple[PivotPoint, ...] = ()
    bear_chain: tuple[PivotPoint, ...] = ()
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
    is_reset: Callable[[float], bool]        # инвалидация цепочки


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
    """Конечный автомат ALERT #1 → ALERT #2 → ALERT #3… с инвалидацией цепочки."""

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
        closes = df["close"].to_numpy(dtype=float)
        times = pd.DatetimeIndex(
            pd.to_datetime(df["open_time"], utc=True)
        ).to_pydatetime()
        last_idx = len(df) - 1

        signals: list[Signal] = []
        bull_chain = self._scan(
            df, rsi_vals, closes, times, symbol, timeframe, _bull_rules(p), signals
        )
        bear_chain: list[PivotPoint] = []
        if p.bearish_enabled:
            bear_chain = self._scan(
                df, rsi_vals, closes, times, symbol, timeframe, _bear_rules(p), signals
            )

        signals.sort(key=lambda s: (s.candle_time, s.type.value))

        last_rsi = float(rsi_vals[last_idx]) if not np.isnan(rsi_vals[last_idx]) else None
        return AnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            signals=signals,
            bull_anchor=bull_chain[-1] if bull_chain else None,
            bear_anchor=bear_chain[-1] if bear_chain else None,
            bull_chain=tuple(bull_chain),
            bear_chain=tuple(bear_chain),
            last_price=float(closes[last_idx]),
            last_rsi=last_rsi,
            last_candle_time=times[last_idx],
            bars=len(df),
        )

    # --- внутреннее ---------------------------------------------------------

    def _scan(
        self,
        df: pd.DataFrame,
        rsi_vals: np.ndarray,
        closes: np.ndarray,
        times,
        symbol: str,
        timeframe: Timeframe,
        rules: _Rules,
        signals: list[Signal],
    ) -> list[PivotPoint]:
        p = self.params
        n = p.fractal_n
        last_idx = len(df) - 1
        price_arr = df[rules.price_source].to_numpy(dtype=float)
        pivot_indices = set(rules.pivots(price_arr, n))

        chain: list[PivotPoint] = []

        for i in range(len(df)):
            r = rsi_vals[i]
            if np.isnan(r):
                continue  # прогрев RSI

            # 1. Протухание цепочки по времени (считаем от последней её точки).
            if chain and i - chain[-1].index > p.max_bars_between:
                chain = []

            # 2. Инвалидация по RSI (бары строго после последней точки).
            if chain and i > chain[-1].index and rules.is_reset(float(r)):
                chain = []

            if i not in pivot_indices:
                continue

            point = PivotPoint(
                index=i,
                time=times[i],
                price=float(price_arr[i]),
                rsi=float(r),
            )
            confirm_idx = i + n  # фрактал подтверждается через N баров

            if not chain:
                if rules.is_extreme_rsi(point.rsi):
                    chain = [point]
                    signals.append(
                        self._make_signal(
                            symbol, timeframe, rules.alert1, rules.direction,
                            point, None, times, closes, confirm_idx, last_idx,
                            1, tuple(chain),
                        )
                    )
                continue

            prev = chain[-1]
            is_divergence = (
                rules.price_broke(point.price, prev.price)
                and rules.rsi_diverged(point.rsi, prev.rsi)
                and rules.rsi_in_zone(point.rsi)
                and (point.index - prev.index) <= p.max_bars_between
            )
            if is_divergence:
                chain.append(point)
                signals.append(
                    self._make_signal(
                        symbol, timeframe, rules.alert2, rules.direction,
                        point, prev, times, closes, confirm_idx, last_idx,
                        len(chain), tuple(chain),
                    )
                )
                if len(chain) >= p.chain_max_points:
                    chain = []   # цепочка дошла до потолка — начинаем заново
                continue

            if rules.is_extreme_rsi(point.rsi):
                # Продлить не вышло, но точка сама по себе в зоне
                # перепроданности/перекупленности → начинаем цепочку с неё.
                chain = [point]
                signals.append(
                    self._make_signal(
                        symbol, timeframe, rules.alert1, rules.direction,
                        point, None, times, closes, confirm_idx, last_idx,
                        1, tuple(chain),
                    )
                )
            # иначе — цепочка сохраняется, ждём следующий фрактал

        return chain

    @staticmethod
    def _make_signal(
        symbol: str,
        timeframe: Timeframe,
        alert_type: AlertType,
        direction: Direction,
        point: PivotPoint,
        reference: PivotPoint | None,
        times,
        closes: np.ndarray,
        confirm_idx: int,
        last_idx: int,
        degree: int = 1,
        chain: tuple[PivotPoint, ...] = (),
    ) -> Signal:
        confirmed = confirm_idx <= last_idx
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
            confirmation_time=times[confirm_idx] if confirmed else None,
            confirmation_price=float(closes[confirm_idx]) if confirmed else None,
            bars_since_confirmation=max(0, last_idx - confirm_idx),
            degree=degree,
            chain=chain,
        )
