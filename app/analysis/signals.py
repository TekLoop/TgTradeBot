"""Детекция сигналов.

Слой ничего не знает ни про Telegram, ни про API провайдеров.
Вход — DataFrame с закрытыми свечами, выход — список Signal.

Движок работает в режиме «переигровки» (replay): на каждом запуске он заново
прогоняет всё окно баров через конечный автомат. Это даёт три полезных свойства:
  * состояние восстанавливается после рестарта и после пропущенных опросов;
  * результат детерминирован и легко тестируется (никакого скрытого состояния);
  * повторная отправка исключается на уровне БД по ключу
    (symbol, timeframe, alert_type, candle_time).

ЦЕПОЧКИ ТОЧЕК И ОПОРНАЯ ТОЧКА. Движок держит цепочку экстремумов
[P0, P1, ... Pn]. ОПОРНАЯ ТОЧКА — это P0 = chain[0]: точка, от которой всё
считается и которая живёт дольше всех. chain[-1] — просто последняя точка.

Опорная точка живёт, пока не случилось одно из трёх:
  1. законное вытеснение: пивот с ценой НИЖЕ и RSI НИЖЕ опорной (ALERT #1);
  2. инвалидация по reset_rsi;
  3. превышение anchor_ttl_bars (возраст считается от chain[0]).
Во всех прочих случаях пивот, не образовавший дивергенцию, молча игнорируется.

max_bars_between_points цепочку НЕ убивает: он ограничивает только расстояние
внутри пары, образующей дивергенцию, и проверяется при обходе цепочки назад.
chain_max_points запрещает продление, но цепочку не трогает.
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
    """ВНИМАНИЕ: строковые значения — часть схемы БД.

    Они попадают в sent_alerts.alert_type и служат частью ключа дедупликации
    (symbol, timeframe, alert_type, candle_time). Переименование любого
    значения — это МИГРАЦИЯ БД, а не рефакторинг. Добавлять новые значения
    можно свободно, менять существующие — нельзя.
    """

    OVERSOLD_PIVOT = "oversold_pivot"          # ALERT #1 (бычий сценарий)
    BULLISH_DIVERGENCE = "bullish_divergence"  # ALERT #2 (бычий сценарий)
    OVERBOUGHT_PIVOT = "overbought_pivot"      # ALERT #1 (медвежий сценарий)
    BEARISH_DIVERGENCE = "bearish_divergence"  # ALERT #2 (медвежий сценарий)
    EXTREME_OVERSOLD = "extreme_oversold"      # Задача 4: RSI в крайней зоне
    EXTREME_OVERBOUGHT = "extreme_overbought"
    WICK_UPPER = "wick_upper"                  # Задача 5: длинный фитиль свечи
    WICK_LOWER = "wick_lower"


#: Типы, по которым заводится запись статистики (см. app/bot/outcomes.py).
#: Extreme-алерты и фитили сюда НЕ входят: у них нет модели входа.
DIVERGENCE_TYPES = frozenset(
    {AlertType.BULLISH_DIVERGENCE, AlertType.BEARISH_DIVERGENCE}
)

#: Алерты, приходящие на своей же свече, без ожидания подтверждения фрактала.
INSTANT_TYPES = frozenset(
    {
        AlertType.EXTREME_OVERSOLD,
        AlertType.EXTREME_OVERBOUGHT,
        AlertType.WICK_UPPER,
        AlertType.WICK_LOWER,
    }
)


class Direction(str, Enum):
    BULL = "bull"
    BEAR = "bear"


@dataclass(frozen=True, slots=True)
class SignalParams:
    """Пороги детектора. Отдельный от pydantic-конфига объект,
    чтобы analysis/ оставался зависимым только от pandas/numpy.

    Для трёх параметров расстояний значение 0 = «проверка выключена».
    """

    rsi_period: int = 14
    fractal_n: int = 2
    oversold: float = 30.0
    overbought: float = 70.0
    divergence_rsi_max: float = 35.0   # верхняя граница RSI для бычьей дивергенции
    divergence_rsi_min: float = 65.0   # нижняя граница RSI для медвежьей дивергенции
    reset_rsi: float = 50.0            # инвалидация бычьей цепочки (RSI выше)
    reset_rsi_bear: float = 50.0       # инвалидация медвежьей цепочки (RSI ниже)

    #: максимальный возраст опорной точки, считается от chain[0], каждый бар
    anchor_ttl_bars: int = 40
    #: максимальное расстояние в паре, образующей дивергенцию (Pj → новый пивот)
    max_bars_between_points: int = 40
    #: минимальное расстояние от chain[-1] до нового пивота (входной фильтр)
    min_bars_between_points: int = 0

    bearish_enabled: bool = False
    chain_max_points: int = 4          # потолок длины цепочки (0 = выключен)

    # --- Задача 4: extreme-алерты (независимый детектор) --------------------
    extreme_alerts_enabled: bool = False
    extreme_oversold: float = 15.0
    extreme_overbought: float = 85.0
    extreme_rearm_rsi_bull: float = 30.0
    extreme_rearm_rsi_bear: float = 70.0
    extreme_rearm_bars: int = 3

    # --- Задача 5: алерты по фитилю (независимый детектор) ------------------
    wick_alerts_enabled: bool = False
    wick_threshold_pct: float = 5.0
    wick_min_body_pct: float = 0.0


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

    @classmethod
    def from_dict(cls, data: dict) -> "PivotPoint":
        raw = data["time"]
        moment = raw if isinstance(raw, datetime) else datetime.fromisoformat(raw)
        return cls(
            index=int(data.get("index", 0)),
            time=moment,
            price=float(data["price"]),
            rsi=float(data["rsi"]),
        )


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    timeframe: Timeframe
    type: AlertType
    price: float
    rsi: float
    candle_time: datetime               # open_time свечи-экстремума = ключ дедупликации
    direction: Direction = Direction.BULL
    reference: PivotPoint | None = None  # точка цепочки, с которой найдена дивергенция
    pivot: PivotPoint | None = None      # текущая точка
    confirmation_time: datetime | None = None  # open_time свечи, подтвердившей фрактал
    confirmation_price: float | None = None    # close той же свечи = «цена входа»

    #: ТРАНСПОРТНОЕ поле, не аналитическое. Отвечает на вопрос «успели ли
    #: сообщить»: зависит от момента запуска опроса, а не от рынка. Используется
    #: только в app/service.py для фильтра notify_max_age_bars. Не путать
    #: с рыночными данными и не заводить на нём статистику.
    bars_since_confirmation: int = 0     # 0 = подтверждено на последней закрытой свече

    degree: int = 1                      # число точек в цепочке: 2 — обычная дивергенция
    chain: tuple[PivotPoint, ...] = ()   # вся цепочка целиком (для сообщения)

    #: Задача 2: признак замены. Позиция в цепочке, начиная с которой точки
    #: выброшены (j+1). None — обычное продление. Живёт только в Signal и
    #: в sent_alerts.payload; в таблицу outcomes НЕ пишется.
    replaced_from: int | None = None
    #: времена выброшенных точек — нужны, чтобы адресно закрыть их записи
    replaced_points: tuple[datetime, ...] = ()

    #: Задача 2: опорная точка, которую вытеснил этот ALERT #1.
    #: None означает, что цепочка была ПУСТА — ограничения по цене в этом
    #: случае нет и быть не должно, сравнивать было не с чем. Приёмочное
    #: утверждение «цена ниже предыдущей опорной» проверяется только для
    #: сигналов, у которых это поле заполнено.
    displaced_anchor: PivotPoint | None = None

    #: Задача 5: процент фитиля и OHLC свечи (open, high, low, close).
    wick_pct: float | None = None
    ohlc: tuple[float, float, float, float] | None = None

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

    @property
    def is_replacement(self) -> bool:
        return self.replaced_from is not None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    symbol: str
    timeframe: Timeframe
    signals: list[Signal] = field(default_factory=list)
    #: ОПОРНАЯ точка = chain[0]. Не путать с последней точкой цепочки.
    bull_anchor: PivotPoint | None = None
    bear_anchor: PivotPoint | None = None
    #: последняя точка цепочки = chain[-1]
    bull_last_point: PivotPoint | None = None
    bear_last_point: PivotPoint | None = None
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
    is_extreme_rsi: Callable[[float], bool]  # условие ALERT #1 на пустой цепочке
    price_broke: Callable[[float, float], bool]   # цена обновила экстремум
    rsi_broke: Callable[[float, float], bool]     # RSI обновил экстремум
    rsi_diverged: Callable[[float, float], bool]  # RSI экстремум НЕ обновил
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
        rsi_broke=lambda new, old: new < old,        # RSI обновил минимум
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
        rsi_broke=lambda new, old: new > old,        # RSI обновил максимум
        rsi_diverged=lambda new, old: new < old,     # RSI максимум НЕ обновил
        rsi_in_zone=lambda r: r >= p.divergence_rsi_min,
        is_reset=lambda r: r < p.reset_rsi_bear,
    )


REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class Window:
    """Предпосчитанное окно: RSI и времена считаются один раз, дальше все
    детекторы работают с готовыми массивами."""

    df: pd.DataFrame
    rsi: np.ndarray
    closes: np.ndarray
    times: object            # массив datetime, индексируется как times[i]
    last_idx: int

    def __len__(self) -> int:
        return len(self.df)


class DivergenceEngine:
    """Конечный автомат ALERT #1 → ALERT #2 → ALERT #3… с инвалидацией цепочки."""

    def __init__(self, params: SignalParams) -> None:
        self.params = params
        # Импорт внутри конструктора, а не на уровне модуля: detectors.py
        # импортирует типы отсюда, и на уровне модуля получился бы цикл.
        # Направление зависимостей от этого не страдает — detectors.py тоже
        # живёт в app/analysis/ и наружу не смотрит.
        from app.analysis.detectors import ExtremeDetector, WickDetector

        self._detectors = (ExtremeDetector(params), WickDetector(params))

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
        window = Window(
            df=df, rsi=rsi_vals, closes=closes, times=times, last_idx=last_idx
        )

        signals: list[Signal] = []
        bull_chain = self._scan(window, symbol, timeframe, _bull_rules(p), signals)
        bear_chain: list[PivotPoint] = []
        if p.bearish_enabled:
            bear_chain = self._scan(window, symbol, timeframe, _bear_rules(p), signals)

        # Независимые детекторы (Задача 4 и 5). Состояния цепочек они не видят
        # и повлиять на опорные точки не могут.
        for detector in self._detectors:
            signals.extend(detector.detect(window, symbol, timeframe))

        signals.sort(key=lambda s: (s.candle_time, s.type.value))

        last_rsi = float(rsi_vals[last_idx]) if not np.isnan(rsi_vals[last_idx]) else None
        return AnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            signals=signals,
            bull_anchor=bull_chain[0] if bull_chain else None,
            bear_anchor=bear_chain[0] if bear_chain else None,
            bull_last_point=bull_chain[-1] if bull_chain else None,
            bear_last_point=bear_chain[-1] if bear_chain else None,
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
        window: Window,
        symbol: str,
        timeframe: Timeframe,
        rules: _Rules,
        signals: list[Signal],
    ) -> list[PivotPoint]:
        """Порядок проверок зафиксирован спецификацией и перестановке
        не подлежит — см. нумерацию шагов в комментариях."""
        p = self.params
        n = p.fractal_n
        rsi_vals = window.rsi
        price_arr = window.df[rules.price_source].to_numpy(dtype=float)
        pivot_indices = set(rules.pivots(price_arr, n))

        chain: list[PivotPoint] = []

        for i in range(len(window)):
            r = rsi_vals[i]
            if np.isnan(r):
                continue  # прогрев RSI

            # --- на каждом баре ---------------------------------------------
            # 1. Возраст опорной точки. Считается от chain[0], а не от chain[-1].
            if chain and p.anchor_ttl_bars > 0 and i - chain[0].index > p.anchor_ttl_bars:
                chain = []

            # 2. Инвалидация по RSI (бары строго после последней точки цепочки).
            if chain and i > chain[-1].index and rules.is_reset(float(r)):
                chain = []

            if i not in pivot_indices:
                continue

            # --- только в момент пивота -------------------------------------
            point = PivotPoint(
                index=i,
                time=window.times[i],
                price=float(price_arr[i]),
                rsi=float(r),
            )
            confirm_idx = i + n  # фрактал подтверждается через N баров

            # 3. Цепочка пуста → кандидат в опорные. Ограничения по цене здесь
            #    нет и быть не может: сравнивать не с чем.
            if not chain:
                if rules.is_extreme_rsi(point.rsi):
                    chain = [point]
                    signals.append(
                        self._make_signal(
                            symbol, timeframe, rules.alert1, rules.direction,
                            point, None, window, confirm_idx, 1, tuple(chain),
                        )
                    )
                continue

            # 4. Минимальное расстояние до последней точки цепочки. Пивот,
            #    отклонённый ТОЛЬКО за то, что пришёл рано, не должен
            #    провалиться в шаг 6 и вытеснить опорную.
            if (
                p.min_bars_between_points > 0
                and point.index - chain[-1].index < p.min_bars_between_points
            ):
                continue

            # 5. Обход цепочки назад: ищем точку, с которой есть дивергенция.
            j = self._find_reference(chain, point, rules)
            if j is not None:
                dropped = chain[j + 1:]
                reference = chain[j]
                chain = chain[: j + 1] + [point]
                signals.append(
                    self._make_signal(
                        symbol, timeframe, rules.alert2, rules.direction,
                        point, reference, window, confirm_idx,
                        len(chain), tuple(chain),
                        replaced_from=(j + 1) if dropped else None,
                        replaced_points=tuple(pt.time for pt in dropped),
                    )
                )
                continue

            # 6. Законное вытеснение опорной: цена НИЖЕ и RSI НИЖЕ опорной.
            #    Проверка is_extreme_rsi здесь избыточна: опорная по построению
            #    имела RSI в зоне, значит более низкий RSI тоже в зоне.
            if rules.price_broke(point.price, chain[0].price) and rules.rsi_broke(
                point.rsi, chain[0].rsi
            ):
                displaced = chain[0]
                chain = [point]
                signals.append(
                    self._make_signal(
                        symbol, timeframe, rules.alert1, rules.direction,
                        point, None, window, confirm_idx, 1, tuple(chain),
                        displaced_anchor=displaced,
                    )
                )
                continue

            # 7. Иначе — молча игнорируем. Цепочка не трогается, алерта нет.

        return chain

    def _find_reference(
        self, chain: list[PivotPoint], point: PivotPoint, rules: _Rules
    ) -> int | None:
        """Обход цепочки назад: индекс j первой подходящей точки Pj или None.

        max_bars_between_points меряется до Pj, а НЕ до chain[-1]: расстояние
        ограничивает пару, образующую дивергенцию. Пара с более ранней точкой
        законна, даже если до последней точки цепочки уже далеко.
        """
        p = self.params
        for j in range(len(chain) - 1, -1, -1):
            candidate = chain[j]
            if not rules.price_broke(point.price, candidate.price):
                continue
            if not rules.rsi_diverged(point.rsi, candidate.rsi):
                continue
            if not rules.rsi_in_zone(point.rsi):
                continue
            if (
                p.max_bars_between_points > 0
                and point.index - candidate.index > p.max_bars_between_points
            ):
                continue
            # Потолок запрещает продление, но замены (j < n) укорачивают
            # цепочку и остаются разрешены при любой длине.
            if p.chain_max_points > 0 and j + 2 > p.chain_max_points:
                continue
            return j
        return None

    @staticmethod
    def _make_signal(
        symbol: str,
        timeframe: Timeframe,
        alert_type: AlertType,
        direction: Direction,
        point: PivotPoint,
        reference: PivotPoint | None,
        window: Window,
        confirm_idx: int,
        degree: int = 1,
        chain: tuple[PivotPoint, ...] = (),
        *,
        replaced_from: int | None = None,
        replaced_points: tuple[datetime, ...] = (),
        displaced_anchor: PivotPoint | None = None,
    ) -> Signal:
        confirmed = confirm_idx <= window.last_idx
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
            confirmation_time=window.times[confirm_idx] if confirmed else None,
            confirmation_price=float(window.closes[confirm_idx]) if confirmed else None,
            bars_since_confirmation=max(0, window.last_idx - confirm_idx),
            degree=degree,
            chain=chain,
            replaced_from=replaced_from,
            replaced_points=replaced_points,
            displaced_anchor=displaced_anchor,
        )
