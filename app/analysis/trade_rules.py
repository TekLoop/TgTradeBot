"""Правила выхода из сделки: лесенка тейков со стопом в безубыток.

Модуль намеренно чистый: ни сети, ни БД, ни Telegram, ни pandas. Вход —
описание сделки и последовательность баров, выход — TradeResult. Бэктест
(app/analysis/replay.py) уже ходит сюда; живой бот (app/bot/outcomes.py)
подключится к этому же модулю следующей итерацией — разъехавшихся копий
логики быть не должно.

МОДЕЛЬ (описана для лонга, для шорта всё зеркально):

    вход   = close свечи, подтвердившей фрактал;
    стоп   = либо от свечи экстремума (sl_set=low, по умолчанию),
             либо процент от входа (sl_set=enter);
    тейки  = три уровня в процентах от входа, объёмы 30% / 30% / остаток;
    переезд стопа после tp1 задаётся sl_after_tp1:
        "entry" — сразу в безубыток (поведение по умолчанию);
        "pivot" — РОВНО на цену пивота, без буфера; в безубыток стоп
                  переезжает только после tp2.

Автомат состояний (sl_after_tp1="pivot"):

    S0  стоп исходный (пивот − buffer), активны stop и tp1
        stop → закрыть 100% по стопу,     reason = "sl"
        tp1  → зафиксировать tp1_size,    стоп → pivot, перейти в S1
    S1  стоп = pivot, активны stop и tp2
        stop → закрыть остаток по pivot,  reason = "pivot"
        tp2  → зафиксировать tp2_size,    стоп → entry, перейти в S2
    S2  стоп = entry, активны stop и tp3
        stop → закрыть остаток по entry,  reason = "be"
        tp3  → зафиксировать остаток,     reason = "tp3"

При sl_after_tp1="entry" стоп из S0 сразу уезжает на entry, состояние S1
работает как S2, и причины "pivot" в выводе не бывает.

ВАЖНО про "pivot": этот стоп стоит НИЖЕ входа, значит выход по нему —
это убыток, а не безубыток. Сделка, взявшая tp1 и выбитая по пивоту,
вполне может закрыться в минус: 30% позиции в плюс не перекрывают 70%
по цене ниже входа. Ради этого причина и вынесена отдельной меткой.

Таймаута нет, новая дивергенция сделку не закрывает: выйти можно только
стопом или по tp3. Пока данные не кончились и ни то, ни другое не случилось,
сделка остаётся открытой (reason = "open") и в средние не попадает.

ИСПОЛНЕНИЕ. Срабатывание по касанию: лонг — стоп при low <= stop, тейк при
high >= tp. Цена исполнения — ровно уровень. Гэпы не моделируются: бар,
открывшийся за уровнем, всё равно считается исполненным по уровню. Это
сознательное упрощение, оно слегка приукрашивает результат.

ПОРЯДОК ВНУТРИ БАРА. Если на баре задет хоть один уровень и переданы
часовые подсвечи (subbars), бар разбирается по ним: так видно, что было
раньше — тейк или стоп, и отрабатывают цепочки вида «tp1 на первом часе,
возврат к входу и безубыток на третьем». Без подсвечек (или при
intrabar_resolution="off") работает консервативное правило: при
одновременном касании первым считается стоп.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Iterable, Protocol, Sequence

from app.analysis.signals import Direction
from app.core.timeframes import Timeframe

# --- причины выхода ---------------------------------------------------------

REASON_SL = "sl"        # стоп на исходном уровне
REASON_PIVOT = "pivot"  # стоп переехал на цену пивота — это ещё минус
REASON_BE = "be"        # стоп в безубытке
REASON_TP3 = "tp3"      # дошли до последнего тейка
REASON_OPEN = "open"    # данные кончились, сделка не закрыта

REASON_LABELS = {
    REASON_SL: "стоп",
    REASON_PIVOT: "стоп на пивоте",
    REASON_BE: "безубыток",
    REASON_TP3: "полный проход (tp3)",
    REASON_OPEN: "не закрыта (конец данных)",
}

#: Порядок для отчётов: сначала худший исход, потом лучший.
REASON_ORDER = (REASON_SL, REASON_PIVOT, REASON_BE, REASON_TP3)

# --- уровни тейков по таймфреймам -------------------------------------------

#: Дефолтные уровни тейков в процентах от входа, ключ — таймфрейм.
#: Единственное место, где эта таблица живёт. Любое значение перебивается
#: через --param (см. app/backtest.py).
TP_DEFAULTS: dict[Timeframe, tuple[float, float, float]] = {
    Timeframe.H4: (5.0, 10.0, 20.0),
    Timeframe.D1: (10.0, 20.0, 50.0),
}

#: Для таймфреймов, которых нет в таблице, берутся значения 4H.
TP_FALLBACK: tuple[float, float, float] = TP_DEFAULTS[Timeframe.H4]


def default_tp_levels(timeframe: Timeframe | str | None) -> tuple[float, float, float]:
    """Строка таблицы для таймфрейма. Неизвестный ТФ → как 4H."""
    if timeframe is None:
        return TP_FALLBACK
    tf = timeframe if isinstance(timeframe, Timeframe) else Timeframe.try_parse(str(timeframe))
    if tf is None:
        return TP_FALLBACK
    return TP_DEFAULTS.get(tf, TP_FALLBACK)


# --- параметры --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeRules:
    """Все параметры лесенки. Значения tp*_pct по умолчанию — как у 4H;
    таблица по таймфреймам разрешается в for_timeframe()."""

    #: low — стоп от свечи экстремума, enter — процент от цены входа
    sl_set: str = "low"
    #: только для sl_set=enter
    sl_percent: float = 5.0
    #: отступ вниз от экстремума, только для sl_set=low
    sl_buffer_pct: float = 1.0

    tp1_pct: float = TP_FALLBACK[0]
    tp2_pct: float = TP_FALLBACK[1]
    tp3_pct: float = TP_FALLBACK[2]

    #: доли позиции в процентах; tp3 забирает остаток
    tp1_size: float = 30.0
    tp2_size: float = 30.0

    #: куда уезжает стоп после tp1: entry — в безубыток, pivot — на цену
    #: пивота без буфера (в безубыток тогда только после tp2)
    sl_after_tp1: str = "entry"

    #: 1h — разрешать порядок внутри бара часовыми свечами, off — не разрешать
    intrabar_resolution: str = "1h"

    def __post_init__(self) -> None:
        if self.sl_after_tp1 not in {"entry", "pivot"}:
            raise ValueError(
                f"sl_after_tp1: ожидалось entry или pivot, "
                f"получено {self.sl_after_tp1!r}"
            )
        if self.sl_set not in {"low", "enter"}:
            raise ValueError(f"sl_set: ожидалось low или enter, получено {self.sl_set!r}")
        if self.intrabar_resolution not in {"1h", "off"}:
            raise ValueError(
                f"intrabar_resolution: ожидалось 1h или off, "
                f"получено {self.intrabar_resolution!r}"
            )
        if self.sl_set == "enter" and not 0 < self.sl_percent < 100:
            raise ValueError("sl_percent должен быть в (0, 100)")
        if self.sl_buffer_pct < 0 or self.sl_buffer_pct >= 100:
            raise ValueError("sl_buffer_pct должен быть в [0, 100)")
        levels = (self.tp1_pct, self.tp2_pct, self.tp3_pct)
        if any(level <= 0 for level in levels):
            raise ValueError("уровни тейков должны быть положительными")
        if not levels[0] < levels[1] < levels[2]:
            raise ValueError(f"уровни тейков должны расти: {levels}")
        if self.tp1_size <= 0 or self.tp2_size <= 0:
            raise ValueError("tp1_size и tp2_size должны быть положительными")
        if self.tp1_size + self.tp2_size >= 100:
            raise ValueError(
                f"tp1_size + tp2_size = {self.tp1_size + self.tp2_size:g} — "
                f"на tp3 ничего не остаётся"
            )

    @property
    def tp3_size(self) -> float:
        """Остаток позиции после двух первых тейков."""
        return 100.0 - self.tp1_size - self.tp2_size

    @property
    def tp_levels(self) -> tuple[float, float, float]:
        return (self.tp1_pct, self.tp2_pct, self.tp3_pct)

    @property
    def tp_sizes(self) -> tuple[float, float, float]:
        return (self.tp1_size, self.tp2_size, self.tp3_size)

    @classmethod
    def for_timeframe(
        cls, timeframe: Timeframe | str | None, overrides: dict | None = None
    ) -> "TradeRules":
        """Таблица уровней по таймфрейму, поверх — то, что пришло в --param.

        Явно переданный tp1_pct всегда сильнее таблицы: таблица разрешается
        один раз на старте прогона, дальше правила уже неизменны.
        """
        levels = default_tp_levels(timeframe)
        values: dict = {
            "tp1_pct": levels[0],
            "tp2_pct": levels[1],
            "tp3_pct": levels[2],
        }
        for key, value in (overrides or {}).items():
            if key not in LADDER_KEYS:
                raise ValueError(f"Неизвестный параметр лесенки: {key}")
            values[key] = value
        return cls(**values)

    def describe(self) -> str:
        """Однострочное описание для шапки отчёта: через месяц по CSV уже
        не восстановить, по каким цифрам он считался."""
        stop = (
            f"стоп от экстремума −{self.sl_buffer_pct:g}%"
            if self.sl_set == "low"
            else f"стоп {self.sl_percent:g}% от входа"
        )
        after = (
            "после tp1 стоп на пивот, после tp2 в безубыток"
            if self.sl_after_tp1 == "pivot"
            else "после tp1 стоп в безубыток"
        )
        return (
            f"{stop} · тейки {self.tp1_pct:g}/{self.tp2_pct:g}/{self.tp3_pct:g}% "
            f"объёмами {self.tp1_size:g}/{self.tp2_size:g}/{self.tp3_size:g}% · "
            f"{after} · "
            f"внутрибарное разрешение {self.intrabar_resolution}"
        )

    def stop_price(self, entry_price: float, pivot_price: float | None,
                   direction: Direction) -> float:
        """Исходный стоп сделки.

        sl_set=low берёт цену ТОЧКИ ДИВЕРГЕНЦИИ (low свечи экстремума для
        лонга, high — для шорта), а не свечи входа: свеча входа стоит на
        fractal_n баров позже и её минимум по построению фрактала выше.
        """
        if self.sl_set == "enter":
            shift = self.sl_percent / 100.0
            return entry_price * (1 - shift) if direction is Direction.BULL \
                else entry_price * (1 + shift)
        if pivot_price is None:
            raise ValueError("sl_set=low требует цену свечи экстремума (signal.pivot)")
        shift = self.sl_buffer_pct / 100.0
        return pivot_price * (1 - shift) if direction is Direction.BULL \
            else pivot_price * (1 + shift)


#: Имена ключей лесенки. Их разбирает отдельная ветка в app/backtest.py:
#: в ParamsConfig они попасть не должны, иначе тут же появятся в валидации
#: живого бота и в /params.
LADDER_KEYS: frozenset[str] = frozenset(f.name for f in fields(TradeRules))


# --- данные сделки ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """Минимум, нужный для проверки касания."""

    time: datetime
    high: float
    low: float


class SubBars(Protocol):
    """Источник подсвечек: что угодно с .get(время бара) → последовательность
    Bar. Обычный dict подходит."""

    def get(self, key: datetime, default=None): ...


@dataclass(frozen=True, slots=True)
class TradeEntry:
    """Точка входа плюс цена экстремума, от которой строится структурный стоп."""

    time: datetime
    price: float
    pivot_price: float | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """Закрытая часть позиции."""

    time: datetime
    price: float
    share: float     # доля позиции, 0..1
    tag: str         # tp1 | tp2 | tp3 | sl | be


@dataclass(frozen=True, slots=True)
class TradeResult:
    direction: Direction
    entry_time: datetime
    entry_price: float
    stop_price: float
    tp_prices: tuple[float, float, float]
    fills: tuple[Fill, ...]
    exit_time: datetime | None
    exit_reason: str
    bars_held: int
    mae_pct: float           # ход против сигнала, <= 0
    mfe_pct: float           # ход в пользу сигнала, >= 0

    @property
    def sign(self) -> float:
        return 1.0 if self.direction is Direction.BULL else -1.0

    @property
    def is_closed(self) -> bool:
        return self.exit_reason != REASON_OPEN

    @property
    def sl_dist_pct(self) -> float:
        return abs(self.entry_price - self.stop_price) / self.entry_price * 100.0

    def pnl_pct(self, price: float) -> float:
        return self.sign * (price / self.entry_price - 1.0) * 100.0

    @property
    def realized_pct(self) -> float:
        """Взвешенный итог по уже закрытым частям. Для открытой сделки —
        то, что зафиксировано на данный момент."""
        return sum(fill.share * self.pnl_pct(fill.price) for fill in self.fills)

    @property
    def result_pct(self) -> float | None:
        """Итог сделки. У незакрытой его нет: остаток позиции ещё в рынке."""
        return self.realized_pct if self.is_closed else None

    @property
    def r_multiple(self) -> float | None:
        distance = self.sl_dist_pct
        result = self.result_pct
        if result is None or not distance:
            return None
        return result / distance

    def fill_time(self, tag: str) -> datetime | None:
        for fill in self.fills:
            if fill.tag == tag:
                return fill.time
        return None

    @property
    def tp1_time(self) -> datetime | None:
        return self.fill_time("tp1")

    @property
    def tp2_time(self) -> datetime | None:
        return self.fill_time("tp2")

    @property
    def tp3_time(self) -> datetime | None:
        return self.fill_time("tp3")

    @property
    def reason_label(self) -> str:
        return REASON_LABELS.get(self.exit_reason, self.exit_reason)


# --- автомат ----------------------------------------------------------------


def simulate_trade(
    entry: TradeEntry,
    direction: Direction,
    bars: Sequence[Bar],
    rules: TradeRules,
    subbars: SubBars | None = None,
) -> TradeResult:
    """Прогоняет сделку по барам и возвращает результат.

    bars — свечи СТРОГО ПОСЛЕ свечи входа: вход берётся по её close, значит
    сама она сработать уже не может. subbars — необязательный индекс часовых
    свечей: .get(время бара) → последовательность Bar внутри этого бара.
    """
    bull = direction is Direction.BULL
    entry_price = float(entry.price)
    if entry_price <= 0:
        raise ValueError("цена входа должна быть положительной")

    stop = rules.stop_price(entry_price, entry.pivot_price, direction)
    if bull and stop >= entry_price:
        raise ValueError(
            f"стоп {stop:g} не ниже цены входа {entry_price:g} — для лонга это ошибка"
        )
    if not bull and stop <= entry_price:
        raise ValueError(
            f"стоп {stop:g} не выше цены входа {entry_price:g} — для шорта это ошибка"
        )

    #: Куда уедет стоп после tp1. "pivot" требует цену пивота и требует,
    #: чтобы она была по нужную сторону от входа; иначе молча работает
    #: безубыток — это заведомо более консервативный вариант.
    stop_after_tp1 = entry_price
    tag_after_tp1 = REASON_BE
    if rules.sl_after_tp1 == "pivot" and entry.pivot_price is not None:
        candidate = float(entry.pivot_price)
        if (bull and candidate < entry_price) or (not bull and candidate > entry_price):
            stop_after_tp1 = candidate
            tag_after_tp1 = REASON_PIVOT

    sign = 1.0 if bull else -1.0
    tp_prices = tuple(
        entry_price * (1.0 + sign * level / 100.0) for level in rules.tp_levels
    )
    sizes = tuple(size / 100.0 for size in rules.tp_sizes)

    state = 0                  # 0 — исходный стоп, 1 — после tp1, 2 — после tp2
    current_stop = stop
    stop_tag = REASON_SL       # метка, с которой закроется текущий стоп
    remaining = 1.0
    fills: list[Fill] = []
    exit_reason = REASON_OPEN
    exit_time: datetime | None = None
    bars_held = 0

    def touches_stop(bar: Bar) -> bool:
        return bar.low <= current_stop if bull else bar.high >= current_stop

    def touches_tp(bar: Bar) -> bool:
        level = tp_prices[state]
        return bar.high >= level if bull else bar.low <= level

    def run(bar: Bar) -> int:
        """Отрабатывает все события внутри одного (под)бара.

        Возвращает число сработавших событий. Внутри бара может пройти целая
        цепочка: tp1 → стоп переехал → новый стоп выбит. При
        одновременном касании стопа и тейка первым считается стоп —
        консервативно.
        """
        nonlocal state, current_stop, stop_tag, remaining, exit_reason, exit_time
        events = 0
        while exit_reason == REASON_OPEN:
            hit_stop = touches_stop(bar)
            hit_tp = touches_tp(bar)
            if not hit_stop and not hit_tp:
                break
            events += 1
            if hit_stop:
                tag = stop_tag
                fills.append(Fill(bar.time, current_stop, remaining, tag))
                remaining = 0.0
                exit_reason = tag
                exit_time = bar.time
                break
            share = remaining if state == 2 else sizes[state]
            fills.append(Fill(bar.time, tp_prices[state], share, f"tp{state + 1}"))
            remaining = max(0.0, remaining - share)
            if state == 2:
                exit_reason = REASON_TP3
                exit_time = bar.time
                break
            if state == 0:
                current_stop = stop_after_tp1
                stop_tag = tag_after_tp1
            else:
                current_stop = entry_price     # после второго тейка — безубыток
                stop_tag = REASON_BE
            state += 1
        return events

    mfe = 0.0
    mae = 0.0
    use_subbars = subbars is not None and rules.intrabar_resolution == "1h"

    for offset, bar in enumerate(bars):
        high_pct = sign * (bar.high / entry_price - 1.0) * 100.0
        low_pct = sign * (bar.low / entry_price - 1.0) * 100.0
        mfe = max(mfe, high_pct, low_pct)
        mae = min(mae, high_pct, low_pct)

        if touches_stop(bar) or touches_tp(bar):
            group: Iterable[Bar] = ()
            if use_subbars:
                group = subbars.get(bar.time) or ()   # type: ignore[union-attr]
            fired = 0
            for sub in group:
                fired += run(sub)
                if exit_reason != REASON_OPEN:
                    break
            if not fired:
                # Либо часовых данных нет, либо они неполны и не показали того,
                # что видно на большом баре. Оба случая — консервативное правило.
                run(bar)

        if exit_reason != REASON_OPEN:
            bars_held = offset + 1
            break
    else:
        bars_held = len(bars)

    return TradeResult(
        direction=direction,
        entry_time=entry.time,
        entry_price=entry_price,
        stop_price=stop,
        tp_prices=tp_prices,  # type: ignore[arg-type]
        fills=tuple(fills),
        exit_time=exit_time,
        exit_reason=exit_reason,
        bars_held=bars_held,
        mae_pct=mae,
        mfe_pct=mfe,
    )
