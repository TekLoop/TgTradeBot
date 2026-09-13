"""Офлайн-оценка сигналов: что было с ценой после каждой дивергенции.

Чистая функция от датафрейма и списка сигналов — ни БД, ни Telegram, ни сети.
Правила выхода живут в app/analysis/trade_rules.py и здесь не дублируются:

    вход  = close свечи, подтвердившей фрактал;
    выход = стоп (исходный или переехавший в безубыток) либо tp3.

Таймаута нет, новая дивергенция открытую сделку НЕ закрывает — поэтому
причин выхода reverse / same / timeout больше не существует. Сделки одной
пары могут перекрываться по времени; чтобы отделить независимые входы от
цепочек «четыре дивергенции подряд — четыре почти одинаковых входа», у
каждой сделки считается признак first_in_series.

Проценты всегда «в сторону сигнала»: для медвежьей падение цены — это плюс.
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from app.analysis.signals import DIVERGENCE_TYPES, Direction, Signal
from app.analysis.trade_rules import (
    REASON_BE,
    REASON_LABELS,
    REASON_OPEN,
    REASON_ORDER,
    REASON_PIVOT,
    REASON_SL,
    REASON_TP3,
    Bar,
    TradeEntry,
    TradeResult,
    TradeRules,
    simulate_trade,
)
from app.core.timeframes import Timeframe

__all__ = [
    "REASON_BE",
    "REASON_LABELS",
    "REASON_OPEN",
    "REASON_ORDER",
    "REASON_PIVOT",
    "REASON_SL",
    "REASON_TP3",
    "INTRABAR_TIMEFRAMES",
    "SubBarIndex",
    "Trade",
    "aggregate",
    "build_bars",
    "build_subbars",
    "evaluate",
    "supports_intrabar",
]


# --- подготовка баров -------------------------------------------------------


def build_bars(df: pd.DataFrame) -> tuple[list[Bar], list[datetime]]:
    """DataFrame → список Bar и список времён (для поиска свечи входа)."""
    times = pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True)).to_pydatetime()
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    bars = [
        Bar(time=times[i], high=float(highs[i]), low=float(lows[i]))
        for i in range(len(df))
    ]
    return bars, list(times)


class SubBarIndex:
    """Часовые свечи, сгруппированные по барам старшего таймфрейма.

    Группы не материализуются: .get(время бара) режет общий список по
    полуинтервалу [t, t + длительность бара) двоичным поиском.
    """

    def __init__(self, bars: list[Bar], times: list[datetime], span: timedelta) -> None:
        self._bars = bars
        self._times = times
        self._span = span

    def __len__(self) -> int:
        return len(self._bars)

    def get(self, key: datetime, default=None):
        left = bisect_left(self._times, key)
        right = bisect_left(self._times, key + self._span)
        if left >= right:
            return default
        return self._bars[left:right]


#: Таймфреймы, для которых включено разрешение порядка часовыми свечами.
#: 1H его собственными свечами не разложить; остальные ТФ в работе не
#: используются, и качать под них часы незачем — там сразу консервативное
#: правило. Захочется расширить — достаточно дописать таймфрейм сюда.
INTRABAR_TIMEFRAMES: frozenset[Timeframe] = frozenset({Timeframe.H4, Timeframe.D1})


def supports_intrabar(timeframe: Timeframe) -> bool:
    return timeframe in INTRABAR_TIMEFRAMES


def build_subbars(hourly: pd.DataFrame | None, timeframe: Timeframe) -> SubBarIndex | None:
    """Индекс часовых свечей для разрешения порядка внутри бара.

    Для таймфреймов вне INTRABAR_TIMEFRAMES возвращается None — дальше
    работает консервативное правило «стоп первым».
    """
    if hourly is None or hourly.empty or not supports_intrabar(timeframe):
        return None
    bars, times = build_bars(hourly)
    return SubBarIndex(bars, times, timeframe.duration)


# --- сделка -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trade:
    """Сигнал + результат его отработки по правилам лесенки."""

    symbol: str
    timeframe: Timeframe
    direction: Direction
    degree: int
    signal_time: datetime       # свеча экстремума
    entry_rsi: float
    result: TradeResult
    first_in_series: bool = True

    # --- делегирование в TradeResult ---------------------------------------

    @property
    def entry_time(self) -> datetime:
        return self.result.entry_time

    @property
    def entry_price(self) -> float:
        return self.result.entry_price

    @property
    def stop_price(self) -> float:
        return self.result.stop_price

    @property
    def sl_dist_pct(self) -> float:
        return self.result.sl_dist_pct

    @property
    def exit_time(self) -> datetime | None:
        return self.result.exit_time

    @property
    def exit_reason(self) -> str:
        return self.result.exit_reason

    @property
    def result_pct(self) -> float | None:
        return self.result.result_pct

    @property
    def realized_pct(self) -> float:
        """Что уже зафиксировано. У открытой сделки result_pct пуст, а эта
        величина показывает взятые тейки — иначе открытые сделки пропадают
        из картины целиком."""
        return self.result.realized_pct

    @property
    def r_multiple(self) -> float | None:
        return self.result.r_multiple

    @property
    def mae_pct(self) -> float:
        return self.result.mae_pct

    @property
    def mfe_pct(self) -> float:
        return self.result.mfe_pct

    @property
    def bars_held(self) -> int:
        return self.result.bars_held

    @property
    def is_closed(self) -> bool:
        return self.result.is_closed

    @property
    def reason_label(self) -> str:
        return self.result.reason_label

    def as_row(self) -> dict:
        result = self.result
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "direction": self.direction.value,
            "degree": self.degree,
            "signal_time": self.signal_time.isoformat(),
            "entry_time": result.entry_time.isoformat(),
            "entry_price": result.entry_price,
            "entry_rsi": round(self.entry_rsi, 2),
            "stop_price": round(result.stop_price, 8),
            "sl_dist_pct": round(result.sl_dist_pct, 4),
            "tp1_price": round(result.tp_prices[0], 8),
            "tp2_price": round(result.tp_prices[1], 8),
            "tp3_price": round(result.tp_prices[2], 8),
            "tp1_time": _iso(result.tp1_time),
            "tp2_time": _iso(result.tp2_time),
            "tp3_time": _iso(result.tp3_time),
            "exit_time": _iso(result.exit_time),
            "exit_reason": result.exit_reason,
            "result_pct": _round(result.result_pct, 3),
            "realized_pct": round(result.realized_pct, 3),
            "r_multiple": _round(result.r_multiple, 3),
            "mae_pct": round(result.mae_pct, 3),
            "mfe_pct": round(result.mfe_pct, 3),
            "bars_held": result.bars_held,
            "first_in_series": int(self.first_in_series),
        }


def _iso(moment: datetime | None) -> str:
    return moment.isoformat() if moment is not None else ""


def _round(value: float | None, digits: int):
    return round(value, digits) if value is not None else ""


# --- прогон -----------------------------------------------------------------


def evaluate(
    df: pd.DataFrame,
    signals: list[Signal],
    *,
    rules: TradeRules,
    subbars: SubBarIndex | None = None,
) -> list[Trade]:
    """Превращает дивергенции в сделки по правилам лесенки.

    Сигналы без точки экстремума (signal.pivot) при sl_set=low пропускаются:
    структурный стоп ставить не от чего. Такие сигналы детектор не выдаёт,
    проверка — страховка от подсунутых вручную данных.
    """
    divergences = [
        s for s in signals
        if s.type in DIVERGENCE_TYPES and s.confirmation_time is not None
    ]
    if not divergences:
        return []

    df = df.reset_index(drop=True)
    bars, times = build_bars(df)
    position = {ts: i for i, ts in enumerate(times)}

    trades: list[Trade] = []
    for signal in sorted(divergences, key=lambda s: s.confirmation_time):
        entry_idx = position.get(_as_utc(signal.confirmation_time))
        if entry_idx is None:
            continue
        entry_price = signal.confirmation_price
        if entry_price is None:
            continue
        if rules.sl_set == "low" and signal.pivot is None:
            continue

        entry = TradeEntry(
            time=times[entry_idx],
            price=float(entry_price),
            pivot_price=float(signal.pivot.price) if signal.pivot is not None else None,
        )
        try:
            result = simulate_trade(
                entry,
                signal.direction,
                bars[entry_idx + 1:],
                rules,
                subbars=subbars,
            )
        except ValueError as exc:
            # Инвариант «вход строго по нужную сторону от стопа» держится
            # построением фрактала. Если он всё же нарушен — это данные,
            # а не повод ронять восьмилетний прогон: пропускаем и говорим вслух.
            print(
                f"  пропущен сигнал {signal.symbol} {signal.timeframe.value} "
                f"{signal.candle_time.isoformat()}: {exc}",
                file=sys.stderr,
            )
            continue
        trades.append(
            Trade(
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                direction=signal.direction,
                degree=signal.degree,
                signal_time=signal.candle_time,
                entry_rsi=float(signal.rsi),
                result=result,
            )
        )

    return mark_series(trades)


def mark_series(trades: list[Trade]) -> list[Trade]:
    """Проставляет first_in_series: на момент входа не было ни одной открытой
    сделки того же направления по этой паре symbol+timeframe."""
    marked: list[Trade] = []
    live: dict[tuple[str, str, str], list[tuple[datetime, datetime | None]]] = {}
    for trade in sorted(trades, key=lambda t: t.entry_time):
        key = (trade.symbol, trade.timeframe.value, trade.direction.value)
        spans = live.setdefault(key, [])
        overlapping = any(
            start <= trade.entry_time and (end is None or end > trade.entry_time)
            for start, end in spans
        )
        spans.append((trade.entry_time, trade.exit_time))
        marked.append(
            Trade(
                symbol=trade.symbol,
                timeframe=trade.timeframe,
                direction=trade.direction,
                degree=trade.degree,
                signal_time=trade.signal_time,
                entry_rsi=trade.entry_rsi,
                result=trade.result,
                first_in_series=not overlapping,
            )
        )
    return marked


def _as_utc(moment: datetime) -> datetime:
    """Ключ для поиска бара: обычный datetime в UTC.

    Времена баров приходят из pandas, времена сигналов — из детектора;
    приводим и то, и другое к одному виду, иначе словарь не совпадёт.
    """
    stamp = pd.Timestamp(moment)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


# --- сводка -----------------------------------------------------------------


def aggregate(trades: list[Trade]) -> dict:
    """Сводка по списку сделок. Незакрытые в средние и винрейт не попадают."""
    closed = [t for t in trades if t.is_closed and t.result_pct is not None]
    results = [t.result_pct for t in closed]
    r_values = [t.r_multiple for t in closed if t.r_multiple is not None]

    shares: dict[str, float] = {}
    counts: dict[str, int] = {}
    for reason in REASON_ORDER:
        n = sum(1 for t in closed if t.exit_reason == reason)
        counts[reason] = n
        shares[reason] = (n / len(closed) * 100.0) if closed else 0.0

    return {
        "count": len(closed),
        "open": len(trades) - len(closed),
        "total": len(trades),
        # Среднее зафиксированного по ВСЕМ сделкам, включая открытые. При
        # далёком tp3 почти каждый победитель остаётся открытым, и среднее
        # по закрытым состоит из одних убытков.
        "realized_all": _mean([t.realized_pct for t in trades]),
        "result": _mean(results),
        "median": _median(results),
        "r": _mean(r_values),
        "wins": sum(1 for v in results if v > 0),
        "win_rate": (sum(1 for v in results if v > 0) / len(results) * 100.0)
        if results else None,
        "mfe": _mean([t.mfe_pct for t in closed]),
        "mae": _mean([t.mae_pct for t in closed]),
        "best": max(results) if results else None,
        "worst": min(results) if results else None,
        "bars": _mean([float(t.bars_held) for t in closed]),
        "sl_dist": _mean([t.sl_dist_pct for t in closed]),
        "reason_counts": counts,
        "reason_shares": shares,
    }


#: Корзины по дистанции стопа. Главный новый разрез отчёта: показывает,
#: зависит ли результат от того, насколько далеко оказался структурный стоп.
SL_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("до 2%", 0.0, 2.0),
    ("2–4%", 2.0, 4.0),
    ("4–7%", 4.0, 7.0),
    ("больше 7%", 7.0, float("inf")),
)


def bucket_of(sl_dist_pct: float) -> str:
    for label, low, high in SL_BUCKETS:
        if low <= sl_dist_pct < high:
            return label
    return SL_BUCKETS[-1][0]


def group_by_sl_distance(trades: list[Trade]) -> dict[str, list[Trade]]:
    groups: dict[str, list[Trade]] = {label: [] for label, _, _ in SL_BUCKETS}
    for trade in trades:
        groups[bucket_of(trade.sl_dist_pct)].append(trade)
    return groups


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
