"""Офлайн-оценка сигналов: что было с ценой после каждой дивергенции.

Чистая функция от датафрейма и списка сигналов — ни БД, ни Telegram, ни сети.
Правила ровно те же, что у живого бота (app/bot/outcomes.py):

    вход  = close свечи, подтвердившей фрактал;
    выход = следующая дивергенция (любого направления) либо таймаут по барам;
    ход в пользу / против сигнала считается от цены входа, минимум — ноль,
    потому что сама свеча входа тоже участвует в отсчёте.

Проценты всегда «в сторону сигнала»: для медвежьей падение цены — это плюс.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from app.analysis.signals import DIVERGENCE_TYPES, Direction, Signal
from app.core.timeframes import Timeframe

#: Причины закрытия сделки.
REASON_REVERSE = "reverse"     # следующая дивергенция противоположного направления
REASON_SAME = "same"           # следующая дивергенция того же направления
REASON_TIMEOUT = "timeout"     # истёк лимит баров
REASON_OPEN = "open"           # данные кончились раньше, сделка не закрыта

REASON_LABELS = {
    REASON_REVERSE: "обратная дивергенция",
    REASON_SAME: "дивергенция того же направления",
    REASON_TIMEOUT: "таймаут",
    REASON_OPEN: "не закрыта (конец данных)",
}


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    timeframe: Timeframe
    direction: Direction
    degree: int
    signal_time: datetime      # свеча экстремума
    entry_time: datetime       # свеча подтверждения
    entry_price: float
    entry_rsi: float
    exit_time: datetime | None
    exit_price: float | None
    exit_reason: str
    bars_held: int
    max_high: float
    max_high_time: datetime
    min_low: float
    min_low_time: datetime

    @property
    def sign(self) -> float:
        return 1.0 if self.direction is Direction.BULL else -1.0

    def pnl_pct(self, price: float | None) -> float | None:
        if price is None or not self.entry_price:
            return None
        return self.sign * (price / self.entry_price - 1.0) * 100.0

    @property
    def favorable_price(self) -> float:
        return self.max_high if self.direction is Direction.BULL else self.min_low

    @property
    def adverse_price(self) -> float:
        return self.min_low if self.direction is Direction.BULL else self.max_high

    @property
    def favorable_time(self) -> datetime:
        return self.max_high_time if self.direction is Direction.BULL else self.min_low_time

    @property
    def favorable_pct(self) -> float:
        return self.pnl_pct(self.favorable_price)  # type: ignore[return-value]

    @property
    def adverse_pct(self) -> float:
        return self.pnl_pct(self.adverse_price)  # type: ignore[return-value]

    @property
    def result_pct(self) -> float | None:
        return self.pnl_pct(self.exit_price)

    @property
    def is_closed(self) -> bool:
        return self.exit_reason != REASON_OPEN

    @property
    def reason_label(self) -> str:
        return REASON_LABELS.get(self.exit_reason, self.exit_reason)

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "direction": self.direction.value,
            "degree": self.degree,
            "signal_time": self.signal_time.isoformat(),
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "entry_rsi": round(self.entry_rsi, 2),
            "exit_time": self.exit_time.isoformat() if self.exit_time else "",
            "exit_price": self.exit_price if self.exit_price is not None else "",
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "favorable_pct": round(self.favorable_pct, 3),
            "adverse_pct": round(self.adverse_pct, 3),
            "result_pct": round(self.result_pct, 3) if self.result_pct is not None else "",
            "bars_to_favorable": self.bars_to_favorable,
        }

    @property
    def bars_to_favorable(self) -> int:
        return max(0, int((self.favorable_time - self.entry_time).total_seconds()
                          // (self.timeframe.minutes * 60)))


def evaluate(
    df: pd.DataFrame,
    signals: list[Signal],
    *,
    timeout_bars: int = 0,
) -> list[Trade]:
    """Превращает дивергенции в сделки. timeout_bars=0 — без таймаута."""
    divergences = [
        s for s in signals
        if s.type in DIVERGENCE_TYPES and s.confirmation_time is not None
    ]
    if not divergences:
        return []

    df = df.reset_index(drop=True)
    times = pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True))
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    py_times = times.to_pydatetime()
    last_idx = len(df) - 1

    position = {ts: i for i, ts in enumerate(times)}
    entries: list[tuple[int, Signal]] = []
    for signal in sorted(divergences, key=lambda s: s.confirmation_time):
        idx = position.get(pd.Timestamp(signal.confirmation_time))
        if idx is not None:
            entries.append((idx, signal))

    trades: list[Trade] = []
    for order, (entry_idx, signal) in enumerate(entries):
        next_idx: int | None = None
        next_signal: Signal | None = None
        for candidate_idx, candidate in entries[order + 1:]:
            if candidate_idx > entry_idx:
                next_idx, next_signal = candidate_idx, candidate
                break

        timeout_idx = entry_idx + timeout_bars if timeout_bars else None
        exit_idx: int | None = None
        reason = REASON_OPEN

        if next_idx is not None and (timeout_idx is None or next_idx <= timeout_idx):
            exit_idx = next_idx
            reason = (
                REASON_REVERSE
                if next_signal is not None and next_signal.direction is not signal.direction
                else REASON_SAME
            )
        elif timeout_idx is not None and timeout_idx <= last_idx:
            exit_idx = timeout_idx
            reason = REASON_TIMEOUT

        # Отсчёт от цены входа: свеча входа даёт затравку, дальше идут только
        # бары строго после неё. Ровно так же считает живой бот, поэтому
        # «ход в пользу» не бывает отрицательным, а «ход против» — плюсовым.
        stop = exit_idx if exit_idx is not None else last_idx
        entry_price = float(closes[entry_idx])
        max_high, high_pos = entry_price, entry_idx
        min_low, low_pos = entry_price, entry_idx
        if stop > entry_idx:
            segment = slice(entry_idx + 1, stop + 1)
            candidate_high = entry_idx + 1 + int(np.argmax(highs[segment]))
            candidate_low = entry_idx + 1 + int(np.argmin(lows[segment]))
            if highs[candidate_high] > max_high:
                max_high, high_pos = float(highs[candidate_high]), candidate_high
            if lows[candidate_low] < min_low:
                min_low, low_pos = float(lows[candidate_low]), candidate_low

        trades.append(
            Trade(
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                direction=signal.direction,
                degree=signal.degree,
                signal_time=signal.candle_time,
                entry_time=py_times[entry_idx],
                entry_price=entry_price,
                entry_rsi=float(signal.rsi),
                exit_time=py_times[exit_idx] if exit_idx is not None else None,
                exit_price=float(closes[exit_idx]) if exit_idx is not None else None,
                exit_reason=reason,
                bars_held=stop - entry_idx,
                max_high=max_high,
                max_high_time=py_times[high_pos],
                min_low=min_low,
                min_low_time=py_times[low_pos],
            )
        )

    return trades


def aggregate(trades: list[Trade]) -> dict:
    """Сводка по списку сделок. Незакрытые в средние не попадают."""
    closed = [t for t in trades if t.is_closed and t.result_pct is not None]
    results = [t.result_pct for t in closed]
    favorable = [t.favorable_pct for t in closed]
    adverse = [t.adverse_pct for t in closed]
    return {
        "count": len(closed),
        "open": len(trades) - len(closed),
        "result": _mean(results),
        "median": _median(results),
        "favorable": _mean(favorable),
        "adverse": _mean(adverse),
        "best": max(favorable) if favorable else None,
        "worst": min(adverse) if adverse else None,
        "wins": sum(1 for v in results if v > 0),
        "win_rate": (sum(1 for v in results if v > 0) / len(results) * 100.0)
        if results else None,
        "bars": _mean([float(t.bars_held) for t in closed]),
        "bars_to_favorable": _mean([float(t.bars_to_favorable) for t in closed]),
    }


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
