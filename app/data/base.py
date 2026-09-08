"""Абстракция источника свечей.

Контракт: get_klines(symbol, timeframe, limit) -> DataFrame
[open_time, open, high, low, close, volume], только ЗАКРЫТЫЕ свечи,
open_time — tz-aware UTC, сортировка по возрастанию.

Провайдер объявляет native_timeframes и supports_symbol(). Если таймфрейм
не поддерживается нативно — базовый класс сам подберёт младший нативный ТФ,
который делит целевой нацело, и соберёт свечи ресемплингом.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone

import pandas as pd

from app.core.timeframes import Timeframe
from app.data.http import HttpClient, ProviderError
from app.data.resampling import normalize_ohlcv, resample_ohlcv

log = logging.getLogger(__name__)


class UnsupportedSymbolError(ProviderError):
    pass


class UnsupportedTimeframeError(ProviderError):
    pass


class _CandleCache:
    """Кэш свечей в памяти. Запись валидна, пока не закрылась новая свеча ТФ."""

    def __init__(self, max_items: int = 512) -> None:
        self._items: OrderedDict[tuple[str, Timeframe], tuple[float, pd.DataFrame]] = (
            OrderedDict()
        )
        self.max_items = max_items
        self.hits = 0
        self.misses = 0

    def get(self, symbol: str, tf: Timeframe, limit: int) -> pd.DataFrame | None:
        key = (symbol, tf)
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        fetched_at, df = item
        fetched_dt = datetime.fromtimestamp(fetched_at, tz=timezone.utc)
        if tf.next_close(fetched_dt) <= datetime.now(timezone.utc):
            self._items.pop(key, None)  # с момента запроса закрылась новая свеча
            self.misses += 1
            return None
        if len(df) < limit:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return df.tail(limit).reset_index(drop=True)

    def put(self, symbol: str, tf: Timeframe, df: pd.DataFrame) -> None:
        key = (symbol, tf)
        self._items[key] = (time.time(), df)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def invalidate(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._items.clear()
            return
        for key in [k for k in self._items if k[0] == symbol]:
            self._items.pop(key, None)


class DataProvider(ABC):
    """Базовый класс провайдера данных."""

    name: str = "abstract"
    max_native_limit: int = 1000

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self._cache = _CandleCache()

    # --- то, что обязан реализовать конкретный провайдер --------------------

    @property
    @abstractmethod
    def native_timeframes(self) -> frozenset[Timeframe]:
        """Таймфреймы, которые провайдер отдаёт напрямую."""

    @abstractmethod
    async def fetch_native(
        self, symbol: str, timeframe: Timeframe, limit: int
    ) -> pd.DataFrame:
        """Запрос к API. Должен вернуть только закрытые свечи."""

    def supports_symbol(self, symbol: str) -> bool:
        """Переопределяется, если провайдер знает свой список инструментов."""
        return True

    # --- общая логика -------------------------------------------------------

    def supports_timeframe(self, timeframe: Timeframe) -> bool:
        return (
            timeframe in self.native_timeframes
            or self.base_timeframe_for(timeframe) is not None
        )

    def base_timeframe_for(self, timeframe: Timeframe) -> Timeframe | None:
        """Младший нативный ТФ, из которого можно собрать целевой."""
        candidates = [tf for tf in self.native_timeframes if tf.divides(timeframe)]
        if not candidates:
            return None
        return max(candidates, key=lambda tf: tf.minutes)

    async def get_klines(
        self, symbol: str, timeframe: Timeframe, limit: int = 300
    ) -> pd.DataFrame:
        if not self.supports_symbol(symbol):
            raise UnsupportedSymbolError(
                f"{self.name}: символ {symbol} не поддерживается"
            )

        if timeframe in self.native_timeframes:
            df = await self._fetch_cached(symbol, timeframe, limit)
            return self._drop_unclosed(df, timeframe)

        base = self.base_timeframe_for(timeframe)
        if base is None:
            raise UnsupportedTimeframeError(
                f"{self.name}: таймфрейм {timeframe.value} недоступен "
                f"(нативные: {sorted(tf.value for tf in self.native_timeframes)})"
            )

        ratio = base.ratio_to(timeframe)
        needed = min((limit + 2) * ratio, self.max_native_limit)
        raw = await self._fetch_cached(symbol, base, needed)
        raw = self._drop_unclosed(raw, base)
        resampled = resample_ohlcv(raw, timeframe, source=base)
        log.debug(
            "%s: %s %s собран ресемплингом из %s (%d → %d баров)",
            self.name, symbol, timeframe.value, base.value, len(raw), len(resampled),
        )
        return resampled.tail(limit).reset_index(drop=True)

    async def _fetch_cached(
        self, symbol: str, timeframe: Timeframe, limit: int
    ) -> pd.DataFrame:
        cached = self._cache.get(symbol, timeframe, limit)
        if cached is not None:
            return cached
        raw = await self.fetch_native(symbol, timeframe, limit)
        df = normalize_ohlcv(raw)
        self._cache.put(symbol, timeframe, df)
        return df

    @staticmethod
    def _drop_unclosed(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Страховка: выкидываем незакрытую свечу, если API её всё-таки отдал."""
        if df.empty:
            return df
        now = pd.Timestamp.now(tz="UTC")
        closed = df["open_time"] + timeframe.duration <= now
        return df.loc[closed].reset_index(drop=True)

    def invalidate_cache(self, symbol: str | None = None) -> None:
        self._cache.invalidate(symbol)

    async def close(self) -> None:
        await self.http.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name}>"
