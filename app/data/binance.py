"""Binance Spot (публичные данные, ключ не требуется).

Эндпоинт: GET {base_url}/api/v3/klines?symbol=&interval=&limit=
Формат ответа — массив массивов:
[open_time_ms, open, high, low, close, volume, close_time_ms, quote_volume,
 trades, taker_base, taker_quote, ignore]

Нативные интервалы: 1h, 2h, 4h, 1d, 1w. Интервала 3h у Binance нет —
он собирается ресемплингом из 1h базовым классом.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.timeframes import Timeframe
from app.data.base import DataProvider
from app.data.http import ProviderError

log = logging.getLogger(__name__)

_INTERVALS: dict[Timeframe, str] = {
    Timeframe.H1: "1h",
    Timeframe.H2: "2h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
    Timeframe.W1: "1w",
}


class BinanceProvider(DataProvider):
    name = "binance"
    max_native_limit = 1000

    def __init__(self, http, base_url: str = "https://api.binance.com") -> None:
        super().__init__(http)
        self.base_url = base_url.rstrip("/")

    @property
    def native_timeframes(self) -> frozenset[Timeframe]:
        return frozenset(_INTERVALS)

    async def fetch_native(
        self, symbol: str, timeframe: Timeframe, limit: int
    ) -> pd.DataFrame:
        interval = _INTERVALS.get(timeframe)
        if interval is None:  # pragma: no cover — защита от рассинхрона карты
            raise ProviderError(f"binance: нет нативного интервала для {timeframe}")

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(min(max(limit, 10), self.max_native_limit)),
        }
        payload = await self.http.get_json(f"{self.base_url}/api/v3/klines", params)
        if not isinstance(payload, list):
            raise ProviderError(f"binance: неожиданный ответ для {symbol}: {payload!r}")

        rows = []
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        for item in payload:
            close_time_ms = int(item[6])
            # close_time у Binance = open_time + interval - 1ms,
            # поэтому незакрытая свеча отсекается строгим сравнением.
            if close_time_ms >= now_ms:
                continue
            rows.append(
                {
                    "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        return pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
