"""CoinAPI (REST v1).

Эндпоинт: GET https://rest.coinapi.io/v1/ohlcv/{symbol_id}/latest
          ?period_id=1HRS&limit=N        + заголовок X-CoinAPI-Key
Поля ответа: time_period_start, time_period_end, price_open, price_high,
             price_low, price_close, volume_traded.

ВАЖНО / ЧЕГО Я НЕ ГАРАНТИРУЮ:
  * symbol_id у CoinAPI — не тикер, а составной идентификатор вида
    EXCHANGE_SPOT_BASE_QUOTE (напр. BINANCE_SPOT_BTC_USDT). Соответствие
    «тикер в конфиге → symbol_id» задаётся вручную в config.yaml
    (providers.coinapi.symbol_map). Проверьте реальные идентификаторы через
    GET /v1/symbols?filter_symbol_id=... — угадывать их нельзя.
  * Доступность токенизированных акций S&P 500 (AAPLX и т.п.) зависит от
    вашего тарифа и от того, какие площадки CoinAPI индексирует. Перед
    прод-запуском проверьте символ вручную одним curl-ом (см. README).
  * Набор period_id: 1HRS/2HRS/3HRS/4HRS/1DAY подтверждены документацией.
    Недельный период я НЕ использую нативно (неочевидна привязка к
    понедельнику) — 1W собирается ресемплингом из 1DAY.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.core.timeframes import Timeframe
from app.data.base import DataProvider
from app.data.http import ProviderError

log = logging.getLogger(__name__)

_PERIODS: dict[Timeframe, str] = {
    Timeframe.H1: "1HRS",
    Timeframe.H2: "2HRS",
    Timeframe.H3: "3HRS",
    Timeframe.H4: "4HRS",
    Timeframe.D1: "1DAY",
}


class CoinAPIProvider(DataProvider):
    name = "coinapi"
    max_native_limit = 1000

    def __init__(
        self,
        http,
        base_url: str = "https://rest.coinapi.io/v1",
        symbol_map: dict[str, str] | None = None,
        strict_symbols: bool = False,
    ) -> None:
        super().__init__(http)
        self.base_url = base_url.rstrip("/")
        self.symbol_map = {k.upper(): v for k, v in (symbol_map or {}).items()}
        self.strict_symbols = strict_symbols

    @property
    def native_timeframes(self) -> frozenset[Timeframe]:
        return frozenset(_PERIODS)

    def supports_symbol(self, symbol: str) -> bool:
        if not self.strict_symbols:
            return True
        return symbol.upper() in self.symbol_map

    def resolve_symbol_id(self, symbol: str) -> str:
        key = symbol.upper()
        if key in self.symbol_map:
            return self.symbol_map[key]
        log.warning(
            "coinapi: для %s нет записи в symbol_map, отправляю тикер как symbol_id — "
            "скорее всего API вернёт 400. Добавьте маппинг в config.yaml",
            symbol,
        )
        return symbol

    async def fetch_native(
        self, symbol: str, timeframe: Timeframe, limit: int
    ) -> pd.DataFrame:
        period = _PERIODS.get(timeframe)
        if period is None:  # pragma: no cover
            raise ProviderError(f"coinapi: нет нативного period_id для {timeframe}")

        symbol_id = self.resolve_symbol_id(symbol)
        url = f"{self.base_url}/ohlcv/{symbol_id}/latest"
        params = {
            "period_id": period,
            "limit": int(min(max(limit, 10), self.max_native_limit)),
            "include_empty_items": "false",
        }
        payload = await self.http.get_json(url, params)
        if not isinstance(payload, list):
            raise ProviderError(f"coinapi: неожиданный ответ для {symbol}: {payload!r}")

        now = pd.Timestamp.now(tz="UTC")
        rows = []
        for item in payload:
            period_end = pd.to_datetime(item["time_period_end"], utc=True)
            if period_end > now:
                continue  # свеча ещё не закрылась
            rows.append(
                {
                    "open_time": pd.to_datetime(item["time_period_start"], utc=True),
                    "open": float(item["price_open"]),
                    "high": float(item["price_high"]),
                    "low": float(item["price_low"]),
                    "close": float(item["price_close"]),
                    "volume": float(item.get("volume_traded") or 0.0),
                }
            )
        return pd.DataFrame(
            rows, columns=["open_time", "open", "high", "low", "close", "volume"]
        )
