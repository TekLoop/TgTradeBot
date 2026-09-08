"""ШАБЛОН нового провайдера.

Как добавить источник данных:
  1. Скопировать этот файл в app/data/<myprovider>.py и переименовать класс.
  2. Заполнить name, native_timeframes, fetch_native.
  3. Зарегистрировать в app/data/registry.py (PROVIDER_FACTORIES).
  4. Описать в config.yaml секцию providers.<name> и указать provider: <name>
     у нужных активов.

Больше ничего в коде править не нужно: ресемплинг недостающих ТФ, кэш,
троттлинг, ретраи и отсечение незакрытой свечи живут в базовом классе.
"""

from __future__ import annotations

import pandas as pd

from app.core.timeframes import Timeframe
from app.data.base import DataProvider
from app.data.http import ProviderError


class TemplateProvider(DataProvider):
    name = "template"
    max_native_limit = 500  # максимум баров за один запрос к вашему API

    def __init__(self, http, base_url: str = "https://api.example.com") -> None:
        super().__init__(http)
        self.base_url = base_url.rstrip("/")

    @property
    def native_timeframes(self) -> frozenset[Timeframe]:
        # Перечислите только то, что API отдаёт САМ.
        # Всё остальное из поддерживаемого списка соберётся ресемплингом,
        # если найдётся нативный ТФ, делящий целевой нацело.
        return frozenset({Timeframe.H1, Timeframe.D1})

    def supports_symbol(self, symbol: str) -> bool:
        # Например: return symbol.upper().endswith("USDT")
        return True

    async def fetch_native(
        self, symbol: str, timeframe: Timeframe, limit: int
    ) -> pd.DataFrame:
        """Должен вернуть DataFrame с колонками
        [open_time, open, high, low, close, volume], только ЗАКРЫТЫЕ свечи.
        open_time — момент ОТКРЫТИЯ свечи в UTC.
        """
        raise ProviderError(
            "TemplateProvider — это заготовка. Реализуйте fetch_native()."
        )

        # Пример типовой реализации:
        #
        # payload = await self.http.get_json(
        #     f"{self.base_url}/candles",
        #     {"symbol": symbol, "tf": timeframe.value, "limit": limit},
        # )
        # rows = [
        #     {
        #         "open_time": pd.to_datetime(item["t"], unit="s", utc=True),
        #         "open": float(item["o"]),
        #         "high": float(item["h"]),
        #         "low": float(item["l"]),
        #         "close": float(item["c"]),
        #         "volume": float(item["v"]),
        #     }
        #     for item in payload["data"]
        # ]
        # return pd.DataFrame(rows)
