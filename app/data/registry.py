"""Сборка провайдеров по конфигу. Единственное место, которое нужно
дополнить при добавлении нового источника данных."""

from __future__ import annotations

import logging
from typing import Callable

from app.config import ProviderConfig
from app.data.base import DataProvider
from app.data.binance import BinanceProvider
from app.data.coinapi import CoinAPIProvider
from app.data.http import HttpClient, Throttler
from app.data.template_provider import TemplateProvider

log = logging.getLogger(__name__)

ProviderFactory = Callable[[ProviderConfig, "SecretsLike"], DataProvider]


class SecretsLike:  # pragma: no cover — только для типизации
    coinapi_key: str


def _http(name: str, cfg: ProviderConfig, headers: dict[str, str] | None = None) -> HttpClient:
    return HttpClient(
        name=name,
        headers=headers,
        timeout=cfg.timeout_seconds,
        retries=cfg.retries,
        throttler=Throttler(rate=cfg.requests_per_minute, per_seconds=60.0),
    )


def _build_binance(cfg: ProviderConfig, secrets) -> DataProvider:
    return BinanceProvider(
        _http("binance", cfg),
        base_url=cfg.base_url or "https://api.binance.com",
    )


def _build_coinapi(cfg: ProviderConfig, secrets) -> DataProvider:
    key = getattr(secrets, "coinapi_key", "") or ""
    if not key:
        log.warning(
            "coinapi: не задан COINAPI_KEY — запросы будут отклоняться сервером"
        )
    return CoinAPIProvider(
        _http("coinapi", cfg, headers={"X-CoinAPI-Key": key, "Accept": "application/json"}),
        base_url=cfg.base_url or "https://rest.coinapi.io/v1",
        symbol_map=cfg.symbol_map,
        strict_symbols=cfg.strict_symbols,
    )


def _build_template(cfg: ProviderConfig, secrets) -> DataProvider:
    return TemplateProvider(_http("template", cfg), base_url=cfg.base_url or "https://api.example.com")


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "binance": _build_binance,
    "coinapi": _build_coinapi,
    "template": _build_template,
}


def build_providers(
    provider_configs: dict[str, ProviderConfig], secrets
) -> dict[str, DataProvider]:
    providers: dict[str, DataProvider] = {}
    for name, cfg in provider_configs.items():
        factory = PROVIDER_FACTORIES.get(name)
        if factory is None:
            log.error(
                "Неизвестный провайдер '%s' в конфиге — пропускаю. Доступны: %s",
                name, ", ".join(sorted(PROVIDER_FACTORIES)),
            )
            continue
        try:
            providers[name] = factory(cfg, secrets)
            log.info("Провайдер %s инициализирован", name)
        except Exception:  # noqa: BLE001 — падение одного провайдера не валит бот
            log.exception("Не удалось инициализировать провайдера %s", name)
    return providers


async def close_providers(providers: dict[str, DataProvider]) -> None:
    for name, provider in providers.items():
        try:
            await provider.close()
        except Exception:  # noqa: BLE001
            log.exception("Ошибка при закрытии провайдера %s", name)
