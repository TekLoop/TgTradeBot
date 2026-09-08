"""Конфигурация: YAML (что отслеживаем и с какими порогами) + .env (секреты).

Добавление нового актива = правка config.yaml, код трогать не нужно.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.analysis.signals import SignalParams
from app.core.timeframes import Timeframe


class ParamsConfig(BaseModel):
    """Пороги детектора + параметры опроса."""

    model_config = ConfigDict(extra="forbid")

    rsi_period: int = Field(14, ge=2, le=200)
    fractal_n: int = Field(2, ge=1, le=20)
    oversold: float = Field(30.0, ge=0, le=100)
    overbought: float = Field(70.0, ge=0, le=100)
    divergence_rsi_max: float = Field(35.0, ge=0, le=100)
    divergence_rsi_min: float = Field(65.0, ge=0, le=100)
    reset_rsi: float = Field(50.0, ge=0, le=100)
    reset_rsi_bear: float = Field(50.0, ge=0, le=100)
    max_bars_between: int = Field(40, ge=2, le=1000)
    bearish_enabled: bool = False
    lookback_bars: int = Field(300, ge=60, le=1000)
    notify_max_age_bars: int = Field(3, ge=0, le=50)

    @model_validator(mode="after")
    def _check_zones(self) -> "ParamsConfig":
        if self.divergence_rsi_max < self.oversold:
            raise ValueError("divergence_rsi_max должен быть >= oversold")
        if self.divergence_rsi_min > self.overbought:
            raise ValueError("divergence_rsi_min должен быть <= overbought")
        return self

    def to_signal_params(self) -> SignalParams:
        return SignalParams(
            rsi_period=self.rsi_period,
            fractal_n=self.fractal_n,
            oversold=self.oversold,
            overbought=self.overbought,
            divergence_rsi_max=self.divergence_rsi_max,
            divergence_rsi_min=self.divergence_rsi_min,
            reset_rsi=self.reset_rsi,
            reset_rsi_bear=self.reset_rsi_bear,
            max_bars_between=self.max_bars_between,
            bearish_enabled=self.bearish_enabled,
        )


#: Параметры, которые можно менять из Telegram командой /params
EDITABLE_PARAMS = tuple(ParamsConfig.model_fields.keys())


class AssetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    provider: str
    timeframes: list[Timeframe] = Field(default_factory=list)
    enabled: bool = True

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("provider")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("timeframes", mode="before")
    @classmethod
    def _parse_tfs(cls, v: Any) -> list[Timeframe]:
        if v is None:
            return []
        return [Timeframe.parse(item) for item in v]


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    base_url: str | None = None
    requests_per_minute: int = Field(60, ge=1, le=6000)
    timeout_seconds: float = Field(15.0, gt=0)
    retries: int = Field(4, ge=0, le=10)
    symbol_map: dict[str, str] = Field(default_factory=dict)
    strict_symbols: bool = False


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    close_delay_seconds: float = Field(15.0, ge=0, le=600)
    jitter_seconds: float = Field(5.0, ge=0, le=120)
    reconcile_interval_seconds: float = Field(60.0, ge=5, le=3600)
    check_on_startup: bool = True


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_path: str = "data/state.db"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    file: str = "logs/bot.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


class SyncConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: удалять из БД активы, которых больше нет в YAML
    remove_missing_assets: bool = False
    #: перезаписывать пороги из YAML при каждом старте (иначе /params имеет приоритет)
    overwrite_params: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[AssetConfig] = Field(default_factory=list)
    params: ParamsConfig = Field(default_factory=ParamsConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)

    @model_validator(mode="after")
    def _check_providers(self) -> "AppConfig":
        for asset in self.assets:
            if asset.provider not in self.providers:
                raise ValueError(
                    f"Актив {asset.symbol}: провайдер '{asset.provider}' "
                    f"не описан в секции providers"
                )
        return self


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден конфиг: {path.resolve()}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


class Secrets:
    """Секреты только из окружения / .env — в YAML их быть не должно."""

    def __init__(self) -> None:
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.coinapi_key = os.getenv("COINAPI_KEY", "").strip()
        self.default_chat_id = _int_or_none(os.getenv("TELEGRAM_CHAT_ID"))
        self.allowed_user_ids = _int_set(os.getenv("TELEGRAM_ALLOWED_USER_IDS", ""))

    def require_telegram_token(self) -> str:
        if not self.telegram_token:
            raise RuntimeError(
                "Не задан TELEGRAM_BOT_TOKEN (положите его в .env)"
            )
        return self.telegram_token


def _int_or_none(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _int_set(value: str) -> set[int]:
    result: set[int] = set()
    for chunk in value.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            try:
                result.add(int(chunk))
            except ValueError:
                continue
    return result
