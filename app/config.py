"""Конфигурация: YAML (что отслеживаем и с какими порогами) + .env (секреты).

Добавление нового актива = правка config.yaml, код трогать не нужно.

ПАРАМЕТРЫ НА ТАЙМФРЕЙМ. Механизм общий для ВСЕХ полей ParamsConfig:
секция `params` в YAML — база, секция `params_by_timeframe.<TF>` — оверлей,
поверх ложатся оверрайды из БД (глобальные, затем на таймфрейм).
Порядок наложения снизу вверх:

    1. дефолты ParamsConfig
    2. секция params из YAML          ─┐ вместе дают базовый ParamsConfig
    3. секция params_by_timeframe[tf]  │
    4. оверрайды из БД, глобальные     │
    5. оверрайды из БД для этого ТФ   ─┘ собирается build_params()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.analysis.signals import SignalParams
from app.core.timeframes import Timeframe


class ParamsConfig(BaseModel):
    """Пороги детектора + параметры опроса.

    Для трёх параметров расстояний действует соглашение «0 = выключено»,
    поэтому у них ge=0, а не ge=2.
    """

    model_config = ConfigDict(extra="forbid")

    rsi_period: int = Field(14, ge=2, le=200)
    fractal_n: int = Field(2, ge=1, le=20)
    oversold: float = Field(30.0, ge=0, le=100)
    overbought: float = Field(70.0, ge=0, le=100)
    divergence_rsi_max: float = Field(35.0, ge=0, le=100)
    divergence_rsi_min: float = Field(65.0, ge=0, le=100)
    reset_rsi: float = Field(50.0, ge=0, le=100)
    reset_rsi_bear: float = Field(50.0, ge=0, le=100)

    #: возраст опорной точки chain[0], проверяется на каждом баре
    anchor_ttl_bars: int = Field(40, ge=0, le=1000)
    #: расстояние внутри пары, образующей дивергенцию
    max_bars_between_points: int = Field(40, ge=0, le=1000)
    #: минимальное расстояние от chain[-1] до нового пивота
    min_bars_between_points: int = Field(0, ge=0, le=1000)

    bearish_enabled: bool = False
    chain_max_points: int = Field(4, ge=0, le=9)

    # --- Задача 4: extreme-алерты -------------------------------------------
    extreme_alerts_enabled: bool = False
    extreme_oversold: float = Field(15.0, ge=0, le=100)
    extreme_overbought: float = Field(85.0, ge=0, le=100)
    extreme_rearm_rsi_bull: float = Field(30.0, ge=0, le=100)
    extreme_rearm_rsi_bear: float = Field(70.0, ge=0, le=100)
    extreme_rearm_bars: int = Field(3, ge=0, le=1000)

    # --- Задача 5: алерты по фитилю -----------------------------------------
    wick_alerts_enabled: bool = False
    wick_threshold_pct: float = Field(5.0, ge=0, le=100)
    wick_min_body_pct: float = Field(0.0, ge=0, le=100)

    lookback_bars: int = Field(300, ge=60, le=1000)
    notify_max_age_bars: int = Field(3, ge=0, le=50)
    outcome_timeout_bars: int = Field(120, ge=0, le=5000)

    @model_validator(mode="after")
    def _check_zones(self) -> "ParamsConfig":
        if self.divergence_rsi_max < self.oversold:
            raise ValueError("divergence_rsi_max должен быть >= oversold")
        if self.divergence_rsi_min > self.overbought:
            raise ValueError("divergence_rsi_min должен быть <= overbought")
        return self

    @model_validator(mode="after")
    def _check_chain_cap(self) -> "ParamsConfig":
        if self.chain_max_points == 1:
            raise ValueError(
                "chain_max_points = 1 бессмысленно: цепочка из одной точки "
                "дивергенцию образовать не может. Допустимо 0 (без потолка) "
                "или значение от 2 до 9"
            )
        return self

    @model_validator(mode="after")
    def _check_distances(self) -> "ParamsConfig":
        # 1. Два пивота-экстремума физически не могут стоять ближе,
        #    чем fractal_n + 1 баров друг от друга.
        if 0 < self.min_bars_between_points < self.fractal_n + 1:
            raise ValueError(
                f"min_bars_between_points должен быть 0 или >= fractal_n + 1 "
                f"(= {self.fractal_n + 1}): при меньшем значении фильтр "
                f"не работает"
            )
        # 2. Правило нестрогое: пара на расстоянии ровно anchor_ttl_bars ещё
        #    жива (TTL убивает при строгом превышении) и проходит фильтр.
        if (
            self.max_bars_between_points
            and self.anchor_ttl_bars
            and self.max_bars_between_points > self.anchor_ttl_bars
        ):
            raise ValueError(
                f"max_bars_between_points ({self.max_bars_between_points}) должен "
                f"быть <= anchor_ttl_bars ({self.anchor_ttl_bars}): иначе часть "
                f"диапазона max недостижима — цепочка умрёт раньше"
            )
        # 3. Движок физически не видит дальше окна.
        if self.anchor_ttl_bars:
            visible = self.lookback_bars - self.rsi_period - 2 * self.fractal_n
            if self.anchor_ttl_bars >= visible:
                raise ValueError(
                    f"anchor_ttl_bars ({self.anchor_ttl_bars}) должен быть < "
                    f"lookback_bars - rsi_period - 2*fractal_n (= {visible}): "
                    f"движок не видит дальше окна"
                )
        # 4. Минимум строго меньше максимума, иначе допустимых пар нет.
        if (
            self.min_bars_between_points
            and self.max_bars_between_points
            and self.min_bars_between_points >= self.max_bars_between_points
        ):
            raise ValueError(
                f"min_bars_between_points ({self.min_bars_between_points}) должен "
                f"быть < max_bars_between_points ({self.max_bars_between_points})"
            )
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
            anchor_ttl_bars=self.anchor_ttl_bars,
            max_bars_between_points=self.max_bars_between_points,
            min_bars_between_points=self.min_bars_between_points,
            bearish_enabled=self.bearish_enabled,
            chain_max_points=self.chain_max_points,
            extreme_alerts_enabled=self.extreme_alerts_enabled,
            extreme_oversold=self.extreme_oversold,
            extreme_overbought=self.extreme_overbought,
            extreme_rearm_rsi_bull=self.extreme_rearm_rsi_bull,
            extreme_rearm_rsi_bear=self.extreme_rearm_rsi_bear,
            extreme_rearm_bars=self.extreme_rearm_bars,
            wick_alerts_enabled=self.wick_alerts_enabled,
            wick_threshold_pct=self.wick_threshold_pct,
            wick_min_body_pct=self.wick_min_body_pct,
        )


#: Параметры, которые можно менять из Telegram командой /params.
#: Собирается из полей модели, поэтому новые поля подхватываются сами.
EDITABLE_PARAMS = tuple(ParamsConfig.model_fields.keys())


def format_validation_error(exc: Exception) -> str:
    """Человекочитаемая причина отказа валидации: поле + сообщение."""
    if not isinstance(exc, ValidationError):
        return str(exc)
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "конфиг"
        message = str(error.get("msg", "")).removeprefix("Value error, ")
        parts.append(f"{location}: {message}")
    return "; ".join(parts) or str(exc)


def build_params(base: ParamsConfig, *layers: dict[str, Any] | None) -> ParamsConfig:
    """Собирает конфиг: база + слои оверрайдов, слева направо.

    Ключи, которых нет в EDITABLE_PARAMS, игнорируются молча — так переживший
    миграцию мусор из старой БД не валит опрос. Результат прогоняется через
    ПОЛНУЮ валидацию: правила связывают параметры между собой, и проверка
    одного значения в отрыве ничего не гарантирует.
    """
    values = base.model_dump()
    for layer in layers:
        if not layer:
            continue
        for key, value in layer.items():
            if key in EDITABLE_PARAMS:
                values[key] = value
    return ParamsConfig.model_validate(values)


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
    #: Оверлеи на таймфрейм. Объявлено явным полем: extra="forbid" не пропустил
    #: бы новую секцию, и конфиг просто не загрузился бы.
    params_by_timeframe: dict[Timeframe, dict[str, Any]] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)

    @field_validator("params_by_timeframe", mode="before")
    @classmethod
    def _parse_by_tf(cls, v: Any) -> dict:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("params_by_timeframe должен быть словарём вида <TF>: {...}")
        out: dict = {}
        for raw_tf, overlay in v.items():
            tf = Timeframe.parse(raw_tf)
            if overlay is None:
                overlay = {}
            if not isinstance(overlay, dict):
                raise ValueError(
                    f"params_by_timeframe.{tf.value}: ожидался набор «ключ: значение»"
                )
            out[tf] = dict(overlay)
        return out

    @model_validator(mode="after")
    def _check_providers(self) -> "AppConfig":
        for asset in self.assets:
            if asset.provider not in self.providers:
                raise ValueError(
                    f"Актив {asset.symbol}: провайдер '{asset.provider}' "
                    f"не описан в секции providers"
                )
        return self

    @model_validator(mode="after")
    def _check_params_by_timeframe(self) -> "AppConfig":
        """Каждая секция валидируется В СОБРАННОМ ВИДЕ, после наложения на базу.

        Битая секция = отказ запуска с указанием таймфрейма и поля. Стартовать
        молча с частично применённым конфигом нельзя.
        """
        for tf, overlay in self.params_by_timeframe.items():
            unknown = sorted(k for k in overlay if k not in EDITABLE_PARAMS)
            if unknown:
                raise ValueError(
                    f"params_by_timeframe.{tf.value}: неизвестные параметры: "
                    f"{', '.join(unknown)}"
                )
            try:
                build_params(self.params, overlay)
            except ValidationError as exc:
                raise ValueError(
                    f"params_by_timeframe.{tf.value}: {format_validation_error(exc)}"
                ) from exc
        return self

    def params_for(self, tf: Timeframe | None) -> ParamsConfig:
        """База + YAML-оверлей таймфрейма, без оверрайдов из БД."""
        if tf is None:
            return self.params
        return build_params(self.params, self.params_by_timeframe.get(tf))


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
