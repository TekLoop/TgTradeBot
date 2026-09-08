"""Оркестратор: связывает data/ + analysis/ + bot/.

Один вызов check_unit() = один цикл «получить свечи → посчитать → отправить».
Ошибка одного актива/провайдера логируется и не выходит за пределы метода.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from app.analysis.signals import AnalysisResult, Direction, DivergenceEngine, Signal
from app.bot.notifier import Notifier
from app.bot.storage import Storage, TrackingUnit
from app.config import EDITABLE_PARAMS, ParamsConfig
from app.core.timeframes import Timeframe
from app.data.base import DataProvider

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckOutcome:
    unit: TrackingUnit
    result: AnalysisResult | None = None
    sent: list[Signal] = None  # type: ignore[assignment]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.sent is None:
            self.sent = []


class TrackingService:
    def __init__(
        self,
        storage: Storage,
        providers: dict[str, DataProvider],
        base_params: ParamsConfig,
        notifier: Notifier | None = None,
    ) -> None:
        self.storage = storage
        self.providers = providers
        self.base_params = base_params
        self.notifier = notifier

    # --- параметры ----------------------------------------------------------

    async def effective_params(self) -> ParamsConfig:
        """YAML-дефолты, поверх которых наложены правки из /params."""
        overrides = await self.storage.get_param_overrides()
        merged = self.base_params.model_dump()
        for key, value in overrides.items():
            if key in EDITABLE_PARAMS:
                merged[key] = value
        try:
            return ParamsConfig.model_validate(merged)
        except Exception:  # noqa: BLE001 — битые оверрайды не должны валить опрос
            log.exception("Некорректные оверрайды параметров, использую YAML-дефолты")
            return self.base_params

    # --- основной цикл ------------------------------------------------------

    async def check_unit(
        self, unit: TrackingUnit, *, notify: bool = True
    ) -> CheckOutcome:
        outcome = CheckOutcome(unit=unit)
        provider = self.providers.get(unit.provider)
        if provider is None:
            outcome.error = f"провайдер '{unit.provider}' недоступен"
            log.error("%s: %s", unit.key, outcome.error)
            return outcome

        params = await self.effective_params()
        try:
            df = await provider.get_klines(
                unit.symbol, unit.timeframe, params.lookback_bars
            )
        except Exception as exc:  # noqa: BLE001 — изолируем падение провайдера
            outcome.error = str(exc)
            log.warning("%s: не удалось получить свечи: %s", unit.key, exc)
            return outcome

        if df.empty:
            outcome.error = "провайдер вернул пустой набор свечей"
            log.warning("%s: %s", unit.key, outcome.error)
            return outcome

        try:
            engine = DivergenceEngine(params.to_signal_params())
            result = engine.analyze(df, unit.symbol, unit.timeframe)
        except Exception as exc:  # noqa: BLE001
            outcome.error = f"ошибка анализа: {exc}"
            log.exception("%s: ошибка анализа", unit.key)
            return outcome

        outcome.result = result

        if not notify:
            return outcome

        seeded = await self.storage.is_seeded(unit.symbol, unit.timeframe)

        for signal in result.signals:
            is_new = await self.storage.try_register_alert(
                unit.symbol,
                unit.timeframe,
                signal.type,
                signal.candle_time,
                payload=_signal_payload(signal),
                notified=False,
            )
            if not is_new:
                continue  # такой сигнал на этой свече уже обрабатывался

            fresh = signal.bars_since_confirmation <= params.notify_max_age_bars
            if not seeded:
                log.info(
                    "%s: первичное наполнение истории — %s на %s записан без отправки",
                    unit.key, signal.type.value, signal.candle_time.isoformat(),
                )
                continue
            if not fresh:
                log.info(
                    "%s: сигнал %s на %s слишком старый (%d баров) — не отправляю",
                    unit.key, signal.type.value, signal.candle_time.isoformat(),
                    signal.bars_since_confirmation,
                )
                continue

            if self.notifier is not None and await self.notifier.send_signal(signal):
                await self.storage.mark_notified(
                    unit.symbol, unit.timeframe, signal.type, signal.candle_time
                )
                outcome.sent.append(signal)
                log.info("%s: отправлен %s", unit.key, signal.type.value)

        await self.storage.save_anchor(
            unit.symbol, unit.timeframe, Direction.BULL, result.bull_anchor
        )
        if params.bearish_enabled:
            await self.storage.save_anchor(
                unit.symbol, unit.timeframe, Direction.BEAR, result.bear_anchor
            )

        if not seeded:
            await self.storage.mark_seeded(unit.symbol, unit.timeframe)

        return outcome

    # --- вспомогательное ----------------------------------------------------

    def provider_supports(self, provider_name: str, symbol: str, tf: Timeframe) -> str | None:
        """Возвращает текст ошибки или None, если пара поддерживается."""
        provider = self.providers.get(provider_name)
        if provider is None:
            return f"провайдер '{provider_name}' не сконфигурирован"
        if not provider.supports_symbol(symbol):
            return f"{provider_name} не поддерживает символ {symbol}"
        if not provider.supports_timeframe(tf):
            return f"{provider_name} не умеет отдавать {tf.value}"
        return None


def _signal_payload(signal: Signal) -> dict:
    payload = {
        "price": signal.price,
        "rsi": signal.rsi,
        "direction": signal.direction.value,
        "confirmation_time": _iso_or_none(signal.confirmation_time),
    }
    if signal.reference is not None:
        payload["reference"] = signal.reference.as_dict()
    if signal.pivot is not None:
        payload["pivot"] = signal.pivot.as_dict()
    return payload


def _iso_or_none(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None
