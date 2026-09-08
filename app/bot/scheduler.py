"""Планировщик.

Одна asyncio-задача на каждую пару (актив × таймфрейм). Задача спит до
момента закрытия своей свечи + close_delay_seconds (+ джиттер), а не тикает
слепым интервалом. Отдельный supervisor-цикл сверяет набор запущенных задач
со списком активных пар в БД, поэтому /add, /remove и /tf работают на лету
без рестарта.

Исключение внутри одной задачи не выходит наружу: пара просто ждёт следующей
свечи, остальные продолжают работать.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone

from app.bot.storage import Storage, TrackingUnit
from app.config import SchedulerConfig
from app.service import TrackingService

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        storage: Storage,
        service: TrackingService,
        config: SchedulerConfig,
    ) -> None:
        self.storage = storage
        self.service = service
        self.config = config
        self._tasks: dict[str, asyncio.Task] = {}
        self._supervisor: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    # --- жизненный цикл -----------------------------------------------------

    async def start(self) -> None:
        self._stopping.clear()
        await self.reconcile()
        self._supervisor = asyncio.create_task(
            self._supervisor_loop(), name="scheduler-supervisor"
        )
        log.info("Планировщик запущен: %d пар в работе", len(self._tasks))

    async def stop(self) -> None:
        self._stopping.set()
        if self._supervisor is not None:
            self._supervisor.cancel()
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        log.info("Планировщик остановлен")

    @property
    def active_keys(self) -> list[str]:
        return sorted(self._tasks)

    # --- синхронизация набора задач ----------------------------------------

    async def reconcile(self) -> None:
        """Приводит запущенные задачи в соответствие с состоянием в БД."""
        try:
            units = await self.storage.active_units()
        except Exception:  # noqa: BLE001
            log.exception("Не удалось прочитать список активных пар")
            return

        desired = {unit.key: unit for unit in units}

        for key in list(self._tasks):
            task = self._tasks[key]
            if key not in desired or task.done():
                if not task.done():
                    task.cancel()
                    log.info("Останавливаю отслеживание %s", key)
                self._tasks.pop(key, None)

        for key, unit in desired.items():
            if key not in self._tasks:
                self._tasks[key] = asyncio.create_task(
                    self._unit_loop(unit), name=f"track:{key}"
                )
                log.info("Запускаю отслеживание %s (%s)", key, unit.provider)

    async def _supervisor_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self.config.reconcile_interval_seconds
                )
                return
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                return
            try:
                await self.reconcile()
            except Exception:  # noqa: BLE001
                log.exception("Ошибка в supervisor-цикле планировщика")

    # --- цикл одной пары ----------------------------------------------------

    async def _unit_loop(self, unit: TrackingUnit) -> None:
        if self.config.check_on_startup:
            # Небольшая рассинхронизация стартовых запросов, чтобы не долбить API пачкой.
            await asyncio.sleep(random.uniform(0.5, 5.0))
            await self._safe_check(unit)

        while not self._stopping.is_set():
            delay = self._seconds_until_next_run(unit)
            log.debug("%s: следующий опрос через %.0f c", unit.key, delay)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            await self._safe_check(unit)

    def _seconds_until_next_run(self, unit: TrackingUnit) -> float:
        now = datetime.now(timezone.utc)
        next_close = unit.timeframe.next_close(now)
        target = (next_close - now).total_seconds() + self.config.close_delay_seconds
        target += random.uniform(0, self.config.jitter_seconds)
        return max(target, 1.0)

    async def _safe_check(self, unit: TrackingUnit) -> None:
        try:
            outcome = await self.service.check_unit(unit)
            if outcome.error:
                log.warning("%s: проверка завершилась с ошибкой: %s", unit.key, outcome.error)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — падение одной пары не трогает остальные
            log.exception("%s: непредвиденная ошибка при проверке", unit.key)
