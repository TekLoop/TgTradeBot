"""Точка входа.

    python -m app.main [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv
from telegram.ext import Application, ApplicationBuilder

from app.bot.handlers import BotHandlers
from app.bot.notifier import Notifier
from app.bot.scheduler import Scheduler
from app.bot.storage import Storage
from app.config import AppConfig, Secrets, load_config
from app.data.registry import build_providers, close_providers
from app.logging_setup import setup_logging
from app.service import TrackingService

log = logging.getLogger(__name__)


async def sync_config_to_storage(config: AppConfig, storage: Storage) -> None:
    """YAML — источник правды для набора активов; ручные правки из Telegram
    сохраняются (см. force_enable=False)."""
    yaml_symbols = {asset.symbol for asset in config.assets if asset.enabled}

    for asset in config.assets:
        if not asset.enabled:
            continue
        await storage.upsert_asset(
            asset.symbol, asset.provider, asset.timeframes, force_enable=False
        )

    if config.sync.remove_missing_assets:
        for record in await storage.list_assets():
            if record.symbol not in yaml_symbols:
                await storage.remove_asset(record.symbol)
                log.info("Актив %s удалён (нет в config.yaml)", record.symbol)

    if config.sync.overwrite_params:
        await storage.clear_params()
        log.info("Оверрайды порогов сброшены (sync.overwrite_params=true)")


def build_application(config: AppConfig, secrets: Secrets) -> Application:
    storage = Storage(config.storage.db_path)
    providers = build_providers(config.providers, secrets)
    if not providers:
        log.warning("Ни один провайдер не инициализирован — сигналов не будет")

    async def on_startup(app: Application) -> None:
        await storage.connect()
        await sync_config_to_storage(config, storage)

        notifier = Notifier(app.bot, storage, secrets.default_chat_id)
        service = TrackingService(storage, providers, config.params, notifier)
        scheduler = Scheduler(storage, service, config.scheduler)

        handlers = BotHandlers(storage, service, scheduler, secrets)
        handlers.register(app)

        app.bot_data["storage"] = storage
        app.bot_data["service"] = service
        app.bot_data["scheduler"] = scheduler

        await scheduler.start()
        me = await app.bot.get_me()
        log.info("Бот @%s запущен", me.username)

    async def on_shutdown(app: Application) -> None:
        scheduler: Scheduler | None = app.bot_data.get("scheduler")
        if scheduler is not None:
            await scheduler.stop()
        await close_providers(providers)
        await storage.close()
        log.info("Остановлено штатно")

    return (
        ApplicationBuilder()
        .token(secrets.require_telegram_token())
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .concurrent_updates(True)
        .build()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RSI divergence Telegram bot")
    parser.add_argument(
        "--config",
        default=os.getenv("CONFIG_PATH", "config.yaml"),
        help="путь к YAML-конфигу (по умолчанию config.yaml)",
    )
    args = parser.parse_args()

    load_dotenv(override=False)
    config = load_config(args.config)
    setup_logging(config.logging)

    log.info("Конфиг загружен: %d активов, %d провайдеров",
             len(config.assets), len(config.providers))

    secrets = Secrets()
    application = build_application(config, secrets)
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
