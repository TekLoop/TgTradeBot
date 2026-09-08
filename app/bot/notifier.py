"""Рассылка уведомлений подписчикам."""

from __future__ import annotations

import logging

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import Forbidden, RetryAfter, TelegramError

from app.analysis.signals import Signal
from app.bot.formatting import format_signal
from app.bot.storage import Storage

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, bot: Bot, storage: Storage, default_chat_id: int | None = None) -> None:
        self.bot = bot
        self.storage = storage
        self.default_chat_id = default_chat_id

    async def targets(self) -> list[int]:
        chats = set(await self.storage.list_subscribers())
        if self.default_chat_id:
            chats.add(self.default_chat_id)
        return sorted(chats)

    async def send_signal(self, signal: Signal) -> bool:
        return await self.send_text(format_signal(signal))

    async def send_text(self, text: str) -> bool:
        chats = await self.targets()
        if not chats:
            log.warning("Нет подписчиков: некому отправлять. Пришлите /start боту.")
            return False

        delivered = False
        for chat_id in chats:
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                delivered = True
            except Forbidden:
                log.warning("Чат %s заблокировал бота — удаляю из подписчиков", chat_id)
                await self.storage.remove_subscriber(chat_id)
            except RetryAfter as exc:
                log.warning("Flood control для чата %s: ждать %s c", chat_id, exc.retry_after)
            except TelegramError:
                log.exception("Не удалось отправить сообщение в чат %s", chat_id)
        return delivered
