"""Telegram-интерфейс: /start, /list, /add, /remove, /tf, /status, /params."""

from __future__ import annotations

import asyncio
import logging
from html import escape

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from app.analysis.signals import Direction
from app.bot.formatting import format_status_line
from app.bot.scheduler import Scheduler
from app.bot.storage import Storage, TrackingUnit
from app.config import EDITABLE_PARAMS, ParamsConfig, Secrets
from app.core.timeframes import ALL_TIMEFRAMES, Timeframe
from app.service import TrackingService

log = logging.getLogger(__name__)

HELP_TEXT = """<b>RSI Divergence Bot</b>

/list — активы и включённые таймфреймы
/add &lt;symbol&gt; &lt;provider&gt; &lt;tf,tf,...&gt; — добавить актив
    пример: <code>/add HYPEUSDT binance 1H,4H,1D</code>
/remove &lt;symbol&gt; — убрать актив
/tf &lt;symbol&gt; &lt;timeframe&gt; on|off — включить/выключить ТФ
    пример: <code>/tf HYPEUSDT 3H on</code>
/status &lt;symbol&gt; — цена, RSI и опорные точки по всем ТФ
/params — показать пороги; <code>/params oversold 28</code> — изменить
/help — эта справка

Таймфреймы: {timeframes}
"""


class BotHandlers:
    def __init__(
        self,
        storage: Storage,
        service: TrackingService,
        scheduler: Scheduler,
        secrets: Secrets,
    ) -> None:
        self.storage = storage
        self.service = service
        self.scheduler = scheduler
        self.secrets = secrets

    # --- регистрация --------------------------------------------------------

    def register(self, app: Application) -> None:
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("list", self.cmd_list))
        app.add_handler(CommandHandler("add", self.cmd_add))
        app.add_handler(CommandHandler("remove", self.cmd_remove))
        app.add_handler(CommandHandler("tf", self.cmd_tf))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("params", self.cmd_params))
        app.add_error_handler(self.on_error)

    # --- доступ -------------------------------------------------------------

    def _allowed(self, update: Update) -> bool:
        allowed = self.secrets.allowed_user_ids
        if not allowed:
            return True
        user = update.effective_user
        return user is not None and user.id in allowed

    async def _guard(self, update: Update) -> bool:
        if self._allowed(update):
            return True
        if update.effective_message:
            await update.effective_message.reply_text("Доступ запрещён.")
        log.warning(
            "Отклонён доступ для user_id=%s",
            update.effective_user.id if update.effective_user else "?",
        )
        return False

    # --- команды ------------------------------------------------------------

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        chat = update.effective_chat
        if chat is not None:
            await self.storage.add_subscriber(chat.id, chat.title or chat.full_name)
        await self._reply(update, "Подписка оформлена ✅\n\n" + self._help_text())

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await self._reply(update, self._help_text())

    async def cmd_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        assets = await self.storage.list_assets()
        if not assets:
            await self._reply(update, "Список активов пуст. Добавьте: /add SYMBOL provider 1H,4H")
            return
        lines = ["<b>Отслеживаемые активы</b>", ""]
        for asset in assets:
            active = asset.active_timeframes()
            off = [tf.value for tf, on in asset.timeframes.items() if not on]
            state = "" if asset.enabled else " (выключен)"
            lines.append(
                f"• <b>{escape(asset.symbol)}</b> [{escape(asset.provider)}]{state}: "
                + (", ".join(tf.value for tf in active) if active else "нет активных ТФ")
            )
            if off:
                lines.append(f"  <i>выключены: {', '.join(sorted(off))}</i>")
        lines += ["", f"Задач в планировщике: {len(self.scheduler.active_keys)}"]
        await self._reply(update, "\n".join(lines))

    async def cmd_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        if len(args) < 2:
            await self._reply(
                update,
                "Формат: <code>/add &lt;symbol&gt; &lt;provider&gt; [tf,tf,...]</code>\n"
                "Пример: <code>/add HYPEUSDT binance 1H,4H,1D</code>",
            )
            return

        symbol = args[0].upper()
        provider = args[1].lower()
        raw_tfs = args[2] if len(args) > 2 else "4H"
        timeframes: list[Timeframe] = []
        for chunk in raw_tfs.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            tf = Timeframe.try_parse(chunk)
            if tf is None:
                await self._reply(update, f"Неизвестный таймфрейм: {escape(chunk)}")
                return
            timeframes.append(tf)

        problems = [
            msg
            for msg in (self.service.provider_supports(provider, symbol, tf) for tf in timeframes)
            if msg
        ]
        if problems:
            await self._reply(update, "Не могу добавить:\n• " + "\n• ".join(map(escape, problems)))
            return

        await self.storage.upsert_asset(symbol, provider, timeframes)
        await self.scheduler.reconcile()
        await self._reply(
            update,
            f"Добавлено: <b>{escape(symbol)}</b> [{escape(provider)}] — "
            + ", ".join(tf.value for tf in timeframes),
        )

    async def cmd_remove(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        if not args:
            await self._reply(update, "Формат: <code>/remove &lt;symbol&gt;</code>")
            return
        symbol = args[0].upper()
        removed = await self.storage.remove_asset(symbol)
        await self.scheduler.reconcile()
        await self._reply(
            update,
            f"Удалено: <b>{escape(symbol)}</b>" if removed else f"Актив {escape(symbol)} не найден",
        )

    async def cmd_tf(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        if len(args) < 3:
            await self._reply(
                update, "Формат: <code>/tf &lt;symbol&gt; &lt;timeframe&gt; on|off</code>"
            )
            return
        symbol = args[0].upper()
        tf = Timeframe.try_parse(args[1])
        if tf is None:
            await self._reply(update, f"Неизвестный таймфрейм: {escape(args[1])}")
            return
        mode = args[2].lower()
        if mode not in {"on", "off"}:
            await self._reply(update, "Последний аргумент — on или off")
            return

        asset = await self.storage.get_asset(symbol)
        if asset is None:
            await self._reply(update, f"Актив {escape(symbol)} не найден. Сначала /add")
            return
        if mode == "on":
            problem = self.service.provider_supports(asset.provider, symbol, tf)
            if problem:
                await self._reply(update, f"Не могу включить: {escape(problem)}")
                return

        await self.storage.set_timeframe(symbol, tf, mode == "on")
        await self.scheduler.reconcile()
        await self._reply(
            update, f"{escape(symbol)} · {tf.value} → <b>{'включён' if mode == 'on' else 'выключен'}</b>"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        if not args:
            await self._reply(update, "Формат: <code>/status &lt;symbol&gt;</code>")
            return
        symbol = args[0].upper()
        asset = await self.storage.get_asset(symbol)
        if asset is None:
            await self._reply(update, f"Актив {escape(symbol)} не найден")
            return
        active = asset.active_timeframes()
        if not active:
            await self._reply(update, f"У {escape(symbol)} нет включённых таймфреймов")
            return

        await self._reply(update, f"Считаю {escape(symbol)}…")

        units = [TrackingUnit(symbol, asset.provider, tf) for tf in active]
        outcomes = await asyncio.gather(
            *(self.service.check_unit(unit, notify=False) for unit in units),
            return_exceptions=True,
        )

        params = await self.service.effective_params()
        lines = [f"<b>{escape(symbol)}</b> [{escape(asset.provider)}]", ""]
        for unit, outcome in zip(units, outcomes):
            if isinstance(outcome, BaseException):
                lines.append(format_status_line(unit.timeframe.value, None, None, None, None, str(outcome)))
                continue
            if outcome.error or outcome.result is None:
                lines.append(
                    format_status_line(
                        unit.timeframe.value, None, None, None, None,
                        outcome.error or "нет данных",
                    )
                )
                continue
            anchor = await self.storage.get_anchor(symbol, unit.timeframe, Direction.BULL)
            lines.append(
                format_status_line(
                    unit.timeframe.value,
                    outcome.result.last_price,
                    outcome.result.last_rsi,
                    outcome.result.last_candle_time,
                    anchor,
                )
            )
        lines += [
            "",
            f"<i>RSI({params.rsi_period}), фрактал N={params.fractal_n}, "
            f"oversold={params.oversold:g}, div_max={params.divergence_rsi_max:g}</i>",
        ]
        await self._reply(update, "\n".join(lines))

    async def cmd_params(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        args = context.args or []
        params = await self.service.effective_params()

        if not args:
            lines = ["<b>Текущие пороги</b>", ""]
            for key in EDITABLE_PARAMS:
                lines.append(f"<code>{key}</code> = <b>{getattr(params, key)}</b>")
            lines += ["", "Изменить: <code>/params oversold 28</code>",
                      "Сбросить всё к config.yaml: <code>/params reset</code>"]
            await self._reply(update, "\n".join(lines))
            return

        if args[0].lower() == "reset":
            await self.storage.clear_params()
            await self._reply(update, "Пороги сброшены к значениям из config.yaml")
            return

        if len(args) < 2:
            await self._reply(update, "Формат: <code>/params &lt;ключ&gt; &lt;значение&gt;</code>")
            return

        key = args[0].strip()
        if key not in EDITABLE_PARAMS:
            await self._reply(
                update,
                f"Неизвестный параметр: <code>{escape(key)}</code>\n"
                f"Доступны: {', '.join(EDITABLE_PARAMS)}",
            )
            return

        value = _coerce(args[1])
        candidate = params.model_dump()
        candidate[key] = value
        try:
            ParamsConfig.model_validate(candidate)
        except Exception as exc:  # noqa: BLE001
            await self._reply(update, f"Некорректное значение: {escape(str(exc)[:300])}")
            return

        await self.storage.set_param(key, value)
        await self._reply(
            update,
            f"<code>{escape(key)}</code>: {getattr(params, key)} → <b>{value}</b>\n"
            f"<i>Применится при следующем опросе.</i>",
        )

    # --- служебное ----------------------------------------------------------

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.exception("Ошибка в обработчике Telegram", exc_info=context.error)

    def _help_text(self) -> str:
        return HELP_TEXT.format(
            timeframes=", ".join(tf.value for tf in ALL_TIMEFRAMES)
        )

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        message = update.effective_message
        if message is None:
            return
        await message.reply_text(
            text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )


def _coerce(raw: str):
    low = raw.strip().lower()
    if low in {"true", "on", "yes"}:
        return True
    if low in {"false", "off", "no"}:
        return False
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
