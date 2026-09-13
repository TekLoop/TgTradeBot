"""Форматирование сообщений Telegram (parse_mode=HTML)."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from app.analysis.signals import INSTANT_TYPES, AlertType, Direction, Signal
from app.bot.outcomes import Outcome
from app.core.timeframes import Timeframe

ALERT_TITLES: dict[AlertType, str] = {
    AlertType.OVERSOLD_PIVOT: "ALERT #1 · Перепроданность",
    AlertType.BULLISH_DIVERGENCE: "ALERT #2 · Бычья дивергенция",
    AlertType.OVERBOUGHT_PIVOT: "ALERT #1 · Перекупленность",
    AlertType.BEARISH_DIVERGENCE: "ALERT #2 · Медвежья дивергенция",
    AlertType.EXTREME_OVERSOLD: "EXTREME · Крайняя перепроданность",
    AlertType.EXTREME_OVERBOUGHT: "EXTREME · Крайняя перекупленность",
    AlertType.WICK_UPPER: "ФИТИЛЬ · Верхний",
    AlertType.WICK_LOWER: "ФИТИЛЬ · Нижний",
}

ALERT_EMOJI: dict[AlertType, str] = {
    AlertType.OVERSOLD_PIVOT: "🟡",
    AlertType.BULLISH_DIVERGENCE: "🟢",
    AlertType.OVERBOUGHT_PIVOT: "🟠",
    AlertType.BEARISH_DIVERGENCE: "🔴",
    AlertType.EXTREME_OVERSOLD: "🧊",
    AlertType.EXTREME_OVERBOUGHT: "🔥",
    AlertType.WICK_UPPER: "🕯",
    AlertType.WICK_LOWER: "🕯",
}

DIRECTION_EMOJI = {Direction.BULL: "🟢", Direction.BEAR: "🔴"}

CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def fmt_price(value: float | None) -> str:
    if value is None:
        return "—"
    abs_v = abs(value)
    if abs_v >= 1000:
        return f"{value:,.2f}".replace(",", " ")
    if abs_v >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.8f}".rstrip("0").rstrip(".")


def fmt_rsi(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def fmt_time(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_date(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%d.%m %H:%M")


def _marker(position: int) -> str:
    return CIRCLED[position] if position < len(CIRCLED) else f"({position + 1})"


def alert_title(signal: Signal) -> str:
    title = ALERT_TITLES.get(signal.type, signal.type.value)
    if signal.is_divergence and signal.degree > 2:
        title = title.replace("ALERT #2", f"ALERT #{signal.degree}") + f" ×{signal.degree}"
    return title


# --- сигнал -----------------------------------------------------------------


def format_signal(signal: Signal) -> str:
    if signal.type in INSTANT_TYPES:
        return _format_instant(signal)

    emoji = ALERT_EMOJI.get(signal.type, "🔔")
    lines = [
        f"{emoji} <b>{escape(signal.symbol)}</b> · <b>{signal.timeframe.value}</b>",
        f"{escape(alert_title(signal))}",
        "",
        f"Цена: <code>{fmt_price(signal.price)}</code>",
        f"RSI: <code>{fmt_rsi(signal.rsi)}</code>",
        f"Свеча (закрытие): {fmt_time(signal.candle_time + signal.timeframe.duration)}",
        f"Экстремум на свече: {fmt_time(signal.candle_time)}",
    ]

    if signal.displaced_anchor is not None:
        old = signal.displaced_anchor
        side = "ниже" if signal.direction is Direction.BULL else "выше"
        lines += [
            "",
            f"<b>Опорная точка сменилась</b> — прежняя: цена "
            f"<code>{fmt_price(old.price)}</code> · RSI <code>{fmt_rsi(old.rsi)}</code> "
            f"({fmt_time(old.time)}). Новая точка {side} и по цене, и по RSI.",
        ]

    if signal.replaced_from is not None:
        position = signal.replaced_from + 1  # человеку считаем с единицы
        lines += [
            "",
            f"<b>Точка {position} заменена</b> — RSI обновил минимум, "
            f"дивергенция пересчитана относительно опорной точки.",
        ]

    if len(signal.chain) > 2:
        lines += ["", f"<b>Цепочка из {len(signal.chain)} точек</b>"]
        for position, point in enumerate(signal.chain):
            lines.append(
                f"{_marker(position)} {fmt_time(point.time)} · цена "
                f"<code>{fmt_price(point.price)}</code> · RSI <code>{fmt_rsi(point.rsi)}</code>"
            )
        first, last = signal.chain[0], signal.chain[-1]
        if first.price:
            total = (last.price / first.price - 1.0) * 100.0
            lines.append(
                f"От ① до {_marker(len(signal.chain) - 1)}: цена <code>{total:+.2f}%</code>, "
                f"RSI <code>{last.rsi - first.rsi:+.2f}</code>"
            )
    elif signal.reference is not None and signal.pivot is not None:
        ref, cur = signal.reference, signal.pivot
        bars = cur.index - ref.index
        lines += [
            "",
            "<b>Точки дивергенции</b>",
            f"① {fmt_time(ref.time)} · цена <code>{fmt_price(ref.price)}</code> "
            f"· RSI <code>{fmt_rsi(ref.rsi)}</code>",
            f"② {fmt_time(cur.time)} · цена <code>{fmt_price(cur.price)}</code> "
            f"· RSI <code>{fmt_rsi(cur.rsi)}</code>",
            f"Δ цена: <code>{fmt_price(signal.price_delta)}</code> "
            f"({signal.price_delta_pct:+.2f}%)",
            f"Δ RSI: <code>{signal.rsi_delta:+.2f}</code>",
            f"Баров между точками: <code>{bars}</code>",
        ]

    if signal.is_divergence and signal.confirmation_price is not None:
        lines += [
            "",
            f"<i>Отсчёт статистики от цены <code>{fmt_price(signal.confirmation_price)}</code> "
            f"(закрытие подтверждающей свечи).</i>",
        ]

    if signal.bars_since_confirmation > 0:
        lines += [
            "",
            f"<i>Сигнал подтверждён {signal.bars_since_confirmation} "
            f"бар(ов) назад (догоняющая отправка).</i>",
        ]

    return "\n".join(lines)


# --- статус -----------------------------------------------------------------


def _format_instant(signal: Signal) -> str:
    """Extreme-алерты и фитили: своя свеча — своё подтверждение."""
    emoji = ALERT_EMOJI.get(signal.type, "🔔")
    lines = [
        f"{emoji} <b>{escape(signal.symbol)}</b> · <b>{signal.timeframe.value}</b>",
        f"{escape(ALERT_TITLES.get(signal.type, signal.type.value))}",
        "",
        f"Свеча (закрытие): {fmt_time(signal.candle_time + signal.timeframe.duration)}",
        f"Открытие свечи: {fmt_time(signal.candle_time)}",
        f"RSI: <code>{fmt_rsi(signal.rsi)}</code>",
    ]

    if signal.wick_pct is not None:
        lines.append(f"Фитиль: <code>{signal.wick_pct:.2f}%</code>")
    if signal.ohlc is not None:
        open_, high, low, close = signal.ohlc
        lines += [
            "",
            f"O <code>{fmt_price(open_)}</code> · H <code>{fmt_price(high)}</code>",
            f"L <code>{fmt_price(low)}</code> · C <code>{fmt_price(close)}</code>",
        ]
    else:
        lines.append(f"Цена: <code>{fmt_price(signal.price)}</code>")

    lines += [
        "",
        "<i>Информационный алерт: приходит сразу на закрытии свечи, "
        "без задержки на подтверждение фрактала — в отличие от ALERT #1 и #2. "
        "Статистика по нему не ведётся.</i>",
    ]
    if signal.bars_since_confirmation > 0:
        lines.append(
            f"<i>Свеча закрылась {signal.bars_since_confirmation} бар(ов) назад "
            f"(догоняющая отправка).</i>"
        )
    return "\n".join(lines)


def format_status_line(
    timeframe: Timeframe,
    price: float | None,
    rsi_value: float | None,
    candle_time: datetime | None,
    anchor: dict | None,
    error: str | None = None,
) -> str:
    if error:
        return f"• <b>{timeframe.value}</b>: ошибка — {escape(error[:120])}"
    return (
        f"• <b>{timeframe.value}</b>: цена <code>{fmt_price(price)}</code>, "
        f"RSI <code>{fmt_rsi(rsi_value)}</code>\n"
        f"  {_anchor_text(timeframe, anchor, candle_time)}\n"
        f"  <i>посл. закрытая свеча: {fmt_time(candle_time)}</i>"
    )


def _anchor_text(
    timeframe: Timeframe, anchor: dict | None, candle_time: datetime | None
) -> str:
    """«опорная … (N баров назад) · последняя … · точек: K».

    При отсутствии chain_json (строка из старой БД) откатывается на одну
    точку и показывает только опорную — без ошибки.
    """
    if not anchor:
        return "опорная точка: нет"

    chain = anchor.get("chain") or []
    if chain:
        first, last = chain[0], chain[-1]
        head = f"опорная {fmt_price(first.price)} / RSI {fmt_rsi(first.rsi)}"
        age = _bars_ago(timeframe, first.time, candle_time)
        if age is not None:
            head += f" ({age} бар(ов) назад)"
        if len(chain) > 1:
            head += f" · последняя {fmt_price(last.price)} / RSI {fmt_rsi(last.rsi)}"
        return f"{head} · точек: {len(chain)}"

    head = (
        f"опорная {fmt_price(anchor.get('price'))} / RSI {fmt_rsi(anchor.get('rsi'))}"
    )
    age = _bars_ago(timeframe, _parse_iso(anchor.get("candle_time")), candle_time)
    if age is not None:
        head += f" ({age} бар(ов) назад)"
    return f"{head} · точек: 1"


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _bars_ago(
    timeframe: Timeframe, moment: datetime | None, now: datetime | None
) -> int | None:
    if moment is None or now is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    minutes = timeframe.minutes
    if minutes <= 0:
        return None
    return max(0, int((now - moment).total_seconds() // (minutes * 60)))


# --- статистика -------------------------------------------------------------


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _aggregate(records: list[Outcome]) -> dict:
    results = [r.result_pct for r in records if r.result_pct is not None]
    favorable = [r.favorable_pct for r in records if r.favorable_pct is not None]
    adverse = [r.adverse_pct for r in records if r.adverse_pct is not None]
    return {
        "count": len(records),
        "result": _avg(results),
        "favorable": _avg(favorable),
        "adverse": _avg(adverse),
        "best": max(favorable) if favorable else None,
        "worst": min(adverse) if adverse else None,
        "wins": sum(1 for v in results if v > 0),
        "bars": _avg([float(r.bars_held) for r in records]),
    }


def format_outcome_line(record: Outcome, short: bool = True) -> str:
    emoji = DIRECTION_EMOJI.get(record.direction, "🔔")
    head = (
        f"{emoji} <b>{escape(record.symbol)}</b> · <b>{record.timeframe.value}</b> "
        f"×{record.degree} · вход <code>{fmt_price(record.entry_price)}</code> "
        f"({fmt_date(record.entry_time)})"
    )
    body = (
        f"  в пользу <code>{fmt_pct(record.favorable_pct)}</code> · "
        f"против <code>{fmt_pct(record.adverse_pct)}</code>"
    )
    if record.is_open:
        body += f" · открыта, {record.bars_held} бар(ов)"
    else:
        body += (
            f" · итог <code>{fmt_pct(record.result_pct)}</code>"
            f"\n  <i>{escape(record.reason_label)}, {fmt_date(record.exit_time)}, "
            f"{record.bars_held} бар(ов)</i>"
        )
    if short:
        return f"{head}\n{body}"
    return f"{head}\n{body}\n  <i>сигнал: {fmt_time(record.signal_time)}</i>"


def format_stats(
    closed: list[Outcome],
    open_records: list[Outcome],
    header: str,
) -> str:
    lines = [f"📊 <b>{escape(header)}</b>"]
    if not closed and not open_records:
        lines += ["", "Пока нет ни одной записи. Статистика заводится "
                      "на каждой подтверждённой дивергенции."]
        return "\n".join(lines)

    lines += [
        "<i>Проценты — в сторону сигнала: для медвежьей падение цены это плюс.</i>",
        "",
        f"Закрыто: <b>{len(closed)}</b> · открыто: <b>{len(open_records)}</b>",
    ]

    if closed:
        agg = _aggregate(closed)
        lines += [
            "",
            f"Ход в пользу: <b>{fmt_pct(agg['favorable'])}</b> "
            f"(лучший {fmt_pct(agg['best'])})",
            f"Ход против: <b>{fmt_pct(agg['adverse'])}</b> "
            f"(худший {fmt_pct(agg['worst'])})",
            f"Итог на выходе: <b>{fmt_pct(agg['result'])}</b> · "
            f"плюсовых {agg['wins']} из {agg['count']}",
        ]
        if agg["bars"]:
            lines.append(f"Баров в записи: <b>{agg['bars']:.0f}</b>")

        by_degree: dict[int, list[Outcome]] = {}
        for record in closed:
            by_degree.setdefault(record.degree, []).append(record)
        if len(by_degree) > 1:
            lines += ["", "<b>По числу точек</b>"]
            for degree in sorted(by_degree):
                agg_d = _aggregate(by_degree[degree])
                lines.append(
                    f"×{degree} ({agg_d['count']}): итог <b>{fmt_pct(agg_d['result'])}</b> · "
                    f"в пользу {fmt_pct(agg_d['favorable'])} · "
                    f"против {fmt_pct(agg_d['adverse'])} · "
                    f"плюсовых {agg_d['wins']}"
                )

        by_symbol: dict[str, list[Outcome]] = {}
        for record in closed:
            by_symbol.setdefault(record.symbol, []).append(record)
        if len(by_symbol) > 1:
            lines += ["", "<b>По активам</b>"]
            for sym in sorted(by_symbol, key=lambda s: len(by_symbol[s]), reverse=True):
                agg_s = _aggregate(by_symbol[sym])
                lines.append(
                    f"{escape(sym)} ({agg_s['count']}): итог <b>{fmt_pct(agg_s['result'])}</b> · "
                    f"в пользу {fmt_pct(agg_s['favorable'])} · "
                    f"против {fmt_pct(agg_s['adverse'])} · "
                    f"плюсовых {agg_s['wins']}"
                )

        by_tf: dict[str, list[Outcome]] = {}
        for record in closed:
            by_tf.setdefault(record.timeframe.value, []).append(record)
        if len(by_tf) > 1:
            lines += ["", "<b>По таймфреймам</b>"]
            for tf_value in sorted(by_tf, key=lambda v: len(by_tf[v]), reverse=True):
                agg_t = _aggregate(by_tf[tf_value])
                lines.append(
                    f"{tf_value} ({agg_t['count']}): итог <b>{fmt_pct(agg_t['result'])}</b> · "
                    f"плюсовых {agg_t['wins']}"
                )

    if open_records:
        lines += ["", "<b>Открытые записи</b>"]
        ordered = sorted(open_records, key=lambda r: (r.symbol, r.timeframe.value))
        for record in ordered[:10]:
            lines.append(format_outcome_line(record))
        if len(ordered) > 10:
            hidden = len(ordered) - 10
            lines.append(f"<i>…и ещё {hidden}. Сузьте: /stats SYMBOL</i>")

    return "\n".join(lines)


def format_history(records: list[Outcome], header: str) -> str:
    lines = [f"🗂 <b>{escape(header)}</b>"]
    if not records:
        lines += ["", "Закрытых записей пока нет."]
        return "\n".join(lines)
    lines += ["<i>Проценты — в сторону сигнала.</i>", ""]
    for record in records:
        lines.append(format_outcome_line(record))
        lines.append("")
    return "\n".join(lines).strip()
