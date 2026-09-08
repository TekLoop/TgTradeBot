"""Форматирование сообщений Telegram (parse_mode=HTML)."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from app.analysis.signals import AlertType, Signal

ALERT_TITLES: dict[AlertType, str] = {
    AlertType.OVERSOLD_PIVOT: "ALERT #1 · Перепроданность",
    AlertType.BULLISH_DIVERGENCE: "ALERT #2 · Бычья дивергенция",
    AlertType.OVERBOUGHT_PIVOT: "ALERT #1 · Перекупленность",
    AlertType.BEARISH_DIVERGENCE: "ALERT #2 · Медвежья дивергенция",
}

ALERT_EMOJI: dict[AlertType, str] = {
    AlertType.OVERSOLD_PIVOT: "🟡",
    AlertType.BULLISH_DIVERGENCE: "🟢",
    AlertType.OVERBOUGHT_PIVOT: "🟠",
    AlertType.BEARISH_DIVERGENCE: "🔴",
}


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


def fmt_time(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_signal(signal: Signal) -> str:
    emoji = ALERT_EMOJI.get(signal.type, "🔔")
    title = ALERT_TITLES.get(signal.type, signal.type.value)
    lines = [
        f"{emoji} <b>{escape(signal.symbol)}</b> · <b>{signal.timeframe.value}</b>",
        f"{escape(title)}",
        "",
        f"Цена: <code>{fmt_price(signal.price)}</code>",
        f"RSI: <code>{fmt_rsi(signal.rsi)}</code>",
        f"Свеча (закрытие): {fmt_time(signal.candle_time + signal.timeframe.duration)}",
        f"Экстремум на свече: {fmt_time(signal.candle_time)}",
    ]

    if signal.reference is not None and signal.pivot is not None:
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

    if signal.bars_since_confirmation > 0:
        lines += [
            "",
            f"<i>Сигнал подтверждён {signal.bars_since_confirmation} "
            f"бар(ов) назад (догоняющая отправка).</i>",
        ]

    return "\n".join(lines)


def format_status_line(
    timeframe_value: str,
    price: float | None,
    rsi_value: float | None,
    candle_time: datetime | None,
    anchor: dict | None,
    error: str | None = None,
) -> str:
    if error:
        return f"• <b>{timeframe_value}</b>: ошибка — {escape(error[:120])}"
    anchor_txt = "нет"
    if anchor:
        anchor_txt = (
            f"цена {fmt_price(anchor.get('price'))} / RSI {fmt_rsi(anchor.get('rsi'))}"
        )
    return (
        f"• <b>{timeframe_value}</b>: цена <code>{fmt_price(price)}</code>, "
        f"RSI <code>{fmt_rsi(rsi_value)}</code>, опорная точка: {anchor_txt}\n"
        f"  <i>посл. закрытая свеча: {fmt_time(candle_time)}</i>"
    )
