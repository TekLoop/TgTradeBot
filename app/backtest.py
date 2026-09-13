"""Офлайн-прогон детектора по истории.

    python -m app.backtest --symbols BTCUSDT --timeframes 4H,1D --years 8

Скрипт ничего не пишет в рабочую БД бота и никуда не отправляет сообщений:
качает свечи, гоняет тот же DivergenceEngine и печатает сводку.

Модель сделки — лесенка тейков со стопом (app/analysis/trade_rules.py):
вход по close подтверждающей свечи, выход только стопом или по tp3.
Таймаута нет, новая дивергенция сделку не закрывает. Живой бот пока живёт
по старым правилам: эта итерация проверяет новые на истории.

Перебор параметров (работает и для порогов детектора, и для уровней лесенки):

    python -m app.backtest --tf 4H --sweep oversold=25,30,35
    python -m app.backtest --tf 4H --sweep tp1_pct=3,5,7 --sweep sl_buffer_pct=0,1,2

Выгрузка всех сделок для ручного разбора:

    python -m app.backtest --tf 4H --csv data/history/trades.csv

История кэшируется в data/cache/*.csv.gz (app/backtest_cache.py), поэтому
повторные прогоны и переборы идут без обращения к сети. С --no-network сеть
запрещена совсем, с --refresh период перекачивается заново.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.analysis.replay import (
    SubBarIndex,
    Trade,
    aggregate,
    build_subbars,
    evaluate,
    group_by_sl_distance,
    supports_intrabar,
)
from app.analysis.signals import DivergenceEngine
from app.analysis.trade_rules import LADDER_KEYS, REASON_ORDER, TradeRules
from app.backtest_cache import CacheError, CacheSettings, load_klines
from app.core.timeframes import Timeframe
from app.data.resampling import resample_ohlcv

DEFAULT_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
DEFAULT_TIMEFRAMES = "4H,1D"

#: Нативные интервалы Binance. 3H собирается из 1H, 1W — из 1D,
#: ровно как в app/data/binance.py.
NATIVE_INTERVALS: dict[Timeframe, str] = {
    Timeframe.H1: "1h",
    Timeframe.H2: "2h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}


# --- загрузка истории -------------------------------------------------------


def base_timeframe(tf: Timeframe) -> Timeframe:
    if tf in NATIVE_INTERVALS:
        return tf
    candidates = [base for base in NATIVE_INTERVALS if base.divides(tf)]
    if not candidates:
        raise ValueError(f"Не из чего собрать {tf.value}")
    return max(candidates, key=lambda base: base.minutes)


def load_history(
    symbol: str,
    tf: Timeframe,
    years: float,
    *,
    settings: CacheSettings,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Отдаёт закрытые свечи нужного ТФ за последние `years` лет."""
    base = base_timeframe(tf)
    interval = NATIVE_INTERVALS[base]
    step_ms = base.minutes * 60_000
    moment = now or datetime.now(timezone.utc)
    start = moment - timedelta(days=365.25 * years)

    window = load_klines(
        symbol, interval, start, step_ms, settings=settings, now=moment
    )
    if window.empty or base is tf:
        return window
    return resample_ohlcv(window, tf, source=base)


# --- параметры --------------------------------------------------------------


def split_ladder(assignments: dict) -> tuple[dict, dict]:
    """Делит разобранные --param на ключи лесенки и всё остальное.

    Ключи лесенки в ParamsConfig попасть не должны: эта модель общая с живым
    ботом, и новые поля тут же появились бы в его валидации и в /params.
    """
    ladder: dict = {}
    params: dict = {}
    for tf, values in assignments.items():
        for key, value in values.items():
            target = ladder if key in LADDER_KEYS else params
            target.setdefault(tf, {})[key] = value
    return ladder, params


def rules_for(tf: Timeframe, ladder: dict, extra: dict | None = None) -> TradeRules:
    """Таблица уровней по таймфрейму + глобальные --param + --param TF:… ."""
    overrides: dict = {}
    overrides.update(ladder.get(None, {}))
    overrides.update(ladder.get(tf, {}))
    overrides.update(extra or {})
    try:
        return TradeRules.for_timeframe(tf, overrides)
    except ValueError as exc:
        raise SystemExit(f"Параметры лесенки для {tf.value}: {exc}") from exc


def load_params(config_path: str, db_path: str | None, overrides: dict):
    """Пороги: config.yaml → params_by_timeframe → /params из state.db → CLI.

    Возвращает (базовый ParamsConfig, {Timeframe: слой оверрайдов}). Слои
    накладываются на базу функцией build_params() из app/config.py — тем же
    кодом, что и в боевом сервисе, чтобы прогон и бот не разъезжались.
    """
    from app.config import EDITABLE_PARAMS, ParamsConfig, load_config

    layers: dict[Timeframe, dict] = {}
    try:
        config = load_config(config_path)
        params = config.params
        for tf, overlay in config.params_by_timeframe.items():
            layers.setdefault(tf, {}).update(overlay)
    except Exception as exc:  # noqa: BLE001 — прогон возможен и без конфига
        print(f"config.yaml не прочитан ({exc}), беру значения по умолчанию",
              file=sys.stderr)
        params = ParamsConfig()

    merged = params.model_dump()

    if db_path and Path(db_path).exists():
        try:
            with sqlite3.connect(db_path) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("pragma table_info(params)").fetchall()
                }
                if "timeframe" in columns:
                    rows = connection.execute(
                        "SELECT key, timeframe, value FROM params"
                    ).fetchall()
                else:  # БД ещё не мигрирована — все записи глобальные
                    rows = [
                        (key, "", value)
                        for key, value in connection.execute(
                            "SELECT key, value FROM params"
                        ).fetchall()
                    ]
            applied = []
            for key, raw_tf, raw in rows:
                if key not in EDITABLE_PARAMS:
                    continue
                value = json.loads(raw)
                tf = Timeframe.try_parse(raw_tf) if raw_tf else None
                if tf is None:
                    merged[key] = value
                    applied.append(key)
                else:
                    layers.setdefault(tf, {})[key] = value
                    applied.append(f"{tf.value}:{key}")
            if applied:
                print(f"Из state.db подтянуты правки /params: {', '.join(applied)}",
                      file=sys.stderr)
        except sqlite3.Error as exc:
            print(f"state.db не прочитан ({exc})", file=sys.stderr)

    merged.update(overrides.get(None, {}))
    for tf, values in overrides.items():
        if tf is not None:
            layers.setdefault(tf, {}).update(values)

    return ParamsConfig.model_validate(merged), layers


def resolve_params(base, layers: dict, tf: Timeframe):
    """Собранный конфиг таймфрейма — тем же build_params(), что и в боте."""
    from app.config import build_params

    return build_params(base, layers.get(tf))


def parse_assignments(items: list[str]) -> dict:
    """key=value → {None: {...}}; 1D:key=value → {Timeframe.D1: {...}}."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Ожидал key=value или TF:key=value, получил: {item}")
        left, raw = item.split("=", 1)
        left = left.strip()
        tf: Timeframe | None = None
        if ":" in left:
            raw_tf, left = left.split(":", 1)
            tf = Timeframe.try_parse(raw_tf.strip())
            if tf is None:
                raise SystemExit(f"Неизвестный таймфрейм в --param: {raw_tf}")
        out.setdefault(tf, {})[left.strip()] = _coerce(raw.strip())
    return out


def parse_sweep(items: list[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Ожидал key=v1,v2, получил: {item}")
        key, raw = item.split("=", 1)
        out[key.strip()] = [_coerce(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    return out


def _coerce(raw: str):
    low = raw.lower()
    if low in {"true", "on", "yes"}:
        return True
    if low in {"false", "off", "no"}:
        return False
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


# --- прогон -----------------------------------------------------------------


def run_one(
    symbol: str,
    tf: Timeframe,
    df: pd.DataFrame,
    params,
    rules: TradeRules,
    subbars: SubBarIndex | None = None,
) -> tuple[list[Trade], list]:
    """(сделки, сигналы). Сигналы нужны для аудита расхождений между прогонами."""
    engine = DivergenceEngine(params.to_signal_params())
    result = engine.analyze(df, symbol, tf)
    trades = evaluate(df, result.signals, rules=rules, subbars=subbars)
    return trades, result.signals


def signal_row(signal) -> dict:
    """Строка выгрузки сигналов. Набор колонок зафиксирован приёмкой:
    сравнение двух прогонов делается обычным диффом двух CSV."""
    return {
        "symbol": signal.symbol,
        "timeframe": signal.timeframe.value,
        "alert_type": signal.type.value,
        "candle_time": signal.candle_time.isoformat(),
        "price": signal.price,
        "rsi": round(signal.rsi, 4),
        "degree": signal.degree,
        "replaced_from": (
            "" if signal.replaced_from is None else signal.replaced_from
        ),
    }


def dump_signals(path: str, signals: list) -> None:
    if not signals:
        print("Нечего выгружать: сигналов нет", file=sys.stderr)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [signal_row(s) for s in signals]
    rows.sort(key=lambda r: (r["symbol"], r["timeframe"], r["candle_time"], r["alert_type"]))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Выгружено сигналов: {len(rows)} → {target}")


def dump_csv(path: str, trades: list[Trade]) -> None:
    if not trades:
        print("Нечего выгружать: сделок нет", file=sys.stderr)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [trade.as_row() for trade in trades]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nВыгружено сделок: {len(rows)} → {target}")


# --- отчёт ------------------------------------------------------------------


def _pct(value: float | None, width: int = 7) -> str:
    return "—".rjust(width) if value is None else f"{value:+.2f}%".rjust(width)


def _num(value: float | None, width: int = 5, digits: int = 0) -> str:
    return "—".rjust(width) if value is None else f"{value:.{digits}f}".rjust(width)


def _rate(value: float | None, width: int = 5) -> str:
    return "—".rjust(width) if value is None else f"{value:.0f}%".rjust(width)


#: Короткие подписи для таблиц: длинные не влезают в колонку.
SHORT_REASONS = {
    "sl": "стоп",
    "pivot": "пивот",
    "be": "безуб",
    "tp3": "тейк 3",
}

#: Колонки причин строятся из REASON_ORDER, а не перечисляются руками:
#: добавили причину в trade_rules — она сама появилась в таблице.
GROUP_HEADER = (
    f"    {'':<18}{'n':>5} {'среднее':>9} {'медиана':>9} {'R':>7} {'плюс':>6}"
    + "".join(f"{SHORT_REASONS[reason]:>7}" for reason in REASON_ORDER)
)


def _group_line(label: str, stats: dict) -> str:
    if not stats["count"]:
        return f"    {label:<18}{0:>5}"
    shares = stats["reason_shares"]
    tail = "".join(_rate(shares[reason], 7) for reason in REASON_ORDER)
    return (
        f"    {label:<18}{stats['count']:>5} {_pct(stats['result'], 9)} "
        f"{_pct(stats['median'], 9)} {_num(stats['r'], 7, 2)} "
        f"{_rate(stats['win_rate'], 6)}{tail}"
    )


def print_groups(title: str, groups: dict[str, list[Trade]]) -> None:
    """Разрез: имя группы → сделки. Пустые группы печатаются как n=0,
    чтобы в таблице было видно, что корзина пуста, а не потеряна."""
    if not groups:
        return
    print(f"\n  {title}")
    print(GROUP_HEADER)
    for label, part in groups.items():
        print(_group_line(label, aggregate(part)))


def print_summary(trades: list[Trade], title: str) -> None:
    stats = aggregate(trades)
    print(f"\n=== {title} ===")
    if not stats["count"]:
        print(f"  закрытых сделок нет (открытых {stats['open']})")
        return
    print(
        f"  закрытых сделок: {stats['count']} · "
        f"среднее {_pct(stats['result'])} · медиана {_pct(stats['median'])} · "
        f"винрейт {stats['win_rate']:.0f}% · среднее в R {stats['r']:+.2f}\n"
        f"  ход в пользу {_pct(stats['mfe'])} · ход против {_pct(stats['mae'])} · "
        f"дистанция стопа {_num(stats['sl_dist'], 5, 2)}% · "
        f"держится {_num(stats['bars'])} бар(ов)\n"
        f"  не закрыто к концу истории: {stats['open']} "
        f"(в средние и винрейт не входят)\n"
        f"  зафиксировано в среднем по всем {stats['total']} сделкам: "
        f"{_pct(stats['realized_all'])} "
        f"(открытые учтены по взятым тейкам)"
    )


def print_breakdowns(trades: list[Trade]) -> None:
    """Разрезы отчёта. Главный из них — по дистанции стопа: он показывает,
    зависит ли результат от того, насколько далеко оказался структурный стоп."""
    if not trades:
        return

    print_groups("по дистанции стопа:", group_by_sl_distance(trades))

    by_reason: dict[str, list[Trade]] = {
        SHORT_REASONS[reason]: [t for t in trades if t.exit_reason == reason]
        for reason in REASON_ORDER
    }
    print_groups("по причине выхода:", by_reason)

    by_direction: dict[str, list[Trade]] = {}
    for trade in trades:
        by_direction.setdefault(trade.direction.value, []).append(trade)
    if len(by_direction) > 1:
        print_groups("по направлению:", dict(sorted(by_direction.items())))

    by_tf: dict[str, list[Trade]] = {}
    for trade in trades:
        by_tf.setdefault(trade.timeframe.value, []).append(trade)
    if len(by_tf) > 1:
        print_groups("по таймфрейму:", dict(sorted(by_tf.items())))

    by_degree: dict[str, list[Trade]] = {}
    for trade in trades:
        by_degree.setdefault(f"×{trade.degree}", []).append(trade)
    if len(by_degree) > 1:
        print_groups("по числу точек:", dict(sorted(by_degree.items())))


def print_report(symbol: str, tf: Timeframe, df: pd.DataFrame, trades: list[Trade]) -> None:
    """Компактная сводка по одной паре: подробные разрезы — в общем отчёте."""
    span = ""
    if not df.empty:
        span = (
            f"{df['open_time'].iloc[0]:%Y-%m-%d} → {df['open_time'].iloc[-1]:%Y-%m-%d}"
        )
    stats = aggregate(trades)
    print(f"\n=== {symbol} · {tf.value} · {len(df)} баров · {span} ===")
    if not stats["count"]:
        print(f"  закрытых сделок нет (открытых {stats['open']})")
        return
    shares = stats["reason_shares"]
    print(
        f"  закрыто {stats['count']} (открыто {stats['open']}) · "
        f"среднее {_pct(stats['result'])} · медиана {_pct(stats['median'])} · "
        f"винрейт {stats['win_rate']:.0f}% · R {stats['r']:+.2f}\n"
        f"  исходы: sl {shares['sl']:.0f}% · be {shares['be']:.0f}% · "
        f"tp3 {shares['tp3']:.0f}%"
    )


def print_full_report(trades: list[Trade]) -> None:
    """Секции 1–6 отчёта: сводка, разрезы, затем то же самое по независимым
    входам (first_in_series)."""
    print_summary(trades, "ИТОГО")
    print_breakdowns(trades)

    first = [t for t in trades if t.first_in_series]
    if first and len(first) != len(trades):
        print_summary(first, "ТОЛЬКО НЕЗАВИСИМЫЕ ВХОДЫ (first_in_series)")
        print_breakdowns(first)


def print_sweep(rows: list[tuple[dict, dict]], keys: list[str]) -> None:
    print("\n=== перебор параметров ===")
    header = " · ".join(f"{key}" for key in keys)
    print(f"{header:<30} {'n':>5} {'среднее':>9} {'медиана':>9} {'R':>7} "
          f"{'плюс':>6} {'sl':>6} {'be':>6} {'tp3':>6}")
    ranked = sorted(
        rows,
        key=lambda row: (row[1]["result"] if row[1]["result"] is not None else -1e9),
        reverse=True,
    )
    for combo, stats in ranked:
        label = " · ".join(f"{combo[key]}" for key in keys)
        if not stats["count"]:
            print(f"{label:<30} {0:>5}")
            continue
        shares = stats["reason_shares"]
        print(
            f"{label:<30} {stats['count']:>5} {_pct(stats['result'], 9)} "
            f"{_pct(stats['median'], 9)} {_num(stats['r'], 7, 2)} "
            f"{_rate(stats['win_rate'], 6)} {_rate(shares['sl'], 6)} "
            f"{_rate(shares['be'], 6)} {_rate(shares['tp3'], 6)}"
        )
    print(
        "\nОсторожно с верхней строкой: при переборе десятков комбинаций лучшая "
        "из них\nвыигрывает отчасти случайно. Смотрите на устойчивость — "
        "хорошо, когда\nсоседние значения параметра дают похожий результат, "
        "а не одинокий пик."
    )


def print_signal_summary(signals: list) -> None:
    """Сводка по сигналам + проверка приёмочного утверждения 2.

    Утверждение проверяется ТОЛЬКО для тех ALERT #1, что пришли при живой
    цепочке (у них заполнен displaced_anchor). Для ALERT #1 на пустой цепочке
    ограничения по цене нет и быть не должно: сравнивать было не с чем.
    """
    if not signals:
        return
    by_type: dict[str, int] = {}
    for signal in signals:
        by_type[signal.type.value] = by_type.get(signal.type.value, 0) + 1

    print("\n=== сигналы ===")
    for name in sorted(by_type):
        print(f"  {name:<22} {by_type[name]}")

    displacements = [s for s in signals if s.displaced_anchor is not None]
    fresh_starts = [
        s for s in signals
        if s.displaced_anchor is None and s.degree == 1 and s.reference is None
        and s.type.value in {"oversold_pivot", "overbought_pivot"}
    ]
    replacements = [s for s in signals if s.replaced_from is not None]
    print(
        f"  из них ALERT #1 с вытеснением опорной: {len(displacements)}, "
        f"на пустой цепочке: {len(fresh_starts)}"
    )
    print(f"  ALERT #2, являющихся заменой точки:    {len(replacements)}")

    violations = []
    for signal in displacements:
        anchor = signal.displaced_anchor
        broke = (
            signal.price < anchor.price
            if signal.direction.value == "bull"
            else signal.price > anchor.price
        )
        if not broke:
            violations.append(signal)

    if violations:
        print(
            f"  !! УТВЕРЖДЕНИЕ 2 НАРУШЕНО: {len(violations)} ALERT #1 вытеснили "
            f"опорную, не обновив её цену — это баг"
        )
        for signal in violations[:10]:
            anchor = signal.displaced_anchor
            print(
                f"     {signal.symbol} {signal.timeframe.value} "
                f"{signal.candle_time.isoformat()}: цена {signal.price} "
                f"против опорной {anchor.price}"
            )
    else:
        print(
            "  утверждение 2: OK — каждый ALERT #1, вытеснивший живую опорную, "
            "обновил её цену"
        )


# --- точка входа ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Прогон RSI-детектора по истории Binance"
    )
    parser.add_argument("--symbols", default="", help="через запятую; по умолчанию — из config.yaml")
    parser.add_argument("--timeframes", "--tf", dest="timeframes", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "config.yaml"))
    parser.add_argument("--db", default="data/state.db", help="откуда взять правки /params")
    parser.add_argument("--no-db-params", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument(
        "--no-network", action="store_true",
        help="работать только на кэше; не хватает данных — падать",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="перекачать историю заново",
    )
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2")
    parser.add_argument("--csv", default="")
    parser.add_argument(
        "--signals-csv", dest="signals_csv", default="",
        help="выгрузить ВСЕ сигналы (не только сделки) для аудита расхождений",
    )
    args = parser.parse_args()

    assignments = parse_assignments(args.param)
    ladder, detector = split_ladder(assignments)
    base_params, layers = load_params(
        args.config, None if args.no_db_params else args.db, detector
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = _symbols_from_config(args.config)
    timeframes = [Timeframe.parse(chunk) for chunk in args.timeframes.split(",") if chunk.strip()]

    sweep = parse_sweep(args.sweep)
    sweep_ladder = {key: values for key, values in sweep.items() if key in LADDER_KEYS}
    sweep_detector = {key: values for key, values in sweep.items() if key not in LADDER_KEYS}

    rules_by_tf = {tf: rules_for(tf, ladder) for tf in timeframes}
    for tf in timeframes:
        params = resolve_params(base_params, layers, tf)
        print(
            f"Пороги {tf.value}: RSI({params.rsi_period}), "
            f"fractal_n={params.fractal_n}, oversold={params.oversold:g}, "
            f"div_max={params.divergence_rsi_max:g}, "
            f"anchor_ttl={params.anchor_ttl_bars}, "
            f"max_between={params.max_bars_between_points}, "
            f"min_between={params.min_bars_between_points}, "
            f"chain_max_points={params.chain_max_points}, "
            f"bearish={'вкл' if params.bearish_enabled else 'выкл'}",
            file=sys.stderr,
        )
        # Через месяц по CSV уже не восстановить, по каким цифрам он считался.
        print(f"Выходы {tf.value}: {rules_by_tf[tf].describe()}", file=sys.stderr)

    settings = CacheSettings(
        cache_dir=Path(args.cache_dir),
        base_url=args.base_url,
        offline=args.no_network,
        refresh=args.refresh,
    )
    needs_hourly = any(
        rules.intrabar_resolution == "1h" and supports_intrabar(tf)
        for tf, rules in rules_by_tf.items()
    )
    for tf, rules in rules_by_tf.items():
        if rules.intrabar_resolution == "1h" and not supports_intrabar(tf):
            print(
                f"{tf.value}: порядок внутри бара часами не разрешается, "
                f"работает правило «стоп первым»",
                file=sys.stderr,
            )

    history: dict[tuple[str, Timeframe], pd.DataFrame] = {}
    hourly: dict[str, pd.DataFrame] = {}
    try:
        for symbol in symbols:
            for tf in timeframes:
                history[(symbol, tf)] = load_history(
                    symbol, tf, args.years, settings=settings
                )
            if needs_hourly:
                # Часовые свечи качаются для тех же активов и того же периода,
                # что и основной таймфрейм, и только когда они реально нужны.
                hourly[symbol] = load_history(
                    symbol, Timeframe.H1, args.years, settings=settings
                )
    except CacheError as exc:
        raise SystemExit(str(exc)) from exc

    subbars: dict[tuple[str, Timeframe], SubBarIndex | None] = {}
    for (symbol, tf) in history:
        rules = rules_by_tf[tf]
        subbars[(symbol, tf)] = (
            build_subbars(hourly.get(symbol), tf)
            if rules.intrabar_resolution == "1h" else None
        )  # build_subbars сам отсеет таймфреймы, которые часами не разложить

    if sweep:
        keys = list(sweep)
        rows: list[tuple[dict, dict]] = []
        for combo_values in itertools.product(*(sweep[key] for key in keys)):
            combo = dict(zip(keys, combo_values))
            ladder_combo = {k: v for k, v in combo.items() if k in sweep_ladder}
            detector_combo = {k: v for k, v in combo.items() if k in sweep_detector}
            trades: list[Trade] = []
            for (symbol, tf), df in history.items():
                if len(df) < 60:
                    continue
                candidate = resolve_params(base_params, layers, tf).model_copy(
                    update=detector_combo
                )
                trades += run_one(
                    symbol, tf, df, candidate,
                    rules_for(tf, ladder, ladder_combo),
                    subbars[(symbol, tf)],
                )[0]
            rows.append((combo, aggregate(trades)))
        print_sweep(rows, keys)
        return

    all_trades: list[Trade] = []
    all_signals: list = []
    for (symbol, tf), df in history.items():
        if len(df) < 60:
            print(f"\n=== {symbol} · {tf.value} === мало данных ({len(df)} баров)")
            continue
        params = resolve_params(base_params, layers, tf)
        trades, signals = run_one(
            symbol, tf, df, params, rules_by_tf[tf], subbars[(symbol, tf)]
        )
        all_trades += trades
        all_signals += signals
        print_report(symbol, tf, df, trades)

    print_signal_summary(all_signals)
    print_full_report(all_trades)

    if args.csv:
        dump_csv(args.csv, all_trades)
    if args.signals_csv:
        dump_signals(args.signals_csv, all_signals)


def _symbols_from_config(config_path: str) -> list[str]:
    try:
        from app.config import load_config

        config = load_config(config_path)
        symbols = [a.symbol for a in config.assets if a.enabled and a.provider == "binance"]
        if symbols:
            return symbols
    except Exception:  # noqa: BLE001
        pass
    return ["BTCUSDT"]


if __name__ == "__main__":
    main()
