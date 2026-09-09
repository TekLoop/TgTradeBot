"""Офлайн-прогон детектора по истории.

    python -m app.backtest --symbols BTCUSDT --timeframes 1H,4H,1D,1W --years 2

Скрипт ничего не пишет в рабочую БД бота и никуда не отправляет сообщений:
качает свечи, гоняет тот же DivergenceEngine и печатает сводку.

Перебор параметров:

    python -m app.backtest --tf 4H --sweep oversold=25,30,35 --sweep fractal_n=2,3

Выгрузка всех сделок для ручного разбора:

    python -m app.backtest --tf 4H --csv data/history/trades.csv

История кэшируется в data/history/*.csv, поэтому повторные прогоны и переборы
идут без обращения к сети. Кэш дописывается свежими барами, а не качается заново.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from app.analysis.replay import REASON_LABELS, REASON_REVERSE, Trade, aggregate, evaluate
from app.analysis.signals import DivergenceEngine
from app.core.timeframes import Timeframe
from app.data.resampling import resample_ohlcv

DEFAULT_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
DEFAULT_TIMEFRAMES = "1H,4H,1D,1W"
REQUEST_LIMIT = 1000
REQUEST_PAUSE = 0.25

#: Нативные интервалы Binance. 3H собирается из 1H, 1W — из 1D,
#: ровно как в app/data/binance.py.
NATIVE_INTERVALS: dict[Timeframe, str] = {
    Timeframe.H1: "1h",
    Timeframe.H2: "2h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}

OHLCV = ["open_time", "open", "high", "low", "close", "volume"]


# --- загрузка истории -------------------------------------------------------


def base_timeframe(tf: Timeframe) -> Timeframe:
    if tf in NATIVE_INTERVALS:
        return tf
    candidates = [base for base in NATIVE_INTERVALS if base.divides(tf)]
    if not candidates:
        raise ValueError(f"Не из чего собрать {tf.value}")
    return max(candidates, key=lambda base: base.minutes)


def _get_json(url: str, retries: int = 4):
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "rsi-divergence-backtest/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 451:
                raise SystemExit(
                    "Binance вернул 451 — этот IP заблокирован. "
                    "Проверьте базовый URL и регион сервера."
                ) from exc
            if exc.code not in {418, 429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 20.0)
    raise RuntimeError("недостижимо")


def fetch_klines(
    symbol: str, interval: str, start_ms: int, base_url: str, step_ms: int
) -> pd.DataFrame:
    """Постраничная выкачка свечей от start_ms до текущего момента."""
    now_ms = int(time.time() * 1000)
    cursor = start_ms
    rows: list[dict] = []
    requests = 0

    while cursor < now_ms:
        query = urlencode(
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "limit": REQUEST_LIMIT,
            }
        )
        payload = _get_json(f"{base_url.rstrip('/')}/api/v3/klines?{query}")
        requests += 1
        if not payload:
            break

        for item in payload:
            if int(item[6]) >= now_ms:
                continue  # свеча ещё не закрылась
            rows.append(
                {
                    "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )

        cursor = int(payload[-1][0]) + step_ms
        print(
            f"  … {symbol} {interval}: {len(rows)} баров",
            file=sys.stderr, end="\r", flush=True,
        )
        if len(payload) < REQUEST_LIMIT:
            break
        time.sleep(REQUEST_PAUSE)

    if requests:
        print(
            f"  {symbol} {interval}: загружено {len(rows)} баров "
            f"за {requests} запрос(ов)   ",
            file=sys.stderr,
        )
    return pd.DataFrame(rows, columns=OHLCV)


def _cache_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    return cache_dir / f"{symbol.upper()}_{interval}.csv"


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Склейка без пустых кусков: пустой кадр сбивает тип колонки времени."""
    filled = [frame for frame in frames if not frame.empty]
    if not filled:
        return pd.DataFrame(columns=OHLCV)
    if len(filled) == 1:
        return filled[0].reset_index(drop=True)
    return pd.concat(filled, ignore_index=True)


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OHLCV)
    df = pd.read_csv(path)
    # В кэше время лежит целыми миллисекундами: никакой возни с разбором
    # строковых таймзон при чтении.
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def load_history(
    symbol: str,
    tf: Timeframe,
    years: float,
    *,
    cache_dir: Path,
    base_url: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Отдаёт закрытые свечи нужного ТФ за последние `years` лет."""
    base = base_timeframe(tf)
    interval = NATIVE_INTERVALS[base]
    step_ms = base.minutes * 60_000
    start = datetime.now(timezone.utc) - timedelta(days=365.25 * years)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, interval)
    cached = _read_cache(path) if use_cache else pd.DataFrame(columns=OHLCV)

    start_ms = int(start.timestamp() * 1000)
    if not cached.empty:
        # Докачиваем только то, чего нет: хвост и, если нужно, начало.
        # Первый бар кэша почти никогда не совпадает со start до миллисекунды —
        # запас в одну свечу, иначе голова качалась бы при каждом прогоне.
        if cached["open_time"].min() - pd.Timestamp(start) > base.duration:
            fresh_head = fetch_klines(symbol, interval, start_ms, base_url, step_ms)
            cached = _concat([fresh_head, cached])
        tail_from = int(cached["open_time"].max().timestamp() * 1000) + step_ms
        fresh_tail = fetch_klines(symbol, interval, tail_from, base_url, step_ms)
        cached = _concat([cached, fresh_tail])
    else:
        cached = fetch_klines(symbol, interval, start_ms, base_url, step_ms)

    if cached.empty:
        return cached

    cached = (
        cached.drop_duplicates(subset="open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if use_cache:
        out = cached.copy()
        out["open_time"] = (
            out["open_time"] - pd.Timestamp("1970-01-01", tz="UTC")
        ) // pd.Timedelta("1ms")
        out.to_csv(path, index=False)

    window = cached.loc[cached["open_time"] >= pd.Timestamp(start)].reset_index(drop=True)
    if base is tf:
        return window
    return resample_ohlcv(window, tf, source=base)


# --- параметры --------------------------------------------------------------


def load_params(config_path: str, db_path: str | None, overrides: dict):
    """Пороги: config.yaml → оверрайды из /params в state.db → аргументы CLI."""
    from app.config import EDITABLE_PARAMS, ParamsConfig, load_config

    try:
        params = load_config(config_path).params
    except Exception as exc:  # noqa: BLE001 — прогон возможен и без конфига
        print(f"config.yaml не прочитан ({exc}), беру значения по умолчанию",
              file=sys.stderr)
        params = ParamsConfig()

    merged = params.model_dump()

    if db_path and Path(db_path).exists():
        try:
            with sqlite3.connect(db_path) as connection:
                rows = connection.execute("SELECT key, value FROM params").fetchall()
            applied = []
            for key, raw in rows:
                if key in EDITABLE_PARAMS:
                    merged[key] = json.loads(raw)
                    applied.append(key)
            if applied:
                print(f"Из state.db подтянуты правки /params: {', '.join(applied)}",
                      file=sys.stderr)
        except sqlite3.Error as exc:
            print(f"state.db не прочитан ({exc})", file=sys.stderr)

    merged.update(overrides)
    return ParamsConfig.model_validate(merged)


def parse_assignments(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Ожидал key=value, получил: {item}")
        key, raw = item.split("=", 1)
        out[key.strip()] = _coerce(raw.strip())
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


# --- прогон и отчёт ---------------------------------------------------------


def run_one(symbol: str, tf: Timeframe, df: pd.DataFrame, params) -> list[Trade]:
    engine = DivergenceEngine(params.to_signal_params())
    result = engine.analyze(df, symbol, tf)
    return evaluate(df, result.signals, timeout_bars=params.outcome_timeout_bars)


def _pct(value: float | None, width: int = 7) -> str:
    return "—".rjust(width) if value is None else f"{value:+.2f}%".rjust(width)


def _num(value: float | None, width: int = 5) -> str:
    return "—".rjust(width) if value is None else f"{value:.0f}".rjust(width)


def print_report(symbol: str, tf: Timeframe, df: pd.DataFrame, trades: list[Trade]) -> None:
    span = ""
    if not df.empty:
        span = (
            f"{df['open_time'].iloc[0]:%Y-%m-%d} → {df['open_time'].iloc[-1]:%Y-%m-%d}"
        )
    print(f"\n=== {symbol} · {tf.value} · {len(df)} баров · {span} ===")

    total = aggregate(trades)
    if not total["count"]:
        print(f"  закрытых сделок нет (открытых {total['open']})")
        return

    print(
        f"  сделок закрыто: {total['count']} (открытых {total['open']})\n"
        f"  итог:         среднее {_pct(total['result'])} · "
        f"медиана {_pct(total['median'])} · "
        f"плюсовых {total['wins']} ({total['win_rate']:.0f}%)\n"
        f"  ход в пользу: {_pct(total['favorable'])} "
        f"(лучший {_pct(total['best'])}) · за {_num(total['bars_to_favorable'])} бар(ов)\n"
        f"  ход против:   {_pct(total['adverse'])} (худший {_pct(total['worst'])})\n"
        f"  держится:     {_num(total['bars'])} бар(ов)"
    )

    by_degree: dict[int, list[Trade]] = {}
    for trade in trades:
        by_degree.setdefault(trade.degree, []).append(trade)
    if len(by_degree) > 1:
        print("  по числу точек:")
        for degree in sorted(by_degree):
            part = aggregate(by_degree[degree])
            if not part["count"]:
                continue
            print(
                f"    ×{degree}: n={part['count']:<4} итог {_pct(part['result'])} · "
                f"в пользу {_pct(part['favorable'])} · против {_pct(part['adverse'])} · "
                f"плюсовых {part['win_rate']:.0f}%"
            )

    by_reason: dict[str, list[Trade]] = {}
    for trade in trades:
        by_reason.setdefault(trade.exit_reason, []).append(trade)
    if len(by_reason) > 1:
        print("  по причине выхода:")
        for reason, part_trades in by_reason.items():
            part = aggregate(part_trades)
            if not part["count"]:
                continue
            print(
                f"    {REASON_LABELS.get(reason, reason):<34} n={part['count']:<4} "
                f"итог {_pct(part['result'])}"
            )

    clean = [t for t in trades if t.exit_reason == REASON_REVERSE]
    clean_stats = aggregate(clean)
    if clean_stats["count"] and clean_stats["count"] != total["count"]:
        print(
            f"  только выход по обратной дивергенции: n={clean_stats['count']} · "
            f"итог {_pct(clean_stats['result'])} · "
            f"плюсовых {clean_stats['win_rate']:.0f}%"
        )


def print_sweep(rows: list[tuple[dict, dict]], keys: list[str]) -> None:
    print("\n=== перебор параметров ===")
    header = " · ".join(f"{key}" for key in keys)
    print(f"{header:<38} {'n':>5} {'итог':>8} {'медиана':>9} {'плюс':>6} "
          f"{'в пользу':>9} {'против':>9}")
    ranked = sorted(
        rows,
        key=lambda row: (row[1]["result"] if row[1]["result"] is not None else -1e9),
        reverse=True,
    )
    for combo, stats in ranked:
        label = " · ".join(f"{combo[key]}" for key in keys)
        if not stats["count"]:
            print(f"{label:<38} {0:>5}")
            continue
        print(
            f"{label:<38} {stats['count']:>5} {_pct(stats['result'], 8)} "
            f"{_pct(stats['median'], 9)} {stats['win_rate']:>5.0f}% "
            f"{_pct(stats['favorable'], 9)} {_pct(stats['adverse'], 9)}"
        )
    print(
        "\nОсторожно с верхней строкой: при переборе десятков комбинаций лучшая "
        "из них\nвыигрывает отчасти случайно. Смотрите на устойчивость — "
        "хорошо, когда\nсоседние значения параметра дают похожий результат, "
        "а не одинокий пик."
    )


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
    parser.add_argument("--cache-dir", default="data/history")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    overrides = parse_assignments(args.param)
    params = load_params(args.config, None if args.no_db_params else args.db, overrides)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = _symbols_from_config(args.config)
    timeframes = [Timeframe.parse(chunk) for chunk in args.timeframes.split(",") if chunk.strip()]

    print(
        f"Пороги: RSI({params.rsi_period}), fractal_n={params.fractal_n}, "
        f"oversold={params.oversold:g}, div_max={params.divergence_rsi_max:g}, "
        f"max_bars_between={params.max_bars_between}, "
        f"chain_max_points={params.chain_max_points}, "
        f"bearish={'вкл' if params.bearish_enabled else 'выкл'}, "
        f"timeout={params.outcome_timeout_bars}",
        file=sys.stderr,
    )

    history: dict[tuple[str, Timeframe], pd.DataFrame] = {}
    for symbol in symbols:
        for tf in timeframes:
            df = load_history(
                symbol, tf, args.years,
                cache_dir=Path(args.cache_dir),
                base_url=args.base_url,
                use_cache=not args.no_cache,
            )
            history[(symbol, tf)] = df

    sweep = parse_sweep(args.sweep)
    if sweep:
        keys = list(sweep)
        rows: list[tuple[dict, dict]] = []
        for combo_values in itertools.product(*(sweep[key] for key in keys)):
            combo = dict(zip(keys, combo_values))
            candidate = params.model_copy(update=combo)
            trades: list[Trade] = []
            for (symbol, tf), df in history.items():
                if len(df) < 60:
                    continue
                trades += run_one(symbol, tf, df, candidate)
            rows.append((combo, aggregate(trades)))
        print_sweep(rows, keys)
        return

    all_trades: list[Trade] = []
    for (symbol, tf), df in history.items():
        if len(df) < 60:
            print(f"\n=== {symbol} · {tf.value} === мало данных ({len(df)} баров)")
            continue
        trades = run_one(symbol, tf, df, params)
        all_trades += trades
        print_report(symbol, tf, df, trades)

    if len(history) > 1:
        total = aggregate(all_trades)
        print(
            f"\n=== ИТОГО по всем парам ===\n"
            f"  закрытых сделок: {total['count']} · "
            f"итог {_pct(total['result'])} · медиана {_pct(total['median'])} · "
            f"плюсовых {total['wins']}"
            if total["count"] else "\n=== ИТОГО === закрытых сделок нет"
        )

    if args.csv:
        dump_csv(args.csv, all_trades)


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
