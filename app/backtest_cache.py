"""Кэш свечей на диске — только для бэктеста.

Часовые данные за восемь лет по десяти активам нельзя качать при каждом
прогоне, поэтому история лежит на диске и дополняется по краям:

    data/cache/{provider}_{symbol}_{interval}.csv.gz

Модуль намеренно стоит в стороне от app/data/: теми провайдерами пользуется
живой бот, и лишний слой на его пути опроса не нужен. Здесь своя простая
выкачка через urllib, свои паузы и свой ретрай — ровно то, что нужно
офлайн-прогону.

Режимы:
    обычный   — читаем кэш, догружаем недостающие начало и хвост, дописываем;
    offline   — только кэш (--no-network), нехватка данных = внятная ошибка;
    refresh   — перекачать период заново (--refresh).

Время в файле хранится целыми миллисекундами: никакой возни с разбором
строковых таймзон при чтении.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import pandas as pd

OHLCV = ["open_time", "open", "high", "low", "close", "volume"]

REQUEST_LIMIT = 1000
#: Пауза между страницами. Binance считает вес запроса, а не только их число;
#: четверть секунды на страницу — это давно проверенный на этом проекте режим.
REQUEST_PAUSE = 0.25
#: Коды, после которых имеет смысл подождать и повторить. 418 Binance выдаёт
#: за игнорирование 429 — это уже известная больная точка проекта.
RETRY_CODES = frozenset({418, 429, 500, 502, 503, 504})
MAX_BACKOFF = 60.0

Fetcher = Callable[[str, str, int, int], pd.DataFrame]


class CacheError(RuntimeError):
    """Кэша не хватает на запрошенный период, а сеть запрещена."""


@dataclass(frozen=True, slots=True)
class CacheSettings:
    cache_dir: Path = Path("data/cache")
    base_url: str = "https://data-api.binance.vision"
    provider: str = "binance"
    offline: bool = False        # --no-network
    refresh: bool = False        # --refresh
    pause: float = REQUEST_PAUSE
    quiet: bool = False

    def path_for(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{self.provider}_{symbol.upper()}_{interval}.csv.gz"


# --- чтение и запись --------------------------------------------------------


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV)


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty()
    df = pd.read_csv(path)
    if df.empty:
        return _empty()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[OHLCV]


def write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["open_time"] = (
        out["open_time"] - pd.Timestamp("1970-01-01", tz="UTC")
    ) // pd.Timedelta("1ms")
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Склейка без пустых кусков: пустой кадр сбивает тип колонки времени."""
    filled = [frame for frame in frames if not frame.empty]
    if not filled:
        return _empty()
    if len(filled) == 1:
        return filled[0].reset_index(drop=True)
    return pd.concat(filled, ignore_index=True)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset="open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


# --- сеть -------------------------------------------------------------------


def _get_json(url: str, retries: int = 5):
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
            if exc.code not in RETRY_CODES or attempt == retries:
                raise
            # Binance сам говорит, сколько ждать, — это указание, а не совет.
            hint = exc.headers.get("Retry-After") if exc.headers else None
            if hint:
                try:
                    delay = max(delay, float(hint))
                except ValueError:
                    pass
            if exc.code in {418, 429}:
                print(
                    f"  Binance {exc.code}: пауза {delay:.0f} с",
                    file=sys.stderr, flush=True,
                )
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay = min(delay * 2, MAX_BACKOFF)
    raise RuntimeError("недостижимо")


def binance_fetcher(
    base_url: str, step_ms: int, pause: float = REQUEST_PAUSE, quiet: bool = False
) -> Fetcher:
    """Постраничная выкачка klines в диапазоне [start_ms, end_ms)."""

    def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        cursor = start_ms
        rows: list[dict] = []
        requests = 0
        while cursor < end_ms:
            query = urlencode(
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": REQUEST_LIMIT,
                }
            )
            payload = _get_json(f"{base_url.rstrip('/')}/api/v3/klines?{query}")
            requests += 1
            if not payload:
                break
            for item in payload:
                if int(item[6]) >= end_ms:
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
            if not quiet:
                print(
                    f"  … {symbol} {interval}: {len(rows)} баров",
                    file=sys.stderr, end="\r", flush=True,
                )
            if len(payload) < REQUEST_LIMIT:
                break
            time.sleep(pause)
        if requests and not quiet:
            print(
                f"  {symbol} {interval}: докачано {len(rows)} баров "
                f"за {requests} запрос(ов)   ",
                file=sys.stderr,
            )
        return pd.DataFrame(rows, columns=OHLCV)

    return fetch


# --- основная функция -------------------------------------------------------


def load_klines(
    symbol: str,
    interval: str,
    start: datetime,
    step_ms: int,
    *,
    settings: CacheSettings,
    fetcher: Fetcher | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Свечи интервала `interval` от `start` до текущего момента.

    Возвращает только закрытые свечи, отсортированные по времени. Кэш
    дописывается на диск, если что-то докачали.
    """
    moment = now or datetime.now(timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    now_ms = int(moment.timestamp() * 1000)
    path = settings.path_for(symbol, interval)

    cached = _empty() if settings.refresh else _tidy(read_cache(path))
    if settings.offline:
        _ensure_covers(cached, symbol, interval, start, step_ms, path)
        tail = cached["open_time"].max()
        if (now_ms - int(tail.timestamp() * 1000)) > 5 * step_ms and not settings.quiet:
            print(
                f"  {symbol} {interval}: офлайн, история обрывается "
                f"{tail:%Y-%m-%d %H:%M} UTC",
                file=sys.stderr,
            )
        return _window(cached, start)

    fetch = fetcher or binance_fetcher(
        settings.base_url, step_ms, settings.pause, settings.quiet
    )
    grown = False

    if cached.empty:
        cached = _tidy(fetch(symbol, interval, start_ms, now_ms))
        grown = not cached.empty
    else:
        head_ms = int(cached["open_time"].min().timestamp() * 1000)
        tail_ms = int(cached["open_time"].max().timestamp() * 1000)
        # Первый бар кэша почти никогда не совпадает со start до миллисекунды:
        # запас в одну свечу, иначе голова качалась бы при каждом прогоне.
        if head_ms - start_ms > step_ms:
            head = fetch(symbol, interval, start_ms, head_ms)
            if not head.empty:
                cached = _concat([head, cached])
                grown = True
        if now_ms - tail_ms > step_ms:
            tail = fetch(symbol, interval, tail_ms + step_ms, now_ms)
            if not tail.empty:
                cached = _concat([cached, tail])
                grown = True
        cached = _tidy(cached)

    if grown and not cached.empty:
        write_cache(path, cached)

    return _window(cached, start)


def _window(df: pd.DataFrame, start: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df["open_time"] >= pd.Timestamp(start)].reset_index(drop=True)


def _ensure_covers(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    start: datetime,
    step_ms: int,
    path: Path,
) -> None:
    """--no-network: молча отдать неполный период нельзя, иначе прогон
    посчитается по другому куску истории, чем предыдущий, и разницу спишут
    на правила выхода."""
    if df.empty:
        raise CacheError(
            f"{symbol} {interval}: кэша нет ({path}). "
            f"Уберите --no-network и дайте прогону скачать историю."
        )
    head = df["open_time"].min()
    if (pd.Timestamp(head) - pd.Timestamp(start)).total_seconds() * 1000 > step_ms:
        raise CacheError(
            f"{symbol} {interval}: в кэше история с {head:%Y-%m-%d}, "
            f"а запрошено с {start:%Y-%m-%d}. Уберите --no-network "
            f"или сократите --years."
        )
