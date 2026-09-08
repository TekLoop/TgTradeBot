"""Общий HTTP-слой для провайдеров: троттлинг, ретраи, таймауты."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import aiohttp

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Ошибка получения данных у провайдера."""


class RateLimitedError(ProviderError):
    pass


class Throttler:
    """Простое ограничение частоты запросов (token bucket)."""

    def __init__(self, rate: int, per_seconds: float = 60.0) -> None:
        self.rate = max(1, rate)
        self.per_seconds = per_seconds
        self._tokens = float(self.rate)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(
                    float(self.rate),
                    self._tokens + elapsed * self.rate / self.per_seconds,
                )
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                await asyncio.sleep(deficit * self.per_seconds / self.rate)


class HttpClient:
    """aiohttp-обёртка: одна сессия на провайдера, ретраи с exponential backoff."""

    RETRYABLE_STATUS = {408, 418, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        name: str,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
        retries: int = 4,
        backoff_base: float = 0.8,
        backoff_max: float = 30.0,
        throttler: Throttler | None = None,
    ) -> None:
        self.name = name
        self.headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.throttler = throttler or Throttler(rate=60)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        headers=self.headers, timeout=self.timeout
                    )
        return self._session

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            await self.throttler.acquire()
            try:
                session = await self._get_session()
                async with session.get(url, params=params) as resp:
                    if resp.status in self.RETRYABLE_STATUS:
                        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                        body = (await resp.text())[:300]
                        last_error = RateLimitedError(
                            f"{self.name}: HTTP {resp.status} {body}"
                        )
                        await self._sleep_backoff(attempt, retry_after)
                        continue
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        raise ProviderError(f"{self.name}: HTTP {resp.status} {body}")
                    return await resp.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                log.warning(
                    "%s: сетевая ошибка (попытка %s/%s): %s",
                    self.name, attempt + 1, self.retries + 1, exc,
                )
                await self._sleep_backoff(attempt, None)

        raise ProviderError(f"{self.name}: запрос не удался: {last_error}")

    async def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            delay = min(retry_after, self.backoff_max)
        else:
            delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        delay += random.uniform(0, delay * 0.25)  # джиттер, чтобы не бить синхронно
        await asyncio.sleep(delay)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
