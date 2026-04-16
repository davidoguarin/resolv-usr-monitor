"""Simple token-bucket rate limiter for async HTTP calls."""
import asyncio
import time


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self._rate = requests_per_second
        self._min_interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
