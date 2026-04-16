"""
GeckoTerminal OHLCV extractor for Uniswap V3 (and any DEX pool supported
by GeckoTerminal's free API).

Endpoint
--------
  GET https://api.geckoterminal.com/api/v2/networks/{network}/pools/{address}/ohlcv/minute
      ?aggregate={minutes}&before_timestamp={unix_ts}&limit=1000&currency=usd&token=base

Response shape:
  {
    "data": {
      "attributes": {
        "ohlcv_list": [
          [timestamp, open, high, low, close, volume_usd],
          ...
        ]
      }
    }
  }

Notes
-----
- Results are returned newest-first; we reverse them after collection.
- Maximum 1000 candles per request.  We paginate by setting
  before_timestamp = oldest_timestamp_seen - 1 until we cover the full
  requested range or the API returns fewer items than the page size.
- Free tier: no API key required; rate-limit gracefully.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

_BASE_URL = "https://api.geckoterminal.com/api/v2"
_HEADERS = {"Accept": "application/json;version=20230302"}


class GeckoTerminalExtractor:
    """Fetch OHLCV from GeckoTerminal for Uniswap V3 pools (bucket size from config or call)."""

    def __init__(self, cfg: dict):
        rate_cfg = cfg["apis"]["rate_limits"]
        self._limiter = RateLimiter(rate_cfg["requests_per_second"])
        self._page_size = 1000          # GeckoTerminal hard max
        self._network = "eth"           # Ethereum mainnet
        self._aggregate_minutes = cfg["extraction"]["interval_seconds"] // 60

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_pool_ohlcv(
        self,
        session: aiohttp.ClientSession,
        pool_cfg: dict,
        ts_start: int,
        ts_end: int,
        aggregate_minutes: Optional[int] = None,
    ) -> list[dict]:
        """Return OHLCV candles for pool_cfg within [ts_start, ts_end].

        aggregate_minutes: GeckoTerminal `aggregate` (default from config interval).
        """
        pool_id = pool_cfg["id"]
        address = pool_cfg["address"].lower()
        agg = (
            aggregate_minutes
            if aggregate_minutes is not None
            else self._aggregate_minutes
        )

        log.info(
            "[%s] Fetching GeckoTerminal OHLCV for %s  (%s → %s) ...",
            pool_id,
            address[:10] + "...",
            datetime.fromtimestamp(ts_start, tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(ts_end, tz=timezone.utc).isoformat(),
        )

        all_candles = await self._paginate(session, address, ts_start, ts_end, agg)
        log.info("[%s] Collected %d raw candles.", pool_id, len(all_candles))

        records = []
        for ts, open_, high, low, close, volume_usd in all_candles:
            records.append(
                {
                    "timestamp_open": int(ts),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume_usdc": float(volume_usd),
                    "base_volume": float(volume_usd),   # GeckoTerminal reports in USD
                    "pool_id": pool_cfg["id"],
                    "dex": pool_cfg["dex"],
                    "token": pool_cfg["token"],
                }
            )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _paginate(
        self,
        session: aiohttp.ClientSession,
        address: str,
        ts_start: int,
        ts_end: int,
        aggregate_minutes: int,
    ) -> list[list]:
        """Collect all candles in [ts_start, ts_end] via before_timestamp pagination."""
        collected: list[list] = []
        before_ts = ts_end + 1  # inclusive upper bound

        while True:
            page = await self._get_page(session, address, before_ts, aggregate_minutes)

            if not page:
                break

            # Page is newest-first; keep only candles within range
            for candle in page:
                candle_ts = candle[0]
                if ts_start <= candle_ts <= ts_end:
                    collected.append(candle)

            oldest_ts = page[-1][0]  # last item = oldest candle on this page

            if oldest_ts <= ts_start:
                break  # we've gone past the start of the window

            if len(page) < self._page_size:
                break  # last page

            # Advance cursor (exclusive: before_ts - 1 to avoid re-fetching boundary)
            before_ts = oldest_ts  # next page will stop before oldest_ts

        # Sort ascending by timestamp (GeckoTerminal returns newest-first)
        collected.sort(key=lambda c: c[0])
        return collected

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _get_page(
        self,
        session: aiohttp.ClientSession,
        address: str,
        before_timestamp: int,
        aggregate_minutes: int,
    ) -> list[list]:
        await self._limiter.acquire()
        url = (
            f"{_BASE_URL}/networks/{self._network}/pools/{address}/ohlcv/minute"
            f"?aggregate={aggregate_minutes}"
            f"&before_timestamp={before_timestamp}"
            f"&limit={self._page_size}"
            f"&currency=usd&token=base"
        )
        log.debug("GET %s", url)
        async with session.get(
            url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status == 429:
                log.warning("GeckoTerminal rate limit hit — waiting 10s ...")
                await asyncio.sleep(10)
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=429
                )
            if resp.status == 403:
                log.error(
                    "GeckoTerminal returned 403 for pool %s — "
                    "this pool may not be supported on the free API tier.",
                    address[:10] + "...",
                )
                return []
            resp.raise_for_status()
            body = await resp.json()

        return body.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
