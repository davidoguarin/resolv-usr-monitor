"""
Curve Finance historical price extractor via the curve-prices API.

Endpoint
--------
  GET https://prices.curve.finance/v1/ohlc/{chain}/{pool_address}
      ?main_token={token_addr}
      &reference_token={token_addr}
      &start={unix_ts}
      &end={unix_ts}
      &step={seconds}

Important: the Curve prices API enforces a minimum step of 3 600 seconds
(1 hour), regardless of what is passed.  5-minute data is NOT available
for Curve pools through this endpoint.  The output is clearly labeled as
hourly in the resulting CSV files.

Response shape:
  {
    "chain":   "ethereum",
    "address": "0x...",
    "data": [
      { "time": 1773964800, "open": 1.0005, "high": 1.0007,
        "low": 0.9999, "close": 1.0003 },
      ...
    ]
  }
"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlencode

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

_PRICES_BASE = "https://prices.curve.finance/v1"
_CURVE_RESOLUTION = 3600       # API minimum, regardless of requested step


class CurveExtractor:
    """Fetch hourly OHLCV from the Curve prices API."""

    def __init__(self, cfg: dict):
        rate_cfg = cfg["apis"]["rate_limits"]
        self._limiter = RateLimiter(rate_cfg["requests_per_second"])
        self._chain = cfg["apis"]["curve"]["chain"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_pool_ohlcv(
        self,
        session: aiohttp.ClientSession,
        pool_cfg: dict,
        ts_start: int,
        ts_end: int,
        step: int = _CURVE_RESOLUTION,
    ) -> list[dict]:
        """Fetch OHLCV candles from the Curve prices API.

        Returns hourly candles (Curve API minimum = 3 600 s).
        """
        pool_id = pool_cfg["id"]
        address = pool_cfg["address"].lower()
        main_token = pool_cfg["main_token"]
        reference_token = pool_cfg["reference_token"]

        if not address:
            log.error("[%s] Pool address is empty — skipping.", pool_id)
            return []

        params = urlencode(
            {
                "main_token": main_token,
                "reference_token": reference_token,
                "start": ts_start,
                "end": ts_end,
                "step": max(step, _CURVE_RESOLUTION),   # honour API minimum
            }
        )
        url = f"{_PRICES_BASE}/ohlc/{self._chain}/{address}?{params}"
        log.info("[%s] Curve prices API: %s", pool_id, url)

        raw = await self._get(session, url)
        log.info("[%s] Received %d hourly candles.", pool_id, len(raw))

        invert = pool_cfg.get("invert_price", False)
        if invert:
            log.info(
                "[%s] invert_price=true — applying 1/price to all OHLC values "
                "(API returns coin0/coin1; we need coin1/coin0 = USDC per token).",
                pool_id,
            )

        records = []
        for c in raw:
            o, h, l, cl = (
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
            )
            if invert:
                # Inverting flips the high/low direction: 1/low becomes the new high
                o, h, l, cl = 1/o, 1/l, 1/h, 1/cl
            records.append(
                {
                    "timestamp_open": int(c["time"]),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": cl,
                    "volume_usdc": 0.0,      # not returned by this endpoint
                    "base_volume": 0.0,
                    "pool_id": pool_id,
                    "dex": "curve",
                    "token": pool_cfg["token"],
                }
            )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _get(self, session: aiohttp.ClientSession, url: str) -> list[dict]:
        await self._limiter.acquire()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 404:
                log.warning("Curve prices API returned 404: %s", url)
                return []
            resp.raise_for_status()
            body = await resp.json()
        return body.get("data", [])
