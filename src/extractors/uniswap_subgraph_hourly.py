"""
Hourly pool stats from the Uniswap V3 subgraph (PoolHourData): TVL, volume, tx count.

Uses THEGRAPH_API_KEY when set (gateway); otherwise the free hosted endpoint
(may be rate-limited).

Curve pools are not supported here — no standard hourly TVL series in this project.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

GRAPH_GW = (
    "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/"
    "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
)

POOL_HOUR_QUERY = """
query PoolHours($pool: String!, $start: Int!, $end: Int!, $first: Int!, $skip: Int!) {
  poolHourDatas(
    where: { pool: $pool, periodStartUnix_gte: $start, periodStartUnix_lte: $end }
    orderBy: periodStartUnix
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    periodStartUnix
    tvlUSD
    volumeUSD
    txCount
    close
    token0Price
    token1Price
  }
}
"""


class UniswapSubgraphHourlyExtractor:
    def __init__(self, cfg: dict):
        rate_cfg = cfg["apis"]["rate_limits"]
        self._limiter = RateLimiter(rate_cfg["requests_per_second"])
        api_key = os.getenv("THEGRAPH_API_KEY", "").strip()
        self._url = (
            GRAPH_GW.format(api_key=api_key)
            if api_key
            else cfg["apis"]["thegraph"]["uniswap_v3_url"]
        )
        self._page = min(int(cfg["apis"]["thegraph"].get("page_size", 1000)), 1000)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _post(self, session: aiohttp.ClientSession, payload: dict) -> dict:
        await self._limiter.acquire()
        async with session.post(
            self._url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
        if body.get("errors"):
            raise RuntimeError(str(body["errors"]))
        return body.get("data", {})

    async def fetch_pool_hourly_stats(
        self,
        session: aiohttp.ClientSession,
        pool_cfg: dict,
        ts_start: int,
        ts_end: int,
    ) -> list[dict]:
        """Return one row per hour with TVL (USD) at end of hour."""
        pool_id = pool_cfg["id"]
        pool_addr = pool_cfg["address"].lower()
        log.info(
            "[%s] Uniswap subgraph poolHourDatas %s → %s ...",
            pool_id,
            datetime.fromtimestamp(ts_start, tz=timezone.utc).date(),
            datetime.fromtimestamp(ts_end, tz=timezone.utc).date(),
        )

        start_h = (ts_start // 3600) * 3600
        end_h = (ts_end // 3600) * 3600

        rows: list[dict] = []
        skip = 0
        while True:
            payload = {
                "query": POOL_HOUR_QUERY,
                "variables": {
                    "pool": pool_addr,
                    "start": start_h,
                    "end": end_h,
                    "first": self._page,
                    "skip": skip,
                },
            }
            data = await self._post(session, payload)
            chunk = data.get("poolHourDatas") or []
            for h in chunk:
                tsu = int(h["periodStartUnix"])
                rows.append(
                    {
                        "timestamp_open": tsu,
                        "datetime_open": datetime.fromtimestamp(
                            tsu, tz=timezone.utc
                        ).isoformat(),
                        "pool_id": pool_id,
                        "dex": "uniswap_v3",
                        "token": pool_cfg["token"],
                        "tvl_usd": float(h.get("tvlUSD") or 0),
                        "volume_usd_hour": float(h.get("volumeUSD") or 0),
                        "tx_count_hour": int(h.get("txCount") or 0),
                        "close_token0_usd": float(h.get("close") or 0),
                        "token0_price": float(h.get("token0Price") or 0),
                        "token1_price": float(h.get("token1Price") or 0),
                    }
                )
            if len(chunk) < self._page:
                break
            skip += self._page
            await asyncio.sleep(0.15)

        log.info("[%s] → %d hourly TVL rows", pool_id, len(rows))
        return rows
