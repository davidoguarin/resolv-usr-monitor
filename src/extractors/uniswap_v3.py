"""
Uniswap V3 historical swap extractor via The Graph subgraph.

Strategy
--------
Query individual swap events for a pool within the target time window using
cursor-based pagination (timestamp_gt on the last seen timestamp) to avoid
The Graph's 5 000-item skip limit.  Swaps are then aggregated into 5-minute
OHLCV buckets by the aggregator module.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GraphQL query – fetch swaps ordered by timestamp (cursor-based pagination)
# ---------------------------------------------------------------------------
_SWAPS_QUERY = """
query PoolSwaps(
  $pool:   String!
  $ts_gte: Int!
  $ts_lte: Int!
  $first:  Int!
) {
  swaps(
    first: $first
    where: {
      pool:          $pool
      timestamp_gte: $ts_gte
      timestamp_lte: $ts_lte
    }
    orderBy:        timestamp
    orderDirection: asc
  ) {
    timestamp
    token0Price
    token1Price
    amount0
    amount1
    sqrtPriceX96
    logIndex
  }
}
"""

# Cursor-based continuation — filter by timestamp_gt to advance the page
_SWAPS_QUERY_CURSOR = """
query PoolSwapsCursor(
  $pool:      String!
  $ts_gt:     Int!
  $ts_lte:    Int!
  $first:     Int!
) {
  swaps(
    first: $first
    where: {
      pool:          $pool
      timestamp_gt:  $ts_gt
      timestamp_lte: $ts_lte
    }
    orderBy:        timestamp
    orderDirection: asc
  ) {
    timestamp
    token0Price
    token1Price
    amount0
    amount1
    sqrtPriceX96
    logIndex
  }
}
"""


class UniswapV3Extractor:
    """Fetches all swap events for one or more pools from The Graph."""

    def __init__(self, cfg: dict):
        api_cfg = cfg["apis"]["thegraph"]
        rate_cfg = cfg["apis"]["rate_limits"]

        api_key = os.getenv("THEGRAPH_API_KEY", "")
        if api_key:
            self._url = api_cfg["uniswap_v3_gateway"].format(api_key=api_key)
            log.info("Using The Graph decentralized network endpoint.")
        else:
            self._url = os.getenv(
                "THEGRAPH_UNISWAP_V3_URL", api_cfg["uniswap_v3_url"]
            )
            log.warning(
                "THEGRAPH_API_KEY not set — using free hosted endpoint "
                "(rate-limited). Set the key in .env for reliability."
            )

        self._page_size: int = api_cfg["page_size"]
        self._limiter = RateLimiter(rate_cfg["requests_per_second"])
        self._retry_attempts: int = rate_cfg["retry_attempts"]
        self._retry_backoff: float = rate_cfg["retry_backoff_seconds"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_pool_swaps(
        self,
        session: aiohttp.ClientSession,
        pool_cfg: dict,
        ts_start: int,
        ts_end: int,
    ) -> list[dict]:
        """Return all swap records for *pool_cfg* between ts_start and ts_end."""
        pool_addr = pool_cfg["address"].lower()
        pool_id = pool_cfg["id"]
        quote_is_token1: bool = pool_cfg["quote_is_token1"]

        log.info(
            "[%s] Fetching Uniswap V3 swaps %s → %s ...",
            pool_id, ts_start, ts_end,
        )

        raw_swaps = await self._paginate(session, pool_addr, ts_start, ts_end)
        log.info("[%s] Retrieved %d raw swaps.", pool_id, len(raw_swaps))

        records = []
        for s in raw_swaps:
            price_usdc = (
                float(s["token1Price"]) if quote_is_token1 else float(s["token0Price"])
            )
            records.append(
                {
                    "timestamp": int(s["timestamp"]),
                    "price_usdc": price_usdc,
                    "amount0": float(s["amount0"]),
                    "amount1": float(s["amount1"]),
                    "pool_id": pool_id,
                    "dex": "uniswap_v3",
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
        pool_addr: str,
        ts_start: int,
        ts_end: int,
    ) -> list[dict]:
        all_swaps: list[dict] = []
        cursor_ts: int | None = None  # None → first page uses ts_gte

        while True:
            if cursor_ts is None:
                variables = {
                    "pool": pool_addr,
                    "ts_gte": ts_start,
                    "ts_lte": ts_end,
                    "first": self._page_size,
                }
                query = _SWAPS_QUERY
            else:
                variables = {
                    "pool": pool_addr,
                    "ts_gt": cursor_ts,
                    "ts_lte": ts_end,
                    "first": self._page_size,
                }
                query = _SWAPS_QUERY_CURSOR

            page = await self._post_graphql(session, query, variables)
            swaps = page.get("swaps", [])

            if not swaps:
                break

            all_swaps.extend(swaps)
            log.debug(
                "  page fetched: %d swaps (total so far: %d)",
                len(swaps), len(all_swaps),
            )

            if len(swaps) < self._page_size:
                break  # last page

            # Advance cursor — multiple swaps can share the same block/timestamp,
            # so move to the timestamp of the last item.
            cursor_ts = int(swaps[-1]["timestamp"])

        return all_swaps

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _post_graphql(
        self,
        session: aiohttp.ClientSession,
        query: str,
        variables: dict,
    ) -> dict[str, Any]:
        await self._limiter.acquire()
        payload = {"query": query, "variables": variables}
        async with session.post(self._url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            body = await resp.json()

        if "errors" in body:
            raise RuntimeError(f"GraphQL errors: {body['errors']}")

        return body.get("data", {})
