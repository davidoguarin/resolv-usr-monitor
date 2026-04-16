"""
Etherscan swap-level extractor for Uniswap V3 pools.

Fetches every Swap event in a block range, decodes sqrtPriceX96 → price,
and returns one record per swap (≈ per-block resolution, often finer).

Swap event ABI
--------------
  event Swap(
      address indexed sender,
      address indexed recipient,
      int256  amount0,
      int256  amount1,
      uint160 sqrtPriceX96,   ← word[2] in data
      uint128 liquidity,      ← word[3]
      int24   tick            ← word[4]
  )

Price formula (token0 / token1 notation)
-----------------------------------------
  price_raw      = (sqrtPriceX96 / 2**96) ** 2
  price_in_usdc  = price_raw * 10 ** (decimals_token0 - decimals_token1)

For both our pools token0 has 18 dec (USDe/USR) and token1 is USDC (6 dec):
  price_usdc = price_raw * 1e12

Pagination
----------
Etherscan caps offset=1000. We advance fromBlock = last_block_seen + 1
until the result comes back with fewer than 1000 items.
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

_BASE = "https://api.etherscan.io/v2/api"
_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
_PAGE_SIZE = 1000


def _to_signed(val: int, bits: int = 256) -> int:
    """Convert unsigned two's-complement integer to signed."""
    if val >= 2 ** (bits - 1):
        val -= 2 ** bits
    return val


def _decode_swap(data_hex: str, dec0: int = 18, dec1: int = 6) -> dict:
    """Decode all five words from a Uniswap V3 Swap event data field.

    Layout (each word = 32 bytes = 64 hex chars):
      word[0]  amount0       int256   tokens0 Δ (signed)
      word[1]  amount1       int256   tokens1 Δ (signed)
      word[2]  sqrtPriceX96  uint160
      word[3]  liquidity     uint128  active in-range liquidity
      word[4]  tick          int24    (signed)
    """
    raw   = data_hex[2:]
    words = [int(raw[i:i+64], 16) for i in range(0, len(raw), 64)]

    amount0       = _to_signed(words[0])           # token0 (USDe/USR, 18 dec)
    amount1       = _to_signed(words[1])           # token1 (USDC, 6 dec)
    sqrt_price_x96 = words[2]
    liquidity     = words[3]                       # uint128, always positive

    price_raw     = (sqrt_price_x96 / 2**96) ** 2
    price_usdc    = price_raw * 10 ** (dec0 - dec1)

    return {
        "price_usdc": price_usdc,
        "liquidity":  liquidity,
        "amount0":    amount0 / 10**dec0,          # human-readable token0
        "amount1":    amount1 / 10**dec1,          # human-readable USDC
    }


class EtherscanSwapExtractor:
    """Fetch per-swap prices from Uniswap V3 pools via Etherscan getLogs."""

    def __init__(self):
        self._api_key = os.getenv("ETHERSCAN_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("ETHERSCAN_API_KEY not set in environment / .env")
        self._limiter = RateLimiter(4)   # stay under free-tier 5 req/s

    async def fetch_swaps(
        self,
        session: aiohttp.ClientSession,
        pool_cfg: dict,
        block_start: int,
        block_end: int,
    ) -> list[dict]:
        """Return per-swap records for pool_cfg between block_start and block_end."""
        pool_id  = pool_cfg["id"]
        address  = pool_cfg["address"].lower()
        token    = pool_cfg["token"]

        log.info(
            "[%s] Fetching Etherscan swap events  blocks %d → %d ...",
            pool_id, block_start, block_end,
        )

        all_logs = await self._paginate(session, address, block_start, block_end)
        log.info("[%s] %d raw swap events collected.", pool_id, len(all_logs))

        records = []
        for entry in all_logs:
            try:
                decoded = _decode_swap(entry["data"])
                ts      = int(entry["timeStamp"], 16)
                block   = int(entry["blockNumber"], 16)
                records.append({
                    "timestamp":  ts,
                    "datetime":   datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "block":      block,
                    "price_usdc": decoded["price_usdc"],
                    "liquidity":  decoded["liquidity"],
                    "amount0":    decoded["amount0"],
                    "amount1":    decoded["amount1"],
                    "pool_id":    pool_id,
                    "dex":        "uniswap_v3",
                    "token":      token,
                    "tx_hash":    entry.get("transactionHash", ""),
                    "log_index":  int(entry.get("logIndex", "0x0"), 16),
                })
            except Exception as exc:
                log.warning("Failed to decode swap log: %s", exc)

        records.sort(key=lambda r: (r["block"], r["log_index"]))
        return records

    async def _paginate(
        self,
        session: aiohttp.ClientSession,
        address: str,
        block_start: int,
        block_end: int,
    ) -> list[dict]:
        all_logs: list[dict] = []
        from_block = block_start

        while from_block <= block_end:
            page = await self._get_logs(session, address, from_block, block_end)

            if not page:
                break

            all_logs.extend(page)
            log.debug("  fetched %d events (total %d, from_block=%d)",
                      len(page), len(all_logs), from_block)

            if len(page) < _PAGE_SIZE:
                break

            # Advance past the last block seen
            from_block = int(page[-1]["blockNumber"], 16) + 1

        return all_logs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _get_logs(
        self,
        session: aiohttp.ClientSession,
        address: str,
        from_block: int,
        to_block: int,
    ) -> list[dict]:
        await self._limiter.acquire()
        params = {
            "chainid":   "1",
            "module":    "logs",
            "action":    "getLogs",
            "address":   address,
            "topic0":    _SWAP_TOPIC,
            "fromBlock": str(from_block),
            "toBlock":   str(to_block),
            "page":      "1",
            "offset":    str(_PAGE_SIZE),
            "apikey":    self._api_key,
        }
        async with session.get(
            _BASE, params=params, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()

        if body.get("status") != "1":
            msg = body.get("message", "")
            result = body.get("result", "")
            # "No records found" is not an error
            if "No records" in msg or result == []:
                return []
            raise RuntimeError(f"Etherscan error: {msg} — {result}")

        return body["result"]
