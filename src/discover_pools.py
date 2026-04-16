"""
Pool discovery helper.

Run this once before extraction to:
  1. Resolve the Curve factory-stable-ng-397 pool address and its coins.
  2. Verify Uniswap V3 pool addresses via The Graph.
  3. Print a ready-to-use config snippet with confirmed addresses.

Usage:
    python -m src.discover_pools
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import aiohttp
import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CURVE_API = "https://api.curve.finance/v1"
GRAPH_URL_FREE = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
GRAPH_URL_GW = (
    "https://gateway.thegraph.com/api/{api_key}"
    "/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
)

POOL_QUERY = """
query PoolInfo($addr: String!) {
  pool(id: $addr) {
    id
    token0 { id symbol decimals }
    token1 { id symbol decimals }
    feeTier
    totalValueLockedUSD
    volumeUSD
  }
}
"""


async def resolve_curve_pool(session: aiohttp.ClientSession, pool_name: str) -> dict | None:
    url = f"{CURVE_API}/pools/ethereum/{pool_name}"
    log.info("Curve API → %s", url)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 404:
                log.warning("Pool '%s' not found (404).", pool_name)
                return None
            r.raise_for_status()
            return (await r.json()).get("data")
    except Exception as exc:
        log.error("Curve API error: %s", exc)
        return None


async def verify_uniswap_pool(
    session: aiohttp.ClientSession, pool_address: str, graph_url: str
) -> dict | None:
    payload = {
        "query": POOL_QUERY,
        "variables": {"addr": pool_address.lower()},
    }
    log.info("The Graph → %s (pool %s)", graph_url, pool_address[:10] + "...")
    try:
        async with session.post(
            graph_url, json=payload, timeout=aiohttp.ClientTimeout(total=20)
        ) as r:
            r.raise_for_status()
            data = (await r.json()).get("data", {})
            return data.get("pool")
    except Exception as exc:
        log.error("The Graph error: %s", exc)
        return None


async def main() -> None:
    load_dotenv()
    api_key = os.getenv("THEGRAPH_API_KEY", "")
    graph_url = (
        GRAPH_URL_GW.format(api_key=api_key) if api_key else GRAPH_URL_FREE
    )

    uniswap_pools = [
        {
            "id": "usde_usdc_uniswap_v3",
            "address": "0xe6d7ebb9f1a9519dc06d557e03c522d53520e76a",
        },
        {
            "id": "usr_usdc_uniswap_v3",
            "address": "0x8bb9cd887dd51c5aa8d7da9e244c94bec035e47c",
        },
    ]
    curve_pools = [
        {"id": "usde_usdc_curve",  "name": "factory-stable-ng-12"},
        {"id": "usr_usdc_curve",   "name": "factory-stable-ng-397"},
    ]

    async with aiohttp.ClientSession(headers={"User-Agent": "RiskMonitor/1.0"}) as session:

        print("\n" + "=" * 65)
        print("UNISWAP V3 POOL VERIFICATION")
        print("=" * 65)
        for p in uniswap_pools:
            info = await verify_uniswap_pool(session, p["address"], graph_url)
            if info:
                print(
                    f"\n  [{p['id']}]"
                    f"\n    address : {info['id']}"
                    f"\n    token0  : {info['token0']['symbol']} ({info['token0']['id']})"
                    f"\n    token1  : {info['token1']['symbol']} ({info['token1']['id']})"
                    f"\n    fee     : {int(info['feeTier'])/10000:.2f}%"
                    f"\n    TVL USD : ${float(info['totalValueLockedUSD']):,.0f}"
                    f"\n    vol USD : ${float(info['volumeUSD']):,.0f}"
                )
            else:
                print(f"\n  [{p['id']}] → NOT FOUND / API unavailable")

        print("\n" + "=" * 65)
        print("CURVE POOL RESOLUTION")
        print("=" * 65)
        for p in curve_pools:
            data = await resolve_curve_pool(session, p["name"])
            if data:
                coins = data.get("coins", [])
                print(
                    f"\n  [{p['id']}]"
                    f"\n    pool_name: {p['name']}"
                    f"\n    address  : {data.get('address', 'n/a')}"
                    f"\n    coins    : {[c['symbol'] for c in coins]}"
                    f"\n    totalTvl : ${float(data.get('usdTotal', 0)):,.0f}"
                )
            else:
                print(f"\n  [{p['id']}] → NOT FOUND / API unavailable")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
