"""
High-resolution swap extractor using Etherscan.

Fetches every Swap event for the Uniswap V3 pools in the March 20-24 2026
window and saves per-swap price records to data/swaps/.

Existing files in data/raw/ and figures/ are NOT touched.

Usage
-----
    python -m src.fetch_swaps
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.extractors.etherscan import EtherscanSwapExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("swaps_extraction.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Block numbers confirmed via Etherscan getblocknobytime
BLOCK_START = 24_694_981   # 2026-03-20 00:00 UTC
BLOCK_END   = 24_730_839   # 2026-03-24 23:59 UTC

OUTPUT_DIR = Path("data/swaps")


def load_config(path: str = "config/pools.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    uniswap_pools = [p for p in cfg["pools"] if p["dex"] == "uniswap_v3"]
    extractor = EtherscanSwapExtractor()

    all_frames: list[pd.DataFrame] = []

    async with aiohttp.ClientSession(headers={"User-Agent": "RiskMonitor/1.0"}) as session:
        tasks = [
            extractor.fetch_swaps(session, pool, BLOCK_START, BLOCK_END)
            for pool in uniswap_pools
        ]
        results = await asyncio.gather(*tasks)

    for pool_cfg, records in zip(uniswap_pools, results):
        if not records:
            log.warning("[%s] No swap records returned.", pool_cfg["id"])
            continue

        df = pd.DataFrame(records)
        out_path = OUTPUT_DIR / f"{pool_cfg['id']}_swaps.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved %s  (%d swaps)", out_path, len(df))

        # Quick stats
        log.info(
            "  price range: %.6f – %.6f USDC  |  blocks %d – %d",
            df["price_usdc"].min(), df["price_usdc"].max(),
            df["block"].min(), df["block"].max(),
        )
        all_frames.append(df)

    if all_frames:
        combined = (
            pd.concat(all_frames, ignore_index=True)
            .sort_values(["block", "log_index"])
            .reset_index(drop=True)
        )
        combined.to_csv(OUTPUT_DIR / "combined_swaps.csv", index=False)
        combined.to_parquet(OUTPUT_DIR / "combined_swaps.parquet", index=False)
        log.info(
            "Combined swaps: %d records across %d pools → data/swaps/combined_swaps.*",
            len(combined), len(all_frames),
        )


def main() -> None:
    load_dotenv()
    asyncio.run(run())


if __name__ == "__main__":
    main()
