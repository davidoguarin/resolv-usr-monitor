"""
Main orchestrator: extract DEX price data for USDe and USR (March 20-24 2026).

Sources
-------
  Uniswap V3 pools → GeckoTerminal API  (5-min OHLCV, free, no key needed)
  Curve pools      → Curve prices API   (1-hour OHLCV, API minimum resolution)

Usage
-----
    python -m src.main
    python -m src.main --config config/pools.yaml --output data/raw

Output
------
  data/raw/
    usde_usdc_uniswap_v3.csv
    usr_usdc_uniswap_v3.csv
    usde_usdc_curve.csv
    usr_usdc_curve.csv
    combined_ohlcv.csv       ← all pools, all tokens
    combined_ohlcv.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.aggregator import (
    OHLCV_COLUMNS,
    normalise_curve,
    normalise_geckoterminal,
)
from src.extractors.curve import CurveExtractor
from src.extractors.geckoterminal import GeckoTerminalExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extraction.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_ts(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp())


# ---------------------------------------------------------------------------
# Per-pool extraction tasks
# ---------------------------------------------------------------------------

async def extract_uniswap_pool(
    extractor: GeckoTerminalExtractor,
    session: aiohttp.ClientSession,
    pool_cfg: dict,
    ts_start: int,
    ts_end: int,
) -> pd.DataFrame:
    try:
        candles = await extractor.fetch_pool_ohlcv(session, pool_cfg, ts_start, ts_end)
        df = normalise_geckoterminal(candles, ts_start, ts_end)
        log.info("[%s] → %d 5-min candles", pool_cfg["id"], len(df))
        return df
    except Exception as exc:
        log.error("[%s] GeckoTerminal extraction failed: %s", pool_cfg["id"], exc)
        return pd.DataFrame(columns=OHLCV_COLUMNS)


async def extract_curve_pool(
    extractor: CurveExtractor,
    session: aiohttp.ClientSession,
    pool_cfg: dict,
    ts_start: int,
    ts_end: int,
) -> pd.DataFrame:
    try:
        candles = await extractor.fetch_pool_ohlcv(session, pool_cfg, ts_start, ts_end)
        df = normalise_curve(candles, ts_start, ts_end)
        log.info("[%s] → %d hourly candles", pool_cfg["id"], len(df))
        return df
    except Exception as exc:
        log.error("[%s] Curve extraction failed: %s", pool_cfg["id"], exc)
        return pd.DataFrame(columns=OHLCV_COLUMNS)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

async def run(cfg: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_start = parse_ts(cfg["extraction"]["start_date"])
    ts_end   = parse_ts(cfg["extraction"]["end_date"])
    interval = cfg["extraction"]["interval_seconds"]

    log.info(
        "Extraction window: %s → %s",
        cfg["extraction"]["start_date"],
        cfg["extraction"]["end_date"],
    )

    gecko_ext = GeckoTerminalExtractor(cfg)
    curve_ext = CurveExtractor(cfg)

    uniswap_pools = [p for p in cfg["pools"] if p["dex"] == "uniswap_v3"]
    curve_pools   = [p for p in cfg["pools"] if p["dex"] == "curve"]

    all_frames: list[pd.DataFrame] = []

    async with aiohttp.ClientSession(headers={"User-Agent": "RiskMonitor/1.0"}) as session:

        # Run all pools concurrently
        uni_tasks = [
            extract_uniswap_pool(gecko_ext, session, p, ts_start, ts_end)
            for p in uniswap_pools
        ]
        curve_tasks = [
            extract_curve_pool(curve_ext, session, p, ts_start, ts_end)
            for p in curve_pools
        ]

        results = await asyncio.gather(*uni_tasks, *curve_tasks)

    all_pool_cfgs = uniswap_pools + curve_pools
    for pool_cfg, df in zip(all_pool_cfgs, results):
        if not df.empty:
            out_path = output_dir / f"{pool_cfg['id']}.csv"
            df.to_csv(out_path, index=False)
            log.info("Saved → %s  (%d rows)", out_path, len(df))
            all_frames.append(df)

    if all_frames:
        combined = (
            pd.concat(all_frames, ignore_index=True)
            .sort_values(["timestamp_open", "token", "pool_id"])
            .reset_index(drop=True)
        )
        csv_path     = output_dir / "combined_ohlcv.csv"
        parquet_path = output_dir / "combined_ohlcv.parquet"
        combined.to_csv(csv_path, index=False)
        combined.to_parquet(parquet_path, index=False)
        log.info(
            "Combined → %d rows, %d pools  |  %s  |  %s",
            len(combined), len(all_frames), csv_path, parquet_path,
        )
        _print_summary(combined)
    else:
        log.warning("No data collected.")


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("EXTRACTION SUMMARY  —  USDe & USR vs USDC  (March 20-24, 2026)")
    print("=" * 72)
    for (token, pool_id), g in df.groupby(["token", "pool_id"]):
        valid = g.dropna(subset=["close"])
        if valid.empty:
            continue
        res = int(g["resolution_sec"].iloc[0])
        label = f"{res//60}-min" if res < 3600 else f"{res//3600}-hour"
        print(
            f"\n  {token} | {pool_id}  [{label}]"
            f"\n    candles  : {len(g)}"
            f"\n    price min: {valid['low'].min():.6f} USDC"
            f"\n    price max: {valid['high'].max():.6f} USDC"
            f"\n    last close: {valid['close'].iloc[-1]:.6f} USDC"
            f"\n    vol USDC : ${valid['volume_usdc'].sum():,.0f}"
        )
    print("\n" + "=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="DEX price monitor extractor")
    parser.add_argument("--config", default="config/pools.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    output_dir = Path(args.output or cfg["extraction"]["output_dir"])

    asyncio.run(run(cfg, output_dir))


if __name__ == "__main__":
    main()
