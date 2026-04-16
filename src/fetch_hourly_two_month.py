"""
Fetch ~2 months of hourly OHLCV (USDe / USR) + Uniswap TVL — separate from data/raw.

Output: config `extraction.output_dir` (default data/hourly_two_month/).

Run (from repo root — or run this file directly; cwd should be repo root for configs)
---
    python -m src.fetch_hourly_two_month
    python -m src.fetch_hourly_two_month --start-date ... --end-date ...

Plot the result:
    python -m src.plot_hourly_two_month
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    os.chdir(_REPO_ROOT)
except OSError:
    pass

import aiohttp
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.aggregator import OHLCV_COLUMNS, normalise_curve, normalise_geckoterminal
from src.extractors.curve import CurveExtractor
from src.extractors.geckoterminal import GeckoTerminalExtractor
from src.extractors.uniswap_subgraph_hourly import UniswapSubgraphHourlyExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fetch_hourly_two_month.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

_HOUR = 3600


def _resample_ohlcv_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    token = df["token"].iloc[0]
    pool_id = df["pool_id"].iloc[0]
    dex = df["dex"].iloc[0]
    d = df.copy()
    d["__t"] = pd.to_datetime(d["timestamp_open"], unit="s", utc=True)
    d = d.set_index("__t").sort_index()
    rs = d.resample("1h", label="left", closed="left")
    agg = pd.DataFrame(
        {
            "open": rs["open"].first(),
            "high": rs["high"].max(),
            "low": rs["low"].min(),
            "close": rs["close"].last(),
            "volume_usdc": rs["volume_usdc"].sum(),
        }
    )
    for col in ("open", "high", "low", "close"):
        agg[col] = agg[col].ffill()
    agg["volume_usdc"] = agg["volume_usdc"].fillna(0.0)
    agg = agg.dropna(subset=["open", "high", "low", "close"], how="all")
    agg = agg.reset_index()
    tcol = agg.columns[0]
    agg["timestamp_open"] = agg[tcol].map(lambda x: int(pd.Timestamp(x).timestamp()))
    agg["datetime_open"] = agg[tcol].map(lambda x: pd.Timestamp(x).isoformat())
    agg["token"] = token
    agg["pool_id"] = pool_id
    agg["dex"] = dex
    agg["resolution_sec"] = _HOUR
    agg["num_swaps"] = 0
    agg = agg.drop(columns=[tcol])
    return agg[
        [
            "timestamp_open",
            "datetime_open",
            "token",
            "pool_id",
            "dex",
            "resolution_sec",
            "open",
            "high",
            "low",
            "close",
            "volume_usdc",
            "num_swaps",
        ]
    ]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(pools_path: Path, window_path: Path) -> dict:
    with open(pools_path) as f:
        base = yaml.safe_load(f)
    with open(window_path) as f:
        win = yaml.safe_load(f)
    return _deep_merge(base, win)


def parse_ts(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp())


async def extract_uniswap_hourly(
    extractor: GeckoTerminalExtractor,
    session: aiohttp.ClientSession,
    pool_cfg: dict,
    ts_start: int,
    ts_end: int,
) -> pd.DataFrame:
    try:
        candles = await extractor.fetch_pool_ohlcv(
            session, pool_cfg, ts_start, ts_end
        )
        df5 = normalise_geckoterminal(candles, ts_start, ts_end)
        return _resample_ohlcv_to_hourly(df5)
    except Exception as exc:
        log.error("[%s] GeckoTerminal → hourly failed: %s", pool_cfg["id"], exc)
        return pd.DataFrame(columns=OHLCV_COLUMNS)


async def extract_curve_hourly(
    extractor: CurveExtractor,
    session: aiohttp.ClientSession,
    pool_cfg: dict,
    ts_start: int,
    ts_end: int,
) -> pd.DataFrame:
    try:
        candles = await extractor.fetch_pool_ohlcv(session, pool_cfg, ts_start, ts_end)
        return normalise_curve(candles, ts_start, ts_end)
    except Exception as exc:
        log.error("[%s] Curve hourly failed: %s", pool_cfg["id"], exc)
        return pd.DataFrame(columns=OHLCV_COLUMNS)


async def extract_uni_tvl(
    extractor: UniswapSubgraphHourlyExtractor,
    session: aiohttp.ClientSession,
    pool_cfg: dict,
    ts_start: int,
    ts_end: int,
) -> pd.DataFrame:
    try:
        rows = await extractor.fetch_pool_hourly_stats(
            session, pool_cfg, ts_start, ts_end
        )
        return pd.DataFrame(rows)
    except Exception as exc:
        log.error("[%s] Subgraph TVL hourly failed: %s", pool_cfg["id"], exc)
        return pd.DataFrame()


async def run(cfg: dict) -> None:
    ext = cfg["extraction"]
    out_root = Path(ext["output_dir"])
    ohlcv_dir = out_root / "ohlcv"
    tvl_dir = out_root / "tvl"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    tvl_dir.mkdir(parents=True, exist_ok=True)

    ts_start = parse_ts(ext["start_date"])
    ts_end = parse_ts(ext["end_date"])

    log.info(
        "Hourly two-month window: %s → %s  |  out %s",
        ext["start_date"],
        ext["end_date"],
        out_root,
    )

    gecko = GeckoTerminalExtractor(cfg)
    curve_ext = CurveExtractor(cfg)
    graph_ext = UniswapSubgraphHourlyExtractor(cfg)

    uniswap_pools = [p for p in cfg["pools"] if p["dex"] == "uniswap_v3"]
    curve_pools = [p for p in cfg["pools"] if p["dex"] == "curve"]

    all_ohlcv: list[pd.DataFrame] = []
    all_tvl: list[pd.DataFrame] = []

    async with aiohttp.ClientSession(headers={"User-Agent": "RiskMonitor/1.0"}) as session:
        uni_tasks = [
            extract_uniswap_hourly(gecko, session, p, ts_start, ts_end)
            for p in uniswap_pools
        ]
        curve_tasks = [
            extract_curve_hourly(curve_ext, session, p, ts_start, ts_end)
            for p in curve_pools
        ]
        uni_frames, curve_frames = await asyncio.gather(
            asyncio.gather(*uni_tasks),
            asyncio.gather(*curve_tasks),
        )

        for pool_cfg, df in zip(uniswap_pools, uni_frames):
            if not df.empty:
                path = ohlcv_dir / f"{pool_cfg['id']}.csv"
                df.to_csv(path, index=False)
                log.info("Saved %s (%d rows)", path, len(df))
                all_ohlcv.append(df)

        for pool_cfg, df in zip(curve_pools, curve_frames):
            if not df.empty:
                path = ohlcv_dir / f"{pool_cfg['id']}.csv"
                df.to_csv(path, index=False)
                log.info("Saved %s (%d rows)", path, len(df))
                all_ohlcv.append(df)

        tvl_tasks = [
            extract_uni_tvl(graph_ext, session, p, ts_start, ts_end)
            for p in uniswap_pools
        ]
        tvl_frames = await asyncio.gather(*tvl_tasks)

        for pool_cfg, tdf in zip(uniswap_pools, tvl_frames):
            if not tdf.empty:
                path = tvl_dir / f"{pool_cfg['id']}_hourly.csv"
                tdf.to_csv(path, index=False)
                log.info("Saved %s (%d rows)", path, len(tdf))
                all_tvl.append(tdf)

    if not all_ohlcv:
        log.warning("No OHLCV collected.")
        return

    combined_o = (
        pd.concat(all_ohlcv, ignore_index=True)
        .sort_values(["timestamp_open", "token", "pool_id"])
        .reset_index(drop=True)
    )
    co_path = ohlcv_dir / "combined_ohlcv.parquet"
    combined_o.to_parquet(co_path, index=False)
    combined_o.to_csv(ohlcv_dir / "combined_ohlcv.csv", index=False)
    log.info("Combined OHLCV → %s (%d rows)", co_path, len(combined_o))

    if all_tvl:
        tvl_all = pd.concat(all_tvl, ignore_index=True)
        o = combined_o.copy()
        t = tvl_all.rename(columns={"volume_usd_hour": "subgraph_volume_usd_hour"})
        merged = o.merge(
            t[
                [
                    "pool_id",
                    "timestamp_open",
                    "tvl_usd",
                    "subgraph_volume_usd_hour",
                    "tx_count_hour",
                    "close_token0_usd",
                    "token0_price",
                    "token1_price",
                ]
            ],
            on=["pool_id", "timestamp_open"],
            how="left",
        )
        mpath = out_root / "uniswap_ohlcv_with_tvl.parquet"
        merged.to_parquet(mpath, index=False)
        merged.to_csv(out_root / "uniswap_ohlcv_with_tvl.csv", index=False)
        log.info("Uniswap OHLCV + TVL → %s", mpath)

    log.info("Done. data/raw/ unchanged. Run: python -m src.plot_hourly_two_month")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch hourly two-month OHLCV + Uniswap TVL")
    ap.add_argument("--pools", type=Path, default=Path("config/pools.yaml"))
    ap.add_argument("--window", type=Path, default=Path("config/hourly_two_month.yaml"))
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    load_dotenv()
    cfg = load_merged_config(args.pools, args.window)
    if args.start_date:
        cfg["extraction"]["start_date"] = args.start_date
    if args.end_date:
        cfg["extraction"]["end_date"] = args.end_date
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
