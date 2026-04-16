"""
Fetch USR/USDC Uniswap V3 pool TVL for a UTC date range and plot it on a 1-minute grid.

Important: The Uniswap V3 subgraph only stores **hourly** `poolHourDatas` (TVL at each
hour boundary). There is no native per-minute TVL entity. This script:

  1. Pulls hourly `tvlUSD` via the existing subgraph client.
  2. Reindexes to a 1-minute timeline with **forward-fill** (stair-step = last known hour).

For true minute-level TVL you would need on-chain reconstruction (Mint/Burn/Swap +
liquidity math) or an external indexer — not available in this subgraph.

Run (repo root; optional `THEGRAPH_API_KEY` in `.env` for the gateway)
---
    python -m src.fetch_plot_usr_usdc_tvl_window
    python -m src.fetch_plot_usr_usdc_tvl_window --start 2026-03-21 --end 2026-03-23
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    os.chdir(_REPO_ROOT)
except OSError:
    pass

import aiohttp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.extractors.uniswap_subgraph_hourly import UniswapSubgraphHourlyExtractor


def _pool_cfg(cfg: dict, pool_id: str) -> dict:
    for p in cfg.get("pools", []):
        if p.get("id") == pool_id:
            return p
    raise KeyError(f"No pool id {pool_id!r} in config/pools.yaml")


def _hourly_to_minute_ffill(
    hourly: pd.DataFrame, t0: pd.Timestamp, t1_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """t0 inclusive, t1_exclusive exclusive; each minute gets latest hourly TVL at or before t."""
    if hourly.empty:
        return pd.DataFrame()
    minute_index = pd.date_range(t0, t1_exclusive, freq="1min", inclusive="left")
    minute_df = pd.DataFrame({"t": minute_index})
    hp = hourly.copy()
    hp["t"] = pd.to_datetime(hp["datetime_open"], utc=True)
    hp = hp.sort_values("t").drop_duplicates(subset=["t"], keep="last")[["t", "tvl_usd"]]
    out = pd.merge_asof(minute_df.sort_values("t"), hp, on="t", direction="backward")
    out["tvl_usd"] = out["tvl_usd"].bfill()
    return out.sort_values("t").reset_index(drop=True)


def _plot(df: pd.DataFrame, title: str, subtitle: str, output: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "font.family": "monospace",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(df["t"], df["tvl_usd"] / 1e6, color="#ea580c", linewidth=0.9)
    ax.set_ylabel("TVL (USD millions)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=8, color="#6b7280")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz="UTC"))
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


async def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg_path = Path("config/pools.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pool = _pool_cfg(cfg, args.pool_id)
    if pool.get("dex") != "uniswap_v3":
        raise SystemExit(f"Pool {args.pool_id} is not uniswap_v3")

    t0 = pd.Timestamp(args.start + "T00:00:00", tz="UTC")
    t1_excl = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    ts_start = int(t0.timestamp())
    ts_end = int((t1_excl - pd.Timedelta(seconds=1)).timestamp())

    ext = UniswapSubgraphHourlyExtractor(cfg)
    async with aiohttp.ClientSession() as session:
        rows = await ext.fetch_pool_hourly_stats(session, pool, ts_start, ts_end)

    hourly = pd.DataFrame(rows)
    minute_df = _hourly_to_minute_ffill(hourly, t0, t1_excl)
    if minute_df.empty:
        raise SystemExit("No TVL rows returned (check dates, API key, and pool address).")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    minute_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}  ({len(minute_df):,} rows)")

    title = "USR / USDC Uniswap V3 — TVL (1-min grid, hourly subgraph)"
    subtitle = (
        "Subgraph: poolHourDatas (hourly); values forward-filled between hour marks — not on-chain per-minute TVL."
    )
    _plot(minute_df, title, subtitle, args.output_png)
    print(f"Wrote {args.output_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="USR/USDC pool TVL: subgraph hourly → 1-min plot")
    p.add_argument("--start", default="2026-03-21", help="UTC start date (inclusive), YYYY-MM-DD")
    p.add_argument("--end", default="2026-03-23", help="UTC end date (inclusive), YYYY-MM-DD")
    p.add_argument(
        "--pool-id",
        default="usr_usdc_uniswap_v3",
        help="Pool id in config/pools.yaml",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/usr_usdc_uniswap_v3_tvl_1min.csv"),
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/usr_usdc_uniswap_v3_tvl_1min.png"),
    )
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
