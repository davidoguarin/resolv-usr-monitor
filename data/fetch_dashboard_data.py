#!/usr/bin/env python3
"""
Fetch 6-month dashboard data (Oct 1 2025 → Apr 1 2026).
Run once to populate data/cached/*.csv before deploying the dashboard.

Outputs
-------
  data/cached/resolv_tvl.csv       - Daily Resolv TVL (DeFiLlama)
  data/cached/usr_apy.csv          - Daily stUSR APY (DeFiLlama yields)
  data/cached/usr_price_daily.csv  - Daily USR/USDC OHLCV (GeckoTerminal)
  data/cached/usr_mint_burn.csv    - Daily USR mint & burn (Etherscan Transfer events)

Usage
-----
  python -m data.fetch_dashboard_data          # from project root
  python data/fetch_dashboard_data.py          # direct
"""
from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
DATE_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
DATE_END   = datetime(2026,  4, 1, tzinfo=timezone.utc)
TS_START   = int(DATE_START.timestamp())   # 1759276800
TS_END     = int(DATE_END.timestamp())     # 1775001600

BLOCK_START = 23_479_243   # ~Oct 1 2025 on Ethereum mainnet
BLOCK_END   = 24_781_026   # ~Apr 1 2026

USR_ADDRESS = "0x66a1e37c9b0eaddca17d3662d6c05f4decf3e110"   # USR token
USR_POOL_V3 = "0x8bb9cd887dd51c5aa8d7da9e244c94bec035e47c"   # USR/USDC Uniswap V3

ETHERSCAN_KEY  = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ZERO_TOPIC     = "0x" + "0" * 64   # 32-byte zero address

GECKO_BASE   = "https://api.geckoterminal.com/api/v2"
GECKO_HDR    = {"Accept": "application/json;version=20230302"}
LLAMA_BASE   = "https://api.llama.fi"
YIELDS_BASE  = "https://yields.llama.fi"

OUT_DIR = Path("data/cached")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, headers: dict | None = None,
         pause: float = 0.25, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                log.warning("Rate-limited — sleeping 15s ...")
                time.sleep(15)
                continue
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            log.warning("Attempt %d failed: %s — retrying ...", attempt + 1, exc)
            time.sleep(4 * (attempt + 1))
    return {}


# ─── 1. Resolv TVL (DeFiLlama) ───────────────────────────────────────────────

def fetch_resolv_tvl() -> pd.DataFrame:
    out = OUT_DIR / "resolv_tvl.csv"
    if out.exists():
        log.info("resolv_tvl.csv already cached — skipping.")
        return pd.read_csv(out, parse_dates=["date"])

    log.info("Fetching Resolv TVL from DeFiLlama ...")
    data = _get(f"{LLAMA_BASE}/protocol/resolv")

    rows = []
    # DeFiLlama returns tvl under different keys depending on protocol config
    tvl_series = data.get("tvl") or []
    if not tvl_series:
        # Try totalLiquidityUSD from chainTvls
        for chain_data in data.get("chainTvls", {}).values():
            if chain_data.get("tvl"):
                tvl_series = chain_data["tvl"]
                break

    for entry in tvl_series:
        ts = int(entry["date"])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if TS_START <= ts <= TS_END:
            rows.append({"date": dt.date(), "tvl_usd": float(entry["totalLiquidityUSD"])})

    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(out, index=False)
    log.info("  → %d TVL entries saved.", len(df))
    return df


# ─── 2. stUSR APY (DeFiLlama Yields) ────────────────────────────────────────

def fetch_usr_apy() -> pd.DataFrame:
    out = OUT_DIR / "usr_apy.csv"
    if out.exists():
        log.info("usr_apy.csv already cached — skipping.")
        return pd.read_csv(out, parse_dates=["date"])

    log.info("Fetching stUSR pool from DeFiLlama yields ...")
    pools_data = _get(f"{YIELDS_BASE}/pools")
    pools = pools_data.get("data", [])

    # Find stUSR pool — project="resolv", symbol contains "USR"
    usr_pools = [
        p for p in pools
        if p.get("project", "").lower() == "resolv"
        and "usr" in p.get("symbol", "").lower()
    ]
    log.info("  Found %d resolv USR pool(s): %s",
             len(usr_pools), [(p["pool"], p["symbol"]) for p in usr_pools])

    if not usr_pools:
        log.warning("No stUSR pool found — using 15%% flat APY fallback.")
        dates = pd.date_range(DATE_START, DATE_END - timedelta(days=1), freq="D")
        df = pd.DataFrame({"date": dates, "apy": 15.0})
        df.to_csv(out, index=False)
        return df

    # Use the first match (stUSR is typically the main one)
    pool_id = usr_pools[0]["pool"]
    log.info("  Using pool: %s (%s)", pool_id, usr_pools[0].get("symbol"))

    chart = _get(f"{YIELDS_BASE}/chart/{pool_id}")
    entries = chart.get("data", [])

    rows = []
    for e in entries:
        # timestamp is ISO8601 string like "2025-10-01T00:00:00.000Z"
        try:
            dt = pd.Timestamp(e["timestamp"]).tz_localize("UTC") if e["timestamp"].endswith("Z") is False else pd.Timestamp(e["timestamp"])
        except Exception:
            continue
        ts = int(dt.timestamp())
        if TS_START <= ts <= TS_END:
            rows.append({
                "date": dt.normalize(),
                "apy":  float(e.get("apy") or e.get("apyBase") or 0),
            })

    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")

    if df.empty:
        log.warning("Empty APY data — using 15%% flat fallback.")
        dates = pd.date_range(DATE_START, DATE_END - timedelta(days=1), freq="D")
        df = pd.DataFrame({"date": dates, "apy": 15.0})

    df.to_csv(out, index=False)
    log.info("  → %d APY entries saved.", len(df))
    return df


# ─── 3. USR/USDC Daily Price (GeckoTerminal) ─────────────────────────────────

def fetch_usr_price_daily() -> pd.DataFrame:
    out = OUT_DIR / "usr_price_daily.csv"
    if out.exists():
        log.info("usr_price_daily.csv already cached — skipping.")
        return pd.read_csv(out, parse_dates=["date"])

    log.info("Fetching USR/USDC daily OHLCV from GeckoTerminal ...")
    # Use /ohlcv/day endpoint — aggregate=1 gives 1-day candles
    url = (
        f"{GECKO_BASE}/networks/eth/pools/{USR_POOL_V3}/ohlcv/day"
        f"?aggregate=1&before_timestamp={TS_END + 86400}&limit=366&currency=usd&token=base"
    )
    data = _get(url, headers=GECKO_HDR)
    candles = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])

    rows = []
    for ts, open_, high, low, close, volume in candles:
        if TS_START <= int(ts) <= TS_END:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            rows.append({
                "date":       dt.date(),
                "open":       float(open_),
                "high":       float(high),
                "low":        float(low),
                "close":      float(close),
                "volume_usd": float(volume),
            })

    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(out, index=False)
    log.info("  → %d daily candles saved.", len(df))
    return df


# ─── 4. USR Mint / Burn (Etherscan Transfer events) ──────────────────────────

def _decode_transfer_amount(data_hex: str, decimals: int = 18) -> float:
    raw = data_hex[2:].lstrip("0") or "0"
    return int(raw, 16) / 10**decimals


def _fetch_transfers(direction: str, from_block: int, to_block: int) -> list[dict]:
    """Fetch mint (direction='mint') or burn (direction='burn') Transfer events."""
    if direction == "mint":
        # topic1 = zero address (transfers FROM 0x0 = mints)
        topic1, topic2 = ZERO_TOPIC, None
    else:
        # topic2 = zero address (transfers TO 0x0 = burns)
        topic1, topic2 = None, ZERO_TOPIC

    all_events: list[dict] = []
    cur_block = from_block

    while cur_block <= to_block:
        params: dict = {
            "chainid":   "1",
            "module":    "logs",
            "action":    "getLogs",
            "address":   USR_ADDRESS,
            "topic0":    TRANSFER_TOPIC,
            "fromBlock": str(cur_block),
            "toBlock":   str(to_block),
            "page":      "1",
            "offset":    "1000",
            "apikey":    ETHERSCAN_KEY,
        }
        if topic1:
            params["topic0_1_opr"] = "and"
            params["topic1"] = topic1
        if topic2:
            params["topic0_2_opr"] = "and"
            params["topic2"] = topic2

        try:
            body = _get(ETHERSCAN_BASE, params=params, pause=0.22)
        except Exception as exc:
            log.error("Etherscan error: %s", exc)
            break

        if body.get("status") != "1":
            msg = body.get("message", "")
            if "No records" in msg or body.get("result") == []:
                break
            log.warning("Etherscan non-OK: %s — %s", msg, body.get("result", "")[:80])
            break

        page = body["result"]
        all_events.extend(page)
        log.info("  [%s] fetched %d events (total %d), from_block=%d",
                 direction, len(page), len(all_events), cur_block)

        if len(page) < 1000:
            break

        cur_block = int(page[-1]["blockNumber"], 16) + 1

    return all_events


def fetch_usr_mint_burn() -> pd.DataFrame:
    out = OUT_DIR / "usr_mint_burn.csv"
    if out.exists():
        log.info("usr_mint_burn.csv already cached — skipping.")
        return pd.read_csv(out, parse_dates=["date"])

    if not ETHERSCAN_KEY:
        log.error("ETHERSCAN_API_KEY not set — skipping mint/burn fetch.")
        return pd.DataFrame(columns=["date", "minted", "burned", "net"])

    log.info("Fetching USR mint events from Etherscan (blocks %d → %d) ...",
             BLOCK_START, BLOCK_END)
    mint_events = _fetch_transfers("mint", BLOCK_START, BLOCK_END)

    log.info("Fetching USR burn events from Etherscan ...")
    burn_events = _fetch_transfers("burn", BLOCK_START, BLOCK_END)

    def events_to_daily(events: list[dict], col: str) -> pd.DataFrame:
        rows = []
        for ev in events:
            try:
                ts  = int(ev["timeStamp"], 16)
                amt = _decode_transfer_amount(ev["data"])
                # Keep timezone-naive date string for consistent merging
                dt  = datetime.utcfromtimestamp(ts).date()
                rows.append({"date": str(dt), col: amt})
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=["date", col])
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")[col].sum().reset_index()

    mints = events_to_daily(mint_events, "minted")
    burns = events_to_daily(burn_events, "burned")

    # Build full date range (timezone-naive) and merge
    all_dates = pd.DataFrame(
        {"date": pd.date_range(DATE_START.replace(tzinfo=None),
                               (DATE_END - timedelta(days=1)).replace(tzinfo=None),
                               freq="D")}
    )
    df = all_dates.merge(mints, on="date", how="left").merge(burns, on="date", how="left")
    df["minted"] = df["minted"].fillna(0.0)
    df["burned"] = df["burned"].fillna(0.0)
    df["net"]    = df["minted"] - df["burned"]

    df.to_csv(out, index=False)
    log.info("  → %d days of mint/burn data saved (mints=%d events, burns=%d events).",
             len(df), len(mint_events), len(burn_events))
    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("Dashboard data fetch: %s → %s", DATE_START.date(), DATE_END.date())
    log.info("=" * 60)

    fetch_resolv_tvl()
    fetch_usr_apy()
    fetch_usr_price_daily()
    fetch_usr_mint_burn()

    log.info("All done — cached files in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
