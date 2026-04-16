"""
USR/USDC pool depeg (per-swap price) vs Ethereum base fee over a UTC time window.

Price data: reads `data/swaps/usr_usdc_uniswap_v3_swaps.csv` (same as plot_zoom).
  Run `python -m src.fetch_swaps` first if that file is missing or stale.

Base fee: JSON-RPC `eth_feeHistory` + `eth_getBlockByNumber`.
  Tries **ETH_RPC_URL** first (if set), then several public endpoints in order — Cloudflare
  often returns `-32046 Cannot fulfill request`; Infura/Alchemy URLs in `.env` are ideal.

Environment
------------
  ETH_RPC_URL   optional; when unset, cycles through built-in public RPCs until one works

Outputs (defaults — override with flags)
-----------------------------------------
  data/usr_depeg_basefee_swaps.csv
  data/usr_depeg_basefee_blocks.csv
  figures/usr_depeg_vs_basefee.png

Example (Mar 22 2026 00:00 UTC → Mar 23 2026 00:00 UTC)
-------------------------------------------------------
    python -m src.fetch_plot_usr_depeg_basefee
    python -m src.fetch_plot_usr_depeg_basefee \\
        --start 2026-03-22T00:00:00Z --end 2026-03-23T00:00:00Z
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

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
from dotenv import load_dotenv

# Cloudflare is last — it often rejects JSON-RPC with -32046 for many clients/IPs.
_FALLBACK_RPCS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
    "https://ethereum.publicnode.com",
    "https://1rpc.io/eth",
    "https://cloudflare-eth.com",
]
C_PRICE = "#ea580c"
C_BASE = "#64748b"
C_THRESH = "#dc2626"
DEPEG_THRESHOLD = 0.99


def _rpc_url_candidates(cli_override: str) -> list[str]:
    o = (cli_override or "").strip()
    if o:
        return [o]
    env = os.getenv("ETH_RPC_URL", "").strip()
    seen: set[str] = set()
    out: list[str] = []
    for u in [env] + _FALLBACK_RPCS:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def _rpc(
    session: aiohttp.ClientSession, url: str, method: str, params: list
) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    async with session.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "RiskAnalysisFramework/1.0",
        },
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        resp.raise_for_status()
        body = await resp.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    return body["result"]


async def _pick_working_rpc(
    session: aiohttp.ClientSession, candidates: list[str]
) -> str:
    last: Optional[BaseException] = None
    for url in candidates:
        try:
            bn = await _rpc(session, url, "eth_blockNumber", [])
            int(bn[2:], 16)
            return url
        except BaseException as exc:
            last = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(
        "No working Ethereum JSON-RPC endpoint. Set ETH_RPC_URL in `.env` to a provider "
        f"(Infura, Alchemy, publicnode, etc.). Last error: {last!r}"
    )


async def _eth_block_number(session: aiohttp.ClientSession, url: str) -> int:
    return int((await _rpc(session, url, "eth_blockNumber", []))[2:], 16)


async def _block_ts(session: aiohttp.ClientSession, url: str, bn: int) -> int:
    b = await _rpc(session, url, "eth_getBlockByNumber", [hex(bn), False])
    return int(b["timestamp"], 16)


async def _find_block_ge_ts(
    session: aiohttp.ClientSession, url: str, ts: int, hi: int
) -> int:
    """Smallest block number with timestamp >= ts."""
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if await _block_ts(session, url, mid) < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


async def _find_block_lt_ts(
    session: aiohttp.ClientSession, url: str, ts: int, head: int
) -> int:
    """Largest block number with timestamp < ts."""
    ge = await _find_block_ge_ts(session, url, ts, head + 1)
    return max(0, ge - 1)


async def _fetch_base_fee_series(
    session: aiohttp.ClientSession, url: str, b_lo: int, b_hi: int
) -> pd.DataFrame:
    """One row per block in [b_lo, b_hi] with base_fee_gwei (timestamp interpolated per chunk)."""
    rows: list[tuple[int, int, float]] = []
    cur = b_hi
    while cur >= b_lo:
        chunk = min(1024, cur - b_lo + 1)
        res = await _rpc(
            session, url, "eth_feeHistory", [hex(chunk), hex(cur), []]
        )
        oldest = int(res["oldestBlock"], 16)
        bfs = res["baseFeePerGas"]
        ts_a = await _block_ts(session, url, oldest)
        last_bn = min(oldest + chunk - 1, cur)
        ts_b = await _block_ts(session, url, last_bn)
        denom = max(chunk - 1, 1)
        for i in range(chunk):
            bn = oldest + i
            if bn < b_lo or bn > b_hi:
                continue
            wei = int(bfs[i], 16)
            gwei = wei / 1e9
            t_unix = int(ts_a + (ts_b - ts_a) * (i / denom))
            rows.append((bn, t_unix, gwei))
        cur = oldest - 1
        await asyncio.sleep(0.05)

    df = pd.DataFrame(rows, columns=["block", "t_unix", "base_fee_gwei"])
    df = df.drop_duplicates(subset=["block"]).sort_values("block")
    df["t"] = pd.to_datetime(df["t_unix"], unit="s", utc=True)
    return df


def _load_swaps(path: Path, t0: pd.Timestamp, t1_excl: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    m = (df["dt"] >= t0) & (df["dt"] < t1_excl)
    return df.loc[m].sort_values("dt")


def _price_ylim(swaps: pd.DataFrame, threshold: float) -> tuple[float, float]:
    """Stablecoin-friendly y-limits; ignores rare sqrtPrice glitches (e.g. >>1 USDC/USR)."""
    if swaps.empty:
        return 0.94, 1.02
    s = swaps["price_usdc"]
    p_lo = float(s.quantile(0.01))
    p_hi = float(s.quantile(0.99))
    if p_hi - p_lo < 1e-9:
        p_lo, p_hi = float(s.min()), float(s.max())
    span = max(p_hi - p_lo, 0.04)
    pad = max(span * 0.10, 0.02)
    y_lo = max(0.0, p_lo - pad)
    y_hi = p_hi + pad
    y_lo = min(y_lo, threshold - 0.06)
    y_hi = max(y_hi, threshold + 0.06, 1.01)
    y_hi = min(y_hi, 1.65)
    return y_lo, y_hi


def _plot(
    swaps: pd.DataFrame,
    base: pd.DataFrame,
    t0: pd.Timestamp,
    t1_excl: pd.Timestamp,
    out: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "font.family": "monospace",
        }
    )
    fig, ax_p = plt.subplots(figsize=(13, 5.2))
    ax_g = ax_p.twinx()
    ax_g.set_zorder(1)
    ax_p.set_zorder(2)
    ax_p.patch.set_visible(False)

    if not base.empty:
        ax_g.plot(
            base["t"],
            base["base_fee_gwei"],
            color=C_BASE,
            linewidth=0.85,
            alpha=0.9,
            label="Base fee (gwei)",
        )
    ax_g.set_ylabel("Base fee (gwei)", color=C_BASE, fontsize=10)
    ax_g.tick_params(axis="y", labelcolor=C_BASE)
    ax_g.spines["right"].set_edgecolor("#d1d5db")

    if not swaps.empty:
        # Scatter only: a line plot connects through rare sqrtPrice outliers (~tens USDC/USR)
        # and draws misleading spikes even when y-limits are correct (clips to axis edge).
        ax_p.scatter(
            swaps["dt"],
            swaps["price_usdc"],
            color=C_PRICE,
            s=6,
            alpha=0.55,
            linewidths=0,
            label=f"USR/USDC swap price ({len(swaps):,} swaps)",
            zorder=4,
            clip_on=True,
        )
    ax_p.axhline(
        DEPEG_THRESHOLD,
        color=C_THRESH,
        linewidth=1.2,
        linestyle="--",
        label=f"Threshold ({DEPEG_THRESHOLD})",
        zorder=5,
    )
    ax_p.set_ylabel("Price (USDC / USR)", color=C_PRICE, fontsize=10)
    ax_p.tick_params(axis="y", labelcolor=C_PRICE)
    ax_p.spines["left"].set_edgecolor(C_PRICE)
    ax_p.set_xlabel("Time (UTC)", fontsize=10)
    ax_p.set_xlim(t0, t1_excl)

    y_lo, y_hi = _price_ylim(swaps, DEPEG_THRESHOLD)
    ax_p.set_ylim(y_lo, y_hi)

    title = (
        f"USR/USDC depeg vs L1 base fee  |  {t0:%Y-%m-%d %H:%M} → {t1_excl:%Y-%m-%d %H:%M} UTC"
    )
    ax_p.set_title(title, fontsize=11, pad=10)

    h1, l1 = ax_p.get_legend_handles_labels()
    h2, l2 = ax_g.get_legend_handles_labels()
    ax_p.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.95)

    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz="UTC"))
    fig.autofmt_xdate()
    ax_p.set_ylim(y_lo, y_hi)

    fig.text(
        0.5,
        0.01,
        "Price: sqrtPriceX96 → USDC/USR (scatter clipped); y-scale from 1–99% swap quantiles "
        "(rare arb ticks omitted). Base fee: eth_feeHistory.",
        ha="center",
        fontsize=6.5,
        color="#6b7280",
    )
    fig.subplots_adjust(bottom=0.14, right=0.88, left=0.08, top=0.9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


async def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    t0 = pd.Timestamp(args.start)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    t1_excl = pd.Timestamp(args.end)
    if t1_excl.tzinfo is None:
        t1_excl = t1_excl.tz_localize("UTC")
    else:
        t1_excl = t1_excl.tz_convert("UTC")

    if t1_excl <= t0:
        raise SystemExit("--end must be after --start")

    swaps = _load_swaps(args.swaps_csv, t0, t1_excl)
    candidates = _rpc_url_candidates(args.rpc_url or "")

    async with aiohttp.ClientSession() as session:
        rpc = await _pick_working_rpc(session, candidates)
        print(f"Using RPC: {rpc}", file=sys.stderr)
        head = await _eth_block_number(session, rpc)
        ts0 = int(t0.timestamp())
        ts1 = int(t1_excl.timestamp())
        b_lo = await _find_block_ge_ts(session, rpc, ts0, head + 1)
        b_hi = await _find_block_lt_ts(session, rpc, ts1, head)
        if b_hi < b_lo:
            raise SystemExit("Empty block range (check dates vs chain history).")
        base = await _fetch_base_fee_series(session, rpc, b_lo, b_hi)

    if swaps.empty:
        print("Warning: no swaps in window — check", args.swaps_csv, file=sys.stderr)

    args.output_swaps_csv.parent.mkdir(parents=True, exist_ok=True)
    swaps_out = swaps.drop(columns=["dt"], errors="ignore")
    swaps_out.to_csv(args.output_swaps_csv, index=False)
    print(f"Wrote {args.output_swaps_csv}")
    base.to_csv(args.output_base_csv, index=False)
    print(f"Wrote {args.output_base_csv}")

    _plot(swaps, base, t0, t1_excl, args.output_png)
    print(f"Wrote {args.output_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="USR/USDC depeg vs ETH base fee (dual axis)")
    p.add_argument(
        "--start",
        default="2026-03-22T00:00:00+00:00",
        help="UTC window start (inclusive)",
    )
    p.add_argument(
        "--end",
        default="2026-03-23T00:00:00+00:00",
        help="UTC window end (exclusive)",
    )
    p.add_argument(
        "--swaps-csv",
        type=Path,
        default=Path("data/swaps/usr_usdc_uniswap_v3_swaps.csv"),
    )
    p.add_argument(
        "--output-swaps-csv",
        type=Path,
        default=Path("data/usr_depeg_basefee_swaps.csv"),
    )
    p.add_argument(
        "--output-base-csv",
        type=Path,
        default=Path("data/usr_depeg_basefee_blocks.csv"),
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/usr_depeg_vs_basefee.png"),
    )
    p.add_argument(
        "--rpc-url",
        default="",
        help="Use only this JSON-RPC URL (no automatic fallbacks)",
    )
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
