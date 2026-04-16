"""
Plot hourly two-month extract (data from fetch_hourly_two_month).

Two figures (no fetch changes):
  • Depeg: USDe / USR hourly close, optional threshold line (default 0.99)
  • TVL: Uniswap pools only (merged parquet/csv or tvl/*_hourly.csv)

`hourly_two_month_depeg.png` always ends **24 March** (UTC), set below — not affected
by `--until-date`. `--until-date` applies only to the TVL figure.

Run
---
    python -m src.plot_hourly_two_month
    python -m src.plot_hourly_two_month --threshold 0.99

Open **`figures/hourly_two_month_depeg.png`** (not `hourly_two_month.png`). If the preview
looks stale, close the tab and reopen, or check the “Generated … UTC” stamp on the figure.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    os.chdir(_REPO_ROOT)
except OSError:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d1d5db",
        "axes.labelcolor": "#1f2937",
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
        "text.color": "#1f2937",
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#d1d5db",
        "font.family": "monospace",
    }
)

C_USDE = "#2563eb"
C_USR = "#ea580c"
C_THRESH = "#dc2626"
STYLES = {"uniswap_v3": "-", "curve": "--"}
WIDTH = {"uniswap_v3": 1.0, "curve": 1.15}

# Last calendar day included on hourly_two_month_depeg.png (UTC, inclusive through 23:59:59).
HOURLY_TWO_MONTH_DEPEG_END_DATE = "2026-03-24"


def _load_ohlcv(data_dir: Path) -> pd.DataFrame:
    """Prefer per-pool `ohlcv/*.csv` (source of truth); `combined_*` can be stale/partial."""
    ohlcv_dir = data_dir / "ohlcv"
    if not ohlcv_dir.is_dir():
        raise FileNotFoundError(f"Missing {ohlcv_dir}")

    pool_csvs = sorted(
        p
        for p in ohlcv_dir.glob("*.csv")
        if p.name != "combined_ohlcv.csv"
    )
    if pool_csvs:
        parts = [pd.read_csv(p) for p in pool_csvs]
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp_open", "pool_id"], keep="last")
        return (
            df.sort_values(["timestamp_open", "token", "pool_id"])
            .reset_index(drop=True)
        )

    pq = ohlcv_dir / "combined_ohlcv.parquet"
    csv = ohlcv_dir / "combined_ohlcv.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No OHLCV in {ohlcv_dir}. Run: python -m src.fetch_hourly_two_month"
    )


def _load_merged(data_dir: Path) -> Optional[pd.DataFrame]:
    pq = data_dir / "uniswap_ohlcv_with_tvl.parquet"
    csv = data_dir / "uniswap_ohlcv_with_tvl.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _load_tvl_frames(data_dir: Path) -> Optional[pd.DataFrame]:
    """TVL rows for plotting: merged file if useful, else raw tvl/*_hourly.csv."""
    merged = _load_merged(data_dir)
    if merged is not None and merged["tvl_usd"].notna().any():
        return merged.loc[merged["tvl_usd"].notna()].copy()

    tvl_dir = data_dir / "tvl"
    if not tvl_dir.is_dir():
        return None
    parts = list(tvl_dir.glob("*_hourly.csv"))
    if not parts:
        return None
    return pd.concat([pd.read_csv(p) for p in sorted(parts)], ignore_index=True)


def _end_of_day_utc(iso_date: str) -> pd.Timestamp:
    """Inclusive end: last instant of that calendar day in UTC (23:59:59)."""
    d = pd.Timestamp(iso_date)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    return d.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)


def _filter_until(df: pd.DataFrame, tcol: str, until_end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df[tcol] <= until_end].copy()


def make_depeg_figure(
    ohlcv: pd.DataFrame,
    output: Path,
    *,
    until_end: pd.Timestamp,
    threshold: float,
) -> None:
    ohlcv = ohlcv.copy()
    ohlcv["t"] = pd.to_datetime(ohlcv["datetime_open"], utc=True)
    ohlcv = _filter_until(ohlcv, "t", until_end)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)

    for ax, token, color in zip(axes, ["USDe", "USR"], [C_USDE, C_USR]):
        sub = ohlcv[ohlcv["token"] == token].dropna(subset=["close"])
        for (pool_id, dex), g in sub.groupby(["pool_id", "dex"]):
            g = g.sort_values("t")
            ax.plot(
                g["t"],
                g["close"],
                linestyle=STYLES.get(dex, "-"),
                linewidth=WIDTH.get(dex, 1.0),
                alpha=0.9,
                label=f"{pool_id} ({dex})",
                color=color,
            )
        ax.axhline(
            threshold,
            color=C_THRESH,
            linewidth=1.2,
            linestyle="--",
            label=f"Threshold ({threshold})",
        )
        ax.set_ylabel(f"{token}\nUSDC / token")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
        ax.set_title(f"{token} — hourly close", fontsize=10, pad=6)

    # sharex hides top-panel date labels by default — show dates on USDe too
    xfmt = mdates.DateFormatter("%Y-%m-%d", tz="UTC")
    for ax in axes:
        ax.xaxis.set_major_formatter(xfmt)
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    axes[-1].set_xlabel("Time (UTC)")

    fig.suptitle(
        f"Hourly depeg (through {until_end.date()} UTC)",
        fontsize=12,
        fontweight="medium",
        y=1.02,
    )
    fig.subplots_adjust(hspace=0.35, top=0.90, bottom=0.12)
    fig.text(
        0.01,
        0.02,
        "Source: fetch_hourly_two_month → Gecko/Curve OHLCV",
        fontsize=6.5,
        color="#6b7280",
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(
        0.99,
        0.02,
        f"Generated {stamp}",
        fontsize=5.5,
        ha="right",
        va="bottom",
        color="#9ca3af",
    )

    out_path = output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_tvl_figure(
    tvl: pd.DataFrame,
    output: Path,
    *,
    until_end: pd.Timestamp,
) -> None:
    tvl = tvl.copy()
    tvl["t"] = pd.to_datetime(tvl["datetime_open"], utc=True)
    tvl = _filter_until(tvl, "t", until_end)
    if tvl.empty:
        raise ValueError("No TVL rows after date filter.")

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.tab10
    for i, (pool_id, g) in enumerate(tvl.groupby("pool_id")):
        g = g.sort_values("t")
        ax.plot(
            g["t"],
            g["tvl_usd"] / 1e6,
            linewidth=1.1,
            color=cmap(i % 10),
            label=f"{pool_id}",
        )
    ax.set_ylabel("TVL (M USD)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_title(
        f"Uniswap V3 pool TVL — hourly (subgraph)\nthrough {until_end.date()} UTC",
        fontsize=11,
        pad=8,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz="UTC"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    fig.subplots_adjust(bottom=0.15, top=0.88)
    fig.text(
        0.01,
        0.02,
        "Source: Uniswap V3 subgraph poolHourDatas (or tvl/*_hourly.csv)",
        fontsize=6.5,
        color="#6b7280",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot hourly two-month extract (depeg + TVL)")
    ap.add_argument("--data", type=Path, default=Path("data/hourly_two_month"))
    ap.add_argument(
        "--until-date",
        default="2026-03-21",
        help="Last calendar day for the TVL figure only (UTC). Depeg plot uses HOURLY_TWO_MONTH_DEPEG_END_DATE.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Horizontal depeg reference (USDC per token)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("figures/hourly_two_month_depeg.png"),
        help="Depeg figure output path",
    )
    ap.add_argument(
        "--output-tvl",
        type=Path,
        default=Path("figures/hourly_two_month_tvl.png"),
        help="TVL-only figure output path",
    )
    ap.add_argument(
        "--skip-tvl",
        action="store_true",
        help="Do not write the TVL figure",
    )
    args = ap.parse_args()

    until_depeg = _end_of_day_utc(HOURLY_TWO_MONTH_DEPEG_END_DATE)
    until_tvl = _end_of_day_utc(args.until_date)

    ohlcv = _load_ohlcv(args.data)
    make_depeg_figure(
        ohlcv, args.output, until_end=until_depeg, threshold=args.threshold
    )
    dep = args.output.resolve()
    print(f"Saved → {dep}  ({dep.stat().st_size:,} bytes)")

    if not args.skip_tvl:
        tvl = _load_tvl_frames(args.data)
        if tvl is not None and not tvl.empty and "tvl_usd" in tvl.columns:
            try:
                make_tvl_figure(tvl, args.output_tvl, until_end=until_tvl)
                print(f"Saved → {args.output_tvl}")
            except ValueError as e:
                print(f"TVL plot skipped: {e}")
        else:
            print("TVL plot skipped: no tvl_usd data (merged file empty or tvl/*.csv missing).")


if __name__ == "__main__":
    main()
