"""
High-resolution depeg plot — per-swap prices from Etherscan.

Two-panel figure (same layout as depeg_liquidity.png, independent file):
  Top   : Per-swap price — USDe and USR vs USDC (each swap = scatter + faint line)
           Dashed red threshold at 0.98 USDC
  Bottom: Per-minute swap count (proxy for on-chain activity / liquidity depth)

Usage
-----
    python -m src.plot_highres
    python -m src.plot_highres --output figures/depeg_highres.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── style (light background) ────────────────────────────────────────────────
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

C_USDE   = "#2563eb"
C_USR    = "#ea580c"
C_THRESH = "#dc2626"
C_EXPLOIT_MARK = "#a16207"
DEPEG_THRESHOLD = 0.98


def load_swaps(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usde = pd.read_csv(data_dir / "usde_usdc_uniswap_v3_swaps.csv")
    usr  = pd.read_csv(data_dir / "usr_usdc_uniswap_v3_swaps.csv")
    for df in (usde, usr):
        df["dt"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("dt", inplace=True)
        df.sort_index(inplace=True)
    return usde, usr


def make_figure(usde: pd.DataFrame, usr: pd.DataFrame, output: Path) -> None:
    fig, (ax_price, ax_act) = plt.subplots(
        2, 1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        sharex=True,
    )

    # ── TOP: per-swap price ────────────────────────────────────────────────
    # USDe: scatter (sparse) + line
    ax_price.scatter(
        usde.index, usde["price_usdc"],
        color=C_USDE, s=6, alpha=0.7, linewidths=0,
        label=f"USDe / USDC  ({len(usde):,} swaps, per-tx)",
        zorder=4,
    )
    ax_price.plot(
        usde.index, usde["price_usdc"],
        color=C_USDE, linewidth=0.5, alpha=0.4,
    )

    # USR (Resolv): faint line + small markers (one per swap → tx timing visible)
    ax_price.plot(
        usr.index, usr["price_usdc"],
        color=C_USR, linewidth=0.4, alpha=0.35, zorder=3,
    )
    ax_price.scatter(
        usr.index, usr["price_usdc"],
        color=C_USR, s=4, alpha=0.5, linewidths=0,
        label=f"USR / USDC  ({len(usr):,} swaps, per-tx)",
        zorder=4,
    )

    # 0.98 withdrawal threshold
    ax_price.axhline(
        DEPEG_THRESHOLD,
        color=C_THRESH, linewidth=1.4, linestyle="--",
        label=f"Withdrawal threshold  ({DEPEG_THRESHOLD} USDC)",
        zorder=5,
    )

    # shade depeg zone
    ax_price.fill_between(
        usr.index, usr["price_usdc"].clip(upper=DEPEG_THRESHOLD), DEPEG_THRESHOLD,
        where=(usr["price_usdc"] < DEPEG_THRESHOLD),
        color=C_USR, alpha=0.10,
    )

    # clip y-axis so the USR spike doesn't squash the interesting region
    ax_price.set_ylim(-0.02, 1.08)

    # exploit annotation
    exploit_ts = pd.Timestamp("2026-03-22 02:30", tz="UTC")
    ax_price.axvline(exploit_ts, color=C_EXPLOIT_MARK, linewidth=0.8, linestyle=":", alpha=0.85)
    ax_price.text(
        exploit_ts, 1.04, "  USR exploit\n  (first swap)",
        color=C_EXPLOIT_MARK, fontsize=7.5, va="top",
    )

    # annotate the out-of-range spike
    spike = usr[usr["price_usdc"] > 1.08]
    if not spike.empty:
        ax_price.annotate(
            f"max spike: ${usr['price_usdc'].max():.1f}\n(axis clipped)",
            xy=(spike.index[0], 1.065),
            color=C_USR, fontsize=7.5, ha="left",
            arrowprops=dict(arrowstyle="->", color=C_USR, lw=1.0),
            xytext=(spike.index[0], 1.04),
        )

    ax_price.set_ylabel("Price (USDC)", fontsize=10)
    ax_price.set_title(
        "Stablecoin Depeg Monitor — Per-Swap Resolution  |  March 20-24, 2026",
        fontsize=12, pad=10,
    )
    ax_price.legend(fontsize=8.5, loc="lower left")

    # ── BOTTOM: per-minute swap count ─────────────────────────────────────
    roll_usde = usde["price_usdc"].resample("1min").count().rename("n")
    roll_usr  = usr["price_usdc"].resample("1min").count().rename("n")

    ax_act.fill_between(roll_usde.index, roll_usde.values,
                        color=C_USDE, alpha=0.35,
                        label="USDe — swaps per minute")
    ax_act.plot(roll_usde.index, roll_usde.values, color=C_USDE, linewidth=0.7)

    ax_act.fill_between(roll_usr.index, roll_usr.values,
                        color=C_USR, alpha=0.35,
                        label="USR — swaps per minute")
    ax_act.plot(roll_usr.index, roll_usr.values, color=C_USR, linewidth=0.7)

    ax_act.axvline(exploit_ts, color=C_EXPLOIT_MARK, linewidth=0.8, linestyle=":", alpha=0.85)
    ax_act.set_ylabel("Swaps / Minute", fontsize=10)
    ax_act.set_xlabel("Date (UTC)", fontsize=10)
    ax_act.legend(fontsize=8.5, loc="upper left")

    # x-axis ticks
    ax_act.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax_act.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18], tz="UTC"))
    ax_act.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz="UTC"))

    fig.text(
        0.01, 0.005,
        "Source: Etherscan getLogs → Uniswap V3 Swap events, sqrtPriceX96 decoded per-tx  |  "
        "USDe 696 swaps · USR 19,062 swaps  |  existing data/raw/ and figures/ untouched",
        fontsize=6.5, color="#6b7280",
    )

    plt.tight_layout(rect=[0, 0.018, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/swaps")
    parser.add_argument("--output", default="figures/depeg_highres.png")
    args = parser.parse_args()

    usde, usr = load_swaps(Path(args.data))
    make_figure(usde, usr, Path(args.output))


if __name__ == "__main__":
    main()
