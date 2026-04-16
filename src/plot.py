"""
Depeg & liquidity visualisation — USDe and USR vs USDC (March 20-24 2026).

Two-panel figure
----------------
  Top   : Price depeg plot
            - Uniswap V3 only (5-min, GeckoTerminal)
            - Dashed red line at 0.98 USDC  ← withdrawal threshold
  Bottom: Liquidity panel
            - Rolling 24-hour USD volume for each Uniswap V3 pool
              (proxy for on-chain liquidity depth; Curve volume unavailable)

Usage
-----
    python -m src.plot
    python -m src.plot --output figures/depeg_liquidity.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── style (light background) ───────────────────────────────────────────────
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

# ── palette ────────────────────────────────────────────────────────────────
C_USDE = "#2563eb"   # blue
C_USR  = "#ea580c"   # orange
C_THRESH = "#dc2626" # red  (0.98 line)
C_EXPLOIT_MARK = "#a16207"
ALPHA_FILL = 0.12

DEPEG_THRESHOLD = 0.98


def load_uniswap(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usde = pd.read_csv(data_dir / "usde_usdc_uniswap_v3.csv")
    usr  = pd.read_csv(data_dir / "usr_usdc_uniswap_v3.csv")
    for df in (usde, usr):
        df["dt"] = pd.to_datetime(df["datetime_open"], utc=True)
        df.set_index("dt", inplace=True)
        df.sort_index(inplace=True)
    return usde, usr


def rolling_volume(df: pd.DataFrame, window: str = "24h") -> pd.Series:
    """Rolling sum of volume_usdc over *window*."""
    return df["volume_usdc"].fillna(0).rolling(window, min_periods=1).sum()


def make_figure(
    usde: pd.DataFrame,
    usr: pd.DataFrame,
    output: Path | None,
) -> None:
    fig, (ax_price, ax_liq) = plt.subplots(
        2, 1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        sharex=True,
    )

    # ── TOP: depeg plot ────────────────────────────────────────────────────
    ax_price.plot(
        usde.index, usde["close"],
        color=C_USDE, linewidth=0.9, label="USDe / USDC  (Uniswap V3 5-min)",
    )
    ax_price.plot(
        usr.index, usr["close"],
        color=C_USR, linewidth=0.9, label="USR / USDC  (Uniswap V3 5-min)",
    )

    # 0.98 withdrawal threshold
    ax_price.axhline(
        DEPEG_THRESHOLD,
        color=C_THRESH, linewidth=1.4, linestyle="--",
        label=f"Withdrawal threshold  ({DEPEG_THRESHOLD} USDC)",
        zorder=5,
    )

    # shade depeg zone beneath threshold
    ax_price.fill_between(
        usr.index, usr["close"], DEPEG_THRESHOLD,
        where=(usr["close"] < DEPEG_THRESHOLD),
        color=C_USR, alpha=ALPHA_FILL,
    )
    ax_price.fill_between(
        usde.index, usde["close"], DEPEG_THRESHOLD,
        where=(usde["close"] < DEPEG_THRESHOLD),
        color=C_USDE, alpha=ALPHA_FILL,
    )

    # y-axis: clip view to 0-1.05 (collapse the USR spike for readability)
    ax_price.set_ylim(-0.02, 1.08)
    ax_price.set_ylabel("Price (USDC)", fontsize=10)
    ax_price.set_title(
        "Stablecoin Depeg Monitor — USDe & USR  |  March 20-24, 2026",
        fontsize=12, pad=10, color="#111827",
    )

    # annotate USR spike out-of-range with a broken-axis indicator
    first_spike = usr[usr["close"] > 1.08].index
    if not first_spike.empty:
        ax_price.annotate(
            "USR spike →\n(price > 1.05,\naxis clipped)",
            xy=(first_spike[0], 1.06),
            xytext=(first_spike[0], 1.02),
            arrowprops=dict(arrowstyle="->", color=C_USR, lw=1.2),
            fontsize=7.5, color=C_USR, ha="center",
        )

    ax_price.legend(fontsize=8.5, loc="lower left")

    # event annotation
    exploit_start = pd.Timestamp("2026-03-22 02:30", tz="UTC")
    ax_price.axvline(exploit_start, color=C_EXPLOIT_MARK, linewidth=0.8, linestyle=":", alpha=0.85)
    ax_price.text(
        exploit_start, 1.04,
        "  USR exploit\n  detected",
        color=C_EXPLOIT_MARK, fontsize=7.5, va="top",
    )

    # ── BOTTOM: liquidity (rolling 24h volume) ─────────────────────────────
    rv_usde = rolling_volume(usde)
    rv_usr  = rolling_volume(usr)

    ax_liq.fill_between(
        usde.index, rv_usde / 1e6,
        color=C_USDE, alpha=0.35, label="USDe — 24h rolling volume (Uniswap V3)",
    )
    ax_liq.plot(usde.index, rv_usde / 1e6, color=C_USDE, linewidth=0.7)

    ax_liq.fill_between(
        usr.index, rv_usr / 1e6,
        color=C_USR, alpha=0.35, label="USR — 24h rolling volume (Uniswap V3)",
    )
    ax_liq.plot(usr.index, rv_usr / 1e6, color=C_USR, linewidth=0.7)

    # mark exploit on liquidity panel too
    ax_liq.axvline(exploit_start, color=C_EXPLOIT_MARK, linewidth=0.8, linestyle=":", alpha=0.85)

    ax_liq.set_ylabel("24h Rolling Volume (M USD)", fontsize=10)
    ax_liq.set_xlabel("Date (UTC)", fontsize=10)
    ax_liq.legend(fontsize=8.5, loc="upper left")
    ax_liq.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))

    # x-axis formatting
    ax_liq.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax_liq.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18], tz="UTC"))
    ax_liq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz="UTC"))

    # footnote
    fig.text(
        0.01, 0.01,
        "Source: GeckoTerminal (Uniswap V3 5-min OHLCV)  |  "
        "Curve pools excluded from depeg plot (unreliable during exploit due to low TVL ~$80K)",
        fontsize=6.5, color="#6b7280",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Depeg & liquidity plot")
    parser.add_argument("--data",   default="data/raw",          help="Directory with CSV files")
    parser.add_argument("--output", default="figures/depeg_liquidity.png")
    args = parser.parse_args()

    data_dir = Path(args.data)
    usde, usr = load_uniswap(data_dir)
    make_figure(usde, usr, Path(args.output))


if __name__ == "__main__":
    main()
