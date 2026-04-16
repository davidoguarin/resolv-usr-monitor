"""
Two zoom figures (dual y-axis each): price + swaps/minute on one time axis.

  • figures/zoom_ethena.png — USDe only, **same ±hours window** as Resolv (centered on
    first USR swap below threshold) so x-axis dates match the Resolv figure.
  • figures/zoom_resolv.png — USR only, anchored at first swap below threshold USDC
  • figures/zoom_ethena_0050_0300utc.png / zoom_resolv_0050_0300utc.png — same charts,
    fixed **00:50–03:00 UTC** on the anchor’s UTC calendar day.

Left axis: price (USDC) with scatter per swap + faint line. Right axis: swap count
resampled to 1-minute buckets. Same light theme as plot_highres.

Usage
-----
    python -m src.plot_zoom
    python -m src.plot_zoom --hours 3 --output-ethena figures/zoom_ethena.png \\
        --output-resolv figures/zoom_resolv.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ── style (match plot_highres light theme) ──────────────────────────────────
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
# Slightly darker hues for swaps/minute (right axis) so volume reads more clearly
C_USDE_VOL = "#1d4ed8"
C_USR_VOL = "#c2410c"
C_THRESH = "#dc2626"
C_CROSS = "#a16207"

DEPEG_THRESHOLD = 0.99

# Fixed narrow zoom: [day0 + start, day0 + end) on the anchor’s UTC calendar day.
ZOOM_FIXED_START_AFTER_MIDNIGHT = pd.Timedelta(hours=0, minutes=50)
ZOOM_FIXED_END_AFTER_MIDNIGHT = pd.Timedelta(hours=3)

_VOL_COLOR = {C_USDE: C_USDE_VOL, C_USR: C_USR_VOL}


def load_swaps(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usde = pd.read_csv(data_dir / "usde_usdc_uniswap_v3_swaps.csv")
    usr = pd.read_csv(data_dir / "usr_usdc_uniswap_v3_swaps.csv")
    for df in (usde, usr):
        df["dt"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("dt", inplace=True)
        df.sort_index(inplace=True)
    return usde, usr


def first_threshold_cross(usr: pd.DataFrame, threshold: float) -> pd.Timestamp:
    below = usr["price_usdc"] < threshold
    if not below.any():
        raise ValueError(f"No USR swap with price < {threshold} in data.")
    return usr.loc[below].index[0]


def fixed_utc_hour_window(t_anchor: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Same UTC calendar day as anchor; swaps in [t0, t1) per ZOOM_FIXED_*_AFTER_MIDNIGHT."""
    day0 = t_anchor.normalize()
    t0 = day0 + ZOOM_FIXED_START_AFTER_MIDNIGHT
    t1 = day0 + ZOOM_FIXED_END_AFTER_MIDNIGHT
    return t0, t1


def make_figure_dual(
    df: pd.DataFrame,
    *,
    issuer: str,
    pair_label: str,
    color: str,
    t_anchor: pd.Timestamp,
    threshold: float,
    output: Path,
    footnote_extra: str,
    vline_label: str,
    vol_ymax: Optional[float] = None,
    half_window: Optional[pd.Timedelta] = None,
    t_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> None:
    if t_window is not None:
        t0, t1 = t_window
        z = df.loc[(df.index >= t0) & (df.index < t1)]
    else:
        if half_window is None:
            raise ValueError("Provide half_window or t_window")
        t0, t1 = t_anchor - half_window, t_anchor + half_window
        z = df.loc[(df.index >= t0) & (df.index <= t1)]
    if z.empty:
        raise ValueError(f"No swaps for {issuer} in window {t0} → {t1}.")

    fig, ax_price = plt.subplots(figsize=(14, 5.8))
    ax_vol = ax_price.twinx()

    # Volume behind price (z-order)
    ax_vol.set_zorder(1)
    ax_price.set_zorder(2)
    ax_price.patch.set_visible(False)

    vol_color = _VOL_COLOR.get(color, color)

    roll = z["price_usdc"].resample("1min").count().rename("n")
    ax_vol.fill_between(
        roll.index,
        roll.values,
        color=vol_color,
        alpha=0.42,
        linewidth=0,
        label=f"{pair_label} — swaps / minute",
        zorder=1,
    )
    ax_vol.plot(
        roll.index, roll.values, color=vol_color, linewidth=0.9, alpha=0.92, zorder=2
    )
    ax_vol.set_ylabel("Swaps / minute", fontsize=10, color=vol_color)
    ax_vol.tick_params(axis="y", labelcolor=vol_color)
    ax_vol.spines["right"].set_edgecolor("#d1d5db")
    ax_vol.grid(False)

    ax_price.plot(
        z.index,
        z["price_usdc"],
        color=color,
        linewidth=0.4,
        alpha=0.4,
        zorder=3,
    )
    ax_price.scatter(
        z.index,
        z["price_usdc"],
        color=color,
        s=7,
        alpha=0.6,
        linewidths=0,
        label=f"{pair_label} — price (USDC), {len(z):,} swaps",
        zorder=5,
    )

    ax_price.axhline(
        threshold,
        color=C_THRESH,
        linewidth=1.4,
        linestyle="--",
        label=f"Threshold  ({threshold} USDC)",
        zorder=4,
    )
    ax_price.fill_between(
        z.index,
        z["price_usdc"].clip(upper=threshold),
        threshold,
        where=(z["price_usdc"] < threshold),
        color=color,
        alpha=0.12,
        zorder=2,
    )

    ax_price.axvline(
        t_anchor,
        color=C_CROSS,
        linewidth=1.0,
        linestyle=":",
        alpha=0.9,
        label=vline_label,
        zorder=6,
    )

    # Raw min/max include rare sqrtPrice glitches (e.g. ~46 USDC/USR) that blow up padding
    # and squash the real depeg band. Use robust quantiles for the y scale.
    s = z["price_usdc"]
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
    ax_price.set_ylim(y_lo, y_hi)

    vmax = float(roll.max()) if len(roll) else 0.0
    auto_top = max(vmax * 1.12, 1.0)
    top = max(vol_ymax, auto_top) if vol_ymax is not None else auto_top
    ax_vol.set_ylim(0, top)

    ax_price.set_ylabel("Price (USDC)", fontsize=10, color="#1f2937")
    ax_price.set_xlabel("Time (UTC)", fontsize=10)
    if t_window is not None:
        ax_price.set_title(
            f"{issuer} — {t0:%Y-%m-%d}  {t0:%H:%M}–{t1:%H:%M} UTC",
            fontsize=11,
            pad=12,
        )
    else:
        assert half_window is not None
        hrs = int(half_window.total_seconds() // 3600)
        ax_price.set_title(
            f"{issuer} — ±{hrs}h zoom  |  {t0:%Y-%m-%d %H:%M} → {t1:%H:%M} UTC",
            fontsize=11,
            pad=12,
        )

    h_p, l_p = ax_price.get_legend_handles_labels()
    h_v, l_v = ax_vol.get_legend_handles_labels()
    ax_price.legend(
        h_p + h_v,
        l_p + l_v,
        fontsize=8.5,
        loc="lower left",
        framealpha=0.95,
    )

    span = t1 - t0
    if span <= pd.Timedelta(hours=4):
        ax_price.set_xlim(t0, t1)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
        if span <= pd.Timedelta(hours=2):
            ax_price.xaxis.set_major_locator(
                mdates.MinuteLocator(
                    byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
                )
            )
            ax_price.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        else:
            ax_price.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax_price.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    else:
        ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz="UTC"))
        ax_price.xaxis.set_minor_locator(
            mdates.MinuteLocator(byminute=[0, 15, 30, 45])
        )
        ax_price.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %d\n%H:%M", tz="UTC")
        )

    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.18)
    fig.text(
        0.08,
        0.02,
        footnote_extra
        + "  |  Etherscan getLogs → Uniswap V3 Swap, sqrtPriceX96 → price",
        fontsize=6.5,
        color="#6b7280",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dual-axis zoom plots (Ethena + Resolv), price + swaps/minute",
    )
    p.add_argument("--data", default="data/swaps", type=Path)
    p.add_argument("--hours", type=float, default=3.0, help="Hours before and after anchor")
    p.add_argument("--threshold", type=float, default=DEPEG_THRESHOLD)
    p.add_argument("--output-ethena", default="figures/zoom_ethena.png", type=Path)
    p.add_argument("--output-resolv", default="figures/zoom_resolv.png", type=Path)
    p.add_argument(
        "--output-ethena-hour",
        default="figures/zoom_ethena_0050_0300utc.png",
        type=Path,
        help="Fixed 00:50–03:00 UTC (anchor day) USDe figure",
    )
    p.add_argument(
        "--output-resolv-hour",
        default="figures/zoom_resolv_0050_0300utc.png",
        type=Path,
        help="Fixed 00:50–03:00 UTC (anchor day) USR figure",
    )
    args = p.parse_args()

    usde, usr = load_swaps(args.data)
    half = pd.Timedelta(hours=args.hours)

    try:
        t_anchor = first_threshold_cross(usr, args.threshold)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    hrs = int(half.total_seconds() // 3600)
    shared_foot = (
        f"Shared window: ±{hrs}h around first USR swap < {args.threshold} USDC "
        f"at {t_anchor} UTC"
    )

    # ── Ethena (USDe): same time window as Resolv ───────────────────────────
    try:
        make_figure_dual(
            usde,
            issuer="Ethena (USDe)",
            pair_label="USDe / USDC",
            color=C_USDE,
            t_anchor=t_anchor,
            half_window=half,
            threshold=args.threshold,
            output=args.output_ethena,
            footnote_extra=shared_foot,
            vline_label="USR first swap < threshold",
        )
    except ValueError as e:
        print(f"Ethena plot skipped: {e}", file=sys.stderr)

    # ── Resolv (USR): anchor at first cross below threshold ─────────────────
    make_figure_dual(
        usr,
        issuer="Resolv (USR)",
        pair_label="USR / USDC",
        color=C_USR,
        t_anchor=t_anchor,
        half_window=half,
        threshold=args.threshold,
        output=args.output_resolv,
        footnote_extra=(
            f"Anchor: first swap below {args.threshold} USDC at {t_anchor} UTC"
        ),
        vline_label="First swap < threshold",
        vol_ymax=200.0,
    )

    # ── Fixed 02:00–03:00 UTC on anchor’s calendar day (narrow zoom) ─────────
    t_h0, t_h1 = fixed_utc_hour_window(t_anchor)
    hour_foot = (
        f"Window {t_h0:%Y-%m-%d} {t_h0:%H:%M}–{t_h1:%H:%M} UTC  |  anchor: first USR < "
        f"{args.threshold} at {t_anchor} UTC"
    )
    try:
        make_figure_dual(
            usde,
            issuer="Ethena (USDe)",
            pair_label="USDe / USDC",
            color=C_USDE,
            t_anchor=t_anchor,
            half_window=None,
            t_window=(t_h0, t_h1),
            threshold=args.threshold,
            output=args.output_ethena_hour,
            footnote_extra=hour_foot,
            vline_label="USR first swap < threshold",
        )
    except ValueError as e:
        print(f"Ethena fixed-band plot skipped: {e}", file=sys.stderr)

    try:
        make_figure_dual(
            usr,
            issuer="Resolv (USR)",
            pair_label="USR / USDC",
            color=C_USR,
            t_anchor=t_anchor,
            half_window=None,
            t_window=(t_h0, t_h1),
            threshold=args.threshold,
            output=args.output_resolv_hour,
            footnote_extra=hour_foot,
            vline_label="First swap < threshold",
            vol_ymax=200.0,
        )
    except ValueError as e:
        print(f"Resolv fixed-band plot skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
