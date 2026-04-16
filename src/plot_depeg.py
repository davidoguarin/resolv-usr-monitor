"""
Plot USDe and USR depeg vs USDC from combined OHLCV (close price ≈ USDC per 1 token).

Reads output from `python -m src.main` (combined_ohlcv.parquet or .csv).

Usage
-----
    python -m src.plot_depeg
    python -m src.plot_depeg --input data/raw/combined_ohlcv.csv --output figures/depeg.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def plot_depeg(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["t"] = pd.to_datetime(df["datetime_open"], utc=True)
    df = df.dropna(subset=["close"])

    tokens = ["USDe", "USR"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    fig.suptitle("Stablecoin peg vs USDC (DEX pool close)", fontsize=13, fontweight="medium")

    styles = {
        "uniswap_v3": {"linestyle": "-", "linewidth": 1.0, "alpha": 0.9},
        "curve": {"linestyle": "--", "linewidth": 1.2, "alpha": 0.85},
    }

    for ax, token in zip(axes, tokens):
        sub = df[df["token"] == token]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {token}", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("USDC / token")
            continue

        for (pool_id, dex), g in sub.groupby(["pool_id", "dex"]):
            g = g.sort_values("t")
            kw = styles.get(dex, {"linestyle": "-", "linewidth": 1.0, "alpha": 0.8})
            label = f"{pool_id} ({dex})"
            ax.plot(g["t"], g["close"], label=label, **kw)

        ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle=":", label="$1.00 peg")
        ax.set_ylabel(f"{token}\nUSDC / token")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot USDe / USR depeg from combined OHLCV")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to combined_ohlcv.parquet or .csv (default: parquet then csv under data/raw)",
    )
    parser.add_argument(
        "--output",
        default="figures/depeg_usde_usr.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
    else:
        raw = Path("data/raw")
        pq, csv = raw / "combined_ohlcv.parquet", raw / "combined_ohlcv.csv"
        in_path = pq if pq.exists() else csv
        if not in_path.exists():
            raise SystemExit(
                f"No default input found. Run `python -m src.main` first, or pass --input. "
                f"Looked for {pq} and {csv}."
            )

    df = load_ohlcv(in_path)
    plot_depeg(df, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
