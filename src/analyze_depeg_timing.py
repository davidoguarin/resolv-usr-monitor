"""
Time from peg band to the 0.98 threshold (per Uniswap V3 swap timestamps).

This answers: "How much clock time passed between the last swap that still looked
pegged and the first swap that printed at or below 0.98 USDC?"

Important
---------
- Times are **between on-chain Swap events** in this pool only. If nobody traded for
  an hour, the gap is "no observations", not necessarily "price stood still".
- Still useful for prevention: a **long** gap with a sudden first print below 0.98
  means monitoring had little on-chain warning in this venue; **short** gap means
  the move showed up quickly in consecutive swaps.

Usage
-----
    python -m src.analyze_depeg_timing
    python -m src.analyze_depeg_timing --peg-band 0.995 --threshold 0.98
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def timing_to_threshold(
    df: pd.DataFrame,
    *,
    peg_band: float,
    threshold: float,
    slip_band: float | None = 0.999,
) -> dict:
    """
    df must have columns: datetime (or dt), price_usdc, sorted by time.
    Returns dict with keys or None if no breach.
    """
    work = df.sort_values("dt").reset_index(drop=True)
    prices = work["price_usdc"].astype(float)

    below = prices < threshold
    if not below.any():
        return {"breach": False}

    first_breach_idx = int(below.to_numpy().argmax())
    prior = work.iloc[:first_breach_idx]
    if prior.empty:
        return {
            "breach": True,
            "first_breach_dt": work.loc[first_breach_idx, "dt"],
            "first_breach_price": float(prices.iloc[first_breach_idx]),
            "last_pegged_dt": None,
            "last_pegged_price": None,
            "delta": None,
            "note": "First row in window is already below threshold.",
        }

    in_band = prior["price_usdc"] >= peg_band
    if not in_band.any():
        last_pegged_dt = None
        last_pegged_price = None
    else:
        rel = prior.loc[in_band].index[-1]
        last_pegged_dt = work.loc[rel, "dt"]
        last_pegged_price = float(work.loc[rel, "price_usdc"])

    first_breach_dt = work.loc[first_breach_idx, "dt"]
    first_breach_price = float(prices.iloc[first_breach_idx])

    delta = None
    if last_pegged_dt is not None:
        delta = first_breach_dt - last_pegged_dt

    # Optional: first time price dipped under slip_band (e.g. 0.999) before breach
    first_slip_dt = None
    first_slip_price = None
    slip_delta_to_breach = None
    if slip_band is not None:
        slip_mask = (prices < slip_band).to_numpy()
        slip_mask[first_breach_idx + 1 :] = False
        if slip_mask.any():
            slip_idx = int(slip_mask.argmax())
            first_slip_dt = work.loc[slip_idx, "dt"]
            first_slip_price = float(prices.iloc[slip_idx])
            slip_delta_to_breach = first_breach_dt - first_slip_dt

    return {
        "breach": True,
        "last_pegged_dt": last_pegged_dt,
        "last_pegged_price": last_pegged_price,
        "first_breach_dt": first_breach_dt,
        "first_breach_price": first_breach_price,
        "delta": delta,
        "first_slip_dt": first_slip_dt,
        "first_slip_price": first_slip_price,
        "slip_delta_to_breach": slip_delta_to_breach,
    }


def _fmt_td(td: pd.Timedelta | None) -> str:
    if td is None:
        return "—"
    total = td.total_seconds()
    if total < 0:
        return str(td)
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts) + f"  ({total:,.0f} s total)"


def main() -> None:
    p = argparse.ArgumentParser(description="Peg → 0.98 threshold timing from swap CSVs")
    p.add_argument("--data", default="data/swaps", type=Path)
    p.add_argument("--peg-band", type=float, default=0.995, help="Still 'pegged' if price >= this")
    p.add_argument("--threshold", type=float, default=0.98, help="Breach when price < this")
    p.add_argument(
        "--slip-band",
        type=float,
        default=0.999,
        help="Report first dip below this before breach (0=disable)",
    )
    args = p.parse_args()
    slip = None if args.slip_band <= 0 else args.slip_band

    files = {
        "USDe": args.data / "usde_usdc_uniswap_v3_swaps.csv",
        "USR": args.data / "usr_usdc_uniswap_v3_swaps.csv",
    }

    print()
    print("Peg → threshold timing  (Uniswap V3 pool, per swap)")
    print(f"  Peg band   : price >= {args.peg_band:g} USDC")
    print(f"  Threshold  : first swap with price < {args.threshold:g} USDC")
    if slip:
        print(f"  Slip marker: first swap with price < {slip:g} before/at breach")
    print()

    for token, path in files.items():
        print(f"── {token}  ({path.name}) ──")
        if not path.exists():
            print(f"  Missing file: {path}")
            continue
        df = pd.read_csv(path)
        df["dt"] = pd.to_datetime(df["datetime"], utc=True)
        r = timing_to_threshold(df, peg_band=args.peg_band, threshold=args.threshold, slip_band=slip)

        if not r["breach"]:
            print(f"  No swap below {args.threshold:g} USDC in this extract.")
            print()
            continue

        print(f"  Last swap still >= {args.peg_band:g} (before first < {args.threshold:g}):")
        if r["last_pegged_dt"] is not None:
            print(f"    {r['last_pegged_dt']}  |  price = {r['last_pegged_price']:.6f} USDC")
        else:
            print("    (none in file before breach — window starts already depegged)")

        print(f"  First swap below {args.threshold:g}:")
        print(f"    {r['first_breach_dt']}  |  price = {r['first_breach_price']:.6f} USDC")

        print(f"  Wall-clock span (last peg-band swap → first < {args.threshold:g}):")
        print(f"    {_fmt_td(r['delta'])}")

        if slip and r.get("first_slip_dt") is not None:
            print(f"  First slip (< {slip:g}) in that episode:")
            print(f"    {r['first_slip_dt']}  |  price = {r['first_slip_price']:.6f} USDC")
            print(f"  Span (first slip → first < {args.threshold:g}):")
            print(f"    {_fmt_td(r['slip_delta_to_breach'])}")
            if r["slip_delta_to_breach"] is not None and r["slip_delta_to_breach"].total_seconds() == 0:
                print(
                    f"    → No earlier swap printed below {slip:g}; the first observable slip "
                    f"was already through {args.threshold:g}."
                )

        if r.get("note"):
            print(f"  Note: {r['note']}")
        print()

    print(
        "Interpretation: longer gaps mean fewer on-chain prints before the breach; "
        "prevention would need other signals (mempool, other venues, protocol metrics)."
    )
    print()


if __name__ == "__main__":
    main()
