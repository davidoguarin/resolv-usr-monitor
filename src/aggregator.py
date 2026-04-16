"""
OHLCV normalisation and padding for the two data sources.

GeckoTerminal (Uniswap V3)
  - Arrives pre-aggregated OHLCV (e.g. 5-min or 1-hour via `aggregate`).
  - Pass through with schema normalisation + gap padding (`interval` = bucket width).

Curve prices API
  - Arrives as 1-hour OHLCV candles (API minimum resolution).
  - Pass through with schema normalisation + gap padding at 1-hour intervals.
  - Clearly tagged as hourly in the `resolution` column.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLUMNS = [
    "timestamp_open",    # Unix timestamp of bucket start (UTC)
    "datetime_open",     # ISO 8601 string (UTC)
    "token",
    "pool_id",
    "dex",
    "resolution_sec",    # actual candle width in seconds
    "open",
    "high",
    "low",
    "close",
    "volume_usdc",
    "num_swaps",         # 0 when not available
]

_CURVE_RESOLUTION = 3600      # Curve prices API minimum
_UNISWAP_RESOLUTION = 300     # 5 min from GeckoTerminal


def normalise_geckoterminal(
    candles: list[dict],
    ts_start: int,
    ts_end: int,
    interval: int = _UNISWAP_RESOLUTION,
) -> pd.DataFrame:
    """Normalise GeckoTerminal candles to the shared schema (`interval` = seconds)."""
    if not candles:
        log.warning("No GeckoTerminal candles — returning empty DataFrame.")
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(candles)
    df = df.rename(columns={"timestamp_open": "timestamp_open"})
    df["resolution_sec"] = interval
    df["num_swaps"] = 0

    df = _pad_and_enrich(df, ts_start, ts_end, interval, candles[0])
    return df[OHLCV_COLUMNS].sort_values("timestamp_open").reset_index(drop=True)


def normalise_curve(
    candles: list[dict],
    ts_start: int,
    ts_end: int,
) -> pd.DataFrame:
    """Normalise Curve hourly candles to the shared schema."""
    if not candles:
        log.warning("No Curve candles — returning empty DataFrame.")
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(candles)
    df["resolution_sec"] = _CURVE_RESOLUTION
    df["num_swaps"] = 0

    df = _pad_and_enrich(df, ts_start, ts_end, _CURVE_RESOLUTION, candles[0])
    return df[OHLCV_COLUMNS].sort_values("timestamp_open").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_and_enrich(
    df: pd.DataFrame,
    ts_start: int,
    ts_end: int,
    interval: int,
    sample_row: dict,
) -> pd.DataFrame:
    """Fill missing buckets, add datetime_open column."""
    # Build full index of expected bucket timestamps
    bucket_start = (ts_start // interval) * interval
    bucket_end   = (ts_end   // interval) * interval
    all_buckets = range(bucket_start, bucket_end + interval, interval)
    full_index = pd.DataFrame({"timestamp_open": list(all_buckets)})

    df = full_index.merge(df, on="timestamp_open", how="left")

    # Fill metadata columns from sample_row (they're constant per pool)
    for col in ("pool_id", "dex", "token", "resolution_sec"):
        df[col] = df[col].ffill().bfill().fillna(sample_row.get(col, ""))

    # Forward-fill prices into empty buckets (no-trade periods)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].ffill()

    df["volume_usdc"] = df["volume_usdc"].fillna(0.0)
    df["num_swaps"]   = df["num_swaps"].fillna(0).astype(int)
    df["resolution_sec"] = df["resolution_sec"].fillna(interval).astype(int)

    df["datetime_open"] = df["timestamp_open"].apply(
        lambda ts: datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    )
    return df
