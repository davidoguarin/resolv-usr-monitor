"""Deprecated: use `python -m src.fetch_hourly_two_month` instead."""
from __future__ import annotations

import warnings

from src.fetch_hourly_two_month import main

if __name__ == "__main__":
    warnings.warn(
        "src.hourly_history is deprecated — run: python -m src.fetch_hourly_two_month",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
