from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .preprocess import select_close_prices


def local_loader(
    *,
    path: str | Path,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    max_missing_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Read a Yahoo Finance-style CSV snapshot and return a close-price matrix.

    Returns a DataFrame with:
    - index: dates
    - columns: tickers
    - values: close prices
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True, low_memory=False)
    return select_close_prices(
        df,
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
        max_missing_ratio=max_missing_ratio,
    )
