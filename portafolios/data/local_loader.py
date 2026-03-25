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
) -> pd.DataFrame:
    """
    Lee un CSV con formato estilo yfinance y devuelve una matriz de cierres.

    Retorna un DataFrame con:
    - index: fechas
    - columns: tickers
    - values: precios de cierre
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return select_close_prices(
        df,
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
    )
