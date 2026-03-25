from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .local_loader import local_loader
from .yfinance_loader import yfinance_loader


def get_loader(source: str = "yfinance"):
    """
    Devuelve una funcion loader compatible con `Portfolio`.
    """
    source_normalized = source.strip().lower()

    if source_normalized in {"yfinance", "yf", "remote"}:
        return yfinance_loader
    if source_normalized in {"local", "csv"}:
        return local_loader

    raise ValueError(f"Fuente de datos no reconocida: {source}")


def load_prices(
    *,
    source: str = "yfinance",
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    path: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Carga precios desde Yahoo Finance o desde un CSV local.
    """
    loader = get_loader(source)

    if loader is local_loader:
        if path is None:
            raise ValueError("Para source='local' debes proporcionar `path`.")
        return loader(
            path=path,
            tickers=tickers,
            start=start,
            end=end,
            prefer_adj_close=prefer_adj_close,
            freq=freq,
            **kwargs,
        )

    return loader(
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
        **kwargs,
    )
