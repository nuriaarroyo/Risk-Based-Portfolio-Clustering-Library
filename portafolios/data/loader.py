from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .base import BaseDataLoader, StandardizedData
from .local_loader import local_loader
from .sources import CSVLoader, YFinanceLoader
from .yfinance_loader import yfinance_loader


def get_loader(source: str = "yfinance"):
    """
    Return a loader function compatible with `Portfolio`.
    """
    source_normalized = source.strip().lower()

    if source_normalized in {"yfinance", "yf", "remote"}:
        return yfinance_loader
    if source_normalized in {"local", "csv"}:
        return local_loader

    raise ValueError(f"Fuente de datos no reconocida: {source}")


def build_data_loader(
    *,
    source: str = "yfinance",
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    max_missing_ratio: float = 0.05,
    path: str | Path | None = None,
    **kwargs,
) -> BaseDataLoader:
    """
    Build an object-oriented loader with standardized output.
    """
    source_normalized = source.strip().lower()

    if source_normalized in {"local", "csv"}:
        if path is None:
            raise ValueError("Para source='local' debes proporcionar `path`.")
        return CSVLoader(
            path=path,
            tickers=tickers,
            start=start,
            end=end,
            prefer_adj_close=prefer_adj_close,
            freq=freq,
            max_missing_ratio=max_missing_ratio,
        )

    if source_normalized in {"yfinance", "yf", "remote"}:
        if tickers is None:
            raise ValueError("Para source='yfinance' debes proporcionar `tickers`.")
        return YFinanceLoader(
            tickers=tickers,
            start=start,
            end=end,
            prefer_adj_close=prefer_adj_close,
            freq=freq,
            max_missing_ratio=max_missing_ratio,
            **kwargs,
        )

    raise ValueError(f"Fuente de datos no reconocida: {source}")


def get_data(
    *,
    source: str = "yfinance",
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    max_missing_ratio: float = 0.05,
    path: str | Path | None = None,
    **kwargs,
) -> StandardizedData:
    """
    Return a standardized data object that is independent of the source.
    """
    loader = build_data_loader(
        source=source,
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
        max_missing_ratio=max_missing_ratio,
        path=path,
        **kwargs,
    )
    return loader.get_data()


def load_prices(
    *,
    source: str = "yfinance",
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    max_missing_ratio: float = 0.05,
    path: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Unified loader for direct use with `Portfolio`.

    Examples:
    - Local:
      `loader=load_prices, loader_kwargs={"source": "local", "path": "..."}`
    - Yahoo Finance:
      `loader=load_prices, loader_kwargs={"source": "yfinance"}`
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
            max_missing_ratio=max_missing_ratio,
            **kwargs,
        )

    return loader(
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
        max_missing_ratio=max_missing_ratio,
        **kwargs,
    )


# Short alias that emphasizes use as a Portfolio loader.
portfolio_loader = load_prices
