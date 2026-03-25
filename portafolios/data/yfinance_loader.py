from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from .preprocess import select_close_prices


def yfinance_loader(
    *,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    repair: bool = True,
    threads: bool = True,
    ignore_tz: bool = True,
    batch_size: int = 100,
    progress: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Descarga precios desde Yahoo Finance y devuelve una matriz de cierres.

    La descarga se divide en bloques para soportar listas grandes de tickers.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar `yfinance`. Instalala para usar `yfinance_loader`."
        ) from exc

    ticker_list = _normalize_tickers(tickers)
    if not ticker_list:
        raise ValueError("Debes proporcionar al menos un ticker para descargar desde Yahoo Finance.")
    if batch_size <= 0:
        raise ValueError("`batch_size` debe ser un entero positivo.")

    frames: list[pd.DataFrame] = []
    for batch in _chunked(ticker_list, batch_size):
        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            repair=repair,
            threads=threads,
            progress=progress,
            group_by="ticker",
            ignore_tz=ignore_tz,
            **kwargs,
        )
        if raw is None or raw.empty:
            continue
        frames.append(_ensure_multiindex_columns(raw, batch))

    if not frames:
        raise ValueError("Yahoo Finance no devolvio datos para los tickers solicitados.")

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index(axis=1)

    return select_close_prices(
        combined,
        tickers=ticker_list,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
    )


def _normalize_tickers(tickers: Optional[Iterable[str]]) -> list[str]:
    if tickers is None:
        return []

    clean: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        clean.append(symbol)
    return clean


def _chunked(items: Sequence[str], size: int):
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def _ensure_multiindex_columns(df: pd.DataFrame, batch: Sequence[str]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        return df

    if len(batch) != 1:
        raise ValueError("Respuesta inesperada de Yahoo Finance: columnas simples para varios tickers.")

    ticker = batch[0]
    out = df.copy()
    out.columns = pd.MultiIndex.from_product([[ticker], out.columns])
    return out
