from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def select_close_prices(
    df: pd.DataFrame,
    *,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    fill_missing: bool = True,
    max_missing_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Normaliza un DataFrame estilo yfinance a una matriz de precios de cierre.

    Acepta columnas simples o MultiIndex y devuelve solo cierres por ticker.
    """
    if df.empty:
        raise ValueError("El DataFrame de entrada esta vacio.")

    prices = df.copy()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    if isinstance(prices.columns, pd.MultiIndex):
        prices = _extract_close_from_multiindex(prices, prefer_adj_close=prefer_adj_close)
    else:
        prices.columns = [str(col) for col in prices.columns]

    prices = prices.loc[:, ~prices.columns.duplicated()].copy()

    if tickers is not None:
        requested = [str(t) for t in tickers]
        keep = [ticker for ticker in requested if ticker in prices.columns]
        if not keep:
            raise ValueError("Ninguno de los tickers solicitados esta disponible en los datos.")
        prices = prices[keep]

    if start:
        prices = prices[prices.index >= pd.to_datetime(start)]
    if end:
        prices = prices[prices.index <= pd.to_datetime(end)]

    if not 0.0 <= max_missing_ratio <= 1.0:
        raise ValueError("`max_missing_ratio` debe estar entre 0 y 1.")

    if not prices.empty:
        missing_ratio = prices.isna().mean(axis=0)
        prices = prices.loc[:, missing_ratio <= max_missing_ratio]

    if fill_missing:
        # Fill forward first to avoid using future values when a past value exists.
        # Backward fill only resolves leading gaps that ffill cannot cover.
        prices = prices.ffill().bfill()

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise ValueError("No quedaron precios disponibles tras filtrar y limpiar los datos.")

    if freq:
        prices = prices.asfreq(freq, method="pad")

    return prices


def _extract_close_from_multiindex(df: pd.DataFrame, *, prefer_adj_close: bool) -> pd.DataFrame:
    preferred_field = "Adj Close" if prefer_adj_close else "Close"
    fallback_field = "Close" if prefer_adj_close else "Adj Close"

    direct = _pick_columns(df, field_name=preferred_field)
    if direct.empty:
        direct = _pick_columns(df, field_name=fallback_field)
    if direct.empty:
        raise ValueError("No se encontraron columnas 'Adj Close' ni 'Close' en el DataFrame.")

    return direct


def _pick_columns(df: pd.DataFrame, *, field_name: str) -> pd.DataFrame:
    selected: list[tuple] = []
    rename_map: dict[tuple, str] = {}

    for col in df.columns:
        parts = tuple(str(level) for level in col)
        if len(parts) < 2:
            continue

        if parts[-1] == field_name:
            selected.append(col)
            rename_map[col] = parts[0]
        elif parts[0] == field_name:
            selected.append(col)
            rename_map[col] = parts[-1]

    if not selected:
        return pd.DataFrame(index=df.index)

    out = df[selected].copy()
    out.columns = [rename_map[col] for col in selected]
    return out
