from __future__ import annotations

from pathlib import Path
import time
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
    cache_dir: str | Path | None = None,
    save_path: str | Path | None = None,
    use_saved_data: bool = False,
    save_download: bool = False,
    fallback_to_saved_data: bool = True,
    max_retries: int = 3,
    retry_wait: float = 2.0,
    timeout: int = 20,
    **kwargs,
) -> pd.DataFrame:
    """
    Descarga precios desde Yahoo Finance y devuelve una matriz de cierres.

    La descarga se divide en bloques para soportar listas grandes de tickers.
    Opcionalmente puede guardar la descarga completa en CSV y reutilizarla.
    """
    try:
        import yfinance as yf
        import yfinance.cache as yfc
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar `yfinance`. Instalala para usar `yfinance_loader`."
        ) from exc

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    yfc.set_cache_location(str(resolved_cache_dir))

    ticker_list = _normalize_tickers(tickers)
    if not ticker_list:
        raise ValueError("Debes proporcionar al menos un ticker para descargar desde Yahoo Finance.")
    if batch_size <= 0:
        raise ValueError("`batch_size` debe ser un entero positivo.")
    if max_retries < 1:
        raise ValueError("`max_retries` debe ser al menos 1.")
    if retry_wait < 0:
        raise ValueError("`retry_wait` no puede ser negativo.")

    resolved_save_path = Path(save_path) if save_path is not None else None
    if use_saved_data:
        return _load_saved_prices(
            save_path=resolved_save_path,
            tickers=ticker_list,
            start=start,
            end=end,
            prefer_adj_close=prefer_adj_close,
            freq=freq,
        )

    frames: list[pd.DataFrame] = []
    batch_errors: list[str] = []
    for batch in _chunked(ticker_list, batch_size):
        try:
            raw = _download_batch(
                yf=yf,
                batch=batch,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                repair=repair,
                threads=threads,
                progress=progress,
                ignore_tz=ignore_tz,
                timeout=timeout,
                max_retries=max_retries,
                retry_wait=retry_wait,
                extra_kwargs=kwargs,
            )
        except RuntimeError as exc:
            batch_errors.append(str(exc))
            continue
        if raw is None or raw.empty:
            batch_errors.append(_describe_batch_failure(batch))
            continue
        frames.append(_ensure_multiindex_columns(raw, batch))

    if not frames:
        if fallback_to_saved_data and resolved_save_path is not None and resolved_save_path.exists():
            return _load_saved_prices(
                save_path=resolved_save_path,
                tickers=ticker_list,
                start=start,
                end=end,
                prefer_adj_close=prefer_adj_close,
                freq=freq,
            )
        details = " | ".join(batch_errors) if batch_errors else "sin detalle adicional"
        raise ValueError(
            "Yahoo Finance no devolvio datos para los tickers solicitados. "
            f"Posibles causas: bloqueo de red, rate limit temporal, tickers invalidos o rango sin datos. "
            f"Detalles: {details}"
        )

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index(axis=1)

    if save_download:
        if resolved_save_path is None:
            raise ValueError("Si `save_download=True`, debes proporcionar `save_path`.")
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(resolved_save_path)

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


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)

    return Path(__file__).resolve().parents[2] / ".yfinance-cache"


def _load_saved_prices(
    *,
    save_path: Path | None,
    tickers: list[str],
    start: Optional[str],
    end: Optional[str],
    prefer_adj_close: bool,
    freq: Optional[str],
) -> pd.DataFrame:
    if save_path is None:
        raise ValueError("Debes proporcionar `save_path` para usar datos guardados.")
    if not save_path.exists():
        raise FileNotFoundError(f"No existe el archivo guardado: {save_path}")

    saved = pd.read_csv(save_path, header=[0, 1], index_col=0, parse_dates=True)
    return select_close_prices(
        saved,
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
    )


def _download_batch(
    *,
    yf,
    batch: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
    repair: bool,
    threads: bool,
    progress: bool,
    ignore_tz: bool,
    timeout: int,
    max_retries: int,
    retry_wait: float,
    extra_kwargs: dict,
) -> pd.DataFrame | None:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=list(batch),
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                repair=repair,
                threads=threads,
                progress=progress,
                group_by="ticker",
                ignore_tz=ignore_tz,
                timeout=timeout,
                **extra_kwargs,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            last_error = exc

        if attempt < max_retries and retry_wait > 0:
            time.sleep(retry_wait * attempt)

    if last_error is not None:
        raise RuntimeError(
            f"Fallo la descarga del bloque {list(batch)}: {_classify_error_message(str(last_error))}"
        ) from last_error

    return None


def _describe_batch_failure(batch: Sequence[str]) -> str:
    return f"bloque {list(batch)} sin datos"


def _classify_error_message(message: str) -> str:
    lowered = message.lower()

    if "too many requests" in lowered or "429" in lowered or "rate limit" in lowered:
        return "posible rate limit de Yahoo Finance"
    if "unable to open database file" in lowered or "database" in lowered or "cache" in lowered:
        return "problema con cache local de yfinance"
    if "failed to connect" in lowered or "connectionerror" in lowered or "timeout" in lowered:
        return "problema de red o bloqueo de conexion hacia Yahoo"
    if "not found" in lowered or "no data" in lowered:
        return "ticker invalido o rango sin datos"

    return message
