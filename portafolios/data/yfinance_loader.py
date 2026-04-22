from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Iterable, Optional, Sequence

import pandas as pd

from .local_loader import local_loader
from .preprocess import normalize_ticker_label, select_close_prices


def yfinance_loader(
    *,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prefer_adj_close: bool = True,
    freq: Optional[str] = None,
    max_missing_ratio: float = 0.05,
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
    catalog_path: str | Path | None = None,
    fallback_to_saved_data: bool = True,
    max_retries: int = 3,
    retry_wait: float = 2.0,
    timeout: int = 20,
    metadata_sink: dict[str, object] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Download prices from Yahoo Finance and return a close-price matrix.

    Downloads are split into batches so large ticker lists remain manageable.
    The loader can optionally save the full download as CSV, reuse it through
    `local_loader`, and register snapshots in a JSON catalog.
    """
    ticker_list = _normalize_tickers(tickers)
    if not ticker_list:
        raise ValueError("You must provide at least one ticker to download from Yahoo Finance.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    if max_retries < 1:
        raise ValueError("`max_retries` must be at least 1.")
    if retry_wait < 0:
        raise ValueError("`retry_wait` cannot be negative.")

    resolved_save_path = _normalize_path(save_path) if save_path is not None else None
    resolved_catalog_path = _resolve_catalog_path(
        catalog_path=catalog_path,
        save_path=resolved_save_path,
    )
    _update_metadata_sink(
        metadata_sink,
        requested_origin="saved_snapshot" if use_saved_data else "yfinance_download",
        save_path=str(resolved_save_path) if resolved_save_path is not None else None,
        catalog_path=str(resolved_catalog_path) if resolved_catalog_path is not None else None,
        used_saved_data=bool(use_saved_data),
        download_attempted=not use_saved_data,
        download_succeeded=False,
    )
    if use_saved_data:
        prices = _load_saved_prices(
            save_path=resolved_save_path,
            tickers=ticker_list,
            start=start,
            end=end,
            prefer_adj_close=prefer_adj_close,
            freq=freq,
            max_missing_ratio=max_missing_ratio,
        )
        _register_snapshot(
            catalog_path=resolved_catalog_path,
            save_path=resolved_save_path,
            prices=prices,
            tickers=ticker_list,
            start=start,
            end=end,
            interval=interval,
            freq=freq,
            prefer_adj_close=prefer_adj_close,
            max_missing_ratio=max_missing_ratio,
            origin="existing_snapshot",
        )
        _update_metadata_sink(
            metadata_sink,
            resolved_origin="saved_snapshot",
            used_saved_data=True,
        )
        return prices

    try:
        import yfinance as yf
        import yfinance.cache as yfc
    except ImportError as exc:
        raise ImportError(
            "Could not import `yfinance`. Install it before using `yfinance_loader`."
        ) from exc

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    yfc.set_cache_location(str(resolved_cache_dir))

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
            prices = _load_saved_prices(
                save_path=resolved_save_path,
                tickers=ticker_list,
                start=start,
                end=end,
                prefer_adj_close=prefer_adj_close,
                freq=freq,
                max_missing_ratio=max_missing_ratio,
            )
            _register_snapshot(
                catalog_path=resolved_catalog_path,
                save_path=resolved_save_path,
                prices=prices,
                tickers=ticker_list,
                start=start,
                end=end,
                interval=interval,
                freq=freq,
                prefer_adj_close=prefer_adj_close,
                max_missing_ratio=max_missing_ratio,
                origin="existing_snapshot",
            )
            _update_metadata_sink(
                metadata_sink,
                resolved_origin="saved_snapshot_fallback",
                used_saved_data=True,
                download_attempted=True,
            )
            return prices
        details = " | ".join(batch_errors) if batch_errors else "no additional detail"
        raise ValueError(
            "Yahoo Finance did not return data for the requested tickers. "
            f"Possible causes: network blocking, temporary rate limits, invalid tickers, or a date range without data. "
            f"Details: {details}"
        )

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index(axis=1)

    if save_download:
        if resolved_save_path is None:
            raise ValueError("If `save_download=True`, you must provide `save_path`.")
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(resolved_save_path)

    prices = select_close_prices(
        combined,
        tickers=ticker_list,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
        max_missing_ratio=max_missing_ratio,
    )
    if save_download and resolved_save_path is not None:
        _register_snapshot(
            catalog_path=resolved_catalog_path,
            save_path=resolved_save_path,
            prices=prices,
            tickers=ticker_list,
            start=start,
            end=end,
            interval=interval,
            freq=freq,
            prefer_adj_close=prefer_adj_close,
            max_missing_ratio=max_missing_ratio,
            origin="yfinance_download",
        )
    _update_metadata_sink(
        metadata_sink,
        resolved_origin="yfinance_download",
        used_saved_data=False,
        download_attempted=True,
        download_succeeded=True,
    )
    return prices


def _normalize_tickers(tickers: Optional[Iterable[str]]) -> list[str]:
    if tickers is None:
        return []

    clean: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = normalize_ticker_label(ticker)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        clean.append(symbol)
    return clean


def _update_metadata_sink(metadata_sink: dict[str, object] | None, **values: object) -> None:
    if metadata_sink is None:
        return

    for key, value in values.items():
        metadata_sink[key] = value


def _chunked(items: Sequence[str], size: int):
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def _ensure_multiindex_columns(df: pd.DataFrame, batch: Sequence[str]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        return df

    if len(batch) != 1:
        raise ValueError("Unexpected Yahoo Finance response: flat columns were returned for multiple tickers.")

    ticker = batch[0]
    out = df.copy()
    out.columns = pd.MultiIndex.from_product([[ticker], out.columns])
    return out


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return _normalize_path(cache_dir)

    return Path(__file__).resolve().parents[2] / ".yfinance-cache"


def _resolve_catalog_path(
    *,
    catalog_path: str | Path | None,
    save_path: Path | None,
) -> Path | None:
    if catalog_path is not None:
        return _normalize_path(catalog_path)
    if save_path is not None:
        return save_path.parent / "yfinance_snapshot_catalog.json"
    return None


def _load_saved_prices(
    *,
    save_path: Path | None,
    tickers: list[str],
    start: Optional[str],
    end: Optional[str],
    prefer_adj_close: bool,
    freq: Optional[str],
    max_missing_ratio: float,
) -> pd.DataFrame:
    if save_path is None:
        raise ValueError("You must provide `save_path` to use saved data.")
    if not save_path.exists():
        raise FileNotFoundError(f"The saved file does not exist: {save_path}")

    return local_loader(
        path=save_path,
        tickers=tickers,
        start=start,
        end=end,
        prefer_adj_close=prefer_adj_close,
        freq=freq,
        max_missing_ratio=max_missing_ratio,
    )


def _register_snapshot(
    *,
    catalog_path: Path | None,
    save_path: Path | None,
    prices: pd.DataFrame,
    tickers: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    freq: Optional[str],
    prefer_adj_close: bool,
    max_missing_ratio: float,
    origin: str,
) -> None:
    if catalog_path is None or save_path is None:
        return

    payload = _read_catalog(catalog_path)
    snapshots = payload["snapshots"]
    now = _timestamp_utc()
    save_path_str = str(_normalize_path(save_path))
    save_key = _snapshot_key(save_path_str)
    entry = next(
        (
            item
            for item in snapshots
            if isinstance(item, dict) and _snapshot_key(item.get("path")) == save_key
        ),
        None,
    )
    if entry is None:
        entry = {
            "snapshot_name": save_path.stem,
            "path": save_path_str,
            "created_at_utc": now,
            "origin": origin,
        }
        snapshots.append(entry)

    if origin == "yfinance_download":
        entry["origin"] = origin
        entry["downloaded_at_utc"] = now
    else:
        entry.setdefault("origin", origin)

    entry.update(
        {
            "snapshot_name": save_path.stem,
            "path": save_path_str,
            "last_used_at_utc": now,
            "last_requested_tickers": list(tickers),
            "last_requested_start": start,
            "last_requested_end": end,
            "interval": interval,
            "frequency": freq or interval,
            "prefer_adj_close": prefer_adj_close,
            "max_missing_ratio": max_missing_ratio,
            "n_requested_tickers": len(tickers),
            "n_assets": int(prices.shape[1]),
            "n_rows": int(prices.shape[0]),
            "data_start": str(prices.index.min().date()) if not prices.empty else None,
            "data_end": str(prices.index.max().date()) if not prices.empty else None,
            "file_size_bytes": save_path.stat().st_size if save_path.exists() else None,
        }
    )
    entry.setdefault("created_at_utc", now)
    payload["snapshots"] = _deduplicate_snapshots(snapshots)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _read_catalog(catalog_path: Path) -> dict[str, object]:
    default_payload: dict[str, object] = {"version": 1, "snapshots": []}
    if not catalog_path.exists():
        return default_payload

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_payload

    if not isinstance(payload, dict):
        return default_payload

    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        payload["snapshots"] = []
    payload.setdefault("version", 1)
    return payload


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _snapshot_key(path: object) -> str | None:
    if path is None:
        return None
    try:
        return str(_normalize_path(path))
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(path)


def _deduplicate_snapshots(snapshots: Sequence[object]) -> list[dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    for idx, item in enumerate(snapshots):
        if not isinstance(item, dict):
            continue

        key = _snapshot_key(item.get("path")) or f"__snapshot_{idx}"
        normalized_item = dict(item)
        normalized_item["path"] = key if not key.startswith("__snapshot_") else normalized_item.get("path")
        if key in deduped:
            deduped[key].update({field: value for field, value in normalized_item.items() if value is not None})
        else:
            deduped[key] = normalized_item

    return sorted(
        deduped.values(),
        key=lambda item: str(item.get("snapshot_name", "")),
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
