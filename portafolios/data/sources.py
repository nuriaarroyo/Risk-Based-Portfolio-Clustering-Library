from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .base import BaseDataLoader
from .local_loader import local_loader
from .yfinance_loader import yfinance_loader


class CSVLoader(BaseDataLoader):
    def __init__(
        self,
        *,
        path: str | Path,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        prefer_adj_close: bool = True,
        freq: Optional[str] = None,
        max_missing_ratio: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.tickers = list(tickers) if tickers is not None else None
        self.start = start
        self.end = end
        self.prefer_adj_close = prefer_adj_close
        self.freq = freq
        self.max_missing_ratio = max_missing_ratio

    def load_prices(self) -> pd.DataFrame:
        return local_loader(
            path=self.path,
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            prefer_adj_close=self.prefer_adj_close,
            freq=self.freq,
            max_missing_ratio=self.max_missing_ratio,
        )

    def _build_metadata(self, prices: pd.DataFrame) -> dict[str, object]:
        return {
            "source": "csv",
            "origin": "local_csv",
            "path": str(self.path),
            "frequency": self.freq,
            "start": str(prices.index.min()) if not prices.empty else self.start,
            "end": str(prices.index.max()) if not prices.empty else self.end,
            "n_assets": len(prices.columns),
            "n_observations": len(prices),
        }


class YFinanceLoader(BaseDataLoader):
    def __init__(
        self,
        *,
        tickers: Iterable[str],
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
    ) -> None:
        self.tickers = list(tickers)
        self.start = start
        self.end = end
        self.prefer_adj_close = prefer_adj_close
        self.freq = freq
        self.max_missing_ratio = max_missing_ratio
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.repair = repair
        self.threads = threads
        self.ignore_tz = ignore_tz
        self.batch_size = batch_size
        self.progress = progress
        self.cache_dir = cache_dir
        self.save_path = save_path
        self.use_saved_data = use_saved_data
        self.save_download = save_download
        self.catalog_path = catalog_path
        self.fallback_to_saved_data = fallback_to_saved_data
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.timeout = timeout
        self._last_runtime_metadata: dict[str, object] = {}

    def load_prices(self) -> pd.DataFrame:
        runtime_metadata: dict[str, object] = {}
        prices = yfinance_loader(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            prefer_adj_close=self.prefer_adj_close,
            freq=self.freq,
            max_missing_ratio=self.max_missing_ratio,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            repair=self.repair,
            threads=self.threads,
            ignore_tz=self.ignore_tz,
            batch_size=self.batch_size,
            progress=self.progress,
            cache_dir=self.cache_dir,
            save_path=self.save_path,
            use_saved_data=self.use_saved_data,
            save_download=self.save_download,
            catalog_path=self.catalog_path,
            fallback_to_saved_data=self.fallback_to_saved_data,
            max_retries=self.max_retries,
            retry_wait=self.retry_wait,
            timeout=self.timeout,
            metadata_sink=runtime_metadata,
        )
        self._last_runtime_metadata = runtime_metadata
        return prices

    def _build_metadata(self, prices: pd.DataFrame) -> dict[str, object]:
        runtime_metadata = dict(self._last_runtime_metadata)
        return {
            "source": "yfinance",
            "origin": runtime_metadata.get("resolved_origin", "yfinance"),
            "requested_origin": runtime_metadata.get("requested_origin"),
            "frequency": self.freq or self.interval,
            "start": str(prices.index.min()) if not prices.empty else self.start,
            "end": str(prices.index.max()) if not prices.empty else self.end,
            "n_assets": len(prices.columns),
            "n_observations": len(prices),
            "save_path": runtime_metadata.get("save_path")
            or (str(self.save_path) if self.save_path is not None else None),
            "catalog_path": runtime_metadata.get("catalog_path")
            or (str(self.catalog_path) if self.catalog_path is not None else None),
            "used_saved_data": runtime_metadata.get("used_saved_data", self.use_saved_data),
            "download_attempted": runtime_metadata.get("download_attempted"),
            "download_succeeded": runtime_metadata.get("download_succeeded"),
        }
