from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class StandardizedData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: list[str]
    metadata: dict[str, Any]


class BaseDataLoader(ABC):
    @abstractmethod
    def load_prices(self) -> pd.DataFrame:
        """
        Load and return normalized prices.
        """

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        clean_prices = prices.sort_index().ffill()
        returns = clean_prices.pct_change(fill_method=None)
        return returns.dropna(how="all")

    def get_data(self) -> StandardizedData:
        prices = self.load_prices()
        returns = self.compute_returns(prices)
        return StandardizedData(
            prices=prices,
            returns=returns,
            tickers=list(prices.columns),
            metadata=self._build_metadata(prices),
        )

    @abstractmethod
    def _build_metadata(self, prices: pd.DataFrame) -> dict[str, Any]:
        """
        Build metadata for the loaded dataset.
        """
