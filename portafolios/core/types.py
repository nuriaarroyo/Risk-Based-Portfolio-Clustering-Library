from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data.base import StandardizedData


MarketData = StandardizedData


@dataclass(slots=True)
class ConstructionResult:
    name: str
    method_id: str
    display_name: str
    weights: pd.Series
    selected_assets: list[str]
    params: dict[str, Any]
    metrics_insample: dict[str, Any]
    construction_start: Optional[pd.Timestamp] = None
    construction_end: Optional[pd.Timestamp] = None
    backtest_result: Optional[Any] = None
    mc_result: Optional[Any] = None
    notes: Optional[str] = None

    @property
    def method(self) -> str:
        return self.method_id


@dataclass(slots=True)
class BacktestResult:
    construction_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    summary_metrics: dict[str, Any]
    notes: Optional[str] = None


@dataclass(slots=True)
class MonteCarloResult:
    construction_name: str
    horizon: int
    n_simulations: int
    simulated_paths: pd.DataFrame | np.ndarray
    terminal_values: np.ndarray
    summary_metrics: dict[str, Any]
    notes: Optional[str] = None
