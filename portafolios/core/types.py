from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data.base import StandardizedData


MarketData = StandardizedData


@dataclass(slots=True)
class HRPDiagnostics:
    distance_matrix: pd.DataFrame
    clusters: list[list[str]]
    cluster_returns: pd.DataFrame
    cluster_weights: pd.Series
    local_weights: dict[str, pd.Series]
    final_weights: pd.Series
    linkage_matrix: np.ndarray | None = None
    cluster_labels: Optional[pd.Series] = None
    cluster_assets: dict[str, list[str]] = field(default_factory=dict)
    inner_metadata_by_cluster: dict[str, dict[str, Any]] = field(default_factory=dict)
    outer_metadata: dict[str, Any] = field(default_factory=dict)
    distance_name: Optional[str] = None
    clustering_name: Optional[str] = None


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
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def method(self) -> str:
        return self.method_id

    @property
    def hrp_diagnostics(self) -> Optional[HRPDiagnostics]:
        diagnostics = self.diagnostics.get("hrp")
        if isinstance(diagnostics, HRPDiagnostics):
            return diagnostics
        return None


@dataclass(slots=True)
class BacktestResult:
    construction_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    summary_metrics: dict[str, Any]
    drawdown_series: Optional[pd.Series] = None
    notes: Optional[str] = None


@dataclass(slots=True)
class MonteCarloResult:
    construction_name: str
    horizon: int
    n_simulations: int
    simulated_paths: pd.DataFrame | np.ndarray
    terminal_values: np.ndarray
    summary_metrics: dict[str, Any]
    estimation_start: Optional[pd.Timestamp] = None
    estimation_end: Optional[pd.Timestamp] = None
    notes: Optional[str] = None
