from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..core.types import ConstructionResult


class BaseConstructor(ABC):
    method_id = "base_constructor"
    display_name = "Base Constructor"

    @property
    def nombre(self) -> str:
        return self.display_name

    @abstractmethod
    def optimizar(self, returns: pd.DataFrame, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        """
        Receive universe returns and return weights plus metadata.
        """

    def build(self, universe, name: str, **kwargs) -> ConstructionResult:
        returns = kwargs.pop("returns", None)
        if returns is None:
            returns = getattr(universe, "asset_returns", None)

        if returns is None:
            raise RuntimeError("El PortfolioUniverse debe tener retornos preparados antes de construir.")

        weights, meta = self.optimizar(returns, **kwargs)
        weights = weights.reindex(returns.columns).fillna(0.0).sort_index()
        metrics = universe.make_basic_metrics(
            weights,
            returns=returns,
            ann_factor=kwargs.get("ann_factor"),
            rf_per_period=kwargs.get("rf_per_period", getattr(self, "rf_per_period", 0.0)),
        )
        if meta:
            metrics.update({f"meta_{key}": value for key, value in meta.items()})

        construction_start = kwargs.get(
            "construction_start",
            getattr(universe, "construction_start", getattr(universe, "start", None)),
        )
        construction_end = kwargs.get(
            "construction_end",
            getattr(universe, "construction_end", getattr(universe, "end", None)),
        )

        return ConstructionResult(
            name=name,
            method_id=self.method_id,
            display_name=self.display_name,
            weights=weights,
            selected_assets=[asset for asset in weights.index if weights.loc[asset] != 0],
            params=dict(kwargs),
            metrics_insample=metrics,
            construction_start=pd.Timestamp(construction_start) if construction_start is not None else None,
            construction_end=pd.Timestamp(construction_end) if construction_end is not None else None,
            backtest_result=None,
            mc_result=None,
            notes=kwargs.get("notes"),
        )
