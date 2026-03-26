from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..core.types import ConstructionResult


class BaseConstructor(ABC):
    nombre = "base_constructor"

    @abstractmethod
    def optimizar(self, returns: pd.DataFrame, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        """
        Recibe retornos del universo y devuelve pesos + metadatos.
        """

    def build(self, universe, name: str, **kwargs) -> ConstructionResult:
        if getattr(universe, "asset_returns", None) is None:
            raise RuntimeError("El universe debe tener retornos preparados antes de construir.")

        weights, meta = self.optimizar(universe.asset_returns, **kwargs)
        weights = weights.reindex(universe.asset_returns.columns).fillna(0.0).sort_index()
        metrics = universe.make_basic_metrics(weights, ann_factor=kwargs.get("ann_factor"))
        if meta:
            metrics.update({f"meta_{key}": value for key, value in meta.items()})

        construction_start = kwargs.get("construction_start", getattr(universe, "start", None))
        construction_end = kwargs.get("construction_end", getattr(universe, "end", None))

        return ConstructionResult(
            name=name,
            method=self.nombre,
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
