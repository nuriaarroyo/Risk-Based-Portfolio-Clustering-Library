from __future__ import annotations

import pandas as pd

from .base import BaseConstructor
from ..core.types import ConstructionResult


class EqualWeightConstructor(BaseConstructor):
    nombre = "equal_weight"

    def build(self, universe, name: str, **kwargs) -> ConstructionResult:
        columns = list(universe.returns.columns)
        if not columns:
            raise ValueError("El universo no tiene activos disponibles para construir pesos.")

        weights = pd.Series(1.0 / len(columns), index=columns, dtype=float)
        params = dict(kwargs)
        metrics = universe.make_basic_metrics(weights, ann_factor=params.get("ann_factor"))

        return ConstructionResult(
            name=name,
            method=self.nombre,
            weights=weights,
            selected_assets=columns,
            params=params,
            metrics_insample=metrics,
            backtest_result=None,
            mc_result=None,
            notes=params.get("notes"),
        )
