from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseConstructor


class EqualWeightConstructor(BaseConstructor):
    method_id = "equal_weight"
    display_name = "Equal Weight"

    def optimizar(self, returns: pd.DataFrame, **kwargs: Any) -> tuple[pd.Series, dict[str, Any]]:
        if returns is None or returns.empty:
            raise ValueError("The returns DataFrame is empty or None.")

        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
        return weights, {"n_assets": len(returns.columns)}
