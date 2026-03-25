from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..data.base import StandardizedData


MarketData = StandardizedData


@dataclass(slots=True)
class ConstructionResult:
    name: str
    method: str
    weights: pd.Series
    selected_assets: list[str]
    params: dict[str, Any]
    metrics_insample: dict[str, Any]
    backtest_result: Optional[Any] = None
    mc_result: Optional[Any] = None
    notes: Optional[str] = None
