from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portafolios.core.types import ConstructionResult
from portafolios.eval import Backtester, MonteCarloEngine


class DummyUniverse:
    def __init__(self, returns: pd.DataFrame, construction: ConstructionResult) -> None:
        self.returns = returns
        self.constructions = {construction.name: construction}

    def get_construction(self, name: str) -> ConstructionResult:
        return self.constructions[name]

    def list_constructions(self) -> list[str]:
        return list(self.constructions)

    def get_returns_window(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        *,
        dropna: bool = True,
    ) -> pd.DataFrame:
        window = self.returns.copy()
        if start is not None:
            window = window.loc[window.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= pd.Timestamp(end)]
        if dropna:
            window = window.dropna(axis=0, how="any")
        if window.empty:
            raise ValueError("No returns are available in the requested window.")
        return window


def make_universe() -> DummyUniverse:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01, 0.03],
            "MSFT": [0.00, 0.01, 0.02, -0.02],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    construction = ConstructionResult(
        name="equal_weight",
        method_id="equal_weight",
        display_name="Equal Weight",
        weights=pd.Series({"AAPL": 0.5, "MSFT": 0.5}),
        selected_assets=["AAPL", "MSFT"],
        params={},
        metrics_insample={},
        construction_start=pd.Timestamp("2024-01-02"),
        construction_end=pd.Timestamp("2024-01-03"),
    )
    return DummyUniverse(returns=returns, construction=construction)


def test_backtester_defaults_to_first_date_after_construction_end() -> None:
    universe = make_universe()

    result = Backtester(universe, "equal_weight").run()

    assert result.start_date == pd.Timestamp("2024-01-04")
    assert result.end_date == pd.Timestamp("2024-01-05")
    assert list(result.portfolio_returns.index) == list(pd.to_datetime(["2024-01-04", "2024-01-05"]))


def test_monte_carlo_summary_metrics_use_initial_value_baseline() -> None:
    terminal_values = pd.Series([1.5, 2.0, 2.5]).to_numpy()

    metrics = MonteCarloEngine._summary_metrics(terminal_values, initial_value=2.0)

    assert math.isclose(metrics["prob_loss"], 1 / 3)
    assert math.isclose(metrics["mean_terminal_return"], 0.0)


def test_monte_carlo_rejects_non_positive_initial_value() -> None:
    universe = make_universe()
    engine = MonteCarloEngine(universe, "equal_weight", seed=42)

    with pytest.raises(ValueError, match="`initial_value` must be positive."):
        engine.run(horizon=5, n_simulations=10, initial_value=0.0)
