from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import MonteCarloResult


class MonteCarloEngine:
    """
    Simulate the future distribution of a fixed-weight portfolio.
    """

    def __init__(self, universe, construction_name: str, *, seed: Optional[int] = None) -> None:
        self.universe = universe
        self.construction_name = construction_name
        self.construction = universe.get_construction(construction_name)
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        *,
        horizon: int,
        n_simulations: int,
        initial_value: float = 1.0,
        notes: Optional[str] = None,
    ) -> MonteCarloResult:
        if horizon <= 0:
            raise ValueError("`horizon` must be positive.")
        if n_simulations <= 0:
            raise ValueError("`n_simulations` must be positive.")
        if initial_value <= 0:
            raise ValueError("`initial_value` must be positive.")

        estimation_returns = self._estimation_returns()
        weights = self.construction.weights.reindex(estimation_returns.columns).fillna(0.0).values
        mean_vector = estimation_returns.mean().values
        cov_matrix = estimation_returns.cov().values

        simulated_asset_returns = self.rng.multivariate_normal(
            mean=mean_vector,
            cov=cov_matrix,
            size=(horizon, n_simulations),
            check_valid="raise",
        )
        portfolio_returns = np.tensordot(simulated_asset_returns, weights, axes=([2], [0]))
        wealth_paths = initial_value * np.cumprod(1.0 + portfolio_returns, axis=0)
        wealth_paths = np.vstack([np.full(n_simulations, initial_value), wealth_paths])

        path_index = pd.RangeIndex(start=0, stop=horizon + 1, step=1, name="step")
        path_columns = [f"sim_{i}" for i in range(n_simulations)]
        simulated_paths = pd.DataFrame(wealth_paths, index=path_index, columns=path_columns)
        terminal_values = wealth_paths[-1, :]

        result = MonteCarloResult(
            construction_name=self.construction_name,
            horizon=horizon,
            n_simulations=n_simulations,
            simulated_paths=simulated_paths,
            terminal_values=terminal_values,
            summary_metrics=self._summary_metrics(terminal_values, initial_value=initial_value),
            estimation_start=pd.Timestamp(estimation_returns.index.min()),
            estimation_end=pd.Timestamp(estimation_returns.index.max()),
            notes=notes,
        )
        return result

    def attach(self, result: MonteCarloResult) -> None:
        stored = self.universe.get_construction(self.construction_name)
        self.universe.constructions[self.construction_name] = replace(stored, mc_result=result)

    def run_and_attach(
        self,
        *,
        horizon: int,
        n_simulations: int,
        initial_value: float = 1.0,
        notes: Optional[str] = None,
    ) -> MonteCarloResult:
        result = self.run(
            horizon=horizon,
            n_simulations=n_simulations,
            initial_value=initial_value,
            notes=notes,
        )
        self.attach(result)
        return result

    @classmethod
    def run_all(
        cls,
        universe,
        *,
        horizon: int,
        n_simulations: int,
        initial_value: float = 1.0,
        attach: bool = True,
        seed: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> dict[str, MonteCarloResult]:
        results: dict[str, MonteCarloResult] = {}
        for idx, name in enumerate(universe.list_constructions()):
            engine = cls(universe, name, seed=None if seed is None else seed + idx)
            result = engine.run(
                horizon=horizon,
                n_simulations=n_simulations,
                initial_value=initial_value,
                notes=notes,
            )
            if attach:
                engine.attach(result)
            results[name] = result
        return results

    @staticmethod
    def _summary_metrics(
        terminal_values: np.ndarray,
        *,
        initial_value: float,
    ) -> dict[str, float]:
        terminal_returns = (terminal_values / initial_value) - 1.0
        return {
            "mean_terminal_value": float(np.mean(terminal_values)),
            "median_terminal_value": float(np.median(terminal_values)),
            "std_terminal_value": float(np.std(terminal_values, ddof=1)) if len(terminal_values) > 1 else 0.0,
            "min_terminal_value": float(np.min(terminal_values)),
            "max_terminal_value": float(np.max(terminal_values)),
            "prob_loss": float(np.mean(terminal_values < initial_value)),
            "mean_terminal_return": float(np.mean(terminal_returns)),
        }

    def _estimation_returns(self) -> pd.DataFrame:
        start = self.construction.construction_start
        end = self.construction.construction_end
        estimation_returns = self.universe.get_returns_window(start, end)
        if len(estimation_returns) < 2:
            raise ValueError("Monte Carlo requires at least two return observations in the estimation window.")
        if not np.isfinite(estimation_returns.to_numpy(dtype=float)).all():
            raise ValueError("Monte Carlo estimation returns contain non-finite values.")
        return estimation_returns
