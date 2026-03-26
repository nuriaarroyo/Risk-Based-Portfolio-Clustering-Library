from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import BacktestResult


class Backtester:
    """
    Evalua pesos fijos sobre un periodo posterior al de construccion.
    """

    def __init__(self, universe, construction_name: str, *, ann_factor: int = 252) -> None:
        self.universe = universe
        self.construction_name = construction_name
        self.ann_factor = ann_factor
        self.construction = universe.get_construction(construction_name)

    def run(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        notes: Optional[str] = None,
    ) -> BacktestResult:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if end < start:
            raise ValueError("`end_date` debe ser mayor o igual que `start_date`.")

        self._validate_period(start)
        returns_window = self._slice_returns(start, end)
        weights = self.construction.weights.reindex(self.universe.returns.columns).fillna(0.0)

        portfolio_returns = returns_window.mul(weights, axis=1).sum(axis=1)
        portfolio_returns.name = self.construction_name
        cumulative_returns = (1.0 + portfolio_returns).cumprod() - 1.0
        cumulative_returns.name = self.construction_name

        result = BacktestResult(
            construction_name=self.construction_name,
            start_date=start,
            end_date=end,
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            summary_metrics=self._summary_metrics(portfolio_returns, cumulative_returns),
            notes=notes,
        )
        return result

    def attach(self, result: BacktestResult) -> None:
        stored = self.universe.get_construction(self.construction_name)
        self.universe.constructions[self.construction_name] = replace(stored, backtest_result=result)

    def run_and_attach(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        notes: Optional[str] = None,
    ) -> BacktestResult:
        result = self.run(start_date=start_date, end_date=end_date, notes=notes)
        self.attach(result)
        return result

    @classmethod
    def run_all(
        cls,
        universe,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        ann_factor: int = 252,
        attach: bool = True,
        notes: Optional[str] = None,
    ) -> dict[str, BacktestResult]:
        results: dict[str, BacktestResult] = {}
        for name in universe.list_constructions():
            bt = cls(universe, name, ann_factor=ann_factor)
            result = bt.run(start_date=start_date, end_date=end_date, notes=notes)
            if attach:
                bt.attach(result)
            results[name] = result
        return results

    def _validate_period(self, start: pd.Timestamp) -> None:
        construction_end = self.construction.construction_end
        if construction_end is not None and start <= pd.Timestamp(construction_end):
            raise ValueError(
                "El backtest debe iniciar despues del periodo de construccion. "
                f"construction_end={pd.Timestamp(construction_end).date()}, start_date={start.date()}"
            )

    def _slice_returns(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        returns = self.universe.returns
        window = returns.loc[(returns.index >= start) & (returns.index <= end)].copy()
        window = window.dropna(axis=0, how="any")
        if window.empty:
            raise ValueError("No hay retornos disponibles en el periodo de backtest solicitado.")
        return window

    def _summary_metrics(self, portfolio_returns: pd.Series, cumulative_returns: pd.Series) -> dict[str, float]:
        n = len(portfolio_returns)
        total_return = float((1.0 + portfolio_returns).prod() - 1.0)
        annualized_return = float((1.0 + total_return) ** (self.ann_factor / n) - 1.0) if n > 0 else np.nan
        annualized_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(self.ann_factor)) if n > 1 else 0.0
        sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
        wealth = 1.0 + cumulative_returns
        drawdown = wealth / wealth.cummax() - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        return {
            "n_periods": n,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": float(sharpe) if sharpe == sharpe else np.nan,
            "max_drawdown": max_drawdown,
        }
