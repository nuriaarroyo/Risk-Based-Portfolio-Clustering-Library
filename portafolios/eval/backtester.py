from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import BacktestResult
from ..metrics import portfolio as pm


class Backtester:
    """
    Evaluate fixed weights over a period that starts after construction ends.
    """

    def __init__(self, universe, construction_name: str, *, ann_factor: int = 252) -> None:
        self.universe = universe
        self.construction_name = construction_name
        self.ann_factor = ann_factor
        self.construction = universe.get_construction(construction_name)

    def run(
        self,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        notes: Optional[str] = None,
    ) -> BacktestResult:
        start, end = self._resolve_period(start_date=start_date, end_date=end_date)
        if end < start:
            raise ValueError("`end_date` debe ser mayor o igual que `start_date`.")

        self._validate_period(start)
        returns_window = self._slice_returns(start, end)
        weights = self.construction.weights.reindex(self.universe.returns.columns).fillna(0.0)

        portfolio_returns = pm.portfolio_return_series(returns_window, weights)
        portfolio_returns.name = self.construction_name
        cumulative_returns = pm.cumulative_return_series(portfolio_returns)
        cumulative_returns.name = self.construction_name
        drawdown_series = pm.drawdown_series(portfolio_returns)
        drawdown_series.name = self.construction_name

        result = BacktestResult(
            construction_name=self.construction_name,
            start_date=start,
            end_date=end,
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            summary_metrics=self._summary_metrics(portfolio_returns, drawdown_series),
            drawdown_series=drawdown_series,
            notes=notes,
        )
        return result

    def attach(self, result: BacktestResult) -> None:
        stored = self.universe.get_construction(self.construction_name)
        self.universe.constructions[self.construction_name] = replace(stored, backtest_result=result)

    def run_and_attach(
        self,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
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
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
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

    def summarize_window(
        self,
        result: BacktestResult | None = None,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> dict[str, float | str]:
        active_result = result or self.construction.backtest_result
        if active_result is None:
            raise ValueError("No hay BacktestResult disponible para resumir.")

        portfolio_returns = active_result.portfolio_returns
        if start_date is not None:
            portfolio_returns = portfolio_returns.loc[portfolio_returns.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            portfolio_returns = portfolio_returns.loc[portfolio_returns.index <= pd.Timestamp(end_date)]
        if portfolio_returns.empty:
            raise ValueError("No hay retornos del backtest en la ventana solicitada.")

        cumulative_returns = pm.cumulative_return_series(portfolio_returns)
        drawdown_series = pm.drawdown_series(portfolio_returns)
        summary = self._summary_metrics(portfolio_returns, drawdown_series)
        summary.update(
            {
                "window_start": str(portfolio_returns.index.min()),
                "window_end": str(portfolio_returns.index.max()),
            }
        )
        return summary

    def _resolve_period(
        self,
        *,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        available_returns = self.universe.returns
        if available_returns is None or available_returns.empty:
            raise ValueError("El universe no tiene retornos disponibles para backtesting.")

        construction_end = self.construction.construction_end
        if start_date is None:
            if construction_end is not None:
                start = self._next_available_date_after(pd.Timestamp(construction_end))
            else:
                start = pd.Timestamp(available_returns.index.min())
        else:
            start = pd.Timestamp(start_date)

        end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(available_returns.index.max())
        return start, end

    def _next_available_date_after(self, cutoff: pd.Timestamp) -> pd.Timestamp:
        returns_index = self.universe.returns.index
        candidates = returns_index[returns_index > cutoff]
        if len(candidates) == 0:
            raise ValueError(
                "No hay datos fuera de muestra despues del fin de construccion. "
                "Amplia el rango del universe para poder backtestear."
            )
        return pd.Timestamp(candidates.min())

    def _validate_period(self, start: pd.Timestamp) -> None:
        construction_end = self.construction.construction_end
        if construction_end is not None and start <= pd.Timestamp(construction_end):
            raise ValueError(
                "El backtest debe iniciar despues del periodo de construccion. "
                f"construction_end={pd.Timestamp(construction_end).date()}, start_date={start.date()}"
            )

    def _slice_returns(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self.universe.get_returns_window(start, end)

    def _summary_metrics(
        self,
        portfolio_returns: pd.Series,
        drawdown_series: pd.Series | None = None,
    ) -> dict[str, float]:
        n = len(portfolio_returns.dropna())
        total_return = pm.realized_total_return_from_series(portfolio_returns)
        annualized_return = (
            pm.realized_annualized_return_from_series(portfolio_returns, ann_factor=self.ann_factor)
            if n > 0
            else np.nan
        )
        annualized_volatility = pm.realized_annualized_volatility_from_series(
            portfolio_returns,
            ann_factor=self.ann_factor,
        )
        sharpe = pm.sharpe_from_series(portfolio_returns, ann_factor=self.ann_factor)
        drawdown = drawdown_series if drawdown_series is not None else pm.drawdown_series(portfolio_returns)
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        return {
            "n_periods": n,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": float(sharpe) if sharpe == sharpe else np.nan,
            "max_drawdown": max_drawdown,
        }
