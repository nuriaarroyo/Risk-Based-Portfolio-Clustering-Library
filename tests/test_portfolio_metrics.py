from __future__ import annotations

import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portafolios.core.portafolio import PortfolioUniverse
from portafolios.core.types import BacktestResult, ConstructionResult
from portafolios.data.base import StandardizedData
from portafolios.metrics import portfolio as pm


def fresh_test_root(name: str) -> Path:
    root = PROJECT_ROOT / "outputs" / "test_artifacts" / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_universe_with_backtest(root: Path) -> PortfolioUniverse:
    index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 118.8, 106.92, 112.266],
            "MSFT": [100.0, 100.0, 100.0, 101.0, 102.01],
        },
        index=index,
    )
    returns = prices.pct_change().dropna()
    market_data = StandardizedData(
        prices=prices,
        returns=returns,
        tickers=list(prices.columns),
        metadata={"source": "test"},
    )
    universe = PortfolioUniverse(
        loader=market_data,
        tickers=list(prices.columns),
        start=str(index.min().date()),
        end=str(index.max().date()),
        construction_start="2024-01-02",
        construction_end="2024-01-03",
        universe_name="portfolio_metrics_test",
        base_output_dir=root,
        auto_save_data=False,
    ).prepare_data()

    construction = ConstructionResult(
        name="aapl_only",
        method_id="manual",
        display_name="AAPL Only",
        weights=pd.Series({"AAPL": 1.0, "MSFT": 0.0}),
        selected_assets=["AAPL"],
        params={},
        metrics_insample={},
        construction_start=pd.Timestamp("2024-01-02"),
        construction_end=pd.Timestamp("2024-01-03"),
    )
    universe.add_construction(construction, set_active=True)

    backtest_returns = pd.Series(
        [-0.10, 0.05],
        index=pd.to_datetime(["2024-01-04", "2024-01-05"]),
        name="aapl_only",
    )
    backtest = BacktestResult(
        construction_name="aapl_only",
        start_date=pd.Timestamp("2024-01-04"),
        end_date=pd.Timestamp("2024-01-05"),
        portfolio_returns=backtest_returns,
        cumulative_returns=pm.cumulative_return_series(backtest_returns),
        summary_metrics={},
        drawdown_series=pm.drawdown_series(backtest_returns),
    )
    stored = universe.get_construction("aapl_only")
    universe.constructions["aapl_only"] = replace(stored, backtest_result=backtest)
    return universe


def test_risk_contributions_return_true_fractions_when_requested() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})

    component = pm.risk_contributions_from_cov(covariance, weights, as_fraction=False)
    fraction = pm.risk_contributions_from_cov(covariance, weights, as_fraction=True)

    portfolio_vol = math.sqrt(0.5 * 0.5 * 0.04 + 0.5 * 0.5 * 0.09)
    assert math.isclose(float(component.sum()), portfolio_vol)
    assert math.isclose(float(fraction.sum()), 1.0)
    assert all(math.isclose(float(f), float(c / portfolio_vol)) for f, c in zip(fraction, component))


def test_gaussian_cvar_exceeds_var_for_the_same_series() -> None:
    portfolio_returns = pd.Series([-0.03, 0.01, 0.00, 0.02, -0.01, 0.015, -0.02])

    var = pm.var_gaussian_from_series(portfolio_returns, alpha=0.95)
    cvar = pm.cvar_gaussian_from_series(portfolio_returns, alpha=0.95)

    assert cvar > var > 0.0


def test_portfolio_universe_path_metrics_prefer_attached_backtest_results() -> None:
    root = fresh_test_root("portfolio_metric_window")
    try:
        universe = make_universe_with_backtest(root)
        backtest_returns = universe.get_construction("aapl_only").backtest_result.portfolio_returns

        expected_sharpe = pm.sharpe_from_series(backtest_returns, ann_factor=None)
        expected_var = pm.var_gaussian_from_series(backtest_returns, alpha=0.95, ann_factor=None)

        assert math.isclose(
            universe.kpi("sharpe", construction_name="aapl_only", ann_factor=None),
            expected_sharpe,
        )
        assert math.isclose(
            universe.kpi("var", construction_name="aapl_only", ann_factor=None, alpha=0.95),
            expected_var,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_calmar_is_no_longer_exposed_as_a_supported_kpi() -> None:
    root = fresh_test_root("portfolio_metric_calmar")
    try:
        universe = make_universe_with_backtest(root)

        with pytest.raises(ValueError, match="Unrecognized KPI: calmar"):
            universe.kpi("calmar", construction_name="aapl_only")
    finally:
        shutil.rmtree(root, ignore_errors=True)
