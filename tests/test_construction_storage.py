from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portafolios.core.portafolio import PortfolioUniverse
from portafolios.core.types import ConstructionResult, HRPDiagnostics
from portafolios.data.base import StandardizedData


def fresh_test_root(name: str) -> Path:
    root = PROJECT_ROOT / "outputs" / "test_artifacts" / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_universe(root: Path) -> PortfolioUniverse:
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.5, 101.0, 103.0, 104.5],
            "MSFT": [100.0, 100.8, 101.5, 102.0, 103.0, 104.0],
            "JPM": [100.0, 99.5, 100.2, 101.1, 100.8, 101.9],
        },
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        ),
    )
    returns = prices.pct_change().dropna()
    market_data = StandardizedData(
        prices=prices,
        returns=returns,
        tickers=list(prices.columns),
        metadata={"source": "test"},
    )
    return PortfolioUniverse(
        loader=market_data,
        tickers=list(prices.columns),
        start="2024-01-01",
        end="2024-01-08",
        construction_start="2024-01-02",
        construction_end="2024-01-08",
        universe_name="construction_storage_test",
        base_output_dir=root,
        auto_save_data=False,
    ).prepare_data()


def test_save_construction_writes_hrp_and_markowitz_diagnostics() -> None:
    root = fresh_test_root("construction_storage")
    try:
        universe = make_universe(root)

        hrp = ConstructionResult(
            name="hrp_recursive",
            method_id="hrp_recursive",
            display_name="HRP Recursive",
            weights=pd.Series({"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}),
            selected_assets=["AAPL", "MSFT", "JPM"],
            params={"allow_short": False},
            metrics_insample={},
            construction_start=pd.Timestamp("2024-01-02"),
            construction_end=pd.Timestamp("2024-01-08"),
            diagnostics={
                "hrp": HRPDiagnostics(
                    distance_matrix=pd.DataFrame(
                        [[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]],
                        index=["AAPL", "MSFT", "JPM"],
                        columns=["AAPL", "MSFT", "JPM"],
                    ),
                    clusters=[["AAPL", "MSFT"], ["JPM"]],
                    cluster_returns=pd.DataFrame(
                        {
                            "C0": [0.01, 0.02],
                            "C1": [0.005, 0.006],
                        },
                        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
                    ),
                    cluster_weights=pd.Series({"C0": 0.7, "C1": 0.3}),
                    local_weights={
                        "C0": pd.Series({"AAPL": 0.6, "MSFT": 0.4}),
                        "C1": pd.Series({"JPM": 1.0}),
                    },
                    final_weights=pd.Series({"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}),
                    linkage_matrix=np.array([[0.0, 1.0, 0.2, 2.0], [2.0, 3.0, 0.4, 3.0]]),
                    cluster_labels=pd.Series([1, 1, 2], index=["AAPL", "MSFT", "JPM"]),
                    cluster_assets={"C0": ["AAPL", "MSFT"], "C1": ["JPM"]},
                    inner_metadata_by_cluster={"C0": {"n_assets": 2}, "C1": {"n_assets": 1}},
                    outer_metadata={"n_assets": 2},
                    distance_name="deprado",
                    clustering_name="hierarchical",
                )
            },
        )
        markowitz = ConstructionResult(
            name="markowitz",
            method_id="markowitz_max_sharpe",
            display_name="Markowitz Max Sharpe",
            weights=pd.Series({"AAPL": 0.4, "MSFT": 0.4, "JPM": 0.2}),
            selected_assets=["AAPL", "MSFT", "JPM"],
            params={"allow_short": False, "ret_kind": "simple"},
            metrics_insample={
                "meta_success": True,
                "meta_message": "Optimization terminated successfully",
                "meta_objective": 1.23,
                "meta_allow_short": False,
            },
            construction_start=pd.Timestamp("2024-01-02"),
            construction_end=pd.Timestamp("2024-01-08"),
        )

        universe.add_construction(hrp, set_active=False)
        universe.add_construction(markowitz, set_active=False)

        hrp_dir = universe.save_construction("hrp_recursive")
        markowitz_dir = universe.save_construction("markowitz")

        assert (hrp_dir / "diagnostics" / "hrp" / "summary.json").exists()
        assert (hrp_dir / "diagnostics" / "hrp" / "distance_matrix.csv").exists()
        assert (hrp_dir / "diagnostics" / "hrp" / "local_weights.csv").exists()
        assert (hrp_dir / "diagnostics" / "hrp" / "linkage_matrix.csv").exists()
        assert (hrp_dir / "diagnostics" / "hrp" / "cluster_assets.json").exists()

        assert (markowitz_dir / "diagnostics" / "markowitz" / "optimizer_summary.json").exists()
        assert (markowitz_dir / "diagnostics" / "markowitz" / "expected_returns.csv").exists()
        assert (markowitz_dir / "diagnostics" / "markowitz" / "covariance.csv").exists()
        assert (markowitz_dir / "diagnostics" / "markowitz" / "asset_moments.csv").exists()
        assert (markowitz_dir / "diagnostics" / "markowitz" / "portfolio_point.json").exists()
        assert (markowitz_dir / "diagnostics" / "markowitz" / "efficient_frontier_points.csv").exists()

        optimizer_summary = json.loads(
            (markowitz_dir / "diagnostics" / "markowitz" / "optimizer_summary.json").read_text(encoding="utf-8")
        )
        assert optimizer_summary["success"] is True
        assert optimizer_summary["allow_short"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)
