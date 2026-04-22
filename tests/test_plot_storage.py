from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portafolios.core.types import BacktestResult, ConstructionResult, HRPDiagnostics, MonteCarloResult
from portafolios.plots import PortfolioVisualizer
from scripts.run_final_experimental_setup import save_mc_terminal_comparison_plot


class DummyUniversePaths:
    def __init__(self, root: Path) -> None:
        self.output_dir = root / "run"
        self.data_dir = self.output_dir / "data"
        self.constructions_dir = self.output_dir / "constructions"
        self.backtests_dir = self.output_dir / "backtests"
        self.monte_carlo_dir = self.output_dir / "monte_carlo"
        self.plots_dir = self.output_dir / "plots"
        self._construction_names = ["equal_weight"]
        for path in (
            self.output_dir,
            self.data_dir,
            self.constructions_dir,
            self.backtests_dir,
            self.monte_carlo_dir,
            self.plots_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def list_constructions(self) -> list[str]:
        return list(self._construction_names)

    def get_construction_dir(self, construction_name: str) -> Path:
        path = self.constructions_dir / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_backtest_dir(self, construction_name: str) -> Path:
        path = self.backtests_dir / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_mc_dir(self, construction_name: str) -> Path:
        path = self.monte_carlo_dir / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_plot_dir(self, category: str | None = None, construction_name: str | None = None) -> Path:
        path = self.plots_dir
        if category is not None:
            path = path / category
        if construction_name is not None:
            path = path / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_market_data(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def save_all_constructions(self) -> dict[str, Path]:
        return {name: self.get_construction_dir(name) for name in self.list_constructions()}

    def save_all_backtests(self) -> dict[str, Path]:
        return {name: self.get_backtest_dir(name) for name in self.list_constructions()}

    def save_all_monte_carlo(self) -> dict[str, Path]:
        return {name: self.get_mc_dir(name) for name in self.list_constructions()}


class DummyUniversePlots(DummyUniversePaths):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.asset_returns = pd.DataFrame(
            {
                "AAPL": [0.01, 0.02, -0.01, 0.03],
                "MSFT": [0.00, 0.01, 0.02, -0.02],
                "JPM": [0.02, -0.01, 0.01, 0.00],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )
        self.asset_log_returns = self.asset_returns.copy()
        self.correlation = self.asset_returns.corr()
        self.covariance = self.asset_returns.cov()
        hrp = ConstructionResult(
            name="hrp_style",
            method_id="hrp_style",
            display_name="HRP Style",
            weights=pd.Series({"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}),
            selected_assets=["AAPL", "MSFT", "JPM"],
            params={"allow_short": False},
            metrics_insample={},
            construction_start=pd.Timestamp("2024-01-02"),
            construction_end=pd.Timestamp("2024-01-05"),
            diagnostics={
                "hrp": HRPDiagnostics(
                    distance_matrix=pd.DataFrame(
                        [
                            [0.0, 0.2, 0.4],
                            [0.2, 0.0, 0.3],
                            [0.4, 0.3, 0.0],
                        ],
                        index=["AAPL", "MSFT", "JPM"],
                        columns=["AAPL", "MSFT", "JPM"],
                    ),
                    clusters=[["AAPL", "MSFT"], ["JPM"]],
                    cluster_returns=pd.DataFrame(),
                    cluster_weights=pd.Series(dtype=float),
                    local_weights={},
                    final_weights=pd.Series(dtype=float),
                    linkage_matrix=np.array(
                        [
                            [0.0, 1.0, 0.2, 2.0],
                            [2.0, 3.0, 0.4, 3.0],
                        ]
                    ),
                    cluster_labels=pd.Series([1, 1, 2], index=["AAPL", "MSFT", "JPM"]),
                )
            },
            backtest_result=BacktestResult(
                construction_name="hrp_style",
                start_date=pd.Timestamp("2024-01-02"),
                end_date=pd.Timestamp("2024-01-05"),
                portfolio_returns=pd.Series(
                    [0.01, -0.005, 0.015, 0.007],
                    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
                ),
                cumulative_returns=pd.Series(
                    [0.01, 0.00495, 0.02002425, 0.02716441975],
                    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
                ),
                summary_metrics={},
                drawdown_series=pd.Series(
                    [0.0, -0.005, 0.0, 0.0],
                    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
                ),
            ),
            mc_result=MonteCarloResult(
                construction_name="hrp_style",
                horizon=4,
                n_simulations=3,
                simulated_paths=pd.DataFrame(
                    {
                        "path_0": [1.0, 1.02, 1.01, 1.03],
                        "path_1": [1.0, 0.99, 1.00, 1.02],
                        "path_2": [1.0, 1.01, 1.03, 1.04],
                    }
                ),
                terminal_values=np.array([1.03, 1.02, 1.04]),
                summary_metrics={},
            ),
        )
        markowitz = ConstructionResult(
            name="markowitz_max_sharpe",
            method_id="markowitz_max_sharpe",
            display_name="Markowitz Max Sharpe",
            weights=pd.Series({"AAPL": 0.55, "MSFT": 0.30, "JPM": 0.15}),
            selected_assets=["AAPL", "MSFT", "JPM"],
            params={"allow_short": False},
            metrics_insample={},
            construction_start=pd.Timestamp("2024-01-02"),
            construction_end=pd.Timestamp("2024-01-05"),
            backtest_result=hrp.backtest_result,
            mc_result=hrp.mc_result,
        )
        self.constructions = {hrp.name: hrp, markowitz.name: markowitz}
        self._construction_names = list(self.constructions)

    def get_construction(self, construction_name: str) -> ConstructionResult:
        return self.constructions[construction_name]

    def get_returns_window(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        *,
        dropna: bool = True,
    ) -> pd.DataFrame:
        window = self.asset_returns.copy()
        if start is not None:
            window = window.loc[window.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= pd.Timestamp(end)]
        if dropna:
            window = window.dropna(axis=0, how="any")
        return window


def fresh_test_root(name: str) -> Path:
    root = PROJECT_ROOT / "outputs" / "test_artifacts" / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_visualizer_saves_html_inside_dedicated_plot_tree() -> None:
    root = fresh_test_root("plot_tree")
    try:
        universe = DummyUniversePaths(root)
        visualizer = PortfolioVisualizer(universe)

        assert visualizer._resolve_save_path(
            kind="construction",
            construction_name="equal_weight",
            filename="weights_bar.html",
        ) == universe.plots_dir / "constructions" / "equal_weight" / "weights_bar.html"
        assert visualizer._resolve_save_path(
            kind="backtest",
            construction_name="equal_weight",
            filename="backtest.html",
        ) == universe.plots_dir / "backtests" / "equal_weight" / "backtest.html"
        assert visualizer._resolve_save_path(
            kind="monte_carlo",
            construction_name="equal_weight",
            filename="mc_paths.html",
        ) == universe.plots_dir / "monte_carlo" / "equal_weight" / "mc_paths.html"
        assert visualizer._resolve_save_path(
            kind="comparison",
            construction_name=None,
            filename="backtest_comparison.html",
        ) == universe.plots_dir / "comparisons" / "backtest_comparison.html"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_visualizer_cleans_legacy_plot_locations() -> None:
    root = fresh_test_root("legacy_cleanup")
    try:
        universe = DummyUniversePaths(root)
        legacy_paths = [
            universe.get_construction_dir("equal_weight") / "weights_bar.html",
            universe.get_construction_dir("equal_weight") / "weights_pie.html",
            universe.get_construction_dir("equal_weight") / "weights_scatter.html",
            universe.get_backtest_dir("equal_weight") / "backtest.html",
            universe.get_mc_dir("equal_weight") / "mc_paths.html",
            universe.get_mc_dir("equal_weight") / "mc_distribution.html",
            universe.get_plot_dir() / "backtest_comparison.html",
        ]
        for path in legacy_paths:
            path.write_text("legacy", encoding="utf-8")

        PortfolioVisualizer(universe).cleanup_legacy_plot_outputs()

        assert all(not path.exists() for path in legacy_paths)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_visualizer_diagnostic_plot_helpers_save_in_clean_locations() -> None:
    root = fresh_test_root("diagnostic_plots")
    try:
        universe = DummyUniversePlots(root)
        visualizer = PortfolioVisualizer(universe)

        bubble = visualizer.plot_weights_bubble("hrp_style", save_html=True)
        frontier = visualizer.plot_efficient_frontier("markowitz_max_sharpe", save_html=True)
        corr = visualizer.plot_correlation_heatmap(save_html=True)
        cov = visualizer.plot_correlation_heatmap(kind="covariance", save_html=True)
        dendrogram = visualizer.plot_hrp_dendrogram("hrp_style", save_html=True)
        dist_matrix = visualizer.plot_hrp_distance_matrix("hrp_style", save_html=True)
        dist_hist = visualizer.plot_hrp_distance_histogram("hrp_style", save_html=True)
        drawdown = visualizer.plot_drawdown("hrp_style", save_html=True)

        assert isinstance(bubble, go.Figure)
        assert isinstance(frontier, go.Figure)
        assert isinstance(corr, go.Figure)
        assert isinstance(cov, go.Figure)
        assert isinstance(dendrogram, go.Figure)
        assert isinstance(dist_matrix, go.Figure)
        assert isinstance(dist_hist, go.Figure)
        assert isinstance(drawdown, go.Figure)

        assert (universe.plots_dir / "constructions" / "hrp_style" / "weights_bubble.html").exists()
        assert (universe.plots_dir / "constructions" / "markowitz_max_sharpe" / "efficient_frontier.html").exists()
        assert (universe.plots_dir / "comparisons" / "correlation_heatmap.html").exists()
        assert (universe.plots_dir / "comparisons" / "covariance_heatmap.html").exists()
        assert (universe.plots_dir / "constructions" / "hrp_style" / "hrp_dendrogram.html").exists()
        assert (universe.plots_dir / "constructions" / "hrp_style" / "hrp_distance_matrix.html").exists()
        assert (universe.plots_dir / "constructions" / "hrp_style" / "hrp_distance_histogram.html").exists()
        assert (universe.plots_dir / "backtests" / "hrp_style" / "drawdown.html").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_visualizer_uses_consistent_accessible_palette_for_key_plots() -> None:
    root = fresh_test_root("palette_consistency")
    try:
        universe = DummyUniversePlots(root)
        visualizer = PortfolioVisualizer(universe)

        pie = visualizer.plot_weights_pie("hrp_style")
        scatter = visualizer.plot_weights_scatter("hrp_style")
        mc_paths = visualizer.plot_mc_paths("hrp_style", max_paths=3)
        mc_distribution = visualizer.plot_mc_distribution("hrp_style")
        bubble = visualizer.plot_weights_bubble("hrp_style")

        assert pie.data[0]["outsidetextfont"]["color"] == visualizer._TEXT_COLOR
        assert pie.data[0]["marker"]["colors"][0] == visualizer._DISCRETE_COLORS[0]

        assert mc_paths.data[0]["line"]["color"] == visualizer._rgba(visualizer._DISCRETE_COLORS[0], 0.18)
        assert mc_paths.data[-1]["line"]["color"] == visualizer._DISCRETE_COLORS[1]

        assert mc_distribution.data[0]["marker"]["color"] == visualizer._DISCRETE_COLORS[0]
        assert mc_distribution.data[1]["line"]["color"] == visualizer._DISCRETE_COLORS[1]
        assert scatter.data[0]["marker"]["size"] == 14
        bubble_sizes = list(bubble.data[0]["marker"]["size"])
        assert bubble_sizes[0] > bubble_sizes[1] > bubble_sizes[2]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_visualizer_adds_axis_padding_for_label_heavy_plots() -> None:
    root = fresh_test_root("axis_padding")
    try:
        universe = DummyUniversePlots(root)
        visualizer = PortfolioVisualizer(universe)

        bar = visualizer.plot_weights_bar("hrp_style")
        frontier = visualizer.plot_efficient_frontier("markowitz_max_sharpe")
        drawdown = visualizer.plot_drawdown("hrp_style")

        max_weight = max(bar.data[0]["y"])
        assert bar.layout.yaxis.range[1] > max_weight

        frontier_returns = np.concatenate([np.asarray(trace["y"], dtype=float) for trace in frontier.data if len(trace["y"])])
        assert frontier.layout.yaxis.range[1] > float(frontier_returns.max())

        drawdown_values = np.asarray(drawdown.data[0]["y"], dtype=float)
        assert drawdown.layout.yaxis.range[0] < float(drawdown_values.min())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_visualizer_restricts_efficient_frontier_to_markowitz() -> None:
    root = fresh_test_root("frontier_scope")
    try:
        universe = DummyUniversePlots(root)
        visualizer = PortfolioVisualizer(universe)

        with pytest.raises(ValueError, match="only available for Markowitz"):
            visualizer.plot_efficient_frontier("hrp_style")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_save_everything_includes_universe_heatmaps() -> None:
    root = fresh_test_root("save_everything_heatmaps")
    try:
        universe = DummyUniversePlots(root)
        visualizer = PortfolioVisualizer(universe)

        saved = visualizer.save_everything(max_mc_paths=3)

        assert saved["universe_plots"] == [
            "correlation_heatmap.html",
            "covariance_heatmap.html",
        ]
        assert (universe.plots_dir / "comparisons" / "correlation_heatmap.html").exists()
        assert (universe.plots_dir / "comparisons" / "covariance_heatmap.html").exists()
        assert (universe.plots_dir / "constructions" / "hrp_style" / "hrp_dendrogram.html").exists()
        assert (universe.plots_dir / "constructions" / "markowitz_max_sharpe" / "efficient_frontier.html").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_mc_terminal_comparison_plot_uses_comparisons_subfolder() -> None:
    root = fresh_test_root("mc_comparison_plot")
    try:
        run_dir = root / "demo_run"
        terminal_values = pd.DataFrame(
            {
                "equal_weight": [0.9, 1.0, 1.1],
                "markowitz": [0.8, 1.1, 1.3],
            }
        )

        output_path = save_mc_terminal_comparison_plot(run_dir, "demo_run", terminal_values)

        assert output_path == run_dir / "plots" / "comparisons" / "mc_terminal_comparison.html"
        assert output_path.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)
