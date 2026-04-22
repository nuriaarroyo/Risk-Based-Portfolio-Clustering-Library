from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portafolios.plots import PortfolioVisualizer
from scripts.run_final_experimental_setup import save_mc_terminal_comparison_plot


class DummyUniversePaths:
    def __init__(self, root: Path) -> None:
        self.output_dir = root / "run"
        self.constructions_dir = self.output_dir / "constructions"
        self.backtests_dir = self.output_dir / "backtests"
        self.monte_carlo_dir = self.output_dir / "monte_carlo"
        self.plots_dir = self.output_dir / "plots"
        self._construction_names = ["equal_weight"]
        for path in (
            self.output_dir,
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
