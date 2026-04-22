from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class PortfolioVisualizer:
    """
    Plotly visualization layer for a universe with saved constructions.
    """

    _PLOT_DIRS = {
        "construction": "constructions",
        "backtest": "backtests",
        "monte_carlo": "monte_carlo",
        "comparison": "comparisons",
    }

    _LEGACY_PLOT_FILES = {
        "construction": ("weights_bar.html", "weights_pie.html", "weights_scatter.html"),
        "backtest": ("backtest.html",),
        "monte_carlo": ("mc_paths.html", "mc_distribution.html"),
    }

    def __init__(self, universe) -> None:
        self.universe = universe

    def plot_weights_bar(
        self,
        construction_name: str,
        drop_zero: bool = True,
        sort: bool = True,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        weights = self._get_weights(construction_name, drop_zero=drop_zero, sort=sort)
        fig = px.bar(
            x=weights.index,
            y=weights.values,
            labels={"x": "Asset", "y": "Weight"},
            title=f"Weights Bar - {construction_name}",
        )
        fig.update_traces(text=[f"{value:.2%}" for value in weights.values], textposition="outside")
        fig.update_layout(showlegend=False)
        self._maybe_save_html(fig, save_html=save_html, filename=filename, kind="construction", construction_name=construction_name, default_filename="weights_bar.html")
        return fig

    def plot_weights_pie(
        self,
        construction_name: str,
        drop_zero: bool = True,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        weights = self._get_weights(construction_name, drop_zero=drop_zero, sort=True)
        fig = px.pie(
            names=weights.index,
            values=weights.values,
            title=f"Weights Pie - {construction_name}",
            hole=0.3,
        )
        self._maybe_save_html(fig, save_html=save_html, filename=filename, kind="construction", construction_name=construction_name, default_filename="weights_pie.html")
        return fig

    def plot_weights_scatter(
        self,
        construction_name: str,
        drop_zero: bool = True,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        weights = self._get_weights(construction_name, drop_zero=drop_zero, sort=True)
        fig = px.scatter(
            x=weights.index,
            y=weights.values,
            size=np.abs(weights.values),
            labels={"x": "Asset", "y": "Weight", "size": "Abs Weight"},
            title=f"Weights Scatter - {construction_name}",
        )
        fig.update_traces(mode="markers+text", text=weights.index, textposition="top center")
        self._maybe_save_html(fig, save_html=save_html, filename=filename, kind="construction", construction_name=construction_name, default_filename="weights_scatter.html")
        return fig

    def plot_backtest(
        self,
        construction_name: str,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        backtest = self._get_backtest(construction_name)
        series = self._slice_series(backtest.cumulative_returns, start_date=start_date, end_date=end_date)
        series = 1.0 + series
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=construction_name,
            )
        )
        fig.update_layout(
            title=f"Backtest Wealth - {construction_name}",
            xaxis_title="Date",
            yaxis_title="Wealth",
        )
        self._maybe_save_html(fig, save_html=save_html, filename=filename, kind="backtest", construction_name=construction_name, default_filename="backtest.html")
        return fig

    def plot_backtest_comparison(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        available = self._available_backtests()
        if not available:
            raise ValueError("No hay backtests guardados en el universe.")

        wealth_series: dict[str, pd.Series] = {}
        for name, backtest in available.items():
            wealth = (1.0 + backtest.portfolio_returns).cumprod()
            wealth = self._slice_series(wealth, start_date=start_date, end_date=end_date)
            wealth.name = name
            wealth_series[name] = wealth.sort_index()

        comparison_df = pd.concat(wealth_series.values(), axis=1, join="outer").sort_index()
        if comparison_df.empty:
            raise ValueError("No hay datos suficientes para comparar backtests.")

        fig = go.Figure()
        for column in comparison_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_df.index,
                    y=comparison_df[column],
                    mode="lines",
                    name=str(column),
                    line=dict(width=2),
                    connectgaps=False,
                )
            )

        fig.update_layout(
            title="Backtest Comparison - Cumulative Wealth",
            xaxis_title="Date",
            yaxis_title="Cumulative Wealth (start = 1.0)",
            legend=dict(
                title="Construction",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            hovermode="x unified",
            template="plotly_white",
        )

        self._maybe_save_html(
            fig,
            save_html=save_html,
            filename=filename,
            kind="comparison",
            construction_name=None,
            default_filename="backtest_comparison.html",
        )
        return fig

    def plot_mc_paths(
        self,
        construction_name: str,
        max_paths: int = 100,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        mc = self._get_mc(construction_name)
        paths = self._paths_to_frame(mc.simulated_paths)
        if max_paths > 0:
            paths = paths.iloc[:, :max_paths]

        fig = go.Figure()
        for column in paths.columns:
            fig.add_trace(
                go.Scatter(
                    x=paths.index,
                    y=paths[column],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.35,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            title=f"Monte Carlo Paths - {construction_name}",
            xaxis_title="Step",
            yaxis_title="Portfolio Value",
        )
        self._maybe_save_html(fig, save_html=save_html, filename=filename, kind="monte_carlo", construction_name=construction_name, default_filename="mc_paths.html")
        return fig

    def plot_mc_distribution(
        self,
        construction_name: str,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        mc = self._get_mc(construction_name)
        terminal_values = np.asarray(mc.terminal_values, dtype=float)
        if terminal_values.size == 0:
            raise ValueError(f"La construccion '{construction_name}' no tiene valores terminales para graficar.")

        mean_value = float(np.mean(terminal_values))
        median_value = float(np.median(terminal_values))
        p05 = float(np.percentile(terminal_values, 5))
        p95 = float(np.percentile(terminal_values, 95))

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=terminal_values,
                nbinsx=min(60, max(20, terminal_values.size // 15)),
                histnorm="probability density",
                name="Terminal values",
                marker=dict(color="#4C78A8", opacity=0.72),
                opacity=0.85,
            )
        )

        kde_x, kde_y = self._kde_curve(terminal_values)
        if kde_x is not None and kde_y is not None:
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name="Density",
                    line=dict(color="#F58518", width=3),
                )
            )

        self._add_reference_line(fig, mean_value, "Mean", "#E45756")
        self._add_reference_line(fig, median_value, "Median", "#72B7B2")
        self._add_reference_line(fig, p05, "5th pct", "#54A24B")
        self._add_reference_line(fig, p95, "95th pct", "#B279A2")

        fig.update_layout(
            title=f"Monte Carlo Terminal Distribution - {construction_name}",
            xaxis_title="Terminal Value",
            yaxis_title="Density",
            bargap=0.05,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        self._maybe_save_html(
            fig,
            save_html=save_html,
            filename=filename,
            kind="monte_carlo",
            construction_name=construction_name,
            default_filename="mc_distribution.html",
        )
        return fig

    def _get_construction(self, construction_name: str):
        return self.universe.get_construction(construction_name)

    def _get_weights(self, construction_name: str, *, drop_zero: bool, sort: bool) -> pd.Series:
        construction = self._get_construction(construction_name)
        weights = construction.weights.copy()
        if drop_zero:
            weights = weights[weights != 0]
        if weights.empty:
            raise ValueError(f"La construccion '{construction_name}' no tiene pesos para graficar.")
        if sort:
            weights = weights.sort_values(ascending=False)
        return weights

    def _get_backtest(self, construction_name: str):
        construction = self._get_construction(construction_name)
        if construction.backtest_result is None:
            raise ValueError(f"La construccion '{construction_name}' no tiene BacktestResult adjunto.")
        return construction.backtest_result

    def _get_mc(self, construction_name: str):
        construction = self._get_construction(construction_name)
        if construction.mc_result is None:
            raise ValueError(f"La construccion '{construction_name}' no tiene MonteCarloResult adjunto.")
        return construction.mc_result

    def _available_backtests(self) -> dict[str, Any]:
        return {
            name: result.backtest_result
            for name, result in self.universe.constructions.items()
            if result.backtest_result is not None
        }

    def _slice_series(
        self,
        series: pd.Series,
        *,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> pd.Series:
        out = series.copy()
        if start_date is not None:
            out = out.loc[out.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            out = out.loc[out.index <= pd.Timestamp(end_date)]
        if out.empty:
            raise ValueError("No hay datos del backtest en la ventana solicitada.")
        return out

    def _paths_to_frame(self, paths: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(paths, pd.DataFrame):
            return paths
        arr = np.asarray(paths, dtype=float)
        columns = [f"path_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)

    def _kde_curve(self, values: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        n = values.size
        if n < 2:
            return None, None

        std = float(np.std(values, ddof=1))
        if std <= 0:
            return None, None

        bandwidth = 1.06 * std * (n ** (-1 / 5))
        if bandwidth <= 0:
            return None, None

        x = np.linspace(values.min(), values.max(), 300)
        z = (x[:, None] - values[None, :]) / bandwidth
        density = np.exp(-0.5 * z**2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
        return x, density

    def _add_reference_line(self, fig: go.Figure, x_value: float, label: str, color: str) -> None:
        fig.add_vline(x=x_value, line_dash="dash", line_color=color, line_width=2)
        fig.add_annotation(
            x=x_value,
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=0,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            font=dict(color=color),
        )

    def save_all_construction_plots(self) -> dict[str, list[str]]:
        saved: dict[str, list[str]] = {}
        for construction_name in self.universe.list_constructions():
            self.plot_weights_bar(construction_name, save_html=True)
            self.plot_weights_pie(construction_name, save_html=True)
            self.plot_weights_scatter(construction_name, save_html=True)
            saved[construction_name] = [
                "weights_bar.html",
                "weights_pie.html",
                "weights_scatter.html",
            ]
        return saved

    def save_all_backtest_plots(self) -> dict[str, list[str]]:
        saved: dict[str, list[str]] = {}
        available = self._available_backtests()
        for construction_name in available:
            self.plot_backtest(construction_name, save_html=True)
            saved[construction_name] = ["backtest.html"]

        if available:
            self.plot_backtest_comparison(save_html=True)
            saved["_comparison"] = ["backtest_comparison.html"]

        return saved

    def save_all_monte_carlo_plots(self, max_paths: int = 100) -> dict[str, list[str]]:
        saved: dict[str, list[str]] = {}
        for construction_name, result in self.universe.constructions.items():
            if result.mc_result is None:
                continue
            self.plot_mc_paths(construction_name, max_paths=max_paths, save_html=True)
            self.plot_mc_distribution(construction_name, save_html=True)
            saved[construction_name] = [
                "mc_paths.html",
                "mc_distribution.html",
            ]
        return saved

    def save_everything(self, max_mc_paths: int = 100) -> dict[str, Any]:
        market_data_dir = self.universe.data_dir
        self.universe.save_market_data()
        constructions = self.universe.save_all_constructions()
        backtests = self.universe.save_all_backtests()
        monte_carlo = self.universe.save_all_monte_carlo()
        self.cleanup_legacy_plot_outputs()
        construction_plots = self.save_all_construction_plots()
        backtest_plots = self.save_all_backtest_plots()
        monte_carlo_plots = self.save_all_monte_carlo_plots(max_paths=max_mc_paths)

        return {
            "market_data_dir": market_data_dir,
            "constructions": constructions,
            "backtests": backtests,
            "monte_carlo": monte_carlo,
            "construction_plots": construction_plots,
            "backtest_plots": backtest_plots,
            "monte_carlo_plots": monte_carlo_plots,
        }

    def cleanup_legacy_plot_outputs(self) -> None:
        for construction_name in self.universe.list_constructions():
            for filename in self._LEGACY_PLOT_FILES["construction"]:
                (self.universe.get_construction_dir(construction_name) / filename).unlink(missing_ok=True)
            for filename in self._LEGACY_PLOT_FILES["backtest"]:
                (self.universe.get_backtest_dir(construction_name) / filename).unlink(missing_ok=True)
            for filename in self._LEGACY_PLOT_FILES["monte_carlo"]:
                (self.universe.get_mc_dir(construction_name) / filename).unlink(missing_ok=True)

        legacy_comparison_dir = self.universe.get_plot_dir()
        (legacy_comparison_dir / "backtest_comparison.html").unlink(missing_ok=True)

    def _maybe_save_html(
        self,
        fig: go.Figure,
        *,
        save_html: bool,
        filename: str | None,
        kind: str,
        construction_name: str | None,
        default_filename: str,
    ) -> None:
        if not save_html:
            return

        output_path = self._resolve_save_path(
            kind=kind,
            construction_name=construction_name,
            filename=filename or default_filename,
        )
        fig.write_html(str(output_path))

    def _resolve_save_path(
        self,
        *,
        kind: str,
        construction_name: str | None,
        filename: str,
    ):
        if kind == "construction":
            if construction_name is None:
                raise ValueError("`construction_name` is required to save construction plots.")
            base_dir = self.universe.get_plot_dir(
                category=self._PLOT_DIRS[kind],
                construction_name=construction_name,
            )
        elif kind == "backtest":
            if construction_name is None:
                raise ValueError("`construction_name` is required to save backtest plots.")
            base_dir = self.universe.get_plot_dir(
                category=self._PLOT_DIRS[kind],
                construction_name=construction_name,
            )
        elif kind == "monte_carlo":
            if construction_name is None:
                raise ValueError("`construction_name` is required to save Monte Carlo plots.")
            base_dir = self.universe.get_plot_dir(
                category=self._PLOT_DIRS[kind],
                construction_name=construction_name,
            )
        elif kind == "comparison":
            base_dir = self.universe.get_plot_dir(category=self._PLOT_DIRS[kind])
        else:
            raise ValueError(f"Unknown plot kind: {kind}")

        return base_dir / filename
