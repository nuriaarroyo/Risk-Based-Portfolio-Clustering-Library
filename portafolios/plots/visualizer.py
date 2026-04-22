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

    _DISCRETE_COLORS = (
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#000000",  # black
    )
    _TEXT_COLOR = "#243447"
    _GRID_COLOR = "#D9E2EC"
    _BACKGROUND_COLOR = "#FFFFFF"
    _CORRELATION_SCALE = (
        (0.0, "#D55E00"),
        (0.25, "#F1B574"),
        (0.5, "#F7F7F7"),
        (0.75, "#8CC8E8"),
        (1.0, "#0072B2"),
    )
    _SEQUENTIAL_SCALE = (
        (0.0, "#F7FBFF"),
        (0.3, "#CFE8F3"),
        (0.6, "#56B4E9"),
        (1.0, "#0072B2"),
    )

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
        colors = self._colors_for_items(weights.index)
        fig = px.bar(
            x=weights.index,
            y=weights.values,
            labels={"x": "Asset", "y": "Weight"},
            title=f"Weights Bar - {construction_name}",
        )
        fig.update_traces(
            marker=dict(color=colors, line=dict(color=self._BACKGROUND_COLOR, width=1)),
            text=[f"{value:.2%}" for value in weights.values],
            textposition="outside",
            textfont=dict(color=self._TEXT_COLOR),
            hovertemplate="Asset=%{x}<br>Weight=%{y:.2%}<extra></extra>",
        )
        self._apply_base_layout(
            fig,
            title=f"Weights Bar - {construction_name}",
            xaxis_title="Asset",
            yaxis_title="Weight",
            showlegend=False,
        )
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
        colors = self._colors_for_items(weights.index)
        fig = px.pie(
            names=weights.index,
            values=weights.values,
            title=f"Weights Pie - {construction_name}",
            hole=0.3,
            color_discrete_sequence=list(colors),
        )
        fig.update_traces(
            sort=False,
            direction="clockwise",
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(color=self._TEXT_COLOR),
            outsidetextfont=dict(color=self._TEXT_COLOR),
            insidetextfont=dict(color=self._TEXT_COLOR),
            marker=dict(colors=colors, line=dict(color=self._BACKGROUND_COLOR, width=1.5)),
            hovertemplate="Asset=%{label}<br>Weight=%{percent}<extra></extra>",
        )
        self._apply_base_layout(
            fig,
            title=f"Weights Pie - {construction_name}",
            showlegend=False,
            show_grid_x=False,
            show_grid_y=False,
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
        colors = self._colors_for_items(weights.index)
        fig = px.scatter(
            x=weights.index,
            y=weights.values,
            size=np.abs(weights.values),
            labels={"x": "Asset", "y": "Weight", "size": "Abs Weight"},
            title=f"Weights Scatter - {construction_name}",
        )
        fig.update_traces(
            mode="markers+text",
            text=weights.index,
            textposition="top center",
            marker=dict(color=colors, line=dict(color=self._BACKGROUND_COLOR, width=1)),
            textfont=dict(color=self._TEXT_COLOR),
            hovertemplate="Asset=%{x}<br>Weight=%{y:.2%}<extra></extra>",
        )
        self._apply_base_layout(
            fig,
            title=f"Weights Scatter - {construction_name}",
            xaxis_title="Asset",
            yaxis_title="Weight",
            showlegend=False,
        )
        self._maybe_save_html(fig, save_html=save_html, filename=filename, kind="construction", construction_name=construction_name, default_filename="weights_scatter.html")
        return fig

    def plot_weights_bubble(
        self,
        construction_name: str,
        drop_zero: bool = True,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        weights = self._get_weights(construction_name, drop_zero=drop_zero, sort=False)
        returns_frame = self._get_construction_returns(construction_name)
        colors = self._colors_for_items(weights.index)
        asset_stats = pd.DataFrame(
            {
                "asset": weights.index,
                "weight": weights.values,
                "abs_weight": np.abs(weights.values),
                "expected_return": returns_frame[weights.index].mean().values,
                "volatility": returns_frame[weights.index].std().values,
            }
        )
        asset_stats["sharpe_like"] = asset_stats["expected_return"] / asset_stats["volatility"].replace(0, np.nan)

        fig = px.scatter(
            asset_stats,
            x="expected_return",
            y="volatility",
            size="abs_weight",
            text="asset",
            hover_data={
                "asset": False,
                "weight": ":.2%",
                "expected_return": ":.4f",
                "volatility": ":.4f",
                "sharpe_like": ":.3f",
                "abs_weight": False,
            },
            title=f"Weights Bubble - {construction_name}",
            labels={
                "expected_return": "Expected Return",
                "volatility": "Volatility",
                "abs_weight": "Abs Weight",
            },
            size_max=60,
        )
        fig.update_traces(
            mode="markers+text",
            textposition="top center",
            marker=dict(color=colors, line=dict(color=self._BACKGROUND_COLOR, width=1)),
            textfont=dict(color=self._TEXT_COLOR),
        )
        self._apply_base_layout(
            fig,
            title=f"Weights Bubble - {construction_name}",
            xaxis_title="Expected Return",
            yaxis_title="Volatility",
            showlegend=False,
        )
        self._maybe_save_html(
            fig,
            save_html=save_html,
            filename=filename,
            kind="construction",
            construction_name=construction_name,
            default_filename="weights_bubble.html",
        )
        return fig

    def plot_correlation_heatmap(
        self,
        kind: str = "correlation",
        round_decimals: int = 2,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        matrix = self._get_matrix(kind)
        z = matrix.values.astype(float)
        if kind == "correlation":
            zmin, zmax = -1.0, 1.0
            title = "Correlation Heatmap"
            colorscale = self._CORRELATION_SCALE
        else:
            max_abs = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
            zmin, zmax = -max_abs, max_abs
            title = "Covariance Heatmap"
            colorscale = self._CORRELATION_SCALE

        text = np.vectorize(lambda value: f"{value:.{round_decimals}f}")(z)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                text=text,
                hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
                colorbar=dict(title=kind.title()),
            )
        )
        self._apply_base_layout(
            title=title,
            fig=fig,
            xaxis_title="Asset",
            yaxis_title="Asset",
            show_grid_x=False,
            show_grid_y=False,
        )
        self._maybe_save_html(
            fig,
            save_html=save_html,
            filename=filename,
            kind="comparison",
            construction_name=None,
            default_filename=f"{kind}_heatmap.html",
        )
        return fig

    def plot_hrp_distance_matrix(
        self,
        construction_name: str,
        reorder_clusters: bool = True,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        diagnostics = self._get_hrp_diagnostics(construction_name)
        matrix = diagnostics.distance_matrix.copy()
        if reorder_clusters and diagnostics.clusters:
            ordered = [asset for cluster in diagnostics.clusters for asset in cluster if asset in matrix.index]
            if ordered:
                matrix = matrix.loc[ordered, ordered]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values.astype(float),
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale=self._SEQUENTIAL_SCALE,
                colorbar=dict(title="Distance"),
            )
        )
        self._apply_base_layout(
            fig=fig,
            title=f"HRP Distance Matrix - {construction_name}",
            xaxis_title="Asset",
            yaxis_title="Asset",
            show_grid_x=False,
            show_grid_y=False,
        )
        self._maybe_save_html(
            fig,
            save_html=save_html,
            filename=filename,
            kind="construction",
            construction_name=construction_name,
            default_filename="hrp_distance_matrix.html",
        )
        return fig

    def plot_hrp_distance_histogram(
        self,
        construction_name: str,
        bins: int = 40,
        save_html: bool = False,
        filename: str | None = None,
    ) -> go.Figure:
        diagnostics = self._get_hrp_diagnostics(construction_name)
        distance_matrix = diagnostics.distance_matrix.values.astype(float)
        triu = np.triu_indices_from(distance_matrix, k=1)
        distances = distance_matrix[triu]
        if distances.size == 0:
            raise ValueError(f"The HRP diagnostics for '{construction_name}' do not include pairwise distances.")

        fig = go.Figure(
            data=go.Histogram(
                x=distances,
                nbinsx=max(10, bins),
                marker=dict(color=self._DISCRETE_COLORS[0], opacity=0.8),
            )
        )
        self._apply_base_layout(
            fig=fig,
            title=f"HRP Distance Histogram - {construction_name}",
            xaxis_title="Distance",
            yaxis_title="Count",
            bargap=0.05,
        )
        self._maybe_save_html(
            fig,
            save_html=save_html,
            filename=filename,
            kind="construction",
            construction_name=construction_name,
            default_filename="hrp_distance_histogram.html",
        )
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
                line=dict(color=self._DISCRETE_COLORS[0], width=3),
            )
        )
        self._apply_base_layout(
            fig=fig,
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
        colors = self._colors_for_items(comparison_df.columns)
        for idx, column in enumerate(comparison_df.columns):
            fig.add_trace(
                go.Scatter(
                    x=comparison_df.index,
                    y=comparison_df[column],
                    mode="lines",
                    name=str(column),
                    line=dict(color=colors[idx], width=2.5),
                    connectgaps=False,
                )
            )

        self._apply_base_layout(
            fig=fig,
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
        path_color = self._rgba(self._DISCRETE_COLORS[0], 0.18)
        for column in paths.columns:
            fig.add_trace(
                go.Scatter(
                    x=paths.index,
                    y=paths[column],
                    mode="lines",
                    line=dict(color=path_color, width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if not paths.empty:
            fig.add_trace(
                go.Scatter(
                    x=paths.index,
                    y=paths.median(axis=1),
                    mode="lines",
                    name="Median path",
                    line=dict(color=self._DISCRETE_COLORS[1], width=3),
                )
            )

        self._apply_base_layout(
            fig=fig,
            title=f"Monte Carlo Paths - {construction_name}",
            xaxis_title="Step",
            yaxis_title="Portfolio Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
                marker=dict(color=self._DISCRETE_COLORS[0], opacity=0.72),
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
                    line=dict(color=self._DISCRETE_COLORS[1], width=3),
                )
            )

        self._add_reference_line(fig, mean_value, "Mean", self._DISCRETE_COLORS[1])
        self._add_reference_line(fig, median_value, "Median", self._DISCRETE_COLORS[2])
        self._add_reference_line(fig, p05, "5th pct", self._DISCRETE_COLORS[3])
        self._add_reference_line(fig, p95, "95th pct", self._DISCRETE_COLORS[4])

        self._apply_base_layout(
            fig=fig,
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

    def _get_hrp_diagnostics(self, construction_name: str):
        construction = self._get_construction(construction_name)
        diagnostics = construction.hrp_diagnostics
        if diagnostics is None:
            raise ValueError(f"The construction '{construction_name}' does not include HRP diagnostics.")
        return diagnostics

    def _available_backtests(self) -> dict[str, Any]:
        return {
            name: result.backtest_result
            for name, result in self.universe.constructions.items()
            if result.backtest_result is not None
        }

    def _get_construction_returns(self, construction_name: str) -> pd.DataFrame:
        construction = self._get_construction(construction_name)
        if hasattr(self.universe, "get_returns_window"):
            return self.universe.get_returns_window(
                construction.construction_start,
                construction.construction_end,
            )

        returns_frame = getattr(self.universe, "asset_returns", None)
        if returns_frame is None:
            raise ValueError("The universe does not expose returns needed for construction plots.")

        out = returns_frame.copy()
        if construction.construction_start is not None:
            out = out.loc[out.index >= construction.construction_start]
        if construction.construction_end is not None:
            out = out.loc[out.index <= construction.construction_end]
        if out.empty:
            raise ValueError(f"The construction window for '{construction_name}' has no returns available.")
        return out

    def _get_matrix(self, kind: str) -> pd.DataFrame:
        if kind == "correlation":
            matrix = getattr(self.universe, "correlation", None)
            if matrix is None:
                asset_log_returns = getattr(self.universe, "asset_log_returns", None)
                asset_returns = getattr(self.universe, "asset_returns", None)
                if asset_log_returns is not None:
                    matrix = asset_log_returns.corr()
                elif asset_returns is not None:
                    matrix = asset_returns.corr()
        elif kind == "covariance":
            matrix = getattr(self.universe, "covariance", None)
            if matrix is None:
                asset_log_returns = getattr(self.universe, "asset_log_returns", None)
                asset_returns = getattr(self.universe, "asset_returns", None)
                if asset_log_returns is not None:
                    matrix = asset_log_returns.cov()
                elif asset_returns is not None:
                    matrix = asset_returns.cov()
        else:
            raise ValueError("`kind` must be 'correlation' or 'covariance'.")

        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            raise ValueError(f"No {kind} matrix is available to plot.")
        return matrix.loc[matrix.index, matrix.index]

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

    def _colors_for_items(self, items) -> list[str]:
        colors = list(self._DISCRETE_COLORS)
        return [colors[idx % len(colors)] for idx, _ in enumerate(items)]

    def _apply_base_layout(
        self,
        fig: go.Figure,
        *,
        title: str,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        showlegend: bool | None = None,
        legend: dict[str, Any] | None = None,
        hovermode: str | None = None,
        show_grid_x: bool = True,
        show_grid_y: bool = True,
        bargap: float | None = None,
    ) -> None:
        layout_kwargs: dict[str, Any] = {
            "title": title,
            "template": "plotly_white",
            "paper_bgcolor": self._BACKGROUND_COLOR,
            "plot_bgcolor": self._BACKGROUND_COLOR,
            "font": dict(color=self._TEXT_COLOR),
            "colorway": list(self._DISCRETE_COLORS),
        }
        if xaxis_title is not None:
            layout_kwargs["xaxis_title"] = xaxis_title
        if yaxis_title is not None:
            layout_kwargs["yaxis_title"] = yaxis_title
        if showlegend is not None:
            layout_kwargs["showlegend"] = showlegend
        if legend is not None:
            layout_kwargs["legend"] = legend
        if hovermode is not None:
            layout_kwargs["hovermode"] = hovermode
        if bargap is not None:
            layout_kwargs["bargap"] = bargap
        fig.update_layout(**layout_kwargs)
        fig.update_xaxes(showgrid=show_grid_x, gridcolor=self._GRID_COLOR, zerolinecolor=self._GRID_COLOR)
        fig.update_yaxes(showgrid=show_grid_y, gridcolor=self._GRID_COLOR, zerolinecolor=self._GRID_COLOR)

    def _rgba(self, color: str, alpha: float) -> str:
        color = color.lstrip("#")
        if len(color) != 6:
            raise ValueError(f"Expected a 6-digit hex color, got '{color}'.")
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
        return f"rgba({red}, {green}, {blue}, {alpha})"

    def save_all_construction_plots(self) -> dict[str, list[str]]:
        saved: dict[str, list[str]] = {}
        for construction_name in self.universe.list_constructions():
            self.plot_weights_bar(construction_name, save_html=True)
            self.plot_weights_pie(construction_name, save_html=True)
            self.plot_weights_scatter(construction_name, save_html=True)
            self.plot_weights_bubble(construction_name, save_html=True)
            saved[construction_name] = [
                "weights_bar.html",
                "weights_pie.html",
                "weights_scatter.html",
                "weights_bubble.html",
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
